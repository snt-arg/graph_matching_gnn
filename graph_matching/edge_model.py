#!/usr/bin/env python3
"""Importable home for the edge-feature matching architecture and its inference path.

Why this file exists: the architecture was defined inside
``pgm_training_adj_glob_edgefeat.py``, which has **no ``if __name__ == "__main__"`` guard** --
importing it starts deserialising datasets at module scope and dies on a missing
``GNN/preprocessed/graph_matching/equal/original.pkl``. So neither the dashboard nor
``pose_sweep.py`` could reach the class. The class/helper bodies below are lifted from that
script VERBATIM (see the per-block provenance comments); the training script is left untouched,
so it remains the authority on how the weights were produced.

The class is renamed ``MatchingModel_EdgeAwareGATv2SinkhornWBCE``. In the training script it is
called ``MatchingModel_MLPGATv2SinkhornWBCE`` -- the *same name* as the unrelated 7-feature class
in ``PGM_class.py``, but with a different signature (``edge_dim``/``edge_hidden_dim``/
``attention_dropout`` vs ``attn_dropout``) and a different ``forward`` contract. Two different
architectures under one name is how a checkpoint gets loaded into the wrong model, so the name is
made unambiguous here.

What distinguishes this model from the ``PGM_class`` family:
  * node ``x`` is 3-dim ``[type_onehot(2), length_or_-1(1)]`` -- ``center``/``normal`` are
    deliberately absent (arXiv:2409.11972 Sec III-B), which is what removes the absolute-coordinate
    contamination path;
  * all relative geometry lives in a 9-dim ``edge_attr``
    ``[d_ij, cos_phi, sin_phi, cos_alpha, sin_alpha, ROOM_ROOM, ROOM_WS, WS_WS_INTRA, WS_WS_INTER]``
    built by ``edge_features.build_edge_index_and_attr`` and re-projected at every hop;
  * normalisation therefore needs FOUR tensors (``mean``/``std`` for nodes, ``edge_mean``/
    ``edge_std`` for edges), not two. ``EDGE_NORM_COLS == (0,)``, so only ``d_ij`` is z-scored.

``forward`` returns ``(perm_pred_list, all_embeddings)`` where ``perm_pred_list[0]`` is the SOFT
Sinkhorn matrix -- it takes no ``return_soft``/``return_intermediate`` kwargs and ignores
``inference``. ``EdgeMatcher`` below adapts that to the two call contracts the existing consumers
expect.
"""

import json
import os

import networkx as nx
import numpy as np
import pygmtools
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from edge_features import (EDGE_ATTR_DIM, build_edge_index_and_attr,
                           node_features, normalize_edge_attr)

pygmtools.BACKEND = "pytorch"

# Lifted from pgm_training_adj_glob_edgefeat.py:365
node_type_mapping = {"room": [1, 0], "ws": [0, 1]}


# ─── verbatim from pgm_training_adj_glob_edgefeat.py:1149-1168 ───────────────────────────
def _real_nx_to_pyg(graph: nx.DiGraph) -> Data:
    """
    Convert a real (Prior/Online) DiGraph to a PyG Data object with the same reduced
    feature layout as the training graphs: node x = [type(2), length_or_-1(1)],
    edge_attr = the 9-column signed schema from edge_features.py (5 geometry columns
    plus the 4-way edge-type one-hot), including the
    mechanism-1 (wall-anchored room-room) and mechanism-2 (cross-room ws-ws) edges,
    which build_edge_index_and_attr adds itself.
    Original (string) node ids are kept in `node_names` so the ground truth can be applied.
    """
    node_ids = [str(n) for n in graph.nodes()]
    raw_id_map = {n: i for i, n in enumerate(graph.nodes())}  # keyed by raw node objects, for edge_features.py

    x = node_features(graph, node_type_mapping)
    edge_index, edge_attr = build_edge_index_and_attr(graph, raw_id_map)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_names = node_ids
    data.permutation = torch.arange(len(node_ids), dtype=torch.long)
    return data

# ─── verbatim from pgm_training_adj_glob_edgefeat.py:1171-1182 ───────────────────────────
def deserialize_norm_stats(path: str, filename: str = "norm_stats.pt"):
    """
    Load the per-feature (mean, std) saved next to a dataset by dataset_gen.py.
    Returns None if the file is missing (older datasets) so the caller can warn.
    """
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        return None
    stats = torch.load(full_path, map_location="cpu")
    # edge_mean/edge_std are absent in datasets built before edge_attr normalization existed;
    # returning None for them lets the caller fall back to raw (unscaled) edge features.
    return stats["mean"], stats["std"], stats.get("edge_mean"), stats.get("edge_std")

# ─── verbatim from pgm_training_adj_glob_edgefeat.py:1531-1562 ───────────────────────────
class EdgeAwareGATLayer(nn.Module):
    """
    One message-passing hop implementing arXiv:2409.11972 Eq. 4 (node update) + Eq. 5 (edge
    update): v_i^{l+1} = g_v([v_i^l, maxpool_{j in N(i)} GATH(v_i^l, e_ij^l, v_j^l)]);
    e_ij^{l+1} = g_e([v_i^l, e_ij^l, v_j^l]). Both g_v/g_e consume LAYER-INPUT embeddings
    (matching the paper's simultaneity -- neither depends on the other's output within a hop).

    GATH + the max-pool is realized via GATv2Conv(edge_dim=..., aggr='max') directly: PyG's
    GATv2Conv already accepts edge-conditioned multi-head attention (the "H" in GATH is the
    existing `heads` hyperparameter) and forwards `aggr` through **kwargs to MessagePassing,
    replacing the default sum-aggregation with max -- verified this session by running it and
    confirming gradients flow into lin_l/lin_r/lin_edge/att/bias. This avoids hand-rolling
    GATv2's attention math while still updating edge embeddings every hop (unlike a plain
    static-edge_dim shortcut, which never transforms edge_attr across layers).
    """
    def __init__(self, node_in_dim, node_out_dim, edge_in_dim, edge_out_dim, heads=1, attn_dropout=0.0):
        super().__init__()
        self.gath = GATv2Conv(node_in_dim, node_out_dim, heads=heads, concat=False,
                               edge_dim=edge_in_dim, dropout=attn_dropout, aggr='max')
        self.g_v = nn.Sequential(nn.Linear(node_in_dim + node_out_dim, node_out_dim), nn.ReLU())
        self.g_e = nn.Sequential(nn.Linear(node_in_dim + edge_in_dim + node_in_dim, edge_out_dim), nn.ReLU())

    def forward(self, x, edge_index, edge_attr, update_edge=True):
        agg = self.gath(x, edge_index, edge_attr=edge_attr)
        x_new = self.g_v(torch.cat([x, agg], dim=-1))
        if not update_edge:
            # Last hop: edge_attr won't be consumed by any further layer (encode() discards
            # it after the loop), so skip g_e entirely rather than compute-and-discard.
            return x_new, None
        src, dst = edge_index[0], edge_index[1]  # dst == i (target/self), matches edge_features.py's swap
        edge_new = self.g_e(torch.cat([x[dst], edge_attr, x[src]], dim=-1))
        return x_new, edge_new

# ─── verbatim from pgm_training_adj_glob_edgefeat.py:1565-1671, class RENAMED ────────────
class MatchingModel_EdgeAwareGATv2SinkhornWBCE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, sinkhorn_max_iter: int = 10, sinkhorn_tau: float = 1.0,
                 attention_dropout: float = 0.1, dropout_emb: float = 0.1, num_layers: int = 2, heads: int = 1,
                 edge_dim: int = EDGE_ATTR_DIM, edge_hidden_dim: int = 16):
        super().__init__()
        # MLP for initial node feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_emb),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_emb)
        )
        # Project raw edge_attr (edge_dim cols) to edge_hidden_dim once, so every
        # EdgeAwareGATLayer only ever has to know about edge_hidden_dim -- mirrors how
        # self.mlp projects raw node x to hidden_dim before the GNN stack.
        self.edge_proj = nn.Linear(edge_dim, edge_hidden_dim)
        self.gnn = nn.ModuleList()
        dims = [hidden_dim] * num_layers + [out_dim]
        for i in range(num_layers):
            # Always average the heads so the feature-dim stays dims[i+1]
            self.gnn.append(
                EdgeAwareGATLayer(dims[i], dims[i+1], edge_hidden_dim, edge_hidden_dim,
                                  heads=heads, attn_dropout=attention_dropout)
            )
        self.dropout = nn.Dropout(p=dropout_emb)
        # # bilinear weight matrix A per affinity
        # std = 1.0 / math.sqrt(out_dim)
        # self.A = nn.Parameter(torch.randn(out_dim, out_dim) * std)
        # self.temperature = temperature
        # InstanceNorm per-sample
        self.inst_norm = nn.InstanceNorm2d(1, affine=True)
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.sinkhorn_tau = sinkhorn_tau

    def encode(self, x, edge_index, edge_attr):
        edge_attr = self.edge_proj(edge_attr)
        for i, layer in enumerate(self.gnn):
            is_last = i == len(self.gnn) - 1
            x, edge_attr = layer(x, edge_index, edge_attr, update_edge=not is_last)
            if not is_last:
                x = F.relu(x)
                x = self.dropout(x)
                edge_attr = F.relu(edge_attr)
                edge_attr = self.dropout(edge_attr)
        return x  # edge_attr discarded after the last hop -- folded into node embeddings by then

    def forward(self, batch1, batch2, perm_list=None, batch_idx1=None, batch_idx2=None, inference=False):
        device = next(self.parameters()).device
        x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
        x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
        edge_attr1, edge_attr2 = batch1.edge_attr.to(device), batch2.edge_attr.to(device)
        perm_list = [p.to(device) for p in perm_list] if perm_list is not None else None

        batch_idx1 = batch1.batch.to(device) if batch_idx1 is None else batch_idx1.to(device)
        batch_idx2 = batch2.batch.to(device) if batch_idx2 is None else batch_idx2.to(device)

        # Apply MLP before GNN
        h1 = self.mlp(x1)
        h2 = self.mlp(x2)
        h1 = self.encode(h1, edge1, edge_attr1)
        h2 = self.encode(h2, edge2, edge_attr2)

        B = batch_idx1.max().item() + 1
        perm_pred_list = []
        all_embeddings = []

        for b in range(B):
            h1_b = h1[batch_idx1 == b]   # [n1, d]
            h2_b = h2[batch_idx2 == b]   # [n2, d]
            N1, N2 = h1_b.size(0), h2_b.size(0)

            # affinity matrix + normalization + sinkhorn
            sim = torch.matmul(h1_b, h2_b.T) # [n1, n2]
            sim_batched = sim.unsqueeze(0).unsqueeze(1) # [1,1,n1,n2]
            sim_normed = self.inst_norm(sim_batched).squeeze(1) # [1,n1,n2]

            # g1 -> A-graph 
            # g2 -> S-graph (partial)
            transposed = N1 > N2

            if transposed:
                # traspose to use dummy_row
                sim_input = sim_normed.transpose(-2, -1)   # [1, n2, n1]
                nr = torch.tensor([N2], dtype=torch.long, device=device)
                nc = torch.tensor([N1], dtype=torch.long, device=device)
            else:
                sim_input = sim_normed                     # [1, n1, n2]
                nr = torch.tensor([N1], dtype=torch.long, device=device)
                nc = torch.tensor([N2], dtype=torch.long, device=device)

            S = pygmtools.sinkhorn(
                sim_input,
                n1=nr, n2=nc,
                dummy_row=(N1 != N2),
                max_iter=self.sinkhorn_max_iter,
                tau=self.sinkhorn_tau
            )

            if transposed:
                S = S.transpose(-2, -1)   # rollback to [1, n1, n2]

            perm_pred_list.append(S.squeeze(0))  # [n1, n2]
            all_embeddings.append((h1_b, h2_b))

        return perm_pred_list, all_embeddings


# ─── new code below: checkpoint-driven construction + the two call contracts ──────────────

def infer_hparams(state_dict):
    """Recover the architecture dims from the WEIGHTS, not from ``best_trial_results.json``.

    The JSON sidecar is not trustworthy as an architecture description: for
    ``adj_no_glob_65_edge`` it records ``heads: 7`` while every GAT layer in both of its
    checkpoints was built with ``heads=6`` (``gnn.0.gath.att`` is ``(1, 6, 16)`` and
    ``gnn.0.gath.lin_l.weight`` is ``(96, 16)`` = 6x16). Constructing from the JSON and then
    calling ``load_state_dict(strict=True)`` fails with a size mismatch on every layer -- the
    same class of defect already recorded for ``adj_glob_65``'s dataset-dump metadata.

    Everything structural is recoverable from the tensor shapes, so derive it and treat the JSON
    as authoritative only for the two Sinkhorn settings, which leave no trace in the weights.
    """
    sd = state_dict
    layer_ids = sorted({int(k.split(".")[1]) for k in sd if k.startswith("gnn.")})
    num_layers = len(layer_ids)
    last = layer_ids[-1]
    hp = {
        "in_dim":          sd["mlp.0.weight"].shape[1],
        "hidden_dim":      sd["mlp.0.weight"].shape[0],
        "heads":           sd["gnn.0.gath.att"].shape[1],
        "num_layers":      num_layers,
        "out_dim":         sd[f"gnn.{last}.g_v.0.weight"].shape[0],
        "edge_dim":        sd["edge_proj.weight"].shape[1],
        "edge_hidden_dim": sd["edge_proj.weight"].shape[0],
    }
    return {k: int(v) for k, v in hp.items()}


def _coerce_2d(graph):
    """Cut ``center``/``normal`` back to 2 components on a deep copy.

    ``build_edge_index_and_attr`` does plain arithmetic on these, and the stored real graphs mix
    plain lists with numpy arrays and sometimes carry a z component. Same coercion
    ``pose_sweep.Matcher._coerce`` and the dashboard's ``_to_2d`` already apply.
    """
    import copy as _copy
    graph = _copy.deepcopy(graph)
    for _, at in graph.nodes(data=True):
        for key in ("center", "normal"):
            if key in at:
                val = at[key]
                at[key] = (val.tolist() if hasattr(val, "tolist") else list(val))[:2]
    return graph


def _offdiag_cos(h):
    """Mean off-diagonal cosine similarity -- the embedding-collapse probe. Matches
    ``pose_sweep._offdiag_cos`` exactly so numbers stay comparable across models."""
    h = F.normalize(h, dim=1)
    M = (h @ h.T).cpu().numpy()
    n = M.shape[0]
    return float((M.sum() - np.trace(M)) / (n * n - n)) if n > 1 else 0.0


def _f1(pred, gts):
    tp = len(pred & gts)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gts) if gts else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


class EdgeMatcher:
    """Inference wrapper for the edge-feature model, speaking both existing call contracts.

    * ``match(a, s) -> (pairs, ints, a_nodes, s_nodes)`` -- the dashboard's matcher protocol
      (``GnnMatcher`` / ``DryRunMatcher``), with ``ints`` holding
      ``affinity`` / ``sim_normed`` / ``S`` / ``perm`` as numpy arrays.
    * ``evaluate(ga, gs, gt) -> (f1_perm, f1_hung, cos_a, cos_s)`` -- ``pose_sweep.Matcher``'s
      protocol, so the sweep driver can swap this in with no other change.

    **Deliberately NOT cached.** ``GnnMatcher`` keys its cache on ``(name, |V|, |E|)``, which a
    pose change does not alter -- a sweep wrapping it would silently report one F1 repeated at
    every angle (the trap recorded at the top of ``pose_sweep.py``). This model is also the one
    whose whole point is invariance, so a genuinely flat curve and a served cache look identical
    in the output. Removing the cache keeps those two distinguishable.
    """

    name = "gnn-edge"
    display = "edge-feature GNN (EdgeAwareGATv2 + Sinkhorn)"

    def __init__(self, model_dir, checkpoint="best_val_model.pt", device=None):
        self.model_dir = str(model_dir)
        self.device = device or torch.device("cpu")
        ckpt_path = os.path.join(self.model_dir, checkpoint)
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if "model_state_dict" not in ckpt:
            raise RuntimeError(
                f"{ckpt_path} has no 'model_state_dict' key (found {list(ckpt)[:6]}) -- it is "
                "not a checkpoint this loader understands.")
        sd = ckpt["model_state_dict"]
        hp = infer_hparams(sd)
        self.epoch = ckpt.get("epoch")

        params = self._read_params_json()
        for key, derived in hp.items():
            if key in params and int(params[key]) != derived:
                print(f"[WARNING] {os.path.basename(self.model_dir)}: "
                      f"best_trial_results.json says {key}={params[key]} but the checkpoint was "
                      f"built with {key}={derived}. Using the checkpoint.")
        missing = [k for k in ("sinkhorn_max_iter", "sinkhorn_tau") if k not in params]
        if missing:
            raise RuntimeError(
                f"{self.model_dir}: {missing} are not recoverable from the weights and are absent "
                "from the params JSON. Supply best_params.json.")

        self.hparams = dict(hp)
        self.hparams["sinkhorn_max_iter"] = int(params["sinkhorn_max_iter"])
        self.hparams["sinkhorn_tau"] = float(params["sinkhorn_tau"])

        self.model = MatchingModel_EdgeAwareGATv2SinkhornWBCE(
            in_dim=hp["in_dim"], hidden_dim=hp["hidden_dim"], out_dim=hp["out_dim"],
            sinkhorn_max_iter=self.hparams["sinkhorn_max_iter"],
            sinkhorn_tau=self.hparams["sinkhorn_tau"],
            attention_dropout=0.0, dropout_emb=0.0,   # eval only
            num_layers=hp["num_layers"], heads=hp["heads"],
            edge_dim=hp["edge_dim"], edge_hidden_dim=hp["edge_hidden_dim"],
        ).to(self.device)
        self.model.load_state_dict(sd, strict=True)   # strict: the extraction's correctness gate
        self.model.eval()

        stats = deserialize_norm_stats(self.model_dir)
        if stats is None:
            raise RuntimeError(
                f"{self.model_dir}/norm_stats.pt is missing. This model needs FOUR tensors "
                "(mean/std for nodes, edge_mean/edge_std for edges); there is no safe default.")
        self.mean, self.std, self.edge_mean, self.edge_std = stats
        if self.edge_mean is None or self.edge_std is None:
            raise RuntimeError(
                f"{self.model_dir}/norm_stats.pt has no edge_mean/edge_std. The model was trained "
                "on normalized edge_attr (EDGE_NORM_COLS=(0,) scales d_ij); feeding raw edge "
                "features would silently change the metre-scale of every distance it reads.")
        print(f"[INFO] EdgeMatcher: {self.model_dir} @ epoch {self.epoch} | {self.hparams}")

    def _read_params_json(self):
        for name in ("best_params.json", "best_trial_results.json"):
            path = os.path.join(self.model_dir, name)
            if os.path.exists(path):
                with open(path) as fh:
                    data = json.load(fh)
                return data.get("params", data)
        return {}

    def clear_cache(self):
        pass  # nothing is cached, by design -- see the class docstring

    def _prep(self, graph):
        data = _real_nx_to_pyg(_coerce_2d(graph))
        data.x = (data.x - self.mean) / (self.std + 1e-8)
        data.edge_attr = normalize_edge_attr(data.edge_attr, self.edge_mean, self.edge_std)
        return data.to(self.device)

    def _forward(self, ga, gs):
        """-> (S_soft, perm_hard, h1, h2) as torch tensors on self.device."""
        d1, d2 = self._prep(ga), self._prep(gs)
        b1 = torch.zeros(d1.num_nodes, dtype=torch.long, device=self.device)
        b2 = torch.zeros(d2.num_nodes, dtype=torch.long, device=self.device)
        with torch.no_grad():
            # Keyword args are mandatory: this forward is
            # (batch1, batch2, perm_list=None, batch_idx1=None, batch_idx2=None, inference=False),
            # so the positional 3rd slot is perm_list -- NOT batch_idx1 as in PGM_class's class.
            soft_list, embeddings = self.model(
                d1, d2, batch_idx1=b1, batch_idx2=b2, inference=True)
            S = soft_list[0]
            sim = S.unsqueeze(0)
            n1 = torch.tensor([sim.shape[1]], dtype=torch.int32, device=self.device)
            n2 = torch.tensor([sim.shape[2]], dtype=torch.int32, device=self.device)
            # Same hardening as the training script's own real-data path
            # (pgm_training_adj_glob_edgefeat.predict_matching_matrix, use_hungarian=True).
            perm = pygmtools.hungarian(sim, n1=n1, n2=n2).squeeze(0)
        h1, h2 = embeddings[0]
        return S, perm, h1, h2

    def match(self, a, s):
        g1, g2 = a.graph, s.graph
        S, perm, h1, h2 = self._forward(g1, g2)
        affinity_t = h1 @ h2.T
        # The model InstanceNorms the affinity internally before Sinkhorn but never returns it;
        # reuse its own (affine, trained) inst_norm so sim_normed is the real pre-Sinkhorn matrix
        # rather than a re-derived approximation.
        with torch.no_grad():
            sim_normed_t = self.model.inst_norm(
                affinity_t.unsqueeze(0).unsqueeze(1)).squeeze(1).squeeze(0)
        g1_nodes, g2_nodes = list(g1.nodes()), list(g2.nodes())
        perm_np = perm.detach().cpu().numpy()
        ints = {
            "affinity":   affinity_t.detach().cpu().numpy(),
            "sim_normed": sim_normed_t.detach().cpu().numpy(),
            "S":          S.detach().cpu().numpy(),
            "perm":       perm_np,
        }
        rows, cols = np.where(perm_np > 0)
        pairs = {(str(g1_nodes[r]), str(g2_nodes[c])) for r, c in zip(rows, cols)}
        return pairs, ints, g1_nodes, g2_nodes

    def evaluate(self, ga, gs, gt):
        """(f1_perm, f1_hung, cos_a, cos_s) -- pose_sweep.Matcher.evaluate's contract."""
        S, perm, h1, h2 = self._forward(ga, gs)
        an, sn = list(ga.nodes()), list(gs.nodes())
        P = perm.detach().cpu().numpy()
        S_np = S.detach().cpu().numpy()
        pred = {(str(an[i]), str(sn[j])) for i, j in zip(*np.nonzero(P))}
        ri, ci = linear_sum_assignment(-S_np)
        hung = {(str(an[i]), str(sn[j])) for i, j in zip(ri, ci)
                if S_np[i, j] > 1.0 / max(S_np.shape)}
        gts = {(str(x), str(y)) for x, y in gt}
        return _f1(pred, gts), _f1(hung, gts), _offdiag_cos(h1), _offdiag_cos(h2)


def is_edge_model(model_dir):
    """True when the folder's norm_stats.pt carries edge statistics -- i.e. it is an
    edge-feature checkpoint rather than one of the 7-feature node-only models."""
    path = os.path.join(str(model_dir), "norm_stats.pt")
    if not os.path.exists(path):
        return False
    try:
        return "edge_mean" in torch.load(path, map_location="cpu")
    except Exception:
        return False
