#!/usr/bin/env python3
"""Per-layer embedding-similarity probe for the EDGE-FEATURE architecture.

Reproduces, for the edge-feature models, the analysis that localised `adj_glob_65`'s failure to
its FINAL GAT layer: mean off-diagonal cosine similarity of the node embeddings after each
message-passing hop. The recorded signature of the defect was non-monotonic --
`adj_glob_65` (3 layers) on real 47_basement went L1 0.53 -> L2 0.40 (improves) -> L3 0.63
(collapses back up), i.e. the last layer actively re-homogenised the embeddings, whereas 2-layer
`adj_no_glob_65` went 0.23 -> 0.09 monotonically.

`layer_probe.py` cannot be reused here: it calls `conv(x, edge_index)`, the plain GATv2Conv
signature, while this architecture needs `layer(x, edge_index, edge_attr, update_edge=...)` after
an `edge_proj` projection, and it updates edge embeddings at every hop.

ON THE ReLU TRAP, which cost an hour the first time round: in the node-only architecture
`encode` applies ReLU+dropout to every layer EXCEPT the last, so a naive readout of the last layer
gives a PRE-activation embedding (cos 0.86 instead of 0.40) and is not comparable to the interior
layers. Here that trap does NOT bite, and the reason is worth stating rather than assuming:
`EdgeAwareGATLayer.g_v` is `Sequential(Linear, ReLU)`, so EVERY layer output -- last included --
is already non-negative. The extra `F.relu` that `encode` applies to non-last layers is therefore
idempotent, and `dropout` is the identity under `model.eval()`. All four readouts are directly
comparable as they stand. `--check-relu` asserts this instead of trusting it.

A consequence of those terminal ReLUs: embeddings live in the non-negative orthant, so off-diagonal
cosine is bounded BELOW by 0 and runs structurally higher than in the node-only models. Values here
are NOT comparable to the 0.628/0.637 recorded for adj_glob_65 -- only the SHAPE across layers is.
"""

import argparse
import copy
import csv
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/workspace/src/graph_matching/graph_matching")

import edge_model as EM   # noqa: E402
import pose_sweep as ps   # noqa: E402

MROOT = "/root/workspace/src/graph_matching_gnn/GNN/models/partial_graph_matching"
GRAPH_DICTS = "/root/workspace/src/graph_matching/graph_matching/graph_dicts"


def per_layer(model, data, check_relu=False):
    """Mean off-diagonal cosine after each hop. Mirrors encode() exactly."""
    cos, flags = [], []
    with torch.no_grad():
        x = model.mlp(data.x)
        edge_attr = model.edge_proj(data.edge_attr)
        for i, layer in enumerate(model.gnn):
            is_last = i == len(model.gnn) - 1
            x, edge_attr = layer(x, data.edge_index, edge_attr, update_edge=not is_last)
            if check_relu:
                # g_v already ends in ReLU -> the activation encode would add is a no-op
                flags.append(bool(torch.equal(torch.relu(x), x)))
            cos.append(EM._offdiag_cos(x))
            if not is_last:
                x = torch.relu(x)
                x = model.dropout(x)
                edge_attr = torch.relu(edge_attr)
                edge_attr = model.dropout(edge_attr)
    return cos, flags, x


def truncated_f1(matcher, k, ga, gs, gt):
    """F1 with the encoder read out at layer k (1-indexed).

    Shortening self.gnn genuinely makes it a k-layer net. Unlike the node-only architecture this
    needs no ReLU restoration: g_v already activated layer k's output while it was interior, so
    the truncated readout is bit-identical to the full encoder's layer-k value.
    """
    m2 = copy.deepcopy(matcher)
    m2.model = copy.deepcopy(matcher.model)
    m2.model.gnn = nn.ModuleList(list(m2.model.gnn)[:k])
    m2.model.eval()
    return m2.evaluate(ga, gs, gt)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--envs", nargs="+", required=True)
    ap.add_argument("--graph-dicts", default=GRAPH_DICTS)
    ap.add_argument("--check-relu", action="store_true")
    ap.add_argument("--trunc-f1", action="store_true",
                    help="also report F1 when reading out at each layer")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    rows = []
    for name in a.models:
        m = EM.EdgeMatcher(os.path.join(MROOT, name))
        nL = len(m.model.gnn)
        print(f"\n{'='*94}\n{name}  ({nL} layers, hidden {m.hparams['hidden_dim']}, "
              f"out {m.hparams['out_dim']}, heads {m.hparams['heads']})\n{'='*94}")
        hdr = f"{'env':<22}{'side':<6}" + "".join(f"{'L'+str(i+1):>9}" for i in range(nL)) + f"{'shape':>12}"
        print(hdr)
        for env in a.envs:
            ga, gs, gt = ps.load_env(os.path.join(a.graph_dicts, env))
            for side, g in (("A", ga), ("S", gs)):
                data = m._prep(g)
                cos, flags, xfin = per_layer(m.model, data, a.check_relu)
                # ANCHOR: the re-implemented loop must match the model's own encode()
                with torch.no_grad():
                    ref = m.model.encode(m.model.mlp(data.x), data.edge_index,
                                         data.edge_attr)
                ok = torch.allclose(ref, xfin, atol=1e-6)
                trend = "RISE at final" if cos[-1] > min(cos[:-1]) + 1e-9 else "monotone-ish"
                print(f"{env:<22}{side:<6}" + "".join(f"{c:>9.4f}" for c in cos)
                      + f"{trend:>16}" + ("" if ok else "  !! ENCODER MISMATCH !!")
                      + ("" if not a.check_relu else f"  relu-noop={all(flags)}"))
                for i, c in enumerate(cos):
                    rows.append(dict(model=name, env=env, side=side, layer=i + 1,
                                     offdiag_cos=round(c, 6), encoder_match=ok))
            if a.trunc_f1:
                f1s = [truncated_f1(m, k, ga, gs, gt) for k in range(1, nL + 1)]
                print(f"{'':<22}{'F1@k':<6}" + "".join(f"{v:>9.3f}" for v in f1s))
                for k, v in enumerate(f1s, 1):
                    rows.append(dict(model=name, env=env, side="F1", layer=k,
                                     offdiag_cos=round(v, 6), encoder_match=True))

    out = os.path.join(a.out_dir, "edge_layer_probe.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "env", "side", "layer",
                                           "offdiag_cos", "encoder_match"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
