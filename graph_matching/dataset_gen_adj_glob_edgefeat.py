"""
Build a partial-matching dataset with invariant edge features (arXiv:2409.11972), for the
`<experiment>_edgefeat_rooms` experiment consumed by optimization_adj_glob_edgefeat.py and
pgm_training_adj_glob_edgefeat.py.

Works for ANY topology variant. The dataset name is a positional argument and everything
else is derived from it, using the same prefix rule the two sibling scripts already apply
(pgm_training_adj_glob_edgefeat.py, `REFERENCE = ...`): `adj_*` reads original/adj and
validates on real_adj, `fully_*` reads original/fully and validates on real_fully.

Reuses the actual pairing/split/normalize logic from dataset_gen.py, applied to the existing
original/<reference>/original.pkl + noise/<experiment>/noise.pkl (these ARE the
already-assembled reference_graphs/noise_graphs intermediates that dataset_gen.py's
build_incremental_partial_dataset itself produces -- the raw MSD .pt files that function would
otherwise read via deserialize_MSD_dataset are not present in this environment, so this script
starts one step downstream, which is sufficient: raw geometry is untouched either way).

Critically this does NOT reuse the pre-built <experiment>/{train,valid,test}_dataset.pkl under
datasets/Graph-matching/noise/<experiment>/ -- those have normalized (z-scored) `.x`, which
would corrupt distances/angles if used to compute edge features. Pairing is rebuilt from raw
original.pkl/noise.pkl instead, with edge_attr computed BEFORE normalization.

Usage:
    python dataset_gen_adj_glob_edgefeat.py adj_glob_65
    python dataset_gen_adj_glob_edgefeat.py fully_no_glob_65
then, with the SAME experiment name (both scripts take it as a positional arg):
    python optimization_adj_glob_edgefeat.py fully_no_glob_65_edgefeat_rooms
    python pgm_training_adj_glob_edgefeat.py fully_no_glob_65_edgefeat_rooms

On `fully`: intra-room ws-ws is complete rather than a ring, which removes the ring's
edge-SUBSTITUTION behaviour under dropout (43-49% of surviving walls gain a neighbour they did
not have when clean -- worse than a plain subset for max-pool; in a complete graph dropping a
wall cannot create a new adjacency). edge_features.py needs no change for it: verified on 200
synthetic fully graphs (intra-room complete in 1494/1494 rooms, 0 pre-existing cross-room ws-ws
edges so WS_WS_INTER stays exactly mechanism-2 provenance, 0 orphan ws) and all six real_fully
graphs. Mechanism 2 itself is UNCHANGED by the topology -- its candidate set is cross-room, and
the emitted WS_WS_INTER counts are identical between real_adj and real_fully.

Use `fully_no_glob_65`, not `fully_glob_65`: the latter is declared in dataset_gen.py's
INCREMENTAL_DATASETS table but was never built (no noise/, no model dir, and the raw folder it
needs is absent). It is also unnecessary -- `fully_no_glob_65` already carries a per-graph
global rotation, measured on its own valid_dataset.pkl: wall-normal angles mod 90 deg have a
4.7-5.3 deg spread WITHIN each graph but a 24.7 deg spread ACROSS graphs (uniform on [0,90)
would be 25.98). adj_glob_65 measures 24.14 and adj_no_glob_65 23.83 -- i.e. all three are
rotated and the glob/no_glob name does not mean what it says.
"""
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

from edge_features import (build_edge_index_and_attr, compute_edge_mean_std,
                           node_features, normalize_edge_attr)

SEED = 42
node_type_mapping = {"room": [1, 0], "ws": [0, 1]}

# Dataset dump root. Repo-relative by default so a clone with `datasets/` beside it needs no
# configuration; override with GM_DATASET_ROOT for any other layout. Resolves to
# /root/workspace/src/datasets/Graph-matching here, i.e. byte-identical to the old literal.
DATASET_ROOT = os.environ.get(
    "GM_DATASET_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "datasets", "Graph-matching")))

GNN_PATH = "./GNN/"  # relative to graph_matching_gnn/ (the parent of this file's directory), matching
                      # the convention used by optimization_adj_glob_edgefeat.py / pgm_training_adj_glob_edgefeat.py

DEFAULT_EXPERIMENT = "adj_glob_65"


def infer_reference(experiment):
    """Topology variant backing `experiment`: adj_* -> adj, fully_* -> fully, else equal.

    Identical rule to pgm_training_adj_glob_edgefeat.py and optimization_adj_glob_edgefeat.py.
    Keeping the three in agreement is what makes the reference graphs, the noise graphs and the
    real-validation scenes all describe the same topology -- mixing them silently trains on one
    connectivity and validates on another.
    """
    if experiment.startswith("adj"):
        return "adj"
    if experiment.startswith("fully"):
        return "fully"
    return "equal"


def resolve_paths(experiment, reference=None):
    """(original dir, noise dir, output experiment name, reference-copy dir) for `experiment`."""
    reference = reference or infer_reference(experiment)
    return (
        os.path.join(DATASET_ROOT, "original", reference),
        os.path.join(DATASET_ROOT, "noise", experiment),
        f"{experiment}_edgefeat_rooms",
        os.path.join(GNN_PATH, "preprocessed", "graph_matching", reference),
    )


# ─── Copied verbatim from the working-tree dataset_gen.py ──────────────────────────────────
# NOT from a branch: dataset_gen.py's INCREMENTAL_DATASETS table exists in no branch of this
# repo (checked feat_gm_class, gnn_matching, gnn_matching_ja, main), only in the uncommitted
# working tree. feat_gm_class HEAD is ade21e7 and its own dataset_gen.py is an older utility
# module with no dataset table at all -- an earlier note citing commit 94e0e3c was wrong.
# except nx_to_pyg_data_preserve_order, which is replaced by the edge-feature-aware version
# below (node_features/build_edge_index_and_attr from edge_features.py).

def deserialize_graph_matching_dataset(path: str, filename: str = "train_dataset.pkl") -> List:
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    with open(full_path, 'rb') as f:
        pairs = pickle.load(f)
    print(f"Loaded {len(pairs)} pairs from {full_path}")
    return pairs


def serialize_graph_matching_dataset(pairs, path: str, filename: str = "train_dataset.pkl"):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    with open(full_path, 'wb') as f:
        pickle.dump(pairs, f)
    print(f"Serialized {len(pairs)} pairs to {full_path}")


def serialize_norm_stats(mean, std, path: str, filename: str = "norm_stats.pt",
                         edge_mean=None, edge_std=None) -> None:
    os.makedirs(path, exist_ok=True)
    stats = {"mean": mean, "std": std}
    if edge_mean is not None and edge_std is not None:
        # Consumed by _real_nx_to_pyg so real scenes are scaled exactly like training data.
        # Older datasets lack these keys, so loaders must treat them as optional.
        stats["edge_mean"] = edge_mean
        stats["edge_std"] = edge_std
    torch.save(stats, os.path.join(path, filename))
    print(f"Saved normalization stats to {os.path.join(path, filename)}")


def nx_to_pyg_data_preserve_order(graph) -> Data:
    node_ids = list(graph.nodes())
    id_map = {nid: i for i, nid in enumerate(node_ids)}
    x = node_features(graph, node_type_mapping)
    edge_index, edge_attr = build_edge_index_and_attr(graph, id_map)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.name = graph.graph.get('name')
    data.node_names = node_ids
    data.permutation = torch.arange(len(node_ids), dtype=torch.long)
    return data


def generate_matching_pair_as_data(g1, g2, pairs_list, noise_level: int = None) -> None:
    pyg_g1 = nx_to_pyg_data_preserve_order(g1)

    orig_names = list(g2.nodes())
    num_g1 = g1.number_of_nodes()
    num_g2 = len(orig_names)
    perm_indices = torch.randperm(num_g2)

    g2_perm_nodes = {}
    for new_idx, orig_idx in enumerate(perm_indices.tolist()):
        orig_id = orig_names[orig_idx]
        g2_perm_nodes[new_idx] = g2.nodes[orig_id]
    import networkx as nx
    g2_perm = nx.DiGraph()
    g2_perm.graph['name'] = g2.graph.get('name', '')
    for new_idx, attrs in g2_perm_nodes.items():
        g2_perm.add_node(new_idx, **attrs)
    orig_to_new = {orig_names[idx]: new for new, idx in enumerate(perm_indices.tolist())}
    for u, v, data_edge in g2.edges(data=True):
        if u in orig_to_new and v in orig_to_new:
            g2_perm.add_edge(orig_to_new[u], orig_to_new[v], **data_edge)

    pyg_g2 = nx_to_pyg_data_preserve_order(g2_perm)
    pyg_g2.permutation = perm_indices
    pyg_g2.node_names = orig_names
    if noise_level is not None:
        pyg_g2.noise_level = noise_level

    P = torch.zeros((num_g1, num_g2), dtype=torch.float32)
    g1_ids = list(g1.nodes())
    for j, orig_idx in enumerate(perm_indices.tolist()):
        orig_id = orig_names[orig_idx]
        if orig_id in g1_ids:
            i = g1_ids.index(orig_id)
            P[i, j] = 1.0

    pairs_list.append((pyg_g1, pyg_g2, P))


def split_graphs_stratified(pairs, train_frac=0.7, val_frac=0.15, test_frac=0.15,
                            n_bins=5, seed=SEED, stratify_on="g2"):
    assert stratify_on in ("g1", "g2")
    total = train_frac + val_frac + test_frac
    assert abs(total - 1.0) < 1e-6

    if stratify_on == "g1":
        sizes = np.array([g1.num_nodes for g1, g2, P in pairs])
    else:
        sizes = np.array([g2.num_nodes for g1, g2, P in pairs])

    while n_bins > 1:
        try:
            size_bins = pd.qcut(sizes, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            n_bins -= 1
            continue
        counts = np.bincount(size_bins, minlength=n_bins)
        if np.all(counts >= 2):
            break
        n_bins -= 1

    idx = np.arange(len(pairs))
    if n_bins <= 1:
        train_idx, temp_idx = train_test_split(idx, test_size=1 - train_frac, random_state=seed)
        rel_val = val_frac / (val_frac + test_frac)
        val_idx, test_idx = train_test_split(temp_idx, test_size=1 - rel_val, random_state=seed)
    else:
        train_idx, temp_idx = train_test_split(idx, test_size=(1.0 - train_frac),
                                               random_state=seed, stratify=size_bins)
        rel_val = val_frac / (val_frac + test_frac)
        temp_bins = size_bins[temp_idx]
        val_idx, test_idx = train_test_split(temp_idx, test_size=(1.0 - rel_val),
                                             random_state=seed, stratify=temp_bins)

    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx], [pairs[i] for i in test_idx]


def split_pairs_by_apartment(pair_gt_list, original_graphs, train_frac=0.7, val_frac=0.15,
                             test_frac=0.15, n_bins=5, seed=SEED):
    """Group-based (apartment-level) split that avoids leakage across noise levels."""
    rep_pairs = []
    for g1 in original_graphs:
        generate_matching_pair_as_data(g1, g1, rep_pairs)

    try:
        train_rep, val_rep, test_rep = split_graphs_stratified(
            rep_pairs, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
            n_bins=n_bins, seed=seed, stratify_on="g1")
    except ValueError as e:
        print(f"[apartment split] stratification failed ({e}); using random apartment split.")
        train_rep, val_rep, test_rep = split_graphs_stratified(
            rep_pairs, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
            n_bins=1, seed=seed, stratify_on="g1")

    train_names = {p[0].name for p in train_rep}
    val_names = {p[0].name for p in val_rep}
    test_names = {p[0].name for p in test_rep}
    assert train_names.isdisjoint(val_names), "apartment leak train/val"
    assert train_names.isdisjoint(test_names), "apartment leak train/test"
    assert val_names.isdisjoint(test_names), "apartment leak val/test"

    train, val, test = [], [], []
    for pair in pair_gt_list:
        name = pair[0].name
        if name in train_names:
            train.append(pair)
        elif name in val_names:
            val.append(pair)
        elif name in test_names:
            test.append(pair)
        else:
            raise ValueError(f"Apartment '{name}' not assigned to any split")

    print(f"[apartment split] {len(train_names)}/{len(val_names)}/{len(test_names)} apartments "
          f"-> {len(train)}/{len(val)}/{len(test)} pairs (train/val/test)")
    return train, val, test


def compute_mean_std(pairs):
    x_list = []
    for data1, data2, _ in pairs:
        x_list.append(data1.x)
        x_list.append(data2.x)
    x_all = torch.cat(x_list, dim=0)
    return x_all.mean(dim=0), x_all.std(dim=0)


def compute_edge_mean_std_from_pairs(pairs):
    """Train-split edge_attr stats. See edge_features.compute_edge_mean_std for which
    columns are actually touched and why length_w is scaled but not shifted."""
    attrs = []
    for data1, data2, _ in pairs:
        attrs.append(data1.edge_attr)
        attrs.append(data2.edge_attr)
    return compute_edge_mean_std(attrs)


def normalize_data_pairs(pairs, mean, std, edge_mean=None, edge_std=None):
    normalized = []
    for data1, data2, P in pairs:
        data1.x = (data1.x - mean) / (std + 1e-8)
        data2.x = (data2.x - mean) / (std + 1e-8)
        if edge_mean is not None and edge_std is not None:
            data1.edge_attr = normalize_edge_attr(data1.edge_attr, edge_mean, edge_std)
            data2.edge_attr = normalize_edge_attr(data2.edge_attr, edge_mean, edge_std)
        normalized.append((data1, data2, P))
    return normalized


def check_pair_integrity(g1, g2, P, idx, name):
    ctx = f"[{name}] pair {idx}"
    assert P.shape == (g1.num_nodes, g2.num_nodes), f"{ctx}: P shape mismatch"
    uniq = torch.unique(P)
    assert torch.isin(uniq, torch.tensor([0.0, 1.0], dtype=P.dtype)).all(), f"{ctx}: P not binary"
    col_max = P.sum(dim=0).max().item() if P.numel() else 0.0
    row_max = P.sum(dim=1).max().item() if P.numel() else 0.0
    assert col_max <= 1.0 and row_max <= 1.0, f"{ctx}: P not a partial matching"
    n_pos = int(P.sum().item())
    assert n_pos > 0, f"{ctx}: P has no positive matches"
    shared = len(set(g1.node_names) & set(g2.node_names))
    assert n_pos == shared, f"{ctx}: positives ({n_pos}) != shared node ids ({shared})"
    assert torch.isfinite(g1.x).all() and torch.isfinite(g2.x).all(), f"{ctx}: non-finite features"
    assert torch.isfinite(g1.edge_attr).all() and torch.isfinite(g2.edge_attr).all(), f"{ctx}: non-finite edge_attr"


def check_pairs_integrity(pairs, name):
    positives = []
    for i, (g1, g2, P) in enumerate(pairs):
        check_pair_integrity(g1, g2, P, i, name)
        positives.append(int(P.sum().item()))
    if positives:
        print(f"[{name}] integrity OK: {len(pairs)} pairs | positives/pair "
              f"min={min(positives)} max={max(positives)} mean={np.mean(positives):.1f}")


def describe(split, name):
    if not split:
        print(f"{name}: EMPTY (0 pairs)")
        return
    sz = [g1.num_nodes for g1, _, _ in split]
    print(f"{name}: count={len(split)}, nodes min={min(sz)}, max={max(sz)}, mean={np.mean(sz):.1f}")


# ─── Build ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", default=DEFAULT_EXPERIMENT,
                        help=f"Dataset under datasets/Graph-matching/noise/ to build edge "
                             f"features for, e.g. adj_glob_65 or fully_no_glob_65 "
                             f"(default: {DEFAULT_EXPERIMENT}). The output experiment is "
                             f"<experiment>_edgefeat_rooms -- pass THAT to "
                             f"optimization_adj_glob_edgefeat.py / pgm_training_adj_glob_edgefeat.py.")
    parser.add_argument("--reference", default=None,
                        help="Override the topology variant under original/ (adj | fully | "
                             "equal). Default: inferred from the experiment prefix.")
    parser.add_argument("--limit-per-block", type=int, default=None,
                        help="Use only the first N graphs of each noise-level block instead of "
                             "the full ~4895/block. Local validation only. NOTE it subsets AFTER "
                             "loading noise.pkl, so it does NOT lower peak memory: measured on a "
                             "6GiB cgroup, adj (731MB noise.pkl) completes but fully (1.3GB) is "
                             "OOM-killed during the load. Omit for the real run.")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    reference = args.reference or infer_reference(args.experiment)
    original_path, noise_path, base_experiment_name, reference_out_path = resolve_paths(
        args.experiment, reference)
    print(f"experiment: {args.experiment} | reference (A-graph) variant: {reference}")
    print(f"  original: {original_path}")
    print(f"  noise   : {noise_path}")
    print(f"  output  : {base_experiment_name}")
    if not os.path.isdir(noise_path):
        raise SystemExit(
            f"[{args.experiment}] {noise_path} does not exist.\n"
            f"Available: {sorted(os.listdir(os.path.join(DATASET_ROOT, 'noise')))}\n"
            f"Note fully_glob_* is declared in dataset_gen.py's INCREMENTAL_DATASETS but was "
            f"never built; use fully_no_glob_65, which already carries the global rotation.")

    original_graphs = deserialize_graph_matching_dataset(original_path, "original.pkl")
    noise_graphs = deserialize_graph_matching_dataset(noise_path, "noise.pkl")
    assert len(noise_graphs) % len(original_graphs) == 0, \
        "noise.pkl must be an integer number of original.pkl-sized blocks"
    n_orig = len(original_graphs)
    n_blocks = len(noise_graphs) // n_orig
    print(f"original: {n_orig} graphs | noise: {len(noise_graphs)} graphs ({n_blocks} blocks)")

    experiment_name = base_experiment_name
    if args.limit_per_block is not None:
        noise_graphs = [g for block in range(n_blocks)
                        for g in noise_graphs[block * n_orig: block * n_orig + args.limit_per_block]]
        print(f"[--limit-per-block {args.limit_per_block}] using {len(noise_graphs)} of "
              f"{n_blocks * n_orig} noise graphs")
        # Distinct output dir -- never let a locally-truncated validation build shadow the
        # real full-size dataset the training machine is expected to produce.
        experiment_name = f"{base_experiment_name}_LOCALTEST_limit{args.limit_per_block}"
    out_path = os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", experiment_name)

    pair_gt_list = []
    for i, g2 in enumerate(tqdm(noise_graphs, desc="Pair graph generation")):
        ref_idx = i % args.limit_per_block if args.limit_per_block is not None else i % n_orig
        generate_matching_pair_as_data(original_graphs[ref_idx], g2, pair_gt_list)
        del g2  # keep peak memory down on memory-constrained machines

    check_pairs_integrity(pair_gt_list, experiment_name)

    split_reference_graphs = (original_graphs[:args.limit_per_block]
                              if args.limit_per_block is not None else original_graphs)
    train, val, test = split_pairs_by_apartment(pair_gt_list, split_reference_graphs)
    describe(train, "TRAIN")
    describe(val, "VAL")
    describe(test, "TEST")

    mean, std = compute_mean_std(train)
    edge_mean, edge_std = compute_edge_mean_std_from_pairs(train)
    # d_ij is the only normalized column now (EDGE_NORM_COLS == (0,)); every other column is
    # left at mean 0 / std 1 on purpose, so print those as a guard that nothing else moved.
    print(f"edge_attr norm: d_ij mean={edge_mean[0]:.3f} std={edge_std[0]:.3f} | "
          f"all other cols identity: "
          f"{bool((edge_mean[1:] == 0).all() and (edge_std[1:] == 1).all())}")
    train_norm = normalize_data_pairs(train, mean, std, edge_mean, edge_std)
    val_norm = normalize_data_pairs(val, mean, std, edge_mean, edge_std)
    test_norm = normalize_data_pairs(test, mean, std, edge_mean, edge_std)

    serialize_norm_stats(mean, std, out_path, edge_mean=edge_mean, edge_std=edge_std)
    serialize_graph_matching_dataset(train_norm, out_path, "train_dataset.pkl")
    serialize_graph_matching_dataset(val_norm, out_path, "valid_dataset.pkl")
    serialize_graph_matching_dataset(test_norm, out_path, "test_dataset.pkl")
    serialize_graph_matching_dataset(noise_graphs, out_path, "noise.pkl")

    # optimization_adj_glob_edgefeat.py / pgm_training_adj_glob_edgefeat.py also load a raw
    # "original.pkl" from preprocessed/graph_matching/<reference>/ for the ground-truth preview
    # plot only (not fed to the model) -- reuse the same raw graphs, no feature changes needed.
    if not os.path.exists(os.path.join(reference_out_path, "original.pkl")):
        serialize_graph_matching_dataset(original_graphs, reference_out_path, "original.pkl")

    print(f"[{experiment_name}] done -> {out_path}")


if __name__ == "__main__":
    main()
