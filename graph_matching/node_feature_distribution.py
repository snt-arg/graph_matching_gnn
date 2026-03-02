#!/usr/bin/env python3
"""
Node Feature Distribution Analysis

Compares node feature distributions between:
  - Training original graphs  (preprocessed original.pkl(ground_truth)  → Prior equivalent)
  - Training noise graphs     (preprocessed noise.pkl from ws_room_dropout_noise → Online equivalent)
  - Inference graphs: A-graph (Prior) and S-graph (Online)

Comparisons:
  A-graph (Prior)   vs  training original  (ground-truth reference graphs)
  S-graph (Online)  vs  training noise     (perturbed/dropout graphs)

Node features (7-dim):
  [0-1] type_onehot : [1,0]=room, [0,1]=ws
  [2]   center_x
  [3]   center_y
  [4]   normal_x
  [5]   normal_y
  [6]   length        (-1.0 for rooms)

Usage:
  # Training data only
  python node_feature_distribution.py

  # Training + inference graphs (pickled NetworkX DiGraphs saved during inference)
  python node_feature_distribution.py --s_graph /path/to/prior.pkl --a_graph /path/to/online.pkl

  # Save plots to a specific directory
  python node_feature_distribution.py --output_dir /tmp/dist_out
"""

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Hardcoded paths ───────────────────────────────────────────────────────────
GNN_PATH = "/root/workspace/src/graph_matching_gnn/GNN"

# Raw graphs (not normalized) — used for RAW plots
ORIGINAL_PKL_PATH = os.path.join(
    GNN_PATH, "preprocessed", "graph_matching", "equal", "original.pkl"
)
NOISE_PKL_PATH = os.path.join(
    GNN_PATH, "preprocessed", "partial_graph_matching", "ws_room_dropout_noise", "noise.pkl"
)
# Pre-normalized PyG pairs saved from the notebook — used for NORMALIZED plots
TRAIN_DATASET_PATH = os.path.join(
    GNN_PATH, "preprocessed", "partial_graph_matching", "ws_room_dropout_noise", "train_dataset.pkl"
)

GRAPH_DICTS_DIR = "/root/workspace/src/graph_matching/graph_matching/graph_dicts"
OUTPUT_DIR      = "/tmp/dist_out"

#FEATURE_NAMES = ["type_room", "type_ws", "center_x", "center_y",
#                 "normal_x", "normal_y", "length"]

FEATURE_NAMES = ["center_x", "center_y",
                 "normal_x", "normal_y", "length"]
# Offset into the 7-column data matrix: columns 0-1 are type one-hot, features start at 2
FEATURE_OFFSET = 2


NODE_TYPE_MAPPING = {"room": [1, 0], "ws": [0, 1]}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_nx_graph_from_pt(path: str):
    """
    Load a pickled graph from a .pt or .pkl file.
    Handles both plain NetworkX graphs and GraphWrapper objects
    (which expose the underlying DiGraph via .graph).
    """
    import networkx as nx
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # If already a NetworkX graph, return as-is
    if isinstance(obj, nx.Graph):
        return obj
    # GraphWrapper wraps a DiGraph in a .graph attribute
    if hasattr(obj, "graph") and isinstance(obj.graph, nx.Graph):
        return obj.graph
    return obj


def graph_to_feature_matrix(G) -> np.ndarray:
    """
    Extract 7-dim feature matrix from a NetworkX graph whose nodes carry
    'type', 'center', 'normal', 'length' attributes (same contract as
    nx_to_pyg_data_preserve_order in PGM_class.py).
    center_x/y are absolute world coordinates.
    """
    rows = []
    for _, attrs in G.nodes(data=True):
        ntype = attrs.get("type", "")
        if ntype not in NODE_TYPE_MAPPING:
            continue
        center = list(attrs["center"][:2])
        normal = list(attrs["normal"][:2])
        length = -1.0 if ntype == "room" else attrs["length"]

        row = NODE_TYPE_MAPPING[ntype] + [float(center[0]), float(center[1]),
                                          float(normal[0]), float(normal[1]),
                                          float(length)]
        rows.append(row)
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 7), dtype=np.float32)


def load_preprocessed_pkl(path: str) -> Tuple[np.ndarray, int]:
    """
    Load a preprocessed .pkl file containing a list of NetworkX DiGraphs
    (as saved by the training pipeline for original.pkl / noise.pkl).
    Returns (feature_matrix [N,7], n_graphs).
    """
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return np.empty((0, 7), dtype=np.float32), 0

    with open(path, "rb") as f:
        graphs = pickle.load(f)

    if not isinstance(graphs, list):
        graphs = [graphs]

    parts = []
    n_loaded = 0
    for G in graphs:
        try:
            # Unwrap GraphWrapper if needed
            if hasattr(G, "graph") and isinstance(G.graph, nx.Graph):
                G = G.graph
            X = graph_to_feature_matrix(G)
            if len(X) > 0:
                parts.append(X)
            n_loaded += 1
        except Exception as e:
            print(f"[WARN] Could not extract features from graph: {e}")

    X_all = np.concatenate(parts, axis=0) if parts else np.empty((0, 7), dtype=np.float32)
    return X_all, n_loaded


def load_train_dataset_pkl(path: str) -> Tuple[np.ndarray, np.ndarray, int, List[int]]:
    """
    Load a train_dataset.pkl file containing pre-normalized (Data1, Data2, P) tuples.
    Returns (X_original_n [N,7], X_noise_n [N,7], n_pairs, train_indices).
    train_indices: int(data1.name) for each pair — the index of that pair's raw graphs
                   in original.pkl and noise.pkl, used to compute the exact mean/std.
    """
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return np.empty((0, 7), dtype=np.float32), np.empty((0, 7), dtype=np.float32), 0, []

    with open(path, "rb") as f:
        pairs = pickle.load(f)

    x_orig, x_noise, indices = [], [], []
    for data1, data2, _ in pairs:
        x_orig.append(data1.x.numpy())
        x_noise.append(data2.x.numpy())
        indices.append(int(data1.name))

    X_orig  = np.concatenate(x_orig,  axis=0) if x_orig  else np.empty((0, 7), dtype=np.float32)
    X_noise = np.concatenate(x_noise, axis=0) if x_noise else np.empty((0, 7), dtype=np.float32)
    return X_orig, X_noise, len(pairs), indices


def compute_exact_mean_std(train_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the exact mean/std used to normalize train_dataset.pkl.

    train_dataset.pkl stores NORMALIZED features — mean/std cannot be recovered from
    them alone (normalized data has mean≈0, std≈1 by construction).  Instead, each
    Data object stores data1.name = the raw-graph index into original.pkl / noise.pkl.
    We load just those raw graphs, extract their features, and compute mean/std,
    exactly replicating compute_mean_std(train) from PGM_class.

    ddof=1 matches torch.Tensor.std() (Bessel's correction).
    Returns (mean [7], std [7]) as float32 arrays.
    """
    with open(ORIGINAL_PKL_PATH, "rb") as f:
        orig_graphs = pickle.load(f)
    with open(NOISE_PKL_PATH, "rb") as f:
        noise_graphs = pickle.load(f)
    if not isinstance(orig_graphs,  list): orig_graphs  = [orig_graphs]
    if not isinstance(noise_graphs, list): noise_graphs = [noise_graphs]

    def _unwrap(G):
        return G.graph if hasattr(G, "graph") and isinstance(G.graph, nx.Graph) else G

    x_parts = []
    for idx in train_indices:
        for G in (orig_graphs[idx], noise_graphs[idx]):
            X = graph_to_feature_matrix(_unwrap(G))
            if len(X) > 0:
                x_parts.append(X)

    X_all = np.concatenate(x_parts, axis=0).astype(np.float64)
    mean  = X_all.mean(axis=0).astype(np.float32)
    std   = X_all.std(axis=0, ddof=1).astype(np.float32)
    return mean, std


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(X: np.ndarray, label: str) -> Dict:
    stats = {"label": label, "n_nodes": len(X)}
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, FEATURE_OFFSET + i]
        stats[name] = {
            "mean":   float(np.mean(col)),
            "std":    float(np.std(col)),
            "min":    float(np.min(col)),
            "max":    float(np.max(col)),
            "median": float(np.median(col)),
        }
    return stats


def compute_stats_by_type(X: np.ndarray) -> Dict:
    """Return stats split by room/ws node type.
    Uses relative comparison of the two type columns so it works on both
    raw (0/1) and normalized features."""
    mask_room = X[:, 0] > X[:, 1]
    mask_ws   = X[:, 1] > X[:, 0]
    return {
        "room": X[mask_room] if mask_room.any() else np.empty((0, 7)),
        "ws":   X[mask_ws]   if mask_ws.any()   else np.empty((0, 7)),
    }


def compute_distribution_distances(X_ref: np.ndarray, X_query: np.ndarray, label: str) -> Dict:
    dist = {"label": label}
    for i, name in enumerate(FEATURE_NAMES):
        a, b = X_ref[:, FEATURE_OFFSET + i], X_query[:, FEATURE_OFFSET + i]
        dist[name] = {
            "mean_diff": float(np.mean(b) - np.mean(a)),
            "std_ratio": float(np.std(b) / (np.std(a) + 1e-8)),
        }
    return dist



# ── Plotting ──────────────────────────────────────────────────────────────────

def _hist(ax, X, fi, color, label, histtype="stepfilled"):
    if len(X) > 0:
        ax.hist(X[:, FEATURE_OFFSET + fi], bins=50, alpha=0.6, color=color, label=label, density=True, histtype=histtype)


def plot_pair(label_ref: str, X_ref: np.ndarray,
              label_inf: str, X_inf: np.ndarray,
              title: str, by_type: bool = False) -> None:
    if by_type:
        fig, axes = plt.subplots(len(FEATURE_NAMES), 2,          
                                 figsize=(14, 3 * len(FEATURE_NAMES)))
        for fi, fname in enumerate(FEATURE_NAMES):
            for ti, (tname, col) in enumerate([("room", 0), ("ws", 1)]):
                ax = axes[fi][ti]
                ax.set_title(f"{fname} [{tname}]", fontsize=9)
                for X, c, lbl, htype in [(X_ref, "steelblue", label_ref, "stepfilled"),
                                         (X_inf, "coral",     label_inf, "step")]:
                    mask = X[:, col] > X[:, 1 - col]
                    _hist(ax, X[mask], fi, c, lbl, htype)
                ax.legend(fontsize=7)
    else:
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        axes = axes.flatten()
        for fi, fname in enumerate(FEATURE_NAMES):
            ax = axes[fi]
            ax.set_title(fname, fontsize=11)
            _hist(ax, X_ref, fi, "steelblue", label_ref, "stepfilled")
            _hist(ax, X_inf, fi, "coral",     label_inf, "step")
            ax.legend(fontsize=8)
        for ax in axes[len(FEATURE_NAMES):]:
            ax.set_visible(False)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show(block=False)

# ── Printing ──────────────────────────────────────────────────────────────────

def print_stats(X: np.ndarray, label: str) -> None:
    if len(X) == 0:
        print(f"\n  [{label}] — no nodes")
        return
    stats = compute_stats(X, label)
    n = stats["n_nodes"]
    print(f"\n{'='*68}")
    print(f"  {label}  ({n} nodes)")
    print(f"{'='*68}")
    header = f"{'Feature':<14} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'median':>9}"
    print(header)
    print("-" * len(header))
    for name in FEATURE_NAMES:
        s = stats[name]
        print(f"{name:<14} {s['mean']:>9.4f} {s['std']:>9.4f} "
              f"{s['min']:>9.4f} {s['max']:>9.4f} {s['median']:>9.4f}")


def print_distances(dist: Dict, ref_label: str) -> None:
    label = dist["label"]
    print(f"\n{'='*60}")
    print(f"  Distribution distances: {ref_label}  vs  {label}")
    print(f"{'='*60}")
    header = f"{'Feature':<14} {'mean_diff':>10} {'std_ratio':>10}"
    print(header)
    print("-" * len(header))
    for name in FEATURE_NAMES:
        d = dist[name]
        print(f"{name:<14} {d['mean_diff']:>10.4f} {d['std_ratio']:>10.4f}")

###


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Node feature distribution analysis")
    parser.add_argument("--s_graph",
                        default=os.path.join(GRAPH_DICTS_DIR, "Online.pkl"),
                        help="Pickled NetworkX DiGraph for S-graph (Online).")
    parser.add_argument("--a_graph",
                        default=os.path.join(GRAPH_DICTS_DIR, "Prior.pkl"),
                        help="Pickled NetworkX DiGraph for A-graph (Prior).")
    args = parser.parse_args()

    # ── Load training data (preprocessed pkls fed to the network) ─────────────
    print(f"\nLoading training original graphs from:\n  {ORIGINAL_PKL_PATH}")
    X_original, n_orig = load_preprocessed_pkl(ORIGINAL_PKL_PATH)
    print(f"  Loaded {n_orig} graphs → {len(X_original)} total node vectors")

    print(f"\nLoading training noise graphs from:\n  {NOISE_PKL_PATH}")
    X_noise, n_noise = load_preprocessed_pkl(NOISE_PKL_PATH)
    print(f"  Loaded {n_noise} graphs → {len(X_noise)} total node vectors")

    if len(X_original) == 0 and len(X_noise) == 0:
        print("[ERROR] No training node features extracted. Aborting.")
        sys.exit(1)

    # ── Load inference graphs ─────────────────────────────────────────────────
    X_s = np.empty((0, 7), dtype=np.float32)
    X_a = np.empty((0, 7), dtype=np.float32)

    if args.s_graph and os.path.exists(args.s_graph):
        print(f"\nLoading S-graph (Online) from: {args.s_graph}")
        G_s = load_nx_graph_from_pt(args.s_graph)
        X_s = graph_to_feature_matrix(G_s)
        print(f"  {len(X_s)} nodes")
    else:
        print(f"\n[INFO] S-graph not found at {args.s_graph} — skipping.")

    if args.a_graph and os.path.exists(args.a_graph):
        print(f"Loading A-graph (Prior) from: {args.a_graph}")
        G_a = load_nx_graph_from_pt(args.a_graph)
        X_a = graph_to_feature_matrix(G_a)
        print(f"  {len(X_a)} nodes")
    else:
        print(f"[INFO] A-graph not found at {args.a_graph} — skipping.")

    # datasets dict used for box-plot overview (all four together)
    all_datasets: Dict[str, np.ndarray] = {
        "train_original": X_original,
        "train_noise":    X_noise,
        "S-graph":        X_s,
        "A-graph":        X_a,
    }

    # ── Overall statistics ────────────────────────────────────────────────────
    print("\n" + "#" * 68)
    print("  OVERALL NODE FEATURE STATISTICS")
    print("#" * 68)
    for label, X in all_datasets.items():
        print_stats(X, label)

    # ── Statistics by node type ───────────────────────────────────────────────
    print("\n" + "#" * 68)
    print("  STATISTICS BY NODE TYPE")
    print("#" * 68)
    for label, X in all_datasets.items():
        by_type = compute_stats_by_type(X)
        for tname, Xsub in by_type.items():
            print_stats(Xsub, f"{label} / {tname}")

    # ── Distribution distances ────────────────────────────────────────────────
    # Prior (A-graph) compared against training originals
    # Online (S-graph) compared against training noise

    pairs = [
        ("train_original", X_original, "A-graph", X_a),
        ("train_noise",    X_noise,    "S-graph", X_s),
    ]

    print("\n" + "#" * 68)
    print("  A-graph vs train_original  |  S-graph vs train_noise")
    print("#" * 68)
    for ref_lbl, X_ref, inf_lbl, X_inf in pairs:
        if len(X_ref) == 0 or len(X_inf) == 0:
            print(f"\n  [{ref_lbl} vs {inf_lbl}] — not enough data to compare")
            continue
        dist = compute_distribution_distances(X_ref, X_inf, inf_lbl)
        print_distances(dist, ref_lbl)

    # ── Per-type distances ────────────────────────────────────────────────────
    print("\n" + "#" * 68)
    print("  DISTRIBUTION DISTANCES BY NODE TYPE")
    print("#" * 68)
    for ref_lbl, X_ref, inf_lbl, X_inf in pairs:
        if len(X_ref) == 0 or len(X_inf) == 0:
            continue
        ref_by_type = compute_stats_by_type(X_ref)
        inf_by_type = compute_stats_by_type(X_inf)
        for tname in ["room", "ws"]:
            X_r = ref_by_type[tname]
            X_q = inf_by_type[tname]
            if len(X_r) == 0 or len(X_q) == 0:
                print(f"\n  [{ref_lbl}/{tname} vs {inf_lbl}/{tname}] — not enough data")
                continue
            dist = compute_distribution_distances(X_r, X_q, f"{inf_lbl} / {tname}")
            print_distances(dist, f"{ref_lbl} / {tname}")

    # ── Load pre-normalized training data from train_dataset.pkl ─────────────
    print(f"\nLoading pre-normalized training pairs from:\n  {TRAIN_DATASET_PATH}")
    X_original_n, X_noise_n, n_pairs, train_indices = load_train_dataset_pkl(TRAIN_DATASET_PATH)
    print(f"  Loaded {n_pairs} pairs → {len(X_original_n)} original nodes, {len(X_noise_n)} noise nodes")

    # Exact mean/std: load the raw training graphs by index stored in train_dataset.pkl
    print("\nComputing exact mean/std from raw training graphs...")
    train_mean, train_std = compute_exact_mean_std(train_indices)
    print(f"  mean: {train_mean.tolist()}")
    print(f"  std:  {train_std.tolist()}")

    def normalize(X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return X
        return (X - train_mean) / (train_std + 1e-8)

    X_a_n = normalize(X_a)
    X_s_n = normalize(X_s)

    all_datasets_n: Dict[str, np.ndarray] = {
        "train_original": X_original_n,
        "train_noise":    X_noise_n,
        "S-graph":        X_s_n,
        "A-graph":        X_a_n,
    }

    # ── Plots: RAW (before normalization) ────────────────────────────────────
    print("\n--- RAW FEATURE DISTRIBUTIONS (before normalization) ---")
    # plot_pair("train_original [raw]", X_original, "A-graph [raw]", X_a,
    #           "RAW: train_original vs A-graph (Prior)")
    plot_pair("train_original [raw]", X_original, "A-graph [raw]", X_a,
              "RAW per type: train_original vs A-graph (Prior)", by_type=True)
    # plot_pair("train_noise [raw]", X_noise, "S-graph [raw]", X_s,
    #           "RAW: train_noise vs S-graph (Online)")
    plot_pair("train_noise [raw]", X_noise, "S-graph [raw]", X_s,
              "RAW per type: train_noise vs S-graph (Online)", by_type=True)
    #plot_boxplots(all_datasets, "Node Feature Box Plots: RAW")

    # ── Plots: NORMALIZED (what the GNN actually sees) ────────────────────────
    print("\n--- NORMALIZED FEATURE DISTRIBUTIONS (what the GNN sees) ---")
    # plot_pair("train_original [norm]", X_original_n, "A-graph [norm]", X_a_n,
    #           "NORMALIZED: train_original vs A-graph (Prior)")
    plot_pair("train_original [norm]", X_original_n, "A-graph [norm]", X_a_n,
              "NORMALIZED per type: train_original vs A-graph (Prior)", by_type=True)
    # plot_pair("train_noise [norm]", X_noise_n, "S-graph [norm]", X_s_n,
    #           "NORMALIZED: train_noise vs S-graph (Online)")
    plot_pair("train_noise [norm]", X_noise_n, "S-graph [norm]", X_s_n,
              "NORMALIZED per type: train_noise vs S-graph (Online)", by_type=True)
    #plot_boxplots(all_datasets_n, "Node Feature Box Plots: NORMALIZED")

    print("\nDone.")
    plt.show()  # block here to keep all figures open


if __name__ == "__main__":
    main()
