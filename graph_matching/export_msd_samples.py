#!/usr/bin/env python3
"""Export random (Prior, Online) pairs from the MSD test split as dashboard environments.

Writes folders that are format-compatible with `graph_matching/graph_dicts/<env>/` and with
`datasets/Graph-matching/real/<variant>/<env>/`, i.e. a `Prior.pkl` + `Online.pkl` +
`ground_truth.json` triple, so an unseen synthetic pair can be run through the dashboard (and
through `build_real_validation_pairs`) exactly like a real scan.

Everything is derived from `test_dataset.pkl` + `norm_stats.pt` alone.  `noise.pkl` is 765 MB and
gets OOM-killed under this box's 6 GB cgroup, and it is not needed: the stored `x` is a pure
z-score of `[type_onehot(2), center(2), normal(2), length(1)]`, so de-normalising recovers the
true geometry, and wall endpoints follow from `center +/- (length/2) * perp(normal)` (verified
against `original/adj/original.pkl` over 3812 walls: max error 4.8e-12).
"""

import argparse
import json
import os
import pickle
import random

import networkx as nx
import numpy as np
import torch

DATASET_ROOT = "/root/workspace/src/datasets/Graph-matching"
DEFAULT_DATASET = os.path.join(DATASET_ROOT, "noise", "adj_glob_65")
DEFAULT_OUT = "/root/workspace/src/graph_matching/graph_matching/MSD_Samples"

# Column layout of `x`, set by dataset_gen.nx_to_pyg_data_preserve_order.
C_ROOM, C_WS, C_CX, C_CY, C_NX, C_NY, C_LEN = range(7)


def denormalize(x, mean, std):
    """Undo normalize_data_pairs: x_norm = (x - mean) / (std + 1e-8)."""
    return x.numpy().astype(np.float64) * (std + 1e-8) + mean


def build_graph(data, x_raw, name):
    """Rebuild one side of a pair as a bare nx.DiGraph in dashboard schema.

    Row `i` of `x`/`edge_index` belongs to node `node_names[permutation[i]]` -- the dataset
    stores `node_names` unpermuted while shuffling the tensors.  Getting this backwards yields a
    silently scrambled graph, so it is asserted against the ground-truth matrix by the caller.
    """
    node_names = list(data.node_names)
    perm = data.permutation.numpy()
    ids = [str(node_names[perm[i]]) for i in range(len(perm))]

    g = nx.DiGraph()
    g.graph["name"] = name
    for i, nid in enumerate(ids):
        row = x_raw[i]
        is_room = row[C_ROOM] > row[C_WS]
        center = [float(row[C_CX]), float(row[C_CY])]
        if is_room:
            # Rooms carry no orientation and no length; snap to the exact sentinels the rest of
            # the pipeline expects rather than propagating float error from the z-score.
            g.add_node(nid, type="room", center=center, normal=[0.0, 0.0], length=-1.0)
        else:
            nrm = np.array([row[C_NX], row[C_NY]], dtype=np.float64)
            length = float(row[C_LEN])
            norm = np.linalg.norm(nrm)
            tangent = np.array([-nrm[1], nrm[0]]) / norm if norm > 0 else np.zeros(2)
            half = 0.5 * length
            limits = [
                [float(center[0] - half * tangent[0]), float(center[1] - half * tangent[1])],
                [float(center[0] + half * tangent[0]), float(center[1] + half * tangent[1])],
            ]
            g.add_node(nid, type="ws", center=center,
                       normal=[float(nrm[0]), float(nrm[1])], length=length, limits=limits)

    ei = data.edge_index.numpy()
    for src, dst in zip(ei[0], ei[1]):
        g.add_edge(ids[src], ids[dst])
    return g, ids


def build_ground_truth(P, prior_ids, online_ids, prior_graph):
    """Invert the assignment matrix into the dashboard's ground_truth.json schema.

    Rows index Prior (A), columns index Online (S).  The JSON is keyed by the *Online* id, and
    ids carry the `a_`/`s_` prefixes the real environments use (readers strip them).
    """
    rooms, ws = {}, []
    for i, j in P.nonzero().numpy():
        a_id, s_id = prior_ids[int(i)], online_ids[int(j)]
        if prior_graph.nodes[a_id]["type"] == "room":
            rooms["s_" + s_id] = "a_" + a_id
        else:
            ws.append(["s_" + s_id, "a_" + a_id])
    return {"rooms": rooms, "ws": sorted(ws)}


def degree_variance(data):
    """Variance of TOTAL (in+out) node degree.

    The graphs are reciprocal, so in-degree variance would be 4x smaller -- every degree-variance
    figure recorded for this project (real 47_basement A=4.48 / S=4.20) uses the total-degree
    convention, and mixing the two silently rescales the homogeneity comparison.
    """
    n = data.num_nodes
    ei = data.edge_index.numpy()
    deg = np.bincount(ei[0], minlength=n) + np.bincount(ei[1], minlength=n)
    return float(deg.var())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-nodes", type=int, default=None,
                    help="restrict to pairs whose Online side has >= this many nodes")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="restrict to pairs whose Online side has <= this many nodes")
    ap.add_argument("--max-degree-var", type=float, default=None,
                    help="restrict to pairs whose Online side has degree variance <= this")
    args = ap.parse_args()

    test_path = os.path.join(args.dataset_dir, "test_dataset.pkl")
    print("loading %s" % test_path)
    with open(test_path, "rb") as f:
        pairs = pickle.load(f)
    stats = torch.load(os.path.join(args.dataset_dir, "norm_stats.pt"))
    mean, std = stats["mean"].numpy().astype(np.float64), stats["std"].numpy().astype(np.float64)
    print("test split: %d pairs" % len(pairs))

    candidates = list(range(len(pairs)))
    if args.min_nodes or args.max_nodes or args.max_degree_var is not None:
        candidates = [
            k for k in candidates
            if (args.min_nodes is None or pairs[k][1].num_nodes >= args.min_nodes)
            and (args.max_nodes is None or pairs[k][1].num_nodes <= args.max_nodes)
            and (args.max_degree_var is None or degree_variance(pairs[k][1]) <= args.max_degree_var)
        ]
        print("after filters: %d candidate pairs" % len(candidates))

    # Each apartment appears 6x (once per noise level); require distinct apartments so that N
    # samples are N buildings rather than N views of one.
    rng = random.Random(args.seed)
    chosen, seen = [], set()
    for k in rng.sample(candidates, len(candidates)):
        if pairs[k][0].name in seen:
            continue
        seen.add(pairs[k][0].name)
        chosen.append(k)
        if len(chosen) == args.num_samples:
            break
    if len(chosen) < args.num_samples:
        raise SystemExit("only %d distinct apartments available" % len(chosen))

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"dataset_dir": args.dataset_dir, "seed": args.seed,
                "filters": {"min_nodes": args.min_nodes, "max_nodes": args.max_nodes,
                            "max_degree_var": args.max_degree_var},
                "samples": []}

    for k in chosen:
        d_a, d_s, P = pairs[k]
        env = "MSD_%s_lvl%d" % (d_a.name, int(d_s.noise_level))
        env_dir = os.path.join(args.out_dir, env)
        os.makedirs(env_dir, exist_ok=True)

        g_a, ids_a = build_graph(d_a, denormalize(d_a.x, mean, std), env + "_prior")
        g_s, ids_s = build_graph(d_s, denormalize(d_s.x, mean, std), env + "_online")

        # The GT matrix is node-id equality expressed in permuted index space; if the
        # permutation indirection above were wrong these would not line up.
        for i, j in P.nonzero().numpy():
            assert ids_a[int(i)] == ids_s[int(j)], "permutation mismatch in %s" % env

        gt = build_ground_truth(P, ids_a, ids_s, g_a)
        with open(os.path.join(env_dir, "Prior.pkl"), "wb") as f:
            pickle.dump(g_a, f)
        with open(os.path.join(env_dir, "Online.pkl"), "wb") as f:
            pickle.dump(g_s, f)
        with open(os.path.join(env_dir, "ground_truth.json"), "w") as f:
            json.dump(gt, f, indent=2)

        entry = {
            "env": env, "test_index": k, "apartment": d_a.name,
            "noise_level": int(d_s.noise_level),
            "prior": {"nodes": g_a.number_of_nodes(), "edges": g_a.number_of_edges(),
                      "rooms": sum(1 for _, t in g_a.nodes(data="type") if t == "room"),
                      "degree_variance": round(degree_variance(d_a), 3)},
            "online": {"nodes": g_s.number_of_nodes(), "edges": g_s.number_of_edges(),
                       "rooms": sum(1 for _, t in g_s.nodes(data="type") if t == "room"),
                       "degree_variance": round(degree_variance(d_s), 3)},
            "gt_matches": int(P.sum().item()),
        }
        manifest["samples"].append(entry)
        print("wrote %-22s prior %3dn/%4de  online %3dn/%4de  degvar %.2f  gt %d"
              % (env, entry["prior"]["nodes"], entry["prior"]["edges"],
                 entry["online"]["nodes"], entry["online"]["edges"],
                 entry["online"]["degree_variance"], entry["gt_matches"]))

    with open(os.path.join(args.out_dir, "sample_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nmanifest -> %s" % os.path.join(args.out_dir, "sample_manifest.json"))


if __name__ == "__main__":
    main()
