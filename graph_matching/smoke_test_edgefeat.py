"""
CPU-only smoke test for the invariant edge-feature architecture. No GPU, no training run --
exercises every seam identified during design:

  - edge_attr schema (9-column signed layout: 5 geometry columns + a 4-way edge-type
    one-hot; definedness is carried by the angle pairs themselves, which are
    unit-norm-or-zero, so there are no has_phi/has_alpha and no is_wall_anchored flags)
  - SE(2) invariance of both edge_attr AND the augmented edge_index
  - signed alpha/phi: antisymmetry and mirror chirality (an unsigned arccos alpha passes
    everything else here and fails only the mirror check)
  - the i<->j directionality swap, for ws-ws, wall-anchored room-room, and cross-room edges
  - mechanism 1: parallel room-room edges, one per accepted wall; unanchored pairs DROPPED
  - mechanism 2: cross-room ws-ws pairs, reciprocity, no duplication of existing edges
  - edge_attr normalization touching only cols 0 and 5
  - permutation handling, batching, forward/backward gradient flow into the edge path
  - all three real environments, plus a Prior-vs-Online per-column drift table

Uses only the first few graphs of original.pkl/noise.pkl to stay fast.

Usage: python smoke_test_edgefeat.py
"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_features import (EDGE_ATTR_DIM, EDGE_NORM_COLS, EDGE_TYPE_COL0, EDGE_TYPE_NAMES, ROOM_ROOM,
                           ROOM_WS, WS_WS_INTER, WS_WS_INTRA,
                           build_edge_index_and_attr, compute_edge_mean_std,
                           compute_local_frame_edge_features, cross_room_surface_pairs,
                           node_features, normalize_edge_attr, room_membership,
                           same_physical_wall, shared_wall_candidates)

# One root, repo-relative by default; override with GM_DATASET_ROOT. Resolves to
# /root/workspace/src/datasets/Graph-matching here -- same as the old hardcoded literals.
DATASET_ROOT = os.environ.get(
    "GM_DATASET_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "datasets", "Graph-matching")))

ORIGINAL_PATH = os.path.join(DATASET_ROOT, "original", "adj", "original.pkl")
NOISE_PATH = os.path.join(DATASET_ROOT, "noise", "adj_glob_65", "noise.pkl")
REAL_ROOT = os.path.join(DATASET_ROOT, "real", "real_adj")
REAL_DIR = os.path.join(REAL_ROOT, "47_basement")

# `fully` counterparts. The same builder must serve both topologies -- `fully` replaces the
# intra-room ring with complete intra-room connectivity, which is what removes the ring's
# edge-SUBSTITUTION behaviour under dropout. Section 10 asserts the schema invariants there.
FULLY_ORIGINAL_PATH = os.path.join(DATASET_ROOT, "original", "fully", "original.pkl")
FULLY_REAL_ROOT = os.path.join(DATASET_ROOT, "real", "real_fully")

node_type_mapping = {"room": [1, 0], "ws": [0, 1]}

N_CHECK = 0
N_PASS = 0


def check(name, cond):
    global N_CHECK, N_PASS
    N_CHECK += 1
    status = "PASS" if cond else "FAIL"
    if cond:
        N_PASS += 1
    print(f"[{status}] {name}")
    return cond


def load_pkl(path, n=None):
    with open(path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs[:n] if n else graphs


def nx_to_pyg(graph):
    from torch_geometric.data import Data
    node_ids = list(graph.nodes())
    id_map = {nid: i for i, nid in enumerate(node_ids)}
    x = node_features(graph, node_type_mapping)
    edge_index, edge_attr = build_edge_index_and_attr(graph, id_map)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.name = graph.graph.get('name')
    data.node_names = node_ids
    data.permutation = torch.arange(len(node_ids), dtype=torch.long)
    return data


def generate_matching_pair(g1, g2, pairs_list):
    import networkx as nx
    pyg_g1 = nx_to_pyg(g1)
    orig_names = list(g2.nodes())
    num_g1, num_g2 = g1.number_of_nodes(), len(orig_names)
    perm_indices = torch.randperm(num_g2)
    g2_perm = nx.DiGraph()
    g2_perm.graph['name'] = g2.graph.get('name', '')
    for new_idx, orig_idx in enumerate(perm_indices.tolist()):
        g2_perm.add_node(new_idx, **g2.nodes[orig_names[orig_idx]])
    orig_to_new = {orig_names[idx]: new for new, idx in enumerate(perm_indices.tolist())}
    for u, v, data_edge in g2.edges(data=True):
        if u in orig_to_new and v in orig_to_new:
            g2_perm.add_edge(orig_to_new[u], orig_to_new[v], **data_edge)
    pyg_g2 = nx_to_pyg(g2_perm)
    pyg_g2.permutation = perm_indices
    pyg_g2.node_names = orig_names
    P = torch.zeros((num_g1, num_g2), dtype=torch.float32)
    g1_ids = list(g1.nodes())
    for j, orig_idx in enumerate(perm_indices.tolist()):
        orig_id = orig_names[orig_idx]
        if orig_id in g1_ids:
            P[g1_ids.index(orig_id), j] = 1.0
    pairs_list.append((pyg_g1, pyg_g2, P))



def report_edge_drift(load_situational_graph, envs):
    """Per-column |Prior - Online| on ground-truth-matched edges, split by edge class.

    Reports only -- it prints a table rather than asserting, because the expected magnitudes
    differ by two orders of magnitude between the alpha channel (near-exact) and the cross-room
    phi channel (known-noisy), so a single pass/fail threshold would be meaningless.
    """
    import json
    import collections

    buckets = collections.defaultdict(list)
    for env in envs:
        d = os.path.join(REAL_ROOT, env)
        gt_path = os.path.join(d, 'ground_truth.json')
        if not os.path.exists(gt_path):
            continue
        prior = load_situational_graph(os.path.join(d, 'Prior.pkl'))
        online = load_situational_graph(os.path.join(d, 'Online.pkl'))
        with open(gt_path) as f:
            gt = json.load(f)
        strip = lambda x: str(x).split('_', 1)[1] if '_' in str(x) else str(x)
        # ground truth maps S(Online) -> A(Prior)
        s2a = {strip(k): strip(v) for k, v in gt.get('rooms', {}).items()}
        s2a.update({strip(a): strip(b) for a, b in gt.get('ws', [])})

        # Room-room edges must be measured through mechanism 1's wall-anchored frame --
        # calling the descriptor without one leaves phi zero-filled and the row would look
        # perfect for the wrong reason, hiding exactly the regression this guard is for.
        def descriptor(graph, room_ws_map, a, b):
            if graph.nodes[a]['type'] == 'room' and graph.nodes[b]['type'] == 'room':
                walls = shared_wall_candidates(graph, a, b, room_ws_map)
                if walls:
                    disp = (np.asarray(graph.nodes[b]['center'], float).ravel()[:2]
                            - np.asarray(graph.nodes[a]['center'], float).ravel()[:2])
                    frames = []
                    for w in walls:
                        n_w = np.asarray(graph.nodes[w]['normal'], float).ravel()[:2]
                        if float(np.dot(n_w, disp)) < 0.0:
                            n_w = -n_w
                        frames.append(compute_local_frame_edge_features(
                            graph, a, b, ROOM_ROOM, frame_normal=n_w))
                    # average over the parallel edges, which is what max-pool sees a spread of
                    return np.asarray(frames, dtype=float).mean(axis=0)
                kind = ROOM_ROOM
            elif graph.nodes[a]['type'] == 'room' or graph.nodes[b]['type'] == 'room':
                kind = ROOM_WS
            else:
                kind = WS_WS_INTRA
            return np.asarray(compute_local_frame_edge_features(graph, a, b, kind),
                              dtype=float)

        _, prior_room_ws = room_membership(prior)
        _, online_room_ws = room_membership(online)

        p_attr = {}
        for u, v in prior.edges():
            p_attr[(str(u), str(v))] = descriptor(prior, prior_room_ws, u, v)
        for u, v in online.edges():
            au, av = s2a.get(str(u)), s2a.get(str(v))
            if au is None or av is None or (au, av) not in p_attr:
                continue
            o_row = descriptor(online, online_room_ws, u, v)
            p_row = p_attr[(au, av)]
            t_u, t_v = online.nodes[u]['type'], online.nodes[v]['type']
            cls = ('room-room' if t_u == t_v == 'room'
                   else 'ws-ws' if t_u == t_v == 'ws' else 'room/ws')
            buckets[cls].append(np.abs(o_row - p_row))

    cols = ['d', 'cos_phi', 'sin_phi', 'cos_a', 'sin_a']
    print('  (room-room rows use mechanism 1\'s wall-anchored frame)')
    print(f"  {'edge class':<12} {'n':>4} " + ' '.join(f'{c:>9}' for c in cols))
    for cls in sorted(buckets):
        arr = np.vstack(buckets[cls])
        print(f"  {cls:<12} {len(arr):>4} " + ' '.join(f'{v:>9.4f}' for v in arr.mean(0)[:5]))
    if not buckets:
        print("  (no ground-truth-matched edges found)")


def check_fully_topology():
    """Section 10: the same schema invariants, on the `fully` topology.

    edge_features.py is topology-agnostic by design, but nothing enforced that. `fully` differs
    from `adj` in one way that could plausibly break the labelling: intra-room ws-ws is complete
    rather than a ring, so there are many more WS_WS_INTRA rows. Mechanism 2 is unaffected --
    its candidate set is cross-room -- so WS_WS_INTER counts must come out IDENTICAL on the same
    scene under both variants, which is the sharpest check here.

    Only the real scenes are exercised by default: original/fully/original.pkl is 460 MB and a
    6 GiB cgroup cannot hold it alongside the adj graphs already loaded above.
    """
    print("\n=== 10. `fully` topology (complete intra-room ws-ws) ===")
    if not os.path.isdir(FULLY_REAL_ROOT):
        print(f"  (not found at {FULLY_REAL_ROOT} -- skipping)")
        return

    n_types = len(EDGE_TYPE_NAMES)
    inter_adj, inter_fully, intra_adj, intra_fully = 0, 0, 0, 0
    rooms_total, rooms_complete = 0, 0
    all_onehot_ok, all_type_matches_endpoints = True, True
    rr_phi_ok, rr_rows, rr_expected = True, 0, 0

    for env in sorted(os.listdir(FULLY_REAL_ROOT)):
        for side in ("Prior", "Online"):
            path = os.path.join(FULLY_REAL_ROOT, env, f"{side}.pkl")
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as f:
                g = pickle.load(f)
            id_map = {n: i for i, n in enumerate(g.nodes())}
            _, ea = build_edge_index_and_attr(g, id_map)
            block = ea[:, EDGE_TYPE_COL0:EDGE_TYPE_COL0 + n_types]
            if not bool((block.sum(dim=1) == 1).all()):
                all_onehot_ok = False
            kind = block.argmax(dim=1)
            intra_fully += int((kind == WS_WS_INTRA).sum())
            inter_fully += int((kind == WS_WS_INTER).sum())

            # The room-room rule is topology-independent: an edge exists only where a shared
            # wall was accepted, so phi is never the zero vector, and the row count equals the
            # number of accepted walls. Asserted here too because `fully` reaches this builder
            # by a different route than `adj` and nothing else would catch a regression.
            n_phi_rr = (ea[:, 1] ** 2 + ea[:, 2] ** 2)[kind == ROOM_ROOM]
            if n_phi_rr.numel() and not bool((n_phi_rr > 0.5).all()):
                rr_phi_ok = False
            rr_rows += int((kind == ROOM_ROOM).sum())
            _, rw_g = room_membership(g)
            seen_rr = set()
            for a_, b_ in g.edges():
                if g.nodes[a_].get('type') == 'room' and g.nodes[b_].get('type') == 'room':
                    key = tuple(sorted([str(a_), str(b_)]))
                    if key in seen_rr:
                        continue
                    seen_rr.add(key)
                    rr_expected += 2 * len(shared_wall_candidates(g, a_, b_, rw_g))

            # every room's walls must be pairwise connected -- the defining property of `fully`
            _, room_ws = room_membership(g)
            for walls in room_ws.values():
                rooms_total += 1
                k = len(walls)
                present = sum(1 for i, a in enumerate(walls) for b in walls[i + 1:]
                              if g.has_edge(a, b) or g.has_edge(b, a))
                if present == k * (k - 1) // 2:
                    rooms_complete += 1

            # a ROOM_ROOM row must have two rooms at its ends, a ROOM_WS row exactly one, etc.
            # Rebuilt here rather than read off the row so the label is checked, not trusted.
            expected = []
            for u, v in g.edges():
                t_u, t_v = g.nodes[u].get('type'), g.nodes[v].get('type')
                expected.append(ROOM_ROOM if t_u == t_v == 'room'
                                else ROOM_WS if 'room' in (t_u, t_v) else WS_WS_INTRA)
            n_graph_edges = len(expected)
            # mechanism 1 emits parallel rows, so compare as multisets over the graph-edge rows
            from collections import Counter
            got = Counter(int(k) for k in kind[:n_graph_edges].tolist())
            want = Counter(expected)
            if any(k not in (ROOM_ROOM,) and got.get(k, 0) < want.get(k, 0) for k in want):
                all_type_matches_endpoints = False

            adj_path = os.path.join(REAL_ROOT, env, f"{side}.pkl")
            if os.path.exists(adj_path):
                with open(adj_path, 'rb') as f:
                    ga = pickle.load(f)
                _, ea_a = build_edge_index_and_attr(ga, {n: i for i, n in enumerate(ga.nodes())})
                ka = ea_a[:, EDGE_TYPE_COL0:EDGE_TYPE_COL0 + n_types].argmax(dim=1)
                intra_adj += int((ka == WS_WS_INTRA).sum())
                inter_adj += int((ka == WS_WS_INTER).sum())

    check("fully: one-hot sums to exactly 1 on every row", all_onehot_ok)
    check(f"fully: intra-room ws-ws complete in every room ({rooms_complete}/{rooms_total})",
          rooms_total > 0 and rooms_complete == rooms_total)
    check("fully: every graph edge's label agrees with its endpoint node types",
          all_type_matches_endpoints)
    # The sharpest one: `fully` changes intra-room connectivity ONLY, so mechanism 2 -- whose
    # candidates are cross-room -- must find exactly the same wall pairs it finds under `adj`.
    check(f"fully: WS_WS_INTER identical to adj ({inter_fully} vs {inter_adj})",
          inter_adj > 0 and inter_fully == inter_adj)
    check(f"fully: strictly more WS_WS_INTRA than adj ({intra_fully} vs {intra_adj})",
          intra_fully > intra_adj)
    check("fully: every ROOM_ROOM row has phi defined (unanchored pairs dropped here too)",
          rr_phi_ok)
    check(f"fully: ROOM_ROOM rows == 2x accepted shared walls ({rr_rows} vs {rr_expected})",
          rr_rows == rr_expected)


def main():
    print("=== 1. Loading a small slice of real data ===")
    original_graphs = load_pkl(ORIGINAL_PATH, n=5)
    noise_graphs = load_pkl(NOISE_PATH, n=5)  # first block only
    print(f"loaded {len(original_graphs)} original, {len(noise_graphs)} noise graphs")

    print("\n=== 2. edge_attr sanity (9-column signed schema + type one-hot) ===")
    g = original_graphs[0]
    data = nx_to_pyg(g)
    ea = data.edge_attr
    check(f"edge_attr shape == (num_edges, {EDGE_ATTR_DIM})",
          tuple(ea.shape) == (data.edge_index.shape[1], EDGE_ATTR_DIM))
    d_ij = ea[:, 0]
    cos_phi, sin_phi = ea[:, 1], ea[:, 2]
    cos_a, sin_a = ea[:, 3], ea[:, 4]
    type_block = ea[:, EDGE_TYPE_COL0:EDGE_TYPE_COL0 + len(EDGE_TYPE_NAMES)]
    kind = type_block.argmax(dim=1)
    n_phi = cos_phi ** 2 + sin_phi ** 2
    n_alpha = cos_a ** 2 + sin_a ** 2

    check("d_ij >= 0 everywhere", bool((d_ij >= 0).all()))
    # THE property the schema rests on, now that has_phi/has_alpha are gone: an angle pair is
    # either exactly the zero vector ("undefined") or exactly unit norm. Nothing in between is
    # reachable, so definedness needs no separate flag column -- and if this ever broke, the
    # zero vector would stop being an unambiguous sentinel.
    check("phi pair is exactly zero-vector or exactly unit norm",
          bool(((n_phi < 1e-10) | ((n_phi - 1).abs() < 1e-5)).all()))
    check("alpha pair is exactly zero-vector or exactly unit norm",
          bool(((n_alpha < 1e-10) | ((n_alpha - 1).abs() < 1e-5)).all()))
    # The type one-hot must be a genuine one-hot: exactly one 1.0 per row, nothing else.
    check("type block is {0,1} valued",
          bool(((type_block == 0) | (type_block == 1)).all()))
    check("type block sums to exactly 1 on every row",
          bool((type_block.sum(dim=1) == 1).all()))

    # A room-room edge is emitted ONLY when a shared wall was accepted, so its frame is
    # always that wall's normal and phi can never be the zero vector. This is the invariant
    # that replaced the old anchored-vs-fallback machinery: if it ever fails, an unanchored
    # room-room edge has leaked back into the builder.
    rr = kind == ROOM_ROOM
    check("every ROOM_ROOM row has phi defined (no unanchored room-room edge is emitted)",
          bool((n_phi[rr] > 0.5).all()))
    check("ROOM_ROOM rows never carry a defined alpha (rooms have no normal)",
          bool((n_alpha[rr] < 1e-10).all()))
    # Which half of a ROOM_WS pair a row is, is likewise not encoded: phi==0 is exactly the
    # room-framed row (the one the ROOM receives), phi!=0 the wall-framed row (received by the
    # wall). Frame owner == receiver, so no arrow notation -- see build_edge_index_and_attr.
    check("ROOM_WS rows have alpha undefined (one endpoint is always a room)",
          bool((n_alpha[kind == ROOM_WS] < 1e-10).all()))
    # alpha, unlike phi, cannot degenerate on a ws-ws edge: it needs only the two normals,
    # never the centroid separation. (phi additionally needs d_ij > 0, and 2 of 60240
    # synthetic intra rows do have coincident centroids -- so phi is NOT asserted here.)
    check("ws-ws rows always have alpha defined",
          bool((n_alpha[(kind == WS_WS_INTRA) | (kind == WS_WS_INTER)] > 0.5).all()))
    check("x shape == (num_nodes, 3)", tuple(data.x.shape) == (g.number_of_nodes(), 3))
    check("x is finite", bool(torch.isfinite(data.x).all()))

    # Node schema: [type_onehot(2), length_or_-1(1)]. The -1 sentinel must reach the model
    # intact -- 0 would collide with the 12.5% of ws whose length is <= 1 cm.
    x_len = data.x[:, 2]
    is_room = torch.tensor([g.nodes[n]['type'] == 'room' for n in g.nodes()])
    check("room length is exactly -1 (never 0)",
          bool(is_room.sum() == 0 or (x_len[is_room] == -1.0).all()))
    check("ws length is a real value, never the -1 sentinel",
          bool((~is_room).sum() == 0 or (x_len[~is_room] != -1.0).all()))
    check("no node length is exactly 0 (0 is not a sentinel in node x)",
          bool((x_len[is_room] != 0.0).all()))
    check("edge_attr is finite", bool(torch.isfinite(ea).all()))

    print("\n=== 2b. SE(2) invariance (the property the whole design exists for) ===")
    import copy as _copy

    def se2(graph, theta, t):
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        h = _copy.deepcopy(graph)
        for _, attrs in h.nodes(data=True):
            c = np.asarray(attrs['center'], dtype=float).ravel()
            attrs['center'] = np.concatenate([R @ c[:2] + t, c[2:]]) if c.size > 2 else R @ c[:2]
            n = np.asarray(attrs.get('normal', [0., 0.]), dtype=float).ravel()[:2]
            attrs['normal'] = R @ n
            if attrs.get('limits') is not None:
                lims = np.asarray(attrs['limits'], dtype=float)
                attrs['limits'] = [np.concatenate([R @ e[:2] + t, e[2:]]) if e.size > 2
                                   else R @ e[:2] for e in lims]
        return h

    rng = np.random.default_rng(0)
    worst_dev = 0.0
    index_stable = True
    for _ in range(5):
        h = se2(g, float(rng.uniform(-np.pi, np.pi)), rng.uniform(-20, 20, 2))
        ei_h, ea_h = build_edge_index_and_attr(h, {n: i for i, n in enumerate(h.nodes())})
        index_stable &= bool(torch.equal(data.edge_index, ei_h))
        worst_dev = max(worst_dev, float((ea - ea_h).abs().max()))
    check(f"edge_attr invariant under 5 random SE(2) transforms (max dev {worst_dev:.2e})",
          worst_dev < 1e-5)
    check("augmented edge_index is also SE(2)-stable (same wall pairs found)", index_stable)

    print("\n=== 2c. Signed angles carry chirality ===")
    ws_pair = None
    for u, v in g.edges():
        if g.nodes[u]['type'] == 'ws' and g.nodes[v]['type'] == 'ws':
            ws_pair = (u, v)
            break
    if ws_pair is None:
        print("  (no ws-ws edge in sample graph -- skipping)")
    else:
        u, v = ws_pair
        f_uv = compute_local_frame_edge_features(g, u, v, WS_WS_INTRA)
        f_vu = compute_local_frame_edge_features(g, v, u, WS_WS_INTRA)
        check("alpha(i,j) == -alpha(j,i): cos equal, sin opposite",
              abs(f_uv[3] - f_vu[3]) < 1e-6 and abs(f_uv[4] + f_vu[4]) < 1e-6)
        # Mirroring the scene must flip every sine and leave every cosine alone. An UNSIGNED
        # alpha (the old arccos form) would be invariant here -- that is exactly the chirality
        # this schema was changed to preserve.
        mirrored = _copy.deepcopy(g)
        for _, attrs in mirrored.nodes(data=True):
            c = np.asarray(attrs['center'], dtype=float).ravel().copy(); c[1] = -c[1]
            attrs['center'] = c
            n = np.asarray(attrs.get('normal', [0., 0.]), dtype=float).ravel()[:2].copy()
            n[1] = -n[1]; attrs['normal'] = n
            if attrs.get('limits') is not None:
                lims = np.asarray(attrs['limits'], dtype=float).copy(); lims[:, 1] = -lims[:, 1]
                attrs['limits'] = list(lims)
        m_uv = compute_local_frame_edge_features(mirrored, u, v, WS_WS_INTRA)
        check("mirroring flips sin_phi/sin_alpha and preserves the cosines",
              abs(m_uv[1] - f_uv[1]) < 1e-6 and abs(m_uv[2] + f_uv[2]) < 1e-6
              and abs(m_uv[3] - f_uv[3]) < 1e-6 and abs(m_uv[4] + f_uv[4]) < 1e-6)

    print("\n=== 3. Directionality check ===")
    # pick a ws-ws edge with both directions present
    ws_edge = None
    for u, v in g.edges():
        if g.nodes[u]['type'] == 'ws' and g.nodes[v]['type'] == 'ws' and (v, u) in g.edges():
            ws_edge = (u, v)
            break
    if ws_edge is None:
        print("  (no ws-ws reciprocal edge found in sample graph -- skipping)")
    else:
        u, v = ws_edge
        node_ids = list(g.nodes())
        id_map = {nid: i for i, nid in enumerate(node_ids)}
        expected = compute_local_frame_edge_features(g, u, v, WS_WS_INTRA)
        # this should land at PyG column where target == id_map[u] (i.e. u's own frame updates u)
        edge_index, edge_attr = build_edge_index_and_attr(g, id_map)
        target_col = (edge_index[1] == id_map[u]) & (edge_index[0] == id_map[v])
        matched_rows = edge_attr[target_col]
        found = any(torch.allclose(row, torch.tensor(expected, dtype=torch.float32), atol=1e-5) for row in matched_rows)
        check(f"phi_ij/alpha_ij for nx edge ({u},{v}) lands at column (source={v},target={u})", found)

    print("\n=== 3b. Mechanism 1: parallel room-room edges, unanchored pairs dropped ===")
    ws_room, room_ws = room_membership(g)
    node_ids = list(g.nodes())
    id_map = {nid: i for i, nid in enumerate(node_ids)}
    rr_edges = [(u, v) for u, v in g.edges()
                if g.nodes[u]['type'] == 'room' and g.nodes[v]['type'] == 'room']
    expected_cols = 0
    n_dropped = 0
    dropped_pairs = []
    for u, v in rr_edges:
        walls = shared_wall_candidates(g, u, v, room_ws)
        expected_cols += len(walls)
        if not walls:
            n_dropped += 1
            dropped_pairs.append((u, v))
    actual_cols = 0
    for u, v in rr_edges:
        mask = (data.edge_index[0] == id_map[v]) & (data.edge_index[1] == id_map[u])
        actual_cols += int(mask.sum())
    check(f"room-room columns == accepted shared walls, unanchored pairs contribute 0 "
          f"({len(rr_edges)} graph edges -> {expected_cols} cols, {n_dropped} dropped)",
          actual_cols == expected_cols)
    # The edge is DROPPED, not degraded: shared_wall_candidates gates existence now.
    check("an unanchored room-room pair emits no column at all",
          all(int((((data.edge_index[0] == id_map[v]) &
                    (data.edge_index[1] == id_map[u])).sum())) == 0 for u, v in dropped_pairs))
    check("edge_attr rows == edge_index columns after augmentation",
          data.edge_attr.shape[0] == data.edge_index.shape[1])

    print("\n=== 3c. Mechanism 2: cross-room ws-ws edges ===")
    pairs_x = cross_room_surface_pairs(g, ws_room, room_ws)
    print(f"  {len(pairs_x)} cross-room surface pairs found")
    check("every cross-room pair satisfies the ported same_physical_wall rule",
          all(same_physical_wall(g, a, b) for a, b in pairs_x))
    check("cross-room pairs always span two DIFFERENT rooms",
          all(ws_room[a] != ws_room[b] for a, b in pairs_x))
    sym_ok = True
    for a, b in pairs_x:
        fwd = ((data.edge_index[0] == id_map[b]) & (data.edge_index[1] == id_map[a])).sum()
        rev = ((data.edge_index[0] == id_map[a]) & (data.edge_index[1] == id_map[b])).sum()
        sym_ok &= bool(fwd >= 1 and rev >= 1)
    check("both directions emitted for every cross-room pair (graph stays reciprocal)", sym_ok)
    check("cross-room edges are new, not duplicates of existing nx edges",
          all(not g.has_edge(a, b) and not g.has_edge(b, a) for a, b in pairs_x))

    print("\n=== 3d. Directionality for the NEW edge classes ===")
    # Same invariant as section 3, but for a wall-anchored room-room edge and a cross-room
    # edge -- these were added after the original directionality test was written.
    checked_rr = False
    for u, v in rr_edges:
        walls = shared_wall_candidates(g, u, v, room_ws)
        if not walls:
            continue
        mask = (data.edge_index[0] == id_map[v]) & (data.edge_index[1] == id_map[u])
        rows = data.edge_attr[mask]
        check(f"room-room ({u},{v}) lands at (source={v},target={u}) and is wall-anchored",
              bool(rows.shape[0] == len(walls) and (rows[:, EDGE_TYPE_COL0 + ROOM_ROOM] == 1.0).all()))
        checked_rr = True
        break
    if not checked_rr:
        print("  (no wall-anchored room-room edge in sample graph -- skipping)")
    if pairs_x:
        a, b = pairs_x[0]
        expected_ab = torch.tensor(compute_local_frame_edge_features(g, a, b, WS_WS_INTER),
                                   dtype=torch.float32)
        mask = (data.edge_index[0] == id_map[b]) & (data.edge_index[1] == id_map[a])
        rows = data.edge_attr[mask]
        check(f"cross-room ({a},{b}) computed in a's frame lands at target=a",
              any(torch.allclose(r, expected_ab, atol=1e-5) for r in rows))
    else:
        print("  (no cross-room pair in sample graph -- skipping)")

    print("\n=== 3e. edge_attr normalization ===")
    attrs = [nx_to_pyg(gr).edge_attr for gr in original_graphs]
    e_mean, e_std = compute_edge_mean_std(attrs)
    normed = torch.cat([normalize_edge_attr(a, e_mean, e_std) for a in attrs], dim=0)
    raw = torch.cat(attrs, dim=0)
    check("d_ij (col 0) standardized to ~zero mean / unit std",
          abs(float(normed[:, 0].mean())) < 0.15 and abs(float(normed[:, 0].std()) - 1.0) < 0.15)
    check("angle cols 1-4 untouched (still exactly in [-1, 1])",
          bool(torch.equal(normed[:, 1:5], raw[:, 1:5])))
    _t0, _t1 = EDGE_TYPE_COL0, EDGE_TYPE_COL0 + len(EDGE_TYPE_NAMES)
    check(f"type one-hot cols {_t0}-{_t1 - 1} untouched (still exactly {{0,1}})",
          bool(torch.equal(normed[:, _t0:_t1], raw[:, _t0:_t1])))
    # Derived from the constants, not hardcoded: this layout has already shifted twice
    # (length_w removal moved the block from 6-9 to 5-8) and a stale slice here would keep
    # passing while checking the wrong columns.
    check("normalized cols are exactly EDGE_NORM_COLS",
          all(bool(torch.equal(normed[:, c], raw[:, c])) is (c not in EDGE_NORM_COLS)
              for c in range(EDGE_ATTR_DIM)))

    print("\n=== 4. Pairing + permutation ===")
    pairs = []
    for i in range(3):
        generate_matching_pair(original_graphs[i % len(original_graphs)], noise_graphs[i], pairs)
    g1_pyg, g2_pyg, P = pairs[0]
    check("P shape == (|g1|, |g2|)", tuple(P.shape) == (g1_pyg.num_nodes, g2_pyg.num_nodes))
    check("P has at least one positive match", P.sum().item() > 0)
    # spot-check: a permuted node's edge_attr should match some pre-permutation edge's values
    check("g2_pyg.edge_attr has same dtype/shape contract as g1_pyg", g2_pyg.edge_attr.shape[1] == EDGE_ATTR_DIM)

    print("\n=== 5. Batching ===")
    from torch_geometric.data import Batch
    data1_list = [p[0] for p in pairs]
    data2_list = [p[1] for p in pairs]
    batch1 = Batch.from_data_list(data1_list)
    batch2 = Batch.from_data_list(data2_list)
    check("batch1.edge_attr.shape[0] == batch1.edge_index.shape[1]", batch1.edge_attr.shape[0] == batch1.edge_index.shape[1])
    check("batch1.edge_attr.shape[1] == EDGE_ATTR_DIM", batch1.edge_attr.shape[1] == EDGE_ATTR_DIM)

    print("\n=== 6. Forward/backward smoke test ===")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # import the real model class + weighted_bce_loss from the training script without
    # executing its (dataset-loading) module-level code, by exec-ing only the class/function
    # defs we need would be fragile; instead re-declare the tiny bits needed here, matching
    # pgm_training_adj_glob_edgefeat.py's EdgeAwareGATLayer/MatchingModel_MLPGATv2SinkhornWBCE
    # exactly (kept in sync manually -- this is a throwaway dev check, not shipped pipeline code).
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv
    import pygmtools
    pygmtools.BACKEND = 'pytorch'

    class EdgeAwareGATLayer(nn.Module):
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
                return x_new, None
            src, dst = edge_index[0], edge_index[1]
            edge_new = self.g_e(torch.cat([x[dst], edge_attr, x[src]], dim=-1))
            return x_new, edge_new

    class TinyModel(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, edge_dim, edge_hidden_dim, num_layers, heads):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
            self.edge_proj = nn.Linear(edge_dim, edge_hidden_dim)
            self.gnn = nn.ModuleList()
            dims = [hidden_dim] * num_layers + [out_dim]
            for i in range(num_layers):
                self.gnn.append(EdgeAwareGATLayer(dims[i], dims[i+1], edge_hidden_dim, edge_hidden_dim, heads=heads))
            self.inst_norm = nn.InstanceNorm2d(1, affine=True)

        def encode(self, x, edge_index, edge_attr):
            edge_attr = self.edge_proj(edge_attr)
            for i, layer in enumerate(self.gnn):
                is_last = i == len(self.gnn) - 1
                x, edge_attr = layer(x, edge_index, edge_attr, update_edge=not is_last)
            return x

        def forward(self, batch1, batch2):
            h1 = self.encode(self.mlp(batch1.x), batch1.edge_index, batch1.edge_attr)
            h2 = self.encode(self.mlp(batch2.x), batch2.edge_index, batch2.edge_attr)
            sim = torch.matmul(h1, h2.T).unsqueeze(0).unsqueeze(1)
            sim = self.inst_norm(sim).squeeze(0).squeeze(0)
            return h1, h2, sim

    d1, d2, P_single = pairs[0]

    # Deterministic init, and the gradient probe is repeated over several seeds.
    #
    # A model this small can land on an init where the FINAL g_v ReLU is entirely dead: h1 and
    # h2 come out all-zero, so nothing anywhere receives gradient and the edge path looks broken
    # when it is fine. Measured at ~5% of random inits (3/60), and since the model used to be
    # built from whatever RNG state the earlier sections left behind, this check failed
    # intermittently. Such inits are degenerate -- they prove nothing in either direction -- so
    # they are identified and skipped, and the property is asserted on every remaining seed
    # rather than on a single lucky one.
    SEEDS = (0, 1, 2, 3, 4)
    shape_checked = False
    degenerate, edge_grad_ok, bad_params = [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = TinyModel(in_dim=d1.x.size(1), hidden_dim=8, out_dim=4, edge_dim=EDGE_ATTR_DIM,
                          edge_hidden_dim=8, num_layers=2, heads=2)
        h1, h2, sim = model(d1, d2)
        if not shape_checked:
            check("output h1 shape[-1] == out_dim", h1.shape[-1] == 4)
            check("output h2 shape[-1] == out_dim", h2.shape[-1] == 4)
            shape_checked = True
        F.binary_cross_entropy(torch.sigmoid(sim), P_single).backward()

        if bool((h1 == 0).all()) and bool((h2 == 0).all()):
            degenerate.append(seed)          # dead final ReLU -- uninformative, not a failure
            continue

        # The last layer's g_e is intentionally skipped (update_edge=False) since encode()
        # discards edge_attr after the last hop -- its params legitimately have no grad.
        skip_prefix = f"gnn.{len(model.gnn) - 1}.g_e"
        bad_params += [f"seed{seed}:{n}" for n, p in model.named_parameters()
                       if not n.startswith(skip_prefix)
                       and (p.grad is None or not torch.isfinite(p.grad).all())]
        edge_grad_ok.append(any(
            n.startswith("gnn") and ("g_e" in n or "gath.lin_edge" in n)
            and p.grad is not None and p.grad.abs().sum() > 0
            for n, p in model.named_parameters()))

    if degenerate:
        print(f"  (seeds {degenerate} produced an all-zero embedding -- dead final ReLU, skipped)")
    if bad_params:
        print(f"  non-finite/missing grad params: {bad_params}")

    check("at least one non-degenerate init to test", len(edge_grad_ok) > 0)
    check("all params (except the discarded last-layer edge update) have finite grads",
          len(bad_params) == 0)
    check(f"edge-path params (g_e / gath.lin_edge) have NON-ZERO grad on EVERY non-degenerate "
          f"init ({sum(edge_grad_ok)}/{len(edge_grad_ok)}) -- proof edge_attr is in the gradient path",
          len(edge_grad_ok) > 0 and all(edge_grad_ok))

    print("\n=== 7. Determinism ===")
    torch.manual_seed(42)
    pairs_a = []
    for i in range(3):
        generate_matching_pair(original_graphs[i % len(original_graphs)], noise_graphs[i], pairs_a)
    names_a = [(p[0].name, p[1].permutation.tolist()) for p in pairs_a]
    torch.manual_seed(42)
    pairs_b = []
    for i in range(3):
        generate_matching_pair(original_graphs[i % len(original_graphs)], noise_graphs[i], pairs_b)
    names_b = [(p[0].name, p[1].permutation.tolist()) for p in pairs_b]
    check("same seed -> identical pairing/permutation", names_a == names_b)

    print("\n=== 8. Real-validation loader (if reachable) ===")
    if os.path.isdir(REAL_DIR):
        import networkx as nx

        class _GraphWrapper:
            """Stub for situational_graphs_wrapper.GraphWrapper -- only its .graph attr is needed."""
            pass

        class _WrapperUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if 'situational_graphs_wrapper' in module or name == 'GraphWrapper':
                    return _GraphWrapper
                return super().find_class(module, name)

        def load_situational_graph(path):
            with open(path, 'rb') as f:
                obj = _WrapperUnpickler(f).load()
            if isinstance(obj, nx.Graph):
                return obj
            g = getattr(obj, 'graph', None)
            if isinstance(g, nx.Graph):
                return g
            raise TypeError(f"Unexpected object in {path}: {type(obj)}")

        envs = sorted(d for d in os.listdir(REAL_ROOT)
                      if os.path.isdir(os.path.join(REAL_ROOT, d)))
        for env in envs:
            try:
                loaded = {}
                for side in ("Prior", "Online"):
                    graph = load_situational_graph(os.path.join(REAL_ROOT, env, f"{side}.pkl"))
                    loaded[side] = (graph, nx_to_pyg(graph))
                ok = all(d.edge_attr.shape[1] == EDGE_ATTR_DIM and d.edge_attr.shape[0] ==
                         d.edge_index.shape[1] and torch.isfinite(d.edge_attr).all()
                         for _, d in loaded.values())
                n_x = sum(len(cross_room_surface_pairs(gr, *room_membership(gr)))
                          for gr, _ in loaded.values())
                check(f"real scene {env}: Prior+Online convert, {n_x} cross-room pairs", ok)
            except Exception as e:
                check(f"real scene {env} conversion failed: {e}", False)

        print("\n=== 9. Prior-vs-Online drift per edge_attr column (diagnostic) ===")
        # Regression guard with no training run: if WALL_NORMAL_DIST_THRESHOLD drifts out of
        # sync with graph_matching_node.py, or a frame construction regresses, the numbers
        # below move. Baselines measured during design: alpha ~0.002, room-room cos/sin
        # 0.000/0.005, cross-room phi 0.18-0.27 (short edges, midpoint slide dominates).
        report_edge_drift(load_situational_graph, envs)
    else:
        print(f"  (real dir not found at {REAL_DIR} -- skipping)")

    check_fully_topology()

    print(f"\n{N_PASS}/{N_CHECK} checks passed")
    sys.exit(0 if N_PASS == N_CHECK else 1)


if __name__ == "__main__":
    main()
