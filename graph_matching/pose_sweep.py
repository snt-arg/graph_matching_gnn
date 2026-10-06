#!/usr/bin/env python3
"""Pose-robustness sweep for a partial-graph-matching model, per environment.

Answers: *given a robot holding the A-graph as a prior map, walking in from an
arbitrary start pose and building an S-graph, over what range of start HEADINGS
and start POSITIONS does the model still match correctly?*

The robot starts at physical point ``p`` with heading ``phi``; its SLAM frame has
origin at ``p`` and x-axis along ``phi``, so a wall at map coordinate ``a`` is
recorded in S as ``s = R_phi^T (a - p)``.  Therefore:

  * varying HEADING alone rotates S's coordinates about the point that is
    ``(0,0)`` in S's own frame -- i.e. **rotation about the world origin**, the
    origin of the coordinate system whose numbers the model actually reads;
  * varying START POSITION alone **translates** S.

Two independent knobs spanning SE(2).  A is the prior map and never moves.

Note on the dashboard's "Drift <n> deg" button: it pivots on S's GT-matched
centroid (dashboard.py:2607), a point derived from ground truth that a robot
never has.  That is outside the family above -- it mixes a heading change with a
start-position change -- so it is measured here as a clearly-labelled secondary
condition only, to explain the curve the GUI produces rather than to bound the
model's rotation tolerance.

Geometry helpers below reimplement `_transform_b_inplace` / `_gt_rigid_alignment`
/ `_relative_transform` from dashboard.py verbatim, deliberately WITHOUT
importing dashboard -- that would drag in PyQt5 and the matplotlib Qt backend.

Trap worth recording: the dashboard's `GnnMatcher._signature` (dashboard.py:944)
keys its cache only on ``(name, |V|, |E|)``, so a pose change does NOT invalidate
it.  This script calls the model directly and never touches that cache, but any
future sweep that wraps `GnnMatcher` must call `clear_cache()` between poses or
it will silently report one F1 repeated.
"""

import argparse
import copy
import csv
import json
import os
import pickle
import sys
import time

import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/root/workspace/src/graph_matching_gnn/graph_matching")
sys.path.insert(0, "/root/workspace/src/graph_matching/graph_matching")
from PGM_class import (PartialGraphMatching, MatchingModel_MLPGATv2SinkhornWBCE,
                       nx_to_pyg_data_preserve_order, normalize_graph)

GNN_ROOT = "/root/workspace/src/graph_matching_gnn/GNN"
DEFAULT_SAMPLES = "/root/workspace/src/graph_matching/graph_matching/MSD_Samples"


# ── dashboard geometry, verbatim (dashboard.py:206 / :403 / :447) ─────────────

def transform_inplace(g, dx, dy, theta_rad, pivot):
    """Rotate g about `pivot`, then translate by (dx, dy). Mutates in place."""
    c = np.asarray(pivot, dtype=float)
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                  [np.sin(theta_rad),  np.cos(theta_rad)]])
    off = np.array([dx, dy], dtype=float)

    def tp(p):
        p = np.asarray(p, dtype=float)
        xy = R @ (p[:2] - c) + c + off
        return np.concatenate([xy, p[2:]]).tolist() if p.shape[0] >= 3 else xy.tolist()

    for _, at in g.nodes(data=True):
        if "center" in at:
            at["center"] = tp(at["center"])
        if "limits" in at:
            L = np.asarray(at["limits"], dtype=float)
            at["limits"] = np.array([tp(L[i]) for i in range(L.shape[0])]).tolist()
        if "normal" in at:
            n = np.asarray(at["normal"], dtype=float)
            nn = R @ n[:2]
            at["normal"] = (np.concatenate([nn, n[2:]]).tolist()
                            if n.shape[0] >= 3 else nn.tolist())


def gt_points(ga, gs, gt):
    ai = {str(n): n for n in ga.nodes()}
    si = {str(n): n for n in gs.nodes()}
    A, S = [], []
    for x, y in gt:
        na, ns = ai.get(x), si.get(y)
        if na is None or ns is None:
            continue
        A.append(np.asarray(ga.nodes[na]["center"], float)[:2])
        S.append(np.asarray(gs.nodes[ns]["center"], float)[:2])
    return np.array(A), np.array(S)


def gt_rigid_alignment(ga, gs, gt):
    """(dx, dy, theta_deg, pivot) aligning S onto A -- 2D Kabsch on GT pairs."""
    A, S = gt_points(ga, gs, gt)
    if len(A) < 2:
        return 0.0, 0.0, 0.0, None
    ac, sc = A.mean(0), S.mean(0)
    h = (S - sc).T @ (A - ac)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag([1.0, d]) @ u.T
    theta_deg = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
    dx, dy = (ac - sc).tolist()
    return dx, dy, theta_deg, sc


def relative_transform(ga, gs, gt):
    """What the dashboard's "Relative graph informations" box would show."""
    dx, dy, th, _ = gt_rigid_alignment(ga, gs, gt)
    return -dx, -dy, -th


def centers_of(g):
    return {n: np.asarray(at["center"], float)[:2] for n, at in g.nodes(data=True)}


def max_node_shift(g_ref, g_now):
    a, b = centers_of(g_ref), centers_of(g_now)
    return float(max(np.linalg.norm(b[n] - a[n]) for n in a)) if a else 0.0


# ── model ────────────────────────────────────────────────────────────────────

class Matcher:
    def __init__(self, model_name, subfolder="ws_room_dropout_noise_inc"):
        pgm = PartialGraphMatching(
            model_class=MatchingModel_MLPGATv2SinkhornWBCE,
            data_paths={
                "equal": os.path.join(GNN_ROOT, "preprocessed/graph_matching/equal"),
                "partial": os.path.join(GNN_ROOT, "preprocessed/partial_graph_matching", subfolder)},
            model_save_path=os.path.join(GNN_ROOT, "models/partial_graph_matching", model_name),
            device=torch.device("cpu"), in_dim=7, inference_only=True)
        pgm.load_best_model()
        self.model = pgm.model
        self.model.eval()
        self.mean, self.std = pgm.mean, pgm.std

    @staticmethod
    def _coerce(g):
        g = copy.deepcopy(g)
        for _, at in g.nodes(data=True):
            for k in ("center", "normal"):
                if k in at:
                    v = at[k]
                    at[k] = (v.tolist() if hasattr(v, "tolist") else v)[:2]
        return g

    def evaluate(self, ga, gs, gt):
        """(f1_perm, f1_hung, cos_a, cos_s). f1_perm is what the dashboard shows."""
        g1 = nx_to_pyg_data_preserve_order(self._coerce(ga))
        g2 = nx_to_pyg_data_preserve_order(self._coerce(gs))
        g1, g2 = normalize_graph(g1, g2, self.mean, self.std)
        b1 = torch.zeros(g1.num_nodes, dtype=torch.long)
        b2 = torch.zeros(g2.num_nodes, dtype=torch.long)
        with torch.no_grad():
            perm, emb, _s, _aff, sink = self.model(g1, g2, b1, b2, inference=True,
                                                   return_soft=True, return_intermediate=True)
        an, sn = list(ga.nodes()), list(gs.nodes())
        P = perm[0].cpu().numpy()
        S = sink[0].cpu().numpy()
        # dashboard path: _pairs_from_perm (dashboard.py:664)
        pred = {(str(an[i]), str(sn[j])) for i, j in zip(*np.nonzero(P))}
        ri, ci = linear_sum_assignment(-S)
        hung = {(str(an[i]), str(sn[j])) for i, j in zip(ri, ci) if S[i, j] > 1.0 / max(S.shape)}
        gts = {(str(x), str(y)) for x, y in gt}
        h1, h2 = emb[0]
        return _f1(pred, gts), _f1(hung, gts), _offdiag_cos(h1), _offdiag_cos(h2)


def _f1(pred, gts):
    tp = len(pred & gts)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gts) if gts else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def make_matcher(model_name, force_edge=None):
    """Build the right matcher for ``model_name``.

    Two incompatible architectures live side by side under
    ``GNN/models/partial_graph_matching/``: the 7-feature node-only family that ``Matcher``
    above handles, and the edge-feature rework (node x 3-dim, ``edge_attr`` 9-dim, a different
    class and a different ``forward`` contract). They are told apart by whether the folder's
    ``norm_stats.pt`` carries ``edge_mean`` -- a property of the trained artifact itself, so it
    cannot drift out of sync the way a name convention or a config field can.
    """
    folder = os.path.join(GNN_ROOT, "models/partial_graph_matching", model_name)
    import edge_model  # local import: pulls in edge_features, only needed on the edge path
    use_edge = edge_model.is_edge_model(folder) if force_edge is None else force_edge
    if use_edge:
        print(f"[matcher] edge-feature model detected: {model_name}")
        return edge_model.EdgeMatcher(folder)
    print(f"[matcher] node-only (7-feature) model: {model_name}")
    return Matcher(model_name)


def _offdiag_cos(h):
    h = torch.nn.functional.normalize(h, dim=1)
    M = (h @ h.T).cpu().numpy()
    n = M.shape[0]
    return float((M.sum() - np.trace(M)) / (n * n - n)) if n > 1 else 0.0


# ── env loading ──────────────────────────────────────────────────────────────

def load_gt(path):
    if not os.path.exists(path):
        return set()
    raw = json.load(open(path))
    st = lambda v, pre: str(v)[len(pre):] if str(v).startswith(pre) else str(v)
    out = set()
    for online, prior in raw.get("rooms", {}).items():
        if prior != "??":
            out.add((st(prior, "a_"), st(online, "s_")))
    for e in raw.get("ws", []):
        if len(e) == 2 and e[1] != "??":
            out.add((st(e[1], "a_"), st(e[0], "s_")))
    return out


def load_env(d):
    def lg(p):
        o = pickle.load(open(p, "rb"))
        return o.graph if hasattr(o, "graph") and not isinstance(o, nx.Graph) else o
    return lg(f"{d}/Prior.pkl"), lg(f"{d}/Online.pkl"), load_gt(f"{d}/ground_truth.json")


def aligned_pair(ga0, gs0, gt):
    """A, S after the dashboard's "Align centers + rot" -- the registered baseline."""
    ga, gs = copy.deepcopy(ga0), copy.deepcopy(gs0)
    dx, dy, th, piv = gt_rigid_alignment(ga, gs, gt)
    if piv is not None:
        transform_inplace(gs, dx, dy, np.deg2rad(th), piv)
    return ga, gs


def row(env, cond, param, axis, m, ga, gs, gt, gs_ref):
    f1p, f1h, ca, cs = m.evaluate(ga, gs, gt)
    dx, dy, dth = relative_transform(ga, gs, gt)
    return dict(env=env, condition=cond, param=round(float(param), 4), axis=axis,
                f1_perm=round(f1p, 4), f1_hung=round(f1h, 4),
                disp_x=round(dx, 6), disp_y=round(dy, 6), disp_theta=round(dth, 6),
                max_node_shift=round(max_node_shift(gs_ref, gs), 4),
                cos_a=round(ca, 4), cos_s=round(cs, 4))


FIELDS = ["env", "condition", "param", "axis", "f1_perm", "f1_hung",
          "disp_x", "disp_y", "disp_theta", "max_node_shift", "cos_a", "cos_s"]


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def tolerance(curve, frac):
    """Largest |param| in each direction for which F1 has not yet fallen below
    ``frac * baseline``.  Scans outward from 0 and stops at the first breach, so
    a later recovery cannot inflate the reported range.

    ``curve`` maps param -> F1 and must contain 0.0.  Returns (neg, pos) as
    positive magnitudes, or (nan, nan) when the baseline is already 0.
    """
    base = curve[0.0]
    if base <= 0:
        return float("nan"), float("nan")
    thr = frac * base
    out = []
    for side in (+1, -1):
        reach = 0.0
        for p in sorted((p for p in curve if p * side > 0), key=lambda p: abs(p)):
            if curve[p] < thr:
                break
            reach = abs(p)
        out.append(reach)
    return out[1], out[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="adj_glob_65",
                    help="FOLDER name under GNN/models/partial_graph_matching/ "
                         "(not a dashboard alias).")
    ap.add_argument("--edge", dest="edge", action="store_true", default=None,
                    help="Force the edge-feature matcher (3-dim node x + 9-dim edge_attr). "
                         "Default: auto-detected from norm_stats.pt carrying edge_mean.")
    ap.add_argument("--no-edge", dest="edge", action="store_false",
                    help="Force the 7-feature node-only matcher.")
    ap.add_argument("--samples-dir", default=DEFAULT_SAMPLES)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--rot-step", type=float, default=10.0)
    ap.add_argument("--trans-max", type=float, default=60.0)
    ap.add_argument("--trans-step", type=float, default=0.5)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["rotation", "translation", "joint"])
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.samples_dir, "sample_manifest.json")))
    envs = a.envs if a.envs else [s["env"] for s in man["samples"]]
    m = make_matcher(a.model, a.edge)
    t_start = time.time()

    # ── anchor: F1 at the stored pose, and the aligned baseline ──────────────
    print(f"\nmodel {a.model} | {len(envs)} envs\n")
    print(f"{'env':<18} {'|A|':>4} {'|S|':>4} {'stored F1':>10} {'aligned F1':>11} "
          f"{'theta0':>8} {'align dxy':>10}")
    loaded = {}
    for env in envs:
        ga0, gs0, gt = load_env(os.path.join(a.samples_dir, env))
        dx, dy, th, piv = gt_rigid_alignment(ga0, gs0, gt)
        ga, gs = aligned_pair(ga0, gs0, gt)
        f_stored = m.evaluate(ga0, gs0, gt)[0]
        f_align = m.evaluate(ga, gs, gt)[0]
        loaded[env] = (ga, gs, gt, f_align)
        print(f"{env:<18} {ga0.number_of_nodes():4d} {gs0.number_of_nodes():4d} "
              f"{f_stored:10.3f} {f_align:11.3f} {-th:8.1f} {np.hypot(dx, dy):10.2f}")

    # ── verification 3: derived topology is rigid-invariant ──────────────────
    import topology_rules
    print("\n[check] topology re-derivation under rigid transform")
    for env in envs[:3]:
        ga, gs, gt, _ = loaded[env]
        def edges_after(g):
            h = copy.deepcopy(g)
            topology_rules.rebuild_all(h)
            return set(h.edges())
        base = edges_after(gs)
        probes = {}
        for tag, (dxx, dyy, dd, piv) in {
                "rot180@origin": (0, 0, 180.0, (0.0, 0.0)),
                "trans+60x": (60.0, 0, 0.0, (0.0, 0.0)),
                "trans-60y": (0, -60.0, 0.0, (0.0, 0.0))}.items():
            h = copy.deepcopy(gs)
            transform_inplace(h, dxx, dyy, np.deg2rad(dd), piv)
            probes[tag] = edges_after(h) == base
        ok = all(probes.values())
        print(f"  {env:<18} {'INVARIANT' if ok else 'CHANGED  '}  {probes}")

    rot_rows, trans_rows, joint_rows = [], [], []

    # ── Step 4: rotation = robot heading ─────────────────────────────────────
    if "rotation" not in a.skip:
        angles = np.arange(0.0, 360.0, a.rot_step)
        print(f"\n[rotation] {len(angles)} angles x 5 conditions x {len(envs)} envs")
        for env in envs:
            ga, gs, gt, _ = loaded[env]
            _, s_pts = gt_points(ga, gs, gt)
            gt_centroid = s_pts.mean(0) if len(s_pts) else (0.0, 0.0)
            # `drift_recentred` moves the WHOLE pair so its GT centroid lands on the
            # world origin before the Drift rotation is applied, i.e. |q| = 0 by
            # construction. The relative A<->S pose is untouched by that move, so any
            # change it produces is pure absolute-coordinate sensitivity.
            qx, qy = float(gt_centroid[0]), float(gt_centroid[1])
            ga_rc = copy.deepcopy(ga)
            transform_inplace(ga_rc, -qx, -qy, 0.0, (0.0, 0.0))
            gs_rc = copy.deepcopy(gs)
            transform_inplace(gs_rc, -qx, -qy, 0.0, (0.0, 0.0))
            # `s_at_origin` moves ONLY S onto the world origin and leaves A where it
            # is -- the real deployment layout: S expressed in the robot's own SLAM
            # frame (origin at its start point), A in the map frame, nothing aligned.
            # The A<->S offset is therefore |q|, and S's own |q| is 0.
            gs_o = gs_rc

            for cond in ("heading", "drift_button", "both_origin",
                         "drift_recentred", "s_at_origin"):
                for d in angles:
                    g_s = copy.deepcopy(gs)
                    g_a, ref = ga, gs
                    if cond == "heading":
                        transform_inplace(g_s, 0, 0, np.deg2rad(d), (0.0, 0.0))
                    elif cond == "drift_button":
                        transform_inplace(g_s, 0, 0, np.deg2rad(d), gt_centroid)
                    elif cond == "both_origin":
                        transform_inplace(g_s, 0, 0, np.deg2rad(d), (0.0, 0.0))
                        g_a = copy.deepcopy(ga)
                        transform_inplace(g_a, 0, 0, np.deg2rad(d), (0.0, 0.0))
                    elif cond == "drift_recentred":
                        g_a, ref = ga_rc, gs_rc
                        g_s = copy.deepcopy(gs_rc)
                        transform_inplace(g_s, 0, 0, np.deg2rad(d), (0.0, 0.0))
                    else:
                        g_a, ref = ga, gs_o
                        g_s = copy.deepcopy(gs_o)
                        transform_inplace(g_s, 0, 0, np.deg2rad(d), (0.0, 0.0))
                    rot_rows.append(row(env, cond, d, "-", m, g_a, g_s, gt, ref))
            print(f"  {env:<18} done  ({time.time()-t_start:6.1f}s)")
        write_csv(os.path.join(a.out_dir, "pose_sweep_rotation.csv"), rot_rows)

    # ── Step 5: translation = robot start position, X then Y ─────────────────
    if "translation" not in a.skip:
        offs = np.round(np.arange(-a.trans_max, a.trans_max + 1e-9, a.trans_step), 4)
        print(f"\n[translation] {len(offs)} offsets x 2 axes x {len(envs)} envs")
        for env in envs:
            ga, gs, gt, _ = loaded[env]
            for axis in ("x", "y"):
                for t in offs:
                    g_s = copy.deepcopy(gs)
                    transform_inplace(g_s, t if axis == "x" else 0.0,
                                      t if axis == "y" else 0.0, 0.0, (0.0, 0.0))
                    trans_rows.append(row(env, "translation", t, axis, m, ga, g_s, gt, gs))
            print(f"  {env:<18} done  ({time.time()-t_start:6.1f}s)")
        write_csv(os.path.join(a.out_dir, "pose_sweep_translation.csv"), trans_rows)

    # ── Step 6: joint heading x offset grid ──────────────────────────────────
    if "joint" not in a.skip:
        print(f"\n[joint] 5 headings x 6 offsets x {len(envs)} envs")
        for env in envs:
            ga, gs, gt, _ = loaded[env]
            for d in (0, 45, 90, 135, 180):
                for t in (0, 5, 10, 20, 40, 60):
                    g_s = copy.deepcopy(gs)
                    transform_inplace(g_s, 0, 0, np.deg2rad(d), (0.0, 0.0))
                    transform_inplace(g_s, t, 0.0, 0.0, (0.0, 0.0))
                    r = row(env, "joint", t, f"deg{d}", m, ga, g_s, gt, gs)
                    r["condition"] = f"joint_deg{d}"
                    joint_rows.append(r)
            print(f"  {env:<18} done  ({time.time()-t_start:6.1f}s)")
        write_csv(os.path.join(a.out_dir, "pose_sweep_joint.csv"), joint_rows)

    # ── verification 2: the "Relative graph informations" panel ──────────────
    if rot_rows:
        print("\n[check] dashboard 'Relative graph informations' panel")
        bad = [r for r in rot_rows if r["condition"] == "drift_button"
               and (abs(r["disp_x"]) > 1e-6 or abs(r["disp_y"]) > 1e-6
                    or abs(((r["disp_theta"] - r["param"] + 180) % 360) - 180) > 1e-4)]
        print(f"  drift_button rows reading a non-zero displacement: {len(bad)}"
              f" / {sum(1 for r in rot_rows if r['condition']=='drift_button')}"
              f"   -> {'PANEL CORRECT' if not bad else 'BUG'}")
        hd = [r for r in rot_rows if r["condition"] == "heading"]
        mx = max(np.hypot(r["disp_x"], r["disp_y"]) for r in hd)
        print(f"  heading rows: max |disp| = {mx:.2f} m  (non-zero as expected -- a"
              f" heading change does relocate S in map coordinates)")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("TOLERANCE RANGES  (largest excursion holding F1 >= frac x aligned baseline)")
    print("=" * 100)
    hdr = (f"{'env':<18} {'base':>5} | {'heading 95%':>11} {'90%':>11} {'50%':>11} |"
           f" {'X 95% -/+':>13} {'Y 95% -/+':>13}")
    print(hdr)
    for env in envs:
        base = loaded[env][3]
        line = f"{env:<18} {base:5.3f} |"
        if rot_rows:
            c = {r["param"]: r["f1_perm"] for r in rot_rows
                 if r["env"] == env and r["condition"] == "heading"}
            c2 = {p if p <= 180 else p - 360: v for p, v in c.items()}
            for frac in (0.95, 0.90, 0.50):
                n, p = tolerance(c2, frac)
                span = "full 360" if (n >= 170 and p >= 170) else f"{-n:.0f}/{p:+.0f}"
                line += f" {span:>11}"
        line += " |"
        if trans_rows:
            for axis in ("x", "y"):
                c = {r["param"]: r["f1_perm"] for r in trans_rows
                     if r["env"] == env and r["axis"] == axis}
                n, p = tolerance(c, 0.95)
                line += f" {-n:6.1f}/{p:+5.1f}"
        print(line)
    print(f"\ntotal {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
