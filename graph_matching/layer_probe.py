#!/usr/bin/env python3
"""Per-layer embedding-separation probe for a partial-graph-matching model.

Answers: *at which message-passing hop does an environment's node embeddings stop being
distinguishable, and does that track how self-similar the environment is?*

`forward` only ever returns the FINAL layer's embeddings (`all_embeddings` is filled from `h1`/`h2`
after `encode` has run to completion), so per-layer values require re-running the encoder's own
loop.  Two details in that loop are load-bearing:

  * `encode` applies ReLU + dropout to every layer EXCEPT the last, so a naive readout of layer 2
    returns the PRE-activation embedding -- cos 0.86 instead of 0.40 on 47_basement.  The
    post-activation value is the one every recorded figure uses.
  * the published per-layer table is therefore post-activation for L1/L2 and RAW for L3, because
    L3 never gets a ReLU at all.  Reproducing it is the harness's own anchor check (`--anchor`).

Truncation (`--trunc k`) makes layer k the last one, which means the native `encode` would skip its
ReLU -- so the ReLU is re-applied by hand at the readout.  Without that, truncated-F1 is measured
off a representation the trained network never produced.

Geometry, model loading and F1 come from `pose_sweep`; model-free similarity metrics from
`env_similarity_metrics`.  Nothing here re-implements either.
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import types

import numpy as np

# PGM_class does `from moviepy.editor import ImageSequenceClip` at module scope and moviepy 2.x
# removed that submodule, so every import below it dies without this stub.
try:
    import moviepy.editor  # noqa: F401
except Exception:
    import moviepy
    _stub = types.ModuleType("moviepy.editor")
    _stub.ImageSequenceClip = getattr(moviepy, "ImageSequenceClip", None)
    sys.modules["moviepy.editor"] = _stub

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402
import torch.nn.functional as F                                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/workspace/src/graph_matching/graph_matching")

import pose_sweep as ps                                         # noqa: E402
import env_similarity_metrics as esm                            # noqa: E402
from PGM_class import nx_to_pyg_data_preserve_order, normalize_graph  # noqa: E402

GRAPH_DICTS = "/root/workspace/src/graph_matching/graph_matching/graph_dicts"
REAL = ["47_basement", "47_topfloor_modified", "CF12"]
MSD = ["MSD_1479_lvl55", "MSD_3276_lvl55", "MSD_2269_lvl15"]
GEN = ["GEN_base", "GEN_D_cv45", "GEN_D_cv65", "GEN_D_cv85",
       "GEN_R_n11", "GEN_R_n7", "GEN_R_n5", "GEN_M_1", "GEN_M_2", "GEN_M_3"]

# The anchor values every recorded figure in this project is expressed in.  Checked before any new
# number is trusted; a harness that cannot reproduce them is measuring something else.
ANCHORS = {
    "adj_glob_65": {
        "47_basement": dict(f1=0.000, cos=[(0.533, 0.472), (0.400, 0.378), (0.628, 0.637)]),
        "CF12": dict(f1=0.471, cos=[(0.390, 0.444), (0.290, 0.312), (0.463, 0.533)]),
    },
    "adj_no_glob_65": {
        "47_basement": dict(f1=0.950, cos=[None, (0.093, 0.064)]),
    },
}


def to_pyg(matcher, ga, gs):
    g1 = nx_to_pyg_data_preserve_order(matcher._coerce(ga))
    g2 = nx_to_pyg_data_preserve_order(matcher._coerce(gs))
    return normalize_graph(g1, g2, matcher.mean, matcher.std)


def layer_cosines(model, g1, g2):
    """Mean off-diagonal cosine after each GAT layer, post-activation except the final one."""
    out = []
    with torch.no_grad():
        for g in (g1, g2):
            x = model.mlp(g.x)
            per = []
            for i, conv in enumerate(model.gnn):
                x = conv(x, g.edge_index)
                if i < len(model.gnn) - 1:
                    x = F.relu(x)
                    per.append(ps._offdiag_cos(x))
                    x = model.dropout(x)
                else:
                    per.append(ps._offdiag_cos(x))   # L_last is raw: encode never ReLUs it
            out.append(per)
    return out[0], out[1]


def truncated(matcher, k):
    """A copy of the model read out at layer k, with the ReLU `encode` would have skipped.

    `encode` loops over `self.gnn`, so shortening that ModuleList genuinely makes it a k-layer
    network -- but it also promotes layer k to "last", and `encode` deliberately leaves the last
    layer un-activated.  Binding a replacement `encode` on the instance restores the activation
    that layer k had while it was an interior layer.
    """
    m = copy.deepcopy(matcher.model)
    full = len(m.gnn)
    m.gnn = nn.ModuleList(list(m.gnn)[:k])

    def encode(x, edge_index, _m=m, _truncated=(k < full)):
        for i, conv in enumerate(_m.gnn):
            x = conv(x, edge_index)
            if i < len(_m.gnn) - 1:
                x = F.relu(x)
                x = _m.dropout(x)
        return F.relu(x) if _truncated else x

    m.encode = encode
    m.eval()
    sub = copy.copy(matcher)
    sub.model = m
    return sub


def own_centroid(g):
    """The graph's own centre of mass -- the pivot for "turn it around itself"."""
    return np.array([at["center"][:2] for _, at in g.nodes(data=True)], dtype=float).mean(0)


def rotation_sweep(matcher, ga, gs, gt, step):
    """F1 with Online rotated about ITS OWN centroid. A fixed, Online turned in place.

    Not the same knob as `pose_sweep`'s `heading`, which pivots on the world origin.  Rotating a
    cloud whose centroid sits |q| from the origin by delta is algebraically an origin-rotation plus
    a translation of 2|q|sin(delta/2) -- here |q| = 4.75 m, so 180 deg smuggles in 9.5 m of
    displacement.  This series is therefore a diagonal through (rotation, translation), which is
    what "turn the scan around itself" physically is; read it as that, never as a pure rotation
    tolerance.
    """
    piv = own_centroid(gs)
    rows = []
    for deg in np.arange(0.0, 360.0, step):
        g_s = copy.deepcopy(gs)
        ps.transform_inplace(g_s, 0.0, 0.0, math.radians(float(deg)), piv)
        rows.append((float(deg), matcher.evaluate(ga, g_s, gt)[0]))
    return rows


def translation_sweep(matcher, ga, gs, gt, tmax, tstep):
    """F1 with Online displaced along X, then along Y, rotation held at zero."""
    rows = []
    for axis in ("x", "y"):
        for t in np.round(np.arange(-tmax, tmax + 1e-9, tstep), 4):
            g_s = copy.deepcopy(gs)
            ps.transform_inplace(g_s, float(t) if axis == "x" else 0.0,
                                 float(t) if axis == "y" else 0.0, 0.0, (0.0, 0.0))
            rows.append((axis, float(t), matcher.evaluate(ga, g_s, gt)[0]))
    return rows


def env_row(name, path, matcher, matcher_l2, step, tmax=0.0, tstep=0.5):
    ga, gs, gt = ps.load_env(path)
    g1, g2 = to_pyg(matcher, ga, gs)
    cos_a, cos_s = layer_cosines(matcher.model, g1, g2)
    f1p, f1h, final_a, final_s = matcher.evaluate(ga, gs, gt)
    f1_l2 = matcher_l2.evaluate(ga, gs, gt)[0] if matcher_l2 is not None else float("nan")

    ma = esm.all_metrics(gs, automorphisms=False)
    mp = esm.all_metrics(ga, automorphisms=False)
    q = float(np.hypot(*np.array([at["center"][:2] for _, at in gs.nodes(data=True)],
                                 dtype=float).mean(0)))

    row = dict(env=name, nodes_a=ga.number_of_nodes(), nodes_s=gs.number_of_nodes(),
               gt=len(gt), f1=round(f1p, 4), f1_hung=round(f1h, 4), f1_L2=round(f1_l2, 4),
               cv=round(ma["ws length CV"], 4), cv_prior=round(mp["ws length CV"], 4),
               n3=round(ma["|N_3| mean nodes"], 3), components=ma["components"],
               degvar=round(ma["degree variance"], 3),
               wl_classes=ma["WL classes"], len_mean=round(ma["ws length mean"], 3), q=round(q, 3))
    for i, (a, s) in enumerate(zip(cos_a, cos_s), start=1):
        row["cosA_L%d" % i] = round(a, 4)
        row["cosS_L%d" % i] = round(s, 4)
    # `evaluate`'s own final-layer cosine must equal the last entry of the manual loop; if it does
    # not, the re-implemented encoder has drifted from the one the model actually runs.
    assert abs(cos_a[-1] - final_a) < 1e-4 and abs(cos_s[-1] - final_s) < 1e-4, name

    rot = rotation_sweep(matcher, ga, gs, gt, step) if step > 0 else []
    if rot:
        vals = np.array([v for _, v in rot])
        best = int(np.argmax(vals))
        row["rot_best"] = round(float(vals[best]), 4)
        row["rot_best_deg"] = rot[best][0]
        row["rot_mean"] = round(float(vals.mean()), 4)
        row["rot_worst"] = round(float(vals.min()), 4)
        row["rot_at_0"] = round(float(vals[0]), 4)
    tr = translation_sweep(matcher, ga, gs, gt, tmax, tstep) if tmax > 0 else []
    if tr:
        vals = np.array([v for _, _, v in tr])
        row["trans_best"] = round(float(vals.max()), 4)
        row["trans_mean"] = round(float(vals.mean()), 4)
        for axis in ("x", "y"):
            curve = {t: v for a, t, v in tr if a == axis}
            neg, pos = ps.tolerance(curve, 0.95)
            row["tol_%s_neg" % axis] = round(neg, 2)
            row["tol_%s_pos" % axis] = round(pos, 2)
        halves = [row["tol_%s_%s" % (a, d)] for a in ("x", "y") for d in ("neg", "pos")]
        # the all-direction safe radius is the MIN of the four half-ranges, never their mean:
        # these curves are strongly asymmetric and averaging hides the direction that fails first.
        row["safe_radius"] = round(float(min(halves)), 2)
    return row, rot, tr


def check_anchors(model_name, matcher, matcher_l2, step):
    exp = ANCHORS.get(model_name, {})
    if not exp:
        print("[anchor] no recorded values for %s -- skipped" % model_name)
        return
    print("[anchor] %s" % model_name)
    for env, want in exp.items():
        row, _, _ = env_row(env, os.path.join(GRAPH_DICTS, env), matcher, matcher_l2, 0.0)
        ok = abs(row["f1"] - want["f1"]) < 0.01
        parts = []
        for i, pair in enumerate(want["cos"], start=1):
            if pair is None:
                continue
            got = (row["cosA_L%d" % i], row["cosS_L%d" % i])
            hit = abs(got[0] - pair[0]) < 0.005 and abs(got[1] - pair[1]) < 0.005
            ok &= hit
            parts.append("L%d %.3f/%.3f%s" % (i, got[0], got[1], "" if hit else " != %.3f/%.3f" % pair))
        print("  %-22s F1 %.3f (want %.3f)  %s   %s"
              % (env, row["f1"], want["f1"], "  ".join(parts), "OK" if ok else "MISMATCH"))
        if not ok:
            raise SystemExit("anchor check failed for %s/%s" % (model_name, env))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="adj_glob_65")
    ap.add_argument("--envs", nargs="*", default=None, help="env names under --graph-dicts")
    ap.add_argument("--graph-dicts", default=GRAPH_DICTS)
    ap.add_argument("--rot-step", type=float, default=10.0, help="0 disables the rotation sweep")
    ap.add_argument("--trans-max", type=float, default=15.0, help="0 disables the translation sweep")
    ap.add_argument("--trans-step", type=float, default=0.5)
    ap.add_argument("--trunc", type=int, default=2, help="layer to read out for the truncated F1")
    ap.add_argument("--out-prefix", default="layer_probe")
    ap.add_argument("--skip-anchor", action="store_true")
    a = ap.parse_args()

    envs = a.envs if a.envs else [e for e in GEN + REAL + MSD
                                  if os.path.isdir(os.path.join(a.graph_dicts, e))]
    matcher = ps.Matcher(a.model)
    n_layers = len(matcher.model.gnn)
    m_l2 = truncated(matcher, a.trunc) if 0 < a.trunc < n_layers else None
    print("model %s | %d GAT layers | %d envs | truncated readout %s\n"
          % (a.model, n_layers, len(envs), a.trunc if m_l2 else "n/a"))

    if not a.skip_anchor:
        check_anchors(a.model, matcher, m_l2, 0.0)
        print()

    rows, rots, trans = [], [], []
    for e in envs:
        row, rot, tr = env_row(e, os.path.join(a.graph_dicts, e), matcher, m_l2,
                               a.rot_step, a.trans_max, a.trans_step)
        rows.append(row)
        rots += [dict(env=e, deg=d, f1=round(v, 4)) for d, v in rot]
        trans += [dict(env=e, axis=ax, t=t, f1=round(v, 4)) for ax, t, v in tr]
        print("  %-22s cv %.3f  |N3| %5.2f  L3 %.3f/%.3f  F1 %.3f  rot best %.3f @%3.0f  trans best %.3f  safe r %.1f m"
              % (e, row["cv"], row["n3"], row.get("cosA_L%d" % n_layers, float("nan")),
                 row.get("cosS_L%d" % n_layers, float("nan")), row["f1"],
                 row.get("rot_best", float("nan")), row.get("rot_best_deg", float("nan")),
                 row.get("trans_best", float("nan")), row.get("safe_radius", float("nan"))))

    fields = list(rows[0].keys())
    for r in rows:
        for k in fields:
            r.setdefault(k, "")
    out = "%s_%s.csv" % (a.out_prefix, a.model)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s (%d rows)" % (out, len(rows)))
    for tag, data, flds in (("rotation", rots, ["env", "deg", "f1"]),
                            ("translation", trans, ["env", "axis", "t", "f1"])):
        if not data:
            continue
        sp = "%s_%s_%s.csv" % (a.out_prefix, a.model, tag)
        with open(sp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=flds)
            w.writeheader()
            w.writerows(data)
        print("wrote %s (%d rows)" % (sp, len(data)))


if __name__ == "__main__":
    main()
