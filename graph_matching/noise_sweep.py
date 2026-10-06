#!/usr/bin/env python3
"""Rotation and translation sweeps over the generated environments, averaged over noise draws.

Why averaged: with observation noise on, one environment's F1 is a single realisation, and
re-drawing the noise on the SAME environment moves F1 by a median SD of 0.156 (12 draws x 10
environments, some as high as 0.27).  Between-environment gaps in the delivered set are smaller
than that, so a single-draw table would rank noise realisations rather than environments.  Every
cell here is a mean over ``--draws`` independent draws, which cuts the per-cell SD by sqrt(N).

Two sweeps, each a different physical knob:

  rotation     Online turned about ITS OWN centroid ("turn the scan around itself").  Note this is
               not a pure rotation: a cloud whose centroid sits |q| from the world origin, rotated
               by delta, is an origin-rotation plus a translation of 2|q|sin(delta/2) -- at
               |q| = 4.75 m that reaches 9.5 m by 180 deg.  Read it as the diagonal it is.
  translation  Online displaced along X, then Y, rotation held at zero.

The clean (noise-free) pair is the source, so each draw is an independent re-measurement of the
same building rather than noise stacked on noise.
"""

import argparse
import copy
import csv
import math
import os
import random
import sys
import types

import numpy as np

try:
    import moviepy.editor  # noqa: F401
except Exception:
    import moviepy
    _s = types.ModuleType("moviepy.editor")
    _s.ImageSequenceClip = getattr(moviepy, "ImageSequenceClip", None)
    sys.modules["moviepy.editor"] = _s

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/workspace/src/graph_matching/graph_matching")

import pose_sweep as ps                      # noqa: E402
import gen_rect_envs as G                    # noqa: E402
import env_similarity_metrics as esm         # noqa: E402

CLEAN = "/root/workspace/src/graph_matching/graph_matching/Gen_Samples_clean"
GRAPH_DICTS = "/root/workspace/src/graph_matching/graph_matching/graph_dicts"
GEN = ["GEN_base", "GEN_D_cv45", "GEN_D_cv65", "GEN_D_cv85",
       "GEN_R_n11", "GEN_R_n7", "GEN_R_n5", "GEN_M_1", "GEN_M_2", "GEN_M_3"]
REAL = ["47_basement", "47_topfloor_modified", "CF12"]


def draws_for(env_idx, gs_clean, n_draws, scale):
    """One re-measurement per draw. Draw 0 reuses the generator's own seed, so it is the
    realisation actually written to disk (up to the <=5 cm re-placement the noise induces)."""
    out = []
    for k in range(n_draws):
        seed = (90000 + env_idx) if k == 0 else (700000 + 1000 * env_idx + k)
        g = copy.deepcopy(gs_clean)
        G.perturb_online(g, scale, random.Random(seed))
        out.append(g)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="adj_glob_65")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--rot-step", type=float, default=10.0)
    ap.add_argument("--trans-max", type=float, default=15.0)
    ap.add_argument("--trans-step", type=float, default=1.0)
    ap.add_argument("--out-prefix", default="noise_sweep")
    a = ap.parse_args()

    m = ps.Matcher(a.model)
    angles = np.arange(0.0, 360.0, a.rot_step)
    offs = np.round(np.arange(-a.trans_max, a.trans_max + 1e-9, a.trans_step), 3)
    rows, base_rows = [], []

    def sweep(env, tag, ga, variants, gt):
        piv = [np.array([at["center"][:2] for _, at in g.nodes(data=True)],
                        dtype=float).mean(0) for g in variants]
        for i, g0 in enumerate(variants):
            f1, _h, ca, cs = m.evaluate(ga, g0, gt)
            # Achieved CV and |N_3| are recorded PER DRAW: the Online topology is re-derived from
            # re-measured geometry, so a door edge can appear or vanish and |N_3| is a random
            # variable rather than the ladder value it was designed to. The analysis regresses on
            # what each draw actually is.
            base_rows.append(dict(env=env, kind=tag, draw=i, f1=round(f1, 4),
                                  cos_a=round(ca, 4), cos_s=round(cs, 4),
                                  cv=round(esm.m2_dispersion(g0)["ws length CV"], 4),
                                  n3=round(esm.m4_receptive_field(g0)["|N_3| mean nodes"], 3),
                                  components=esm.m1_composition(g0)["components"],
                                  degvar=round(esm.m1_composition(g0)["degree variance"], 3)))
            for d in angles:
                g = copy.deepcopy(g0)
                ps.transform_inplace(g, 0.0, 0.0, math.radians(float(d)), piv[i])
                rows.append(dict(env=env, kind=tag, draw=i, cond="rot", axis="-",
                                 param=float(d), f1=round(m.evaluate(ga, g, gt)[0], 4)))
            for axis in ("x", "y"):
                for t in offs:
                    g = copy.deepcopy(g0)
                    ps.transform_inplace(g, float(t) if axis == "x" else 0.0,
                                         float(t) if axis == "y" else 0.0, 0.0, (0.0, 0.0))
                    rows.append(dict(env=env, kind=tag, draw=i, cond="trans", axis=axis,
                                     param=float(t), f1=round(m.evaluate(ga, g, gt)[0], 4)))

    print(f"model {a.model} | {len(angles)} angles | {len(offs)} offsets x 2 axes "
          f"| {a.draws} noise draws\n")
    for idx, env in enumerate(GEN):
        ga, gs_clean, gt = ps.load_env(os.path.join(CLEAN, env))
        sweep(env, "noisy", ga, draws_for(idx, gs_clean, a.draws, a.noise), gt)
        sweep(env, "clean", ga, [gs_clean], gt)
        got = [r["f1"] for r in base_rows if r["env"] == env and r["kind"] == "noisy"]
        print(f"  {env:<12} at rest: clean "
              f"{[r['f1'] for r in base_rows if r['env']==env and r['kind']=='clean'][0]:.2f}"
              f"   noisy mean {np.mean(got):.2f} (sd {np.std(got):.2f})")
    for env in REAL:                       # a real scan is its own single realisation
        ga, gs, gt = ps.load_env(os.path.join(GRAPH_DICTS, env))
        sweep(env, "real", ga, [gs], gt)
        print(f"  {env:<12} at rest: {[r['f1'] for r in base_rows if r['env']==env][0]:.2f}")

    for name, data, flds in ((f"{a.out_prefix}_sweep.csv", rows,
                              ["env", "kind", "draw", "cond", "axis", "param", "f1"]),
                             (f"{a.out_prefix}_base.csv", base_rows,
                              ["env", "kind", "draw", "f1", "cos_a", "cos_s",
                               "cv", "n3", "components", "degvar"])):
        with open(name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=flds)
            w.writeheader()
            w.writerows(data)
        print(f"wrote {name} ({len(data)} rows)")


if __name__ == "__main__":
    main()
