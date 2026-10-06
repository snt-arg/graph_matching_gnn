#!/usr/bin/env python3
"""Rank two matching models by robustness to RE-MEASUREMENT NOISE, paired per noise draw.

Why noise and not pose: the edge-feature models drop `center`/`normal` from node features and put
all geometry into the 9-dim invariant `edge_attr`, so a global rigid displacement leaves their input
BIT-IDENTICAL (measured: max|d x| = max|d edge_attr| = 0.0 under rotation, translation and both at
once). Every pose sweep is therefore exactly flat for them and cannot rank two such models.

Re-measurement noise does vary the input, and it varies the channel that actually matters. The
recorded real Prior<->Online drift is RE-SEGMENTATION, not displacement -- along-wall 0.247 m vs
across-wall 0.077 m -- and in model sigma the damage is entirely in `length` (0.246 mean / 1.735
max) against a centre drift of 0.016. `length` is the one surviving per-node feature in both
architectures, so this is the axis they are actually exposed on.

Design notes:
  * PAIRED. Both models are evaluated on the SAME perturbed graph for each (env, scale, draw), so a
    difference cannot be a noise-draw artifact. Per-env F1 moves with sd 0.05-0.20 across redraws,
    which would otherwise swamp the effect being measured.
  * Noise is applied ON TOP of each env's stored Online graph. The GEN envs already ship at
    noise_scale 1.0 and the real scans carry real drift, and no clean source exists on disk for
    GEN_H_*. So the x-axis is ADDITIONAL noise, not absolute magnitude -- fine for ranking two
    models on identical inputs, but do not read it as an absolute noise level.
  * perturb_online re-derives the whole edge set via topology_rules.rebuild_all, which is
    deployment-realistic (the robot has no access to the prior map's topology) and makes |N_3| a
    random variable rather than a constant.
  * perturb_online MUTATES IN PLACE -> deepcopy per draw.
"""

import argparse
import copy
import csv
import os
import random
import statistics as st
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/workspace/src/graph_matching/graph_matching")

import edge_model as EM           # noqa: E402
import gen_rect_envs as G         # noqa: E402
import pose_sweep as ps           # noqa: E402

MROOT = "/root/workspace/src/graph_matching_gnn/GNN/models/partial_graph_matching"
GRAPH_DICTS = "/root/workspace/src/graph_matching/graph_matching/graph_dicts"
FIELDS = ["env", "scale", "draw", "model", "f1", "cos_a", "cos_s",
          "len_change_absmean", "centre_shift_mean"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="model FOLDER names")
    ap.add_argument("--envs", nargs="+", required=True)
    ap.add_argument("--graph-dicts", default=GRAPH_DICTS)
    ap.add_argument("--scales", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    matchers = {m: EM.EdgeMatcher(os.path.join(MROOT, m)) for m in a.models}
    for name, m in matchers.items():
        print(f"[model] {name}: tau={m.hparams['sinkhorn_tau']:.4f} "
              f"iter={m.hparams['sinkhorn_max_iter']} hidden={m.hparams['hidden_dim']} "
              f"heads={m.hparams['heads']}")

    rows = []
    for env in a.envs:
        ga, gs, gt = ps.load_env(os.path.join(a.graph_dicts, env))
        print(f"\n[{env}] |A|={ga.number_of_nodes()} |S|={gs.number_of_nodes()} gt={len(gt)}")
        for scale in a.scales:
            per = {m: [] for m in a.models}
            for k in range(a.draws):
                g = copy.deepcopy(gs)
                if scale > 0:
                    # One RNG per (env, scale, draw) -> identical perturbation for every model.
                    # NOTE: must NOT use builtin hash() on a str here. Python salts string
                    # hashing per process (PYTHONHASHSEED), so hash(("env",...)) makes the draws
                    # irreproducible across runs -- the same defect gen_rect_envs has, where a
                    # set-iteration order flips a float's last bit. crc32 is stable.
                    key = f"{env}|{scale:.4f}|{k}".encode()
                    rng = random.Random(zlib.crc32(key) ^ a.seed)
                    nstats = G.perturb_online(g, scale, rng)
                else:
                    nstats = {"len_change_absmean": 0.0, "centre_shift_mean": 0.0}
                for name, m in matchers.items():
                    f1, _f1h, ca, cs = m.evaluate(ga, g, gt)
                    per[name].append(f1)
                    rows.append(dict(env=env, scale=scale, draw=k, model=name,
                                     f1=round(f1, 4), cos_a=round(ca, 4), cos_s=round(cs, 4),
                                     len_change_absmean=round(nstats["len_change_absmean"], 4),
                                     centre_shift_mean=round(nstats["centre_shift_mean"], 4)))
                if scale == 0:
                    break      # scale 0 is deterministic; one draw is the whole story
            msg = "  ".join(
                f"{n.split('_')[-1]}={st.mean(v):.3f}"
                + (f"+-{st.pstdev(v):.3f}" if len(v) > 1 else "")
                for n, v in per.items())
            print(f"   scale {scale:<5} {msg}")

    out = os.path.join(a.out_dir, "noise_robustness.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
