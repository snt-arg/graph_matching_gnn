#!/usr/bin/env python3
"""Model-free, layer-independent characterisation of how self-similar an environment is.

Built to answer one question: how much more self-similar is `47_basement` (adj_glob_65 F1 0.000)
than `CF12` (F1 0.471)?  Every metric here is a property of the graph alone -- no checkpoint is
loaded, so nothing can be confounded by normalisation stats or by how many GAT layers the model
happens to have.

Three axes, deliberately reported as a profile rather than blended into one score (they disagree
on this pair, and that disagreement is itself the finding):

  M1 composition   -- what the graph is made of; walls-per-room is the categorical separator
                      (every real room has exactly 4 walls; 0 of 800 MSD graphs are like that)
  M2 dispersion    -- how varied the node features are, per node type, invariant quantities only
  M3 topology      -- Weisfeiler-Leman run TO CONVERGENCE: how many nodes are structurally
                      distinguishable at all, independent of any layer count

Two conventions that silently corrupt comparisons if mixed, both enforced here:
  * degree means TOTAL (in+out) degree.  These graphs are reciprocal, so in-degree variance is
    exactly 4x smaller.
  * only rotation+translation invariant quantities count.  A rotated copy of a graph carries
    identical information but different raw `center`/`normal` values, so their variance measures
    pose, not similarity.  `--self-test` enforces this empirically.
"""

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import pickle

import networkx as nx
import numpy as np

REAL = "/root/workspace/src/datasets/Graph-matching/real/real_adj"
MSD = "/root/workspace/src/graph_matching/graph_matching/MSD_Samples"
AUTOMORPHISM_CAP = 20000


# ---------------------------------------------------------------- loading

def load_graph(path):
    """Accepts either a bare nx graph or a pickled GraphWrapper, as the real envs mix both."""
    obj = pickle.load(open(path, "rb"))
    if hasattr(obj, "graph") and not isinstance(obj, nx.Graph):
        return obj.graph
    return obj


def load_env(env_dir):
    return (load_graph(os.path.join(env_dir, "Prior.pkl")),
            load_graph(os.path.join(env_dir, "Online.pkl")))


def _arr(v, n=2):
    return np.asarray(list(v)[:n], dtype=float)


def neighbours(g, node):
    return set(g.successors(node)) | set(g.predecessors(node))


def total_degrees(g):
    return np.array([len(neighbours(g, n)) * 2 if False else
                     g.in_degree(n) + g.out_degree(n) for n in g.nodes()], dtype=float)


def entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------- M1: composition

def m1_composition(g):
    types = [t for _, t in g.nodes(data="type")]
    n_room = sum(1 for t in types if t == "room")
    n_ws = sum(1 for t in types if t == "ws")

    walls_per_room = [
        sum(1 for m in neighbours(g, n) if g.nodes[m]["type"] == "ws")
        for n, t in g.nodes(data="type") if t == "room"
    ]
    deg = total_degrees(g)
    comps = sorted((len(c) for c in nx.connected_components(g.to_undirected())), reverse=True)

    return {
        "nodes": float(g.number_of_nodes()),
        "rooms": float(n_room),
        "ws": float(n_ws),
        "ws per room": (n_ws / n_room) if n_room else float("nan"),
        "walls/room mean": float(np.mean(walls_per_room)) if walls_per_room else float("nan"),
        "walls/room STD": float(np.std(walls_per_room)) if walls_per_room else float("nan"),
        "walls/room hist": collections.Counter(walls_per_room),
        "degree mean": float(deg.mean()) if len(deg) else float("nan"),
        "degree variance": float(deg.var()) if len(deg) else float("nan"),
        "degree entropy": entropy(list(collections.Counter(deg.tolist()).values())),
        "components": float(len(comps)),
        "component sizes": comps,
    }


# ---------------------------------------------------------------- M2: dispersion

def m2_dispersion(g):
    """Per-node-type dispersion of rotation+translation invariant quantities.

    Rooms are excluded from the length statistics: they carry the sentinel -1, and pooling them
    with walls makes the distribution bimodal, so the variance ends up measuring the room/wall
    ratio instead of feature diversity.  That is exactly why the naive all-node variance
    (reported as the contrast baseline in `m2_naive_baseline`) fails to separate environments.
    """
    lengths = np.array([float(a.get("length", -1)) for _, a in g.nodes(data=True)
                        if a["type"] == "ws"], dtype=float)
    out = {}
    if len(lengths):
        mu = lengths.mean()
        out["ws length mean"] = float(mu)
        out["ws length STD"] = float(lengths.std())
        out["ws length CV"] = float(lengths.std() / mu) if mu > 0 else float("nan")
        out["ws len dup <5%"] = float(np.mean([
            np.sum(np.abs(lengths - x) <= 0.05 * max(abs(x), 1e-9)) for x in lengths]))
    else:
        out.update({"ws length mean": float("nan"), "ws length STD": float("nan"),
                    "ws length CV": float("nan"), "ws len dup <5%": float("nan")})

    # pairwise distances: invariant under any rigid transform
    pos = np.array([_arr(a["center"]) for _, a in g.nodes(data=True)], dtype=float)
    if len(pos) >= 2:
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        iu = np.triu_indices(len(pos), k=1)
        dv = d[iu]
        out["pair dist CV"] = float(dv.std() / dv.mean()) if dv.mean() > 0 else float("nan")
    else:
        out["pair dist CV"] = float("nan")

    # relative normal angle across ws-ws edges: both normals rotate together, so invariant
    angles = []
    for u, v in g.to_undirected().edges():
        if g.nodes[u]["type"] != "ws" or g.nodes[v]["type"] != "ws":
            continue
        a, b = _arr(g.nodes[u].get("normal", [0, 0])), _arr(g.nodes[v].get("normal", [0, 0]))
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        a, b = a / na, b / nb
        angles.append(math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b)))
    if angles:
        ang = np.asarray(angles)
        R = abs(np.mean(np.exp(1j * ang)))
        out["ws-ws angle circvar"] = float(1.0 - R)
        hist, _ = np.histogram(ang, bins=12, range=(-math.pi, math.pi))
        out["ws-ws angle entropy"] = entropy(hist)
    else:
        out["ws-ws angle circvar"] = float("nan")
        out["ws-ws angle entropy"] = float("nan")
    return out


def m2_naive_baseline(g):
    """The metric as originally proposed: raw per-feature variance over ALL nodes.

    Kept to demonstrate its failure -- it is dominated by the room `length=-1` sentinel and is not
    rotation invariant, so `--self-test` shows it moving under a rigid transform.
    """
    rows = []
    for _, a in g.nodes(data=True):
        c, nrm = _arr(a["center"]), _arr(a.get("normal", [0.0, 0.0]))
        rows.append([c[0], c[1], nrm[0], nrm[1], float(a.get("length", -1))])
    x = np.asarray(rows, dtype=float)
    v = x.var(0)
    # Per-AXIS, as "variance per feature" literally means.  Note that summing cx+cy would give
    # the trace of the covariance, which IS rotation invariant -- averaging the two axes hides
    # the non-invariance rather than fixing it, so the axes are kept separate.
    return {"naive var cx": float(v[0]), "naive var cy": float(v[1]),
            "naive var nx": float(v[2]), "naive var ny": float(v[3]),
            "naive var length(all)": float(v[4]),
            "naive var center (trace)": float(v[0] + v[1])}


# ---------------------------------------------------------------- M3: topology

def wl_to_convergence(g, max_rounds=64):
    """Weisfeiler-Leman colour refinement run until the partition stops refining.

    Stopping at convergence rather than at a fixed k makes this a property of the graph, not of
    any particular network depth.  Initial colour is the node type, so the result answers: how
    many nodes are distinguishable from structure + type alone?
    """
    und = g.to_undirected()
    colour = {n: str(g.nodes[n]["type"]) for n in g}
    prev = len(set(colour.values()))
    rounds = 0
    for r in range(1, max_rounds + 1):
        new = {}
        for n in g:
            sig = "%s|%s" % (colour[n], ",".join(sorted(colour[m] for m in und.neighbors(n))))
            new[n] = hashlib.md5(sig.encode()).hexdigest()[:16]
        cur = len(set(new.values()))
        colour = new
        if cur == prev:          # partition stopped refining
            rounds = r - 1
            break
        prev = cur
        rounds = r
    return colour, rounds


def m3_topology(g, automorphisms=True):
    colour, rounds = wl_to_convergence(g)
    sizes = collections.Counter(colour.values())
    n = g.number_of_nodes()
    uniq = sum(1 for c in colour.values() if sizes[c] == 1)
    out = {
        "WL rounds to converge": float(rounds),
        "WL classes": float(len(sizes)),
        "WL % unique": 100.0 * uniq / n if n else float("nan"),
        "WL largest class": float(max(sizes.values())) if sizes else float("nan"),
        "WL effective nodes": float(math.exp(entropy(list(sizes.values())))),
    }
    if automorphisms:
        from networkx.algorithms.isomorphism import DiGraphMatcher
        gm = DiGraphMatcher(g, g, node_match=lambda a, b: a.get("type") == b.get("type"))
        cnt = sum(1 for _ in itertools.islice(gm.isomorphisms_iter(), AUTOMORPHISM_CAP))
        out["automorphisms"] = float(cnt)
        out["automorphisms capped"] = float(cnt >= AUTOMORPHISM_CAP)
    return out


def m4_receptive_field(g, max_k=3):
    """Mean ABSOLUTE size of the k-hop neighbourhood -- how many nodes each embedding mixes in.

    Reported as a growth curve over k rather than at one fixed depth, so it stays a property of
    the graph.  The absolute count is the meaningful one: expressing it as a *fraction* of the
    connected component normalises away exactly the effect being measured (a 5-node component and
    a 20-node component both reach 100% coverage, while mixing 4x different amounts of signal).
    """
    und = g.to_undirected()
    out = {}
    for k in range(1, max_k + 1):
        sizes = []
        for src in und:
            sp = nx.single_source_shortest_path_length(und, src, cutoff=k)
            sizes.append(len(sp))
        out["|N_%d| mean nodes" % k] = float(np.mean(sizes)) if sizes else float("nan")
    out["N_3 / n"] = out["|N_3| mean nodes"] / g.number_of_nodes() if g.number_of_nodes() else float("nan")
    return out


def all_metrics(g, automorphisms=True):
    m = {}
    m.update(m1_composition(g))
    m.update(m2_dispersion(g))
    m.update(m2_naive_baseline(g))
    m.update(m3_topology(g, automorphisms=automorphisms))
    m.update(m4_receptive_field(g))
    return m


# ---------------------------------------------------------------- rendering

SECTIONS = [
    ("M1  COMPOSITION", ["nodes", "rooms", "ws", "ws per room", "walls/room mean",
                         "walls/room STD", "walls/room hist", "degree mean", "degree variance",
                         "degree entropy", "components", "component sizes"]),
    ("M2  FEATURE DISPERSION  (per type, invariant only)",
     ["ws length mean", "ws length STD", "ws length CV", "ws len dup <5%",
      "pair dist CV", "ws-ws angle circvar", "ws-ws angle entropy"]),
    ("M2b NAIVE BASELINE  (all nodes, per-axis -- shown to fail)",
     ["naive var cx", "naive var cy", "naive var nx", "naive var ny",
      "naive var length(all)", "naive var center (trace)"]),
    ("M3  TOPOLOGY  (WL to convergence -- no layer count involved)",
     ["WL rounds to converge", "WL classes", "WL % unique", "WL largest class",
      "WL effective nodes", "automorphisms"]),
    ("M4  RECEPTIVE FIELD  (absolute nodes mixed per embedding)",
     ["|N_1| mean nodes", "|N_2| mean nodes", "|N_3| mean nodes", "N_3 / n"]),
]


def fmt(v):
    if isinstance(v, collections.Counter):
        return " ".join("%dw:%d" % (k, v[k]) for k in sorted(v)) or "-"
    if isinstance(v, list):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        if v == int(v) and abs(v) < 1e6:
            return "%d" % int(v)
        return "%.3f" % v
    return str(v)


def ratio(a, b):
    if not isinstance(a, float) or not isinstance(b, float):
        return ""
    if math.isnan(a) or math.isnan(b):
        return "-"
    if b == 0:
        return "inf" if a > 0 else "0/0"
    return "%.2fx" % (a / b)


def render(cols, ratio_pairs, title):
    """cols: list of (label, metrics dict).  ratio_pairs: list of (label, i, j) column indices."""
    labels = [c[0] for c in cols]
    w = max(24, max(len(l) for l in labels) + 1)
    head = "%-28s" % "" + "".join("%*s" % (w, l) for l in labels)
    head += "".join("%*s" % (w, r[0]) for r in ratio_pairs)
    print("\n" + "=" * len(head))
    print(title)
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for section, keys in SECTIONS:
        print("\n" + section)
        for k in keys:
            if k not in cols[0][1]:
                continue
            line = "  %-26s" % k + "".join("%*s" % (w, fmt(c[1][k])) for c in cols)
            for _, i, j in ratio_pairs:
                line += "%*s" % (w, ratio(cols[i][1][k], cols[j][1][k]))
            print(line)


# ---------------------------------------------------------------- self-test

def rigid_transform(g, theta, tx, ty):
    """Rotate + translate a graph.  Normals rotate but do not translate."""
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    h = g.copy()
    for n, a in h.nodes(data=True):
        a["center"] = (R @ _arr(a["center"]) + np.array([tx, ty])).tolist()
        if "normal" in a:
            a["normal"] = (R @ _arr(a["normal"])).tolist()
        if "limits" in a:
            a["limits"] = [(R @ _arr(p) + np.array([tx, ty])).tolist() for p in a["limits"]]
    return h


def self_test():
    fails = []

    # WL sanity: a ring of identical nodes is one class; a path separates by symmetry.
    ring = nx.DiGraph()
    for i in range(6):
        ring.add_node(i, type="ws", center=[0.0, 0.0], normal=[1.0, 0.0], length=1.0)
    for i in range(6):
        ring.add_edge(i, (i + 1) % 6); ring.add_edge((i + 1) % 6, i)
    c, _ = wl_to_convergence(ring)
    if len(set(c.values())) != 1:
        fails.append("WL: ring of 6 gave %d classes, expected 1" % len(set(c.values())))

    path = nx.DiGraph()
    for i in range(6):
        path.add_node(i, type="ws", center=[0.0, 0.0], normal=[1.0, 0.0], length=1.0)
    for i in range(5):
        path.add_edge(i, i + 1); path.add_edge(i + 1, i)
    c, _ = wl_to_convergence(path)
    if len(set(c.values())) != 3:
        fails.append("WL: path of 6 gave %d classes, expected 3" % len(set(c.values())))

    # Invariance: every M2/M3 metric must survive a rigid transform; the naive baseline must not.
    rng = np.random.default_rng(0)
    for env in ("47_basement", "CF12"):
        for side, g in zip(("Prior", "Online"), load_env(os.path.join(REAL, env))):
            base = all_metrics(g, automorphisms=False)
            moved = all_metrics(rigid_transform(g, rng.uniform(0, 2 * math.pi),
                                                rng.uniform(-50, 50), rng.uniform(-50, 50)),
                                automorphisms=False)
            for k, v in base.items():
                if not isinstance(v, float) or math.isnan(v):
                    continue
                if k.startswith("naive"):
                    continue
                if abs(v - moved[k]) > 1e-6 * max(1.0, abs(v)):
                    fails.append("%s/%s: %s not invariant (%.6f -> %.6f)"
                                 % (env, side, k, v, moved[k]))
            # The per-axis variances are the literal "variance per feature" and must MOVE
            # under rotation -- that is the whole reason they are invalid here.
            if abs(base["naive var cx"] - moved["naive var cx"]) <= 1e-6:
                fails.append("%s/%s: naive var cx unexpectedly invariant" % (env, side))
            # ...while the trace is genuinely invariant, so it must NOT move.
            if abs(base["naive var center (trace)"] - moved["naive var center (trace)"]) > 1e-6:
                fails.append("%s/%s: center-variance trace should be invariant" % (env, side))

    # Degeneracy: CF12 has an isolated single-node component; nothing may blow up.
    _, cf_online = load_env(os.path.join(REAL, "CF12"))
    if min(len(c) for c in nx.connected_components(cf_online.to_undirected())) != 1:
        fails.append("CF12/Online: expected an isolated single-node component")
    for k, v in all_metrics(cf_online, automorphisms=False).items():
        if isinstance(v, float) and math.isinf(v):
            fails.append("CF12/Online: %s is infinite" % k)

    # Reproduce figures measured independently earlier in the investigation.
    known = [("47_basement", "Online", "degree variance", 4.20, 0.01),
             ("47_basement", "Online", "walls/room STD", 0.00, 1e-9),
             ("CF12", "Online", "components", 6.0, 1e-9),
             ("CF12", "Online", "ws length CV", 0.46, 0.01)]
    for env, side, key, want, tol in known:
        g = load_env(os.path.join(REAL, env))[0 if side == "Prior" else 1]
        got = all_metrics(g, automorphisms=False)[key]
        if abs(got - want) > tol:
            fails.append("%s/%s %s = %.4f, expected %.4f" % (env, side, key, got, want))

    if fails:
        print("SELF-TEST FAILED (%d):" % len(fails))
        for f in fails:
            print("  -", f)
        return 1
    print("SELF-TEST PASSED  (WL sanity, rigid-transform invariance, degeneracy, "
          "reproduction of 4 previously-measured values)")
    return 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-reference", action="store_true",
                    help="omit the secondary MSD reference block")
    ap.add_argument("--envs", nargs="*", default=None,
                    help="explicit env folder paths (default: 47_basement and CF12)")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    env_dirs = args.envs or [os.path.join(REAL, "47_basement"), os.path.join(REAL, "CF12")]
    cols = []
    for d in env_dirs:
        name = os.path.basename(d.rstrip("/"))
        ga, gs = load_env(d)
        cols.append(("%s/Prior" % name, all_metrics(ga)))
        cols.append(("%s/Online" % name, all_metrics(gs)))

    ratios = []
    if len(cols) == 4:
        ratios = [("ratio Prior", 0, 2), ("ratio Online", 1, 3)]
    render(cols, ratios,
           "SELF-SIMILARITY PROFILE   (ratio = %s / %s)"
           % (os.path.basename(env_dirs[0].rstrip("/")),
              os.path.basename(env_dirs[1].rstrip("/")) if len(env_dirs) > 1 else "-"))

    if not args.no_reference and os.path.exists(os.path.join(MSD, "sample_manifest.json")):
        man = json.load(open(os.path.join(MSD, "sample_manifest.json")))
        ref = []
        for s in man["samples"]:
            gs = load_graph(os.path.join(MSD, s["env"], "Online.pkl"))
            ref.append(("%s/Online" % s["env"].replace("MSD_", ""), all_metrics(gs)))
        render(ref, [], "SECONDARY REFERENCE ONLY -- unseen MSD synthetic (all score F1 >= 0.855)."
                        "\nNot part of the comparison; shown so a CF12 value can be read as "
                        "healthy vs merely less bad.")


if __name__ == "__main__":
    main()
