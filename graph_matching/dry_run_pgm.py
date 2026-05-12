"""Dry-run replacement for the GNN-based partial graph matcher.

Mimics PGM_class.PartialGraphMatching and predict_matching_matrix without
loading the trained GATv2 model or running any GNN forward pass. Replaces
everything up to and including the affinity matrix with random data of the
correct shapes; the downstream Sinkhorn + Hungarian post-processing is
re-implemented in numpy/scipy so the dashboard sees a realistic doubly-
stochastic S and a true binary permutation.

Numpy + scipy only — no torch, no torch_geometric, no pygmtools.

Outputs are seeded deterministically from each input pair (graph name + node
count + edge count) so:
  - Repeat calls on the same pair (e.g. MSD's twin predict_matching_matrix
    calls for discrete=True / discrete=False) yield consistent arrays.
  - Re-running the dashboard reproduces the same fake results.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment


EMBED_DIM = 32              # matches MatchingModel_GATv2SinkhornTopK out_dim
SINKHORN_MAX_ITER = 10      # matches model.sinkhorn_max_iter
SINKHORN_TAU = 1.0          # matches model.sinkhorn_tau
INSTANCE_NORM_EPS = 1e-5


class _NumpyMatrixShim:
    """Quacks like a CPU torch tensor for downstream `.cpu().numpy()` chains."""

    __slots__ = ("_arr",)

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape

    def __array__(self, dtype=None):
        return self._arr.astype(dtype) if dtype is not None else self._arr


def _pair_signature(d1, d2):
    """Stable signature for an input pair: (name, N, E) for each side.

    Accepts either a networkx graph (has .number_of_nodes / .number_of_edges,
    name read from .graph['name']) or a PyG Data-like object (.num_nodes,
    .edge_index, .name).
    """
    def _info(d):
        name = None
        if hasattr(d, "graph") and hasattr(d.graph, "get"):
            name = d.graph.get("name")
        if name is None:
            name = getattr(d, "name", None)
        if hasattr(d, "number_of_nodes"):
            return (str(name), int(d.number_of_nodes()), int(d.number_of_edges()))
        n_nodes = int(getattr(d, "num_nodes", 0))
        edge_index = getattr(d, "edge_index", None)
        n_edges = int(edge_index.shape[1]) if edge_index is not None else 0
        return (str(name), n_nodes, n_edges)
    return (_info(d1), _info(d2))


def _seed_from_signature(sig):
    h = hashlib.blake2b(repr(sig).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & 0xFFFFFFFF


def _instance_norm(M):
    mu = M.mean()
    sigma = M.std()
    return (M - mu) / (sigma + INSTANCE_NORM_EPS)


def _sinkhorn(M, max_iter=SINKHORN_MAX_ITER, tau=SINKHORN_TAU):
    """Doubly-stochastic normalization of exp(M / tau) via row/col scaling."""
    S = np.exp(M / tau - (M / tau).max())  # subtract max for numerical stability
    for _ in range(max_iter):
        S = S / np.clip(S.sum(axis=1, keepdims=True), 1e-12, None)
        S = S / np.clip(S.sum(axis=0, keepdims=True), 1e-12, None)
    return S


def _hungarian(S):
    """Binary one-to-one assignment maximizing sum of S entries."""
    rows, cols = linear_sum_assignment(-S)
    P = np.zeros_like(S, dtype=np.float32)
    P[rows, cols] = 1.0
    return P


def _make_intermediates(n1, n2, seed, signature):
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal(size=(n1, EMBED_DIM)).astype(np.float32)
    h2 = rng.standard_normal(size=(n2, EMBED_DIM)).astype(np.float32)
    affinity = (h1 @ h2.T).astype(np.float32)        # raw dot-product similarity
    sim_normed = _instance_norm(affinity).astype(np.float32)
    S = _sinkhorn(sim_normed).astype(np.float32)
    perm = _hungarian(S).astype(np.float32)
    return {
        "signature": signature,
        "seed": int(seed),
        "h1": h1,
        "h2": h2,
        "affinity": affinity,
        "sim_normed": sim_normed,
        "S": S,
        "perm": perm,
    }


# Module-level cache: one entry per unique input pair signature.
# De-dupes the discrete=True / discrete=False back-to-back calls in the MSD path.
_intermediates_cache: dict = {}


def get_intermediates_log():
    """Return the per-pair intermediates produced so far, in insertion order."""
    return list(_intermediates_cache.values())


def clear_intermediates_log():
    _intermediates_cache.clear()


def _get_or_make(data1, data2):
    signature = _pair_signature(data1, data2)
    seed = _seed_from_signature(signature)
    cached = _intermediates_cache.get(seed)
    if cached is not None:
        return cached
    n1 = data1.number_of_nodes() if hasattr(data1, "number_of_nodes") else int(data1.num_nodes)
    n2 = data2.number_of_nodes() if hasattr(data2, "number_of_nodes") else int(data2.num_nodes)
    ints = _make_intermediates(n1, n2, seed, signature)
    _intermediates_cache[seed] = ints
    return ints


def predict_matching_matrix(_model, data1, data2, discrete=True,
                            mc_samples=0, mc_affinity_samples=0, **_kwargs):
    """Drop-in replacement for PGM_class.predict_matching_matrix.

    The `_model` argument is ignored — accepted for signature compatibility.
    All threshold / MC kwargs are accepted and ignored: the dry-run output is
    random, so thresholding it produces no meaningful change. When mc_samples
    or mc_affinity_samples > 0, returns (matrix, uncertainty) with a random
    uncertainty array of the same shape, matching the real function's contract.
    """
    ints = _get_or_make(data1, data2)
    matrix = ints["perm"] if discrete else ints["S"]
    out = _NumpyMatrixShim(matrix)
    if mc_samples > 0 or mc_affinity_samples > 0:
        uncertainty = _NumpyMatrixShim(
            np.abs(np.random.default_rng(ints["seed"] ^ 0xA5A5A5A5)
                   .standard_normal(size=matrix.shape).astype(np.float32))
        )
        return out, uncertainty
    return out


class DryRunPartialGraphMatching:
    """Drop-in replacement for PGM_class.PartialGraphMatching.

    Implements only the subset of the API that matching_synthetic_dataset.py
    actually uses: __init__ (accepts and ignores the same kwargs), no-op
    load_best_model(), and infer_matching(g1, g2, discrete=True, **kwargs).
    """

    def __init__(self, model_class=None, data_paths=None, model_save_path=None,
                 device=None, in_dim=7, **_kwargs):
        self.in_dim = in_dim
        # `model` is the attribute the MSD path passes to predict_matching_matrix.
        # We expose `self.model = None` so call sites don't crash; the function
        # ignores it anyway.
        self.model = None
        # Real PartialGraphMatching exposes `original_graphs`; MSD-mode visualization
        # reads it. Empty list keeps that code path safe to call.
        self.original_graphs = []
        self._data_paths = data_paths
        self._model_save_path = model_save_path

    def load_best_model(self):
        """No-op: there is no model to load."""
        return None

    def infer_matching(self, g1, g2, discrete=True,
                       mc_samples=0, mc_affinity_samples=0, **_kwargs):
        """Same return contract as PartialGraphMatching.infer_matching.

        Returns a tensor-shim (or (shim, uncertainty_shim) when MC is requested)
        whose .cpu().numpy() yields the matching matrix as a numpy array.
        """
        return predict_matching_matrix(
            self.model, g1, g2,
            discrete=discrete,
            mc_samples=mc_samples,
            mc_affinity_samples=mc_affinity_samples,
        )


# Compatibility shim: matching_synthetic_dataset.py also imports the model
# class symbol from PGM_class. Expose a sentinel under the same name so the
# import line works regardless of mode.
MatchingModel_GATv2SinkhornTopK = None
