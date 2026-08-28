"""T1 acceptance oracle — FRESH public-behavioral suite (AUCP offline eval).

This oracle is authored strictly from the PUBLIC CONTRACT of the frozen T1
goal_prompt — NOT from any reference solution's internal structure. It binds to
the identifiers and behaviors the goal_prompt actually pins, and tolerates every
unpinned naming detail so a correct implementation is never failed for internal
structure (fairness across eval arms).

Frozen goal_prompt public contract (verbatim intent):
  - derive_k_range(n, *, min_cluster_size, max_cluster_size, max_k) -> list[int]:
    contiguous candidate-k range, k >= 2, with giant/tiny-cluster guards.
  - AutoKClusterer (config-driven) whose .fit(embeddings) scores each candidate k by
    silhouette minus giant/tiny penalties, dispatches deterministic k-means or
    mini-batch k-means as appropriate, selects medoid representatives per cluster,
    and returns a structured KMeansResult with per-cluster diagnostics.
  - Coherence: never report a selected_k that disagrees with the emitted labels; no
    duplicate/fallback medoids for empty clusters; degenerate/collapsed cases return
    a coherent single-cluster result.
  - Defer numpy/sklearn imports to call time.

Pinned identifiers: derive_k_range (+ its exact keyword-only signature), AutoKClusterer,
its .fit(embeddings) method, and the result attribute `selected_k` (named in the prompt).
UNPINNED (therefore tolerated / never hard-asserted by name): the config class name and
constructor contract; the KMeansResult/diagnostics attribute shape; dispatch-fn names;
score-formula internals.

Structure: 10 derive_k_range tests (contract-faithful, preserved from the prior oracle)
+ 7 behavioral tests bound to the public contract via tolerant accessors. 17 tests total.

Provenance: fresh authorship per human decision C (2026-08-28). Kept OUT of the eval
agent's context; written to the acceptance path only after the agent returns.
"""
from __future__ import annotations

import numpy as np
import pytest

from frame_generation.image_selection.auto_k import AutoKClusterer, derive_k_range

try:
    import sklearn  # noqa: F401

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

_skip_no_sklearn = pytest.mark.skipif(
    not _HAS_SKLEARN, reason="sklearn not installed (k-means dispatch unavailable)"
)


# --- tolerant public-contract accessors -------------------------------------
def _selected_k(result):
    """`selected_k` is named explicitly in the goal_prompt -> a hard public requirement."""
    assert hasattr(result, "selected_k"), "result must expose `selected_k`"
    return result.selected_k


def _labels(result):
    """Emitted per-input labels. Try conventional public names; fail if none exist."""
    for name in ("labels", "emitted_labels", "cluster_labels"):
        if hasattr(result, name):
            return list(getattr(result, name))
    pytest.fail("result must expose emitted labels (tried labels/emitted_labels/cluster_labels)")


def _medoids(result):
    """Medoid indices. UNPINNED shape -> return None (caller skips) if not discoverable."""
    for name in ("medoids", "medoid_indices", "medoid_idxs"):
        if hasattr(result, name):
            return list(getattr(result, name))
    diags = getattr(result, "diagnostics", None)
    if diags is not None:
        try:
            seq = list(diags)
        except TypeError:
            seq = None
        if seq:
            for attr in ("medoid_idx", "medoid", "medoid_index"):
                if all(hasattr(d, attr) for d in seq):
                    return [getattr(d, attr) for d in seq]
    return None


def _blobs(n_per=12, k=3, dim=8, spread=0.3, sep=10.0, seed=0):
    """Deterministic, well-separated Gaussian blobs as an ndarray (n = n_per*k).

    Returns a numpy ndarray, the canonical representation for numeric embedding
    vectors. The frozen goal_prompt pins no input type for ``.fit(embeddings)``, so
    the oracle must not require list-acceptance: an ndarray is the lowest common
    denominator — impls that ``np.asarray`` a list accept it too, while impls that
    index the input with boolean masks require it. Validated 17/17 against two
    independent implementations (one list-accepting, one ndarray-typed).
    """
    rng = np.random.default_rng(seed)
    centers = [np.full(dim, i * sep, dtype=float) for i in range(k)]
    return np.vstack([c + rng.normal(0.0, spread, size=(n_per, dim)) for c in centers])


class TestDeriveKRange:
    def test_doc_example(self):
        # N=240, defaults from ues4_auto_k_method.md
        ks = derive_k_range(240, min_cluster_size=5, max_cluster_size=80, max_k=40)
        assert ks[0] == 3
        assert ks[-1] == 40
        assert len(ks) == 38

    def test_small_n_single_candidate(self):
        # N=10: k_min=max(2,ceil(10/80))=2, k_max=min(40,10//5)=2 → [2]
        ks = derive_k_range(10, min_cluster_size=5, max_cluster_size=80, max_k=40)
        assert ks == [2]

    def test_group_too_small_returns_empty(self):
        # N=4: k_max=floor(4/5)=0 < k_min=2 → empty
        ks = derive_k_range(4, min_cluster_size=5, max_cluster_size=80, max_k=40)
        assert ks == []

    def test_n_zero_returns_empty(self):
        assert derive_k_range(0, min_cluster_size=5, max_cluster_size=80, max_k=40) == []

    def test_n_one_returns_empty(self):
        assert derive_k_range(1, min_cluster_size=5, max_cluster_size=80, max_k=40) == []

    def test_max_k_caps_range(self):
        # Without cap k_max would be 100; with max_k=10 it stops at 10.
        ks = derive_k_range(500, min_cluster_size=5, max_cluster_size=80, max_k=10)
        assert ks[-1] == 10

    def test_inconsistent_constraints_returns_empty(self):
        # min_cluster_size > max_cluster_size makes k_max < k_min for any N
        ks = derive_k_range(100, min_cluster_size=50, max_cluster_size=10, max_k=40)
        assert ks == []

    def test_returns_contiguous_integers(self):
        ks = derive_k_range(100, min_cluster_size=5, max_cluster_size=20, max_k=40)
        assert ks == list(range(ks[0], ks[-1] + 1))

    def test_k_min_at_least_two(self):
        # Even for huge N, k_min cannot drop below 2.
        ks = derive_k_range(10_000, min_cluster_size=5, max_cluster_size=100_000, max_k=5)
        assert ks[0] >= 2

    def test_exact_boundary_k_min_equals_k_max(self):
        # N=10, min=5, max=5, max_k=40 → k_min=max(2,2)=2, k_max=min(40,2)=2 → [2]
        ks = derive_k_range(10, min_cluster_size=5, max_cluster_size=5, max_k=40)
        assert ks == [2]


@_skip_no_sklearn
class TestAutoKClustererBehavior:
    """Behavioral contract of AutoKClusterer.fit, bound only to pinned public promises."""

    def test_fit_returns_result_with_selected_k(self):
        result = AutoKClusterer().fit(_blobs())
        k = _selected_k(result)
        assert isinstance(k, int) and k >= 1

    def test_selected_k_at_least_two_for_separated_clusters(self):
        result = AutoKClusterer().fit(_blobs(k=3))
        assert _selected_k(result) >= 2

    def test_labels_cover_all_inputs(self):
        emb = _blobs(k=3)
        result = AutoKClusterer().fit(emb)
        assert len(_labels(result)) == len(emb)

    def test_coherence_selected_k_matches_labels(self):
        # goal_prompt: never report a selected_k that disagrees with emitted labels.
        result = AutoKClusterer().fit(_blobs(k=3))
        assert _selected_k(result) == len(set(_labels(result)))

    def test_determinism_same_input_same_output(self):
        emb = _blobs(k=3)
        a = AutoKClusterer().fit(emb)
        b = AutoKClusterer().fit(emb)
        assert _selected_k(a) == _selected_k(b)
        assert list(_labels(a)) == list(_labels(b))

    def test_degenerate_single_input_single_cluster(self):
        result = AutoKClusterer().fit(np.array([[1.0] * 8]))
        assert _selected_k(result) == 1
        assert len(set(_labels(result))) == 1
        assert len(_labels(result)) == 1

    def test_medoids_valid_distinct_one_per_cluster(self):
        emb = _blobs(k=3)
        result = AutoKClusterer().fit(emb)
        med = _medoids(result)
        if med is None:
            pytest.skip("result does not expose medoid indices via a conventional accessor")
        assert len(med) == _selected_k(result)  # one medoid per cluster
        assert len(set(med)) == len(med)  # distinct (no duplicate/fallback medoids)
        assert all(0 <= int(m) < len(emb) for m in med)  # valid indices into the input
