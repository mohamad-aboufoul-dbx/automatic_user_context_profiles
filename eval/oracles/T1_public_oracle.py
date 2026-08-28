"""Eval-time acceptance oracle for T1 — public-API contract only.

Derived from: platform 98b8bd7 tests/unit/test_image_selection_auto_k.py
RELAXED to the public-API contract (private-internal tests removed) per
human decision 2026-08-28.

This is the eval-time acceptance oracle, kept OUT of the agent's context
and never agent-authored.  Changes require explicit human authorization and
a new task_definition_hash in eval/tasks/T1.json.

Removals vs. 98b8bd7 (16 tests total):
  - TestPenalties class (8 tests: giant/tiny penalty private helpers)
  - TestSelectMedoids class (7 tests: medoid-selection private helper)
  - TestAutoKClusterer method that imported tiny-penalty private helper at call time (1 test)

Retained (29 tests):
  - TestDeriveKRange (10 tests) — public derive_k_range API
  - TestAutoKClusterer minus the one private-helper test (19 tests)
"""
from __future__ import annotations

import numpy as np
import pytest

from frame_generation.image_selection.auto_k import (
    AutoKClusterer,
    derive_k_range,
)
from frame_generation.image_selection.types import ImageClusteringConfig

try:
    import sklearn  # noqa: F401

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

_skip_no_sklearn = pytest.mark.skipif(
    not _HAS_SKLEARN,
    reason="scikit-learn not installed; install to run k-means tests",
)


# ---------------------------------------------------------------------------
# A. Pure-Python / numpy-only helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# B. AutoKClusterer (requires scikit-learn)
# ---------------------------------------------------------------------------


def _make_embeddings(n_per_cluster: int, n_clusters: int, dim: int = 8, seed: int = 0):
    """Synthetic well-separated unit-normalised embeddings: n_clusters tight blobs."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    vecs = []
    for c in range(n_clusters):
        noise = rng.standard_normal((n_per_cluster, dim)).astype(np.float32) * 0.05
        blob = centers[c] + noise
        blob /= np.linalg.norm(blob, axis=1, keepdims=True)
        vecs.append(blob)
    return np.vstack(vecs)


@_skip_no_sklearn
class TestAutoKClusterer:
    def test_one_label_per_image(self):
        emb = _make_embeddings(10, 3)
        cfg = ImageClusteringConfig(min_cluster_size=3, max_cluster_size=40, max_k=10)
        result = AutoKClusterer(cfg).fit(emb)
        assert len(result.labels) == len(emb)

    def test_labels_cover_all_clusters(self):
        emb = _make_embeddings(10, 3)
        cfg = ImageClusteringConfig(min_cluster_size=3, max_cluster_size=40, max_k=10)
        result = AutoKClusterer(cfg).fit(emb)
        unique_labels = set(result.labels.tolist())
        assert unique_labels == set(range(result.selected_k))

    def test_one_medoid_per_cluster(self):
        emb = _make_embeddings(10, 3)
        cfg = ImageClusteringConfig(min_cluster_size=3, max_cluster_size=40, max_k=10)
        result = AutoKClusterer(cfg).fit(emb)
        assert len(result.medoid_indices) == result.selected_k

    def test_medoid_indices_valid(self):
        emb = _make_embeddings(10, 3)
        cfg = ImageClusteringConfig(min_cluster_size=3, max_cluster_size=40, max_k=10)
        result = AutoKClusterer(cfg).fit(emb)
        for idx in result.medoid_indices:
            assert 0 <= idx < len(emb)

    def test_medoid_indices_distinct(self):
        emb = _make_embeddings(10, 3)
        cfg = ImageClusteringConfig(min_cluster_size=3, max_cluster_size=40, max_k=10)
        result = AutoKClusterer(cfg).fit(emb)
        assert len(set(result.medoid_indices)) == result.selected_k

    def test_medoid_belongs_to_its_cluster(self):
        emb = _make_embeddings(10, 3)
        cfg = ImageClusteringConfig(min_cluster_size=3, max_cluster_size=40, max_k=10)
        result = AutoKClusterer(cfg).fit(emb)
        for cluster_id, midx in enumerate(result.medoid_indices):
            assert result.labels[midx] == cluster_id

    def test_determinism(self):
        emb = _make_embeddings(10, 3)
        cfg = ImageClusteringConfig(min_cluster_size=3, max_cluster_size=40, max_k=10)
        a = AutoKClusterer(cfg).fit(emb)
        b = AutoKClusterer(cfg).fit(emb)
        assert a.selected_k == b.selected_k
        np.testing.assert_array_equal(a.labels, b.labels)
        assert a.medoid_indices == b.medoid_indices

    def test_undersized_group_single_cluster(self):
        # N=4 < min_cluster_size*2 → derive_k_range returns empty → single cluster
        emb = _make_embeddings(2, 2)  # 4 images total
        cfg = ImageClusteringConfig(min_cluster_size=5, max_cluster_size=80, max_k=40)
        result = AutoKClusterer(cfg).fit(emb)
        assert result.selected_k == 1
        assert len(result.labels) == 4
        assert all(lbl == 0 for lbl in result.labels)

    def test_n_less_than_two_raises(self):
        emb = np.array([[1.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError, match=">= 2"):
            AutoKClusterer().fit(emb)

    def test_candidate_k_bounds_respected(self):
        # 50 images, min_cluster_size=5, max_cluster_size=20, max_k=10
        # k_min = max(2, ceil(50/20)) = 3; k_max = min(10, 50//5) = 10
        emb = _make_embeddings(10, 5)  # 50 images
        cfg = ImageClusteringConfig(min_cluster_size=5, max_cluster_size=20, max_k=10)
        result = AutoKClusterer(cfg).fit(emb)
        assert result.diagnostics.candidate_k_min == 3
        assert result.diagnostics.candidate_k_max == 10

    def test_diagnostics_candidate_scores_count(self):
        emb = _make_embeddings(10, 3)
        cfg = ImageClusteringConfig(min_cluster_size=3, max_cluster_size=40, max_k=10)
        result = AutoKClusterer(cfg).fit(emb)
        k_min = result.diagnostics.candidate_k_min
        k_max = result.diagnostics.candidate_k_max
        assert len(result.diagnostics.candidate_scores) == k_max - k_min + 1

    def test_score_formula_correct(self):
        emb = _make_embeddings(10, 3)
        cfg = ImageClusteringConfig(min_cluster_size=3, max_cluster_size=40, max_k=10)
        result = AutoKClusterer(cfg).fit(emb)
        for cs in result.diagnostics.candidate_scores:
            if cs.silhouette != float("-inf"):
                expected = cs.silhouette - cs.giant_penalty - cs.tiny_penalty
                assert abs(cs.score - expected) < 1e-9

    def test_tie_break_lower_k_wins(self):
        # Any k in range that produces the same silhouette; lowest k must win.
        emb = _make_embeddings(5, 4)  # 20 images, 4 tight clusters
        cfg = ImageClusteringConfig(
            min_cluster_size=2,
            max_cluster_size=10,
            max_k=4,
            random_seed=17,
        )
        result = AutoKClusterer(cfg).fit(emb)
        scores = {cs.k: cs.score for cs in result.diagnostics.candidate_scores}
        best_score = max(scores.values())
        winner_ks = [k for k, s in scores.items() if abs(s - best_score) < 1e-9]
        assert result.selected_k == min(winner_ks)

    def test_giant_penalty_fires_in_diagnostics(self):
        # One dominant cluster (20 images) + one tiny (2 images); k=2 forced.
        rng = np.random.default_rng(99)
        vecs = np.vstack([
            rng.standard_normal((20, 8)).astype(np.float32) * 0.01
            + np.array([1.0] + [0.0] * 7),
            rng.standard_normal((2, 8)).astype(np.float32) * 0.01
            + np.array([-1.0] + [0.0] * 7),
        ])
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        # k_range = [2, 2]: max_k=2 forces single candidate
        cfg = ImageClusteringConfig(
            min_cluster_size=2,
            max_cluster_size=22,
            max_k=2,
            giant_cluster_threshold=0.5,
            giant_cluster_penalty=0.5,
        )
        result = AutoKClusterer(cfg).fit(vecs)
        assert result.selected_k == 2
        score_at_k2 = result.diagnostics.candidate_scores[0]
        # 20/22 ≈ 0.91 > 0.5 → giant penalty must have fired
        assert score_at_k2.giant_penalty == pytest.approx(0.5)

    def test_kmeans_dispatch_standard_path(self):
        # N=10 < mini_batch_threshold=2000 → _fit_kmeans used, not _fit_mini_batch
        emb = _make_embeddings(5, 2)  # 10 images
        cfg = ImageClusteringConfig(
            min_cluster_size=2, max_cluster_size=10, max_k=5, mini_batch_threshold=2000
        )
        import frame_generation.image_selection.auto_k as _ak
        from unittest.mock import patch

        with patch.object(_ak, "_fit_mini_batch", wraps=_ak._fit_mini_batch) as mb_mock, \
             patch.object(_ak, "_fit_kmeans", wraps=_ak._fit_kmeans) as km_mock:
            AutoKClusterer(cfg).fit(emb)
            assert km_mock.called
            assert not mb_mock.called

    def test_diagnostics_n_input_matches(self):
        emb = _make_embeddings(10, 3)  # 30 images
        cfg = ImageClusteringConfig(min_cluster_size=3, max_cluster_size=40, max_k=10)
        result = AutoKClusterer(cfg).fit(emb)
        assert result.diagnostics.n_input == 30

    def test_diagnostics_cluster_sizes_sum_to_n(self):
        emb = _make_embeddings(10, 3)
        cfg = ImageClusteringConfig(min_cluster_size=3, max_cluster_size=40, max_k=10)
        result = AutoKClusterer(cfg).fit(emb)
        assert sum(result.diagnostics.cluster_sizes) == 30

    def test_max_k_warning_when_selected_k_hits_cap(self):
        # Force k_range = [2, 2] where max_k=2; selected_k will equal max_k=2
        emb = _make_embeddings(5, 4)  # 20 images
        cfg = ImageClusteringConfig(min_cluster_size=2, max_cluster_size=10, max_k=2)
        result = AutoKClusterer(cfg).fit(emb)
        if result.selected_k == cfg.max_k:
            assert any("MAX_K" in w for w in result.diagnostics.warnings)

    def test_airgap_module_importable_without_sklearn(self):
        # Importing the module must not trigger a sklearn import at module level.
        import importlib
        import sys

        sklearn_backup = {k: v for k, v in sys.modules.items()
                          if k == "sklearn" or k.startswith("sklearn.")}
        for k in list(sklearn_backup):
            del sys.modules[k]
        try:
            sys.modules.pop("frame_generation.image_selection.auto_k", None)
            sys.modules.pop("frame_generation.image_selection", None)
            mod = importlib.import_module("frame_generation.image_selection.auto_k")
            assert hasattr(mod, "AutoKClusterer")
            assert hasattr(mod, "derive_k_range")
        finally:
            sys.modules.update(sklearn_backup)
