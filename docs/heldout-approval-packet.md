# Held-Out Evaluation — Approval Packet

**Project:** Automatic User Context Profiles · branch `dev/asaid`
**Status:** PROPOSAL — human sign-off required (see CHECKLIST "Held-out evaluation gate")
**Scope:** propose cutoff + 3 held-out `/goal` tasks only. No extraction, no `02`, no compilation, no scored runs, no treatment-file edits.
**Decision on file (2026-08-27):** keep the recent UES4 trio; claim scoped to repository-rediscovery savings from general context; three tasks acknowledged as one correlated feature program, reported as directional at N=3.

## Corpus reality check

The *ingested* `raw_conversations_abdullah_said` (`data/raw_abdullah_tab.csv`) is **247 rows / 110 conversations, 2026-07-10 → 2026-08-24** (the `Datetime` column mixes `MM/DD/YYYY` and ISO8601) — **not** the 5,480-row `all_sessions` sheet. Per instruction, **`ASaid7/smart-bites` is excluded entirely**; all prior smart-bites candidates are dropped. All 10 `/goal` launches in this corpus are on the **accessible `~/Projects/platform`** repo (branch `image-selection`, pushed to `origin/image-selection`), the **UES4 image-embedding-clustering** program.

## Approval table

| # | Task | Repo / branch | Pinned pre-solution SHA | Deliverable file(s) | Objective success check | Source conv IDs | Offline-reproducible? |
|---|---|---|---|---|---|---|---|
| **CUTOFF** | `CUTOFF_TS = 2026-08-17T00:00:00Z` (UTC) | — | — | — | precedes all held-out solution leakage (§7) | — | — |
| **T1** | UES4 auto-**k k-means** image-selection production module | `platform` / `image-selection` | `ac6a62b2c84fd228844ea0c1e0af726d2ac74bcc` | `src/frame_generation/image_selection/auto_k.py`, `types.py`, `__init__.py` | `pytest tests/unit/test_image_selection_auto_k.py` green (45 tests, air-gap-safe) + `ruff` clean | `d8d5b84d` (impl+fix); follow-ups `06147d1c`, `15616d81` | ✅ Yes (pure Python) |
| **T2** | UES4 **HDBSCAN** near-duplicate / exemplar diagnostics spike | `platform` / `image-selection` | `ac6a62b2c84fd228844ea0c1e0af726d2ac74bcc` | `notebooks/ues4_hdbscan_spike.py` | Structural: notebook builds + runs documented sections, emits density-cluster diagnostics over SigLIP2 embeddings | `3b41f7ea` | ⚠️ No — needs `autonomy_dev` ws + `/Volumes` data |
| **T3** | UES4 **candidate-blocking + threshold-graph** clustering spike | `platform` / `image-selection` | `ac6a62b2c84fd228844ea0c1e0af726d2ac74bcc` | `notebooks/ues4_candidate_graph_spike.py` | Structural: notebook builds + runs, emits connected-component clusters over blocked candidate pairs | `c7b01e88`, `d7537515`, `cdef3eb6` | ⚠️ No — needs `autonomy_dev` ws + `/Volumes` data |

Solution commits (for exclusion, **never for extraction**): T1 `98b8bd7`, T2 `e0367a3`, T3 `5934492` — **each has parent `ac6a62b`**; all three deliverables verified **absent** at `ac6a62b` (true pre-solution state).

## 1. Standalone, executable `/goal` prompts

Rewritten to be self-contained and contamination-safe — they do **not** instruct reading solution-era ticket docs or inspecting sibling UES4 spike notebooks (both would leak held-out solutions; see §8).

**T1 — auto-k k-means module**

> `/goal` Implement a reusable, air-gap-safe image-selection clustering module at `src/frame_generation/image_selection/`. Add `auto_k.py` and supporting `types.py`. Requirements: (1) `derive_k_range(n, *, min_cluster_size, max_cluster_size, max_k) -> list[int]` deriving a contiguous candidate-k range with k≥2 and giant/tiny-cluster guards; (2) an `AutoKClusterer` (config-driven) whose `.fit(embeddings)` scores each candidate k by silhouette minus giant-cluster and tiny-cluster penalties, dispatches deterministic k-means or mini-batch k-means as appropriate, selects medoid representatives per cluster, and returns a structured `KMeansResult` with per-cluster diagnostics. Coherence contract: never report a `selected_k` that disagrees with the effective emitted labels; no duplicate/fallback medoids for empty clusters; degenerate/collapsed cases return a coherent single-cluster result. Defer numpy/sklearn imports to call time so the module stays importable with no native wheels. Add `tests/unit/test_image_selection_auto_k.py` covering k-range derivation, penalty formulas, medoid validity, and collapse handling. Do not edit code outside `src/frame_generation/image_selection/` and `tests/unit/`. Acceptance: `pytest tests/unit/test_image_selection_auto_k.py` green; `ruff check` clean.

**T2 — HDBSCAN diagnostics spike**

> `/goal` Create `notebooks/ues4_hdbscan_spike.py`, a Databricks-source notebook-style spike that runs **HDBSCAN** over L2-normalized SigLIP2 image embeddings to diagnose near-duplicate / exemplar structure for UES4 image selection. Discover RGB images from the configured dataset roots under `/Volumes/autonomy_dev/autonomy_dev_bronze/dirty_data/KAGGLE/` (handle subdirectories; skip missing/empty dirs as non-errors), exclude mask/label images. Provide clearly-marked configured dataset sections, embedding load/normalize, HDBSCAN fit with min_cluster_size/min_samples config, and cluster-size + noise-point + near-duplicate-pair diagnostics with markdown commentary. Notebook spike artifact only — do not add or modify pipeline source under `src/`. Acceptance: notebook parses as a Databricks source file and, on `autonomy_dev`, runs each dataset section end-to-end producing density-cluster diagnostics.

**T3 — candidate-graph / threshold-blocking spike**

> `/goal` Create `notebooks/ues4_candidate_graph_spike.py`, a Databricks-source notebook-style spike for UES4 **candidate blocking + threshold-graph clustering**. Over L2-normalized SigLIP2 embeddings from the configured `/Volumes/autonomy_dev/autonomy_dev_bronze/dirty_data/KAGGLE/` dataset roots (plus the drone-footage dataset path as a first-class section; handle subdirectories, skip missing/empty as non-errors), build a blocked candidate-pair set, threshold pairs by cosine similarity, form a similarity graph, and cluster via connected components. Provide configured dataset sections, blocking/threshold config, graph construction, connected-component cluster extraction, and cluster-size/edge-density diagnostics with markdown commentary. Notebook spike artifact only — do not add or modify pipeline source under `src/`. Acceptance: notebook parses as a Databricks source file and, on `autonomy_dev`, runs each dataset section producing connected-component clusters.

## 2. Feasibility evidence at `ac6a62b`

- `ac6a62b` = "Merge `feature/ues1-candidate-source-seam` into `main`" (2026-08-18); ancestor of HEAD and of all three solution commits.
- **Pre-solution verified:** `git cat-file -e ac6a62b:<file>` → all three deliverables **absent**; each solution commit's parent is exactly `ac6a62b`.
- **T1 provably feasible:** the real solution (`98b8bd7`) was authored directly on `ac6a62b`; the module is documented "air-gap safe" (deferred numpy/sklearn imports) with 45 passing offline unit tests. Package scaffold `src/frame_generation/__init__.py`, `pyproject.toml` pytest config (`testpaths=["tests"]`), and `Makefile` already exist at base.
- **T2/T3 feasible as real work:** solutions `e0367a3` (1,419 lines) and `5934492` (1,353 lines) were authored on `ac6a62b`; feasibility of the deliverable is proven by their existence. Scored validation depends on live workspace/data (§8).

## 3. Source conversation IDs

- **T1:** `d8d5b84d` (impl + review-fix, Aug 21) + integration follow-ups `06147d1c`, `15616d81` (Aug 22).
- **T2:** `3b41f7ea` (Aug 21).
- **T3:** `c7b01e88`, `d7537515`, `cdef3eb6` (Aug 20).

## 4. Complete feature-cluster exclusion set

The **temporal cutoff (2026-08-17T00:00Z) already excludes the entire UES4 program** — every corpus conversation on/after that instant. The explicit ID exclusions below are belt-and-suspenders and for the record.

- **T1 cluster:** `d8d5b84d`, `06147d1c`, `15616d81`, plus integration convs `dae9f10f…`, `5820684a…`, `69f7003b…`, `bee6864a…`, `54f431ab…`, `36b9bfa3…`, `233a6c24…`.
- **T2 cluster:** `3b41f7ea`, plus HDBSCAN-touching convs `6e4b3402`, `b0f6dbce`.
- **T3 cluster:** `c7b01e88`, `d7537515`, `cdef3eb6`, `97e19ff4`, `137d36c8`, and the Aug-17→19 candidate-graph design convs (`ac12bd5c`, `b7b22d8e`, `4fc7d653`, `477e1874`, `8b3edc4b`, and omni-namespaced `…01a011d0…`, `…01a01216…`, `…01a015eb…`, `…01a015f7…`, `…01a01acb…`, `…01a01b29…`, `…01a01b6f…`, `…01a01b75…`, `…01a01b79…`, `…01a01f70…`, `…01a01f77…`).
- **Shared UES4 program cluster (exclude for all three):** the SigLIP2 embedding-clustering sibling `486c522b` / `81189d5` (+ `07a9305a`), the design megasession `bbd3d128` (Aug 17), the multi-feature launch convs `6e011c33`, `566160042386…`, `38d34d89…`, `07f02c96…`, `84fb0195…`, and the entire Aug 22–24 integration / UES8 / UES9 / UES10 tail (`ecf1b8a` / `900eef4` / `d4c17b1` seam+facade, `9087418` / `f4a1ac9` embeddings store, `23dbfc2` UES10 demo, review NBs) — corpus convs `c6ca284a`, `89e3cdc8`, `b25b2da6`, `6ee20497`, `55e02aa4`, `28339a95…`, `461ad24…`, `ae48ddb4…`, `02214c3c…`, `9ddd02bf…`, `871507b6…`, `5eed8adc`, `4e556a8c`, `da98fe98…`, `01a03458…`, and the remaining `…01a03…` omni rows.

## 5. Independence

Three distinct clustering paradigms with disjoint deliverable files and disjoint source conversations: T1 = centroid/partition (k-means, `src/` module + tests), T2 = density (HDBSCAN, notebook), T3 = graph/connected-components (blocking, notebook). Solving any one yields none of the others' code or algorithm. T1's own contract explicitly states "USE AUTO-K K-MEANS, **not** graph/radius/connected-components," cleanly separating it from T3. All three branch from the same base yet touch non-overlapping paths.

Caveat (accepted on file): the three tasks are one correlated feature week (same data, same embedding pipeline, same repo era). They are independent as *coding deliverables*, not as independent *domains*; results are reported as directional at N=3.

## 6. Success / evaluation criteria

- **T1 (objective, per SPEC §6):** `success = pytest tests/unit/test_image_selection_auto_k.py green AND ruff clean AND no files touched outside src/frame_generation/image_selection/ + tests/unit/`. Efficiency: `exploratory_reads_before_edit`, total tool calls, failed-test cycles.
- **T2/T3 (structural):** notebook parses as Databricks source; each configured dataset section executes on `autonomy_dev` producing the expected diagnostics; no edits under `src/`. **No unit-test gate** — weaker than the SPEC's tests-green ideal (§8).

## 7. Proposed `CUTOFF_TS` and why it precedes all leakage

**`2026-08-17T00:00:00Z`.** The earliest UES4-clustering conversation in the corpus is `ac12bd5c` at 2026-08-17T18:00; the base commit is 2026-08-18; the earliest `/goal` launch and spike commit are 2026-08-20; integration/operationalization run Aug 22–24. A cutoff at the start of Aug 17 sits before the first design discussion (~18 h margin) and therefore before every design note, `/goal`, spike commit, and follow-on for all three tasks. It retains the pre-Aug-17 general platform/workflow/Databricks memories the eval is meant to test.

Note: `starting_commit = ac6a62b` (Aug 18) is a *code* baseline and is independent of the memory-source cutoff — extracting only pre-Aug-17 conversations is strictly conservative.

## 8. Setup requirements & risks

- **Ticket-doc gap (contamination + feasibility):** the raw `/goal` prompts cite `docs/tickets/ues4-handoff.md` / `…operationalization.md` / `universal-exemplar-selection-tickets.md`. These were **not committed at `ac6a62b`** (first added 2026-08-23 in `e5239d3`, solution-era, and their "Proposed production build" sections may reveal solutions). The standalone prompts in §1 are rewritten to embed requirements inline and **not** point at those docs or at sibling spike notebooks.
- **T2/T3 reproducibility:** require the `autonomy_dev` workspace (id `5515503609081527`) and `/Volumes/autonomy_dev/.../KAGGLE/` (+ drone) data, and have **no offline test gate** — below the SPEC's "tests-green" bar.
- **Pre-cutoff algorithm mentions:** 2025 convs `02c8e44d`, `118657a0`, `01f8b7a6`, `11d98f6d`, `11dd3914` mention hdbscan / k-means / blocking in unrelated contexts; kept by the cutoff but should be eyeballed for latent leakage before extraction.

## Requires explicit human confirmation

1. Approve `CUTOFF_TS = 2026-08-17T00:00:00Z` and base commit `ac6a62b…` for all three tasks.
2. Approve the T1/T2/T3 set — incl. accepting that T2/T3 have structural-only, live-workspace acceptance vs. T1's offline `pytest`.
3. Confirm the §4 exclusion ID set + clearing the pre-cutoff 2025 algorithm-mention convs (§8).
4. Confirm the §1 standalone prompt rewrites (raw prompts cannot run at `ac6a62b` — missing ticket docs).

On approval: populate `config/compile_config.json` (`cutoff_ts`, `heldout_conversation_ids`, `domain_in: ["repo:platform","general_workflow"]`) and write `eval/tasks/T1.json…T3.json` — **stopping before any extraction / `02` / compilation.**
