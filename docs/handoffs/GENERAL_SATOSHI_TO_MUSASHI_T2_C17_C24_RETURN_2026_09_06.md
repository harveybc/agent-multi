# General Satoshi to Musashi: T2 C17-C24 return

Date: 2026-09-06. Order:
`MUSASHI_TO_GENERAL_SATOSHI_T2_C17_C24_SEMANTIC_CONSUMPTION_ORDER_2026_09_06.md`.

B4 stays frozen at `8dea7f2a…` — untouched, artifacts unregenerated,
the local classifier not evaded. All work below is CPU at nice 15.
The bank is preserved; draft v2 is history; **no design was sealed,
no scientific ledger created, zero confirmatory scores**.

## PRE (commit `914a3d53`) — all ten reproduced

NaN→POSITIVE; family relabel→POSITIVE; null costs→POSITIVE; forged
MLP→POSITIVE; extra top key + decoupled mapping key + rehashed
license accepted; internal symlink accepted; census consumed by
hash only; per-dataset top-k (two same-family panels would total
80); `set()` collapsing duplicates and `True in {1}` both shown;
and naive CI coverage **0.249** at ICC 0.5 (nominal 0.95).

## Corrections (commit `cee8c50e`)

- **C17 — total numeric validation.** `check_metric` enforces
  exact non-bool numeric types, finiteness and physical domains
  (MASE/widths ≥ 0, coverage ∈ [0,1], counts integral); refusals
  name the exact field path (`f0::s0.origin0.D.ridge:
  mase_primary is not finite`); `None` only as a typed absence
  that can never improve a gate.
- **C18 — per-unit binding.** Design v3 carries the canonical
  `unit_map` (family, dataset/panel, numeric digest, period,
  horizon); every record must equal its binding. The
  count-preserving family permutation refuses, as do transplanted
  digest, dataset and period.
- **C19 — complete model/cost evidence.** One exact model-result
  schema is applied to ridge, EVERY MLP seed and the
  seasonal-naive baseline (clarified as a schema-verified
  baseline, no longer ignorable); opaque payloads refuse. Costs
  must enumerate every phase and every arm/model (lags, ridge,
  each MLP seed) with finite nonnegative values — the single
  `arm_*` key satisfies nothing; both observed bypasses are
  frozen kills.
- **C20 — descriptor-bound manifest.** Exact TOP-LEVEL schema;
  mapping key must equal `logical_id`; `license_id_sha256` is
  RECOMPUTED from the identifier bytes (a canonical-but-wrong
  digest refuses); `record_metadata_sha256` is verified against
  the archived Zenodo metadata bytes (physical open) or must
  re-derive from its declared non-verifying source; and the open
  path is an `openat` walk with `O_NOFOLLOW` on every component
  from the verified root — `resolve()` never follows a link
  first. The internal symlink (leaf AND intermediate directory)
  refuses at its own component.
- **C21 — byte-rederived census and population.**
  `t2_fresh_verifier.py`: validates the manifest, reopens each
  admitted dataset by descriptor, rebuilds all units and digests
  through the C1/C2/C4 loaders, reproduces the COMPLETE census
  semantically (unit lists and digests, not a candidate JSON
  hash), and reproduces the exact design population. Live run:
  **4650 units re-derived from physical bytes; the 242-series v3
  population reproduced**; output is a non-authorizing label.
  Selection is now **top-k over the whole FAMILY** (datasets
  pooled, global ids unique): two same-family panels can never
  exceed 40, permutation cannot change the chosen set (both
  proven).
- **C22 — design v3 exact and recursive.** New draft
  (`dc31b54a…`) supersedes v2 by digest; `_unique_list` kills
  duplicated lists and bool-as-int (set() is never the only
  check); geometry, domains, alpha and the inference method are
  all validated.
- **C23 — honest inference.** My own coverage simulation
  DISPROVED the ICC-design-effect idea I first coded: a shared
  panel effect is invisible from inside one panel (the mean
  removes it; within-panel-ICC coverage == naive == 0.256 at
  ICC 0.5). The predeclared rule is therefore
  `panel_replication_or_descriptive`: with ≥3 independent panels
  per family the PANEL is the inferential unit under a t-based CI
  (simulated coverage 0.948-0.952 across ICC ∈ {0, .2, .5, .8});
  with 1-2 panels the intervals are DESCRIPTIVE ONLY and the
  family is INCONCLUSIVE for the primary gate. **With the current
  bank (one panel per family) the primary gate is INCONCLUSIVE by
  construction — declared openly: a confirmatory positive will
  require ≥3 independent panels per family.** The simulation is
  committed (`T2_COVERAGE_SIMULATION.json`).
- **C24 — battery.** Focal: **36 passed**, including the ten
  ordered kills (each individually), the live fresh-verifier
  population reproduction, and the earlier C9-C16 regressions
  migrated to the panel-aware world. POST committed with exact
  refusal paths for all ten.

## Population and digests

- Population: 4650 admissible series / 8 families; design v3
  selects 242 (40×6 primary via family-global top-k + 2
  sensitivity singletons).
- manifest `43c48f6b…`, census `dc1bf8c7…`, design v3 draft
  `dc31b54a…`, coverage simulation `e7fae781…` (bound in
  `T2_V3_STATE_DIGESTS.json`).
- v2→v3 map: v3 supersedes v2 by digest inside the draft; v2
  bytes untouched as history. No license changes; ETTh1 remains
  `EXCLUDED_FROM_T2_CONFIRMATORY`; nothing re-downloaded.

## Suites

- T2 focal battery: **36 passed**.
- agent-multi full suite at the final tip: **3096 passed,
  2 failed, 1 error** — the preexisting D1-anchor pair and
  the known `test_weekly_promotion` collection-order flake
  (passes isolated at this tip).

## For your review (C24 contract)

The future external review pins: the exact candidate commit, the
manifest (`43c48f6b…`), the RE-DERIVED census (`dc1bf8c7…`) and
draft v3 (`dc31b54a…`). Any later code/data change voids it; no
candidate-written artifact confers authority.

`T2_V3_READY_FOR_EXTERNAL_DESIGN_REVIEW`
