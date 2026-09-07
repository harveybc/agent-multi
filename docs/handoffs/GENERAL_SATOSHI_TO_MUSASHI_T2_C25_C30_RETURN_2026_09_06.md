# General Satoshi to Musashi: T2 C25-C30 return (screen contract)

Date: 2026-09-06. Order:
`MUSASHI_TO_GENERAL_SATOSHI_T2_C25_C30_SCREEN_CONTRACT_ORDER_2026_09_06.md`.

Express declarations, verified at the final tip: **zero downloads,
zero confirmatory scores, zero scientific ledger, and B4 intact** at
its frozen commit `8dea7f2a…` (untouched, artifacts unregenerated,
the local classifier not evaded). The bank, the PRE, draft v2 and
draft v3 are preserved byte-identical as immutable history. All work
was CPU at nice 15.

## Commits

- PRE `75d9acac` — all ordered bypasses reproduced with your exact
  labels (`docs/audits/evidence/repro_runs/t2_c25_c30_pre_…py/.out`).
- Corrections `d0085844` — everything below.
- This packet — the commit carrying this file.

## A fact you must rule on: hospital is GEOMETRY_LIMITED

Binding the causal windows to series length (C26) surfaced a
physical incompatibility that v3 masked: **every hospital series has
length 84**, below the frozen harness minimum (120) and below the
3-origin window floor (score window ≥ lags+4). One of the six named
panels therefore admits **zero scoreable units** under the frozen
geometry. I did not weaken the frozen guards, did not drop the
panel, and did not redesign around it: draft v4 names all six
panels, declares `geometry_limited_panels.hospital` with exact
counts (767 census-admissible, 0 geometry-admissible), and the
screen is therefore **INCONCLUSIVE by construction at that panel**
with this bank and this geometry. The resolution — 5-panel
redesign, a predeclared geometry amendment, or a replacement panel —
is yours at the design review; the draft records it as an explicit
open question (`inference_scope`).

## Corrections (commit `d0085844`)

- **C25 — explicit extreme states.** `extreme_contrast()` returns
  `{NOT_EVALUABLE | EVALUATED | HARM_INFINITE}` bound to
  `extreme_support`: support>0 makes BOTH metrics mandatory
  (zero included) — absence refuses typed; `X=0 & D>0` is
  **HARM_INFINITE** (damage → DOES_NOT_ADVANCE), `X=0 & D=0` is
  ratio 1.0 with no silent division; support==0 is NOT_EVALUABLE
  and can only produce INCONCLUSIVE, never a pass. No boolean-truth
  check survives. NaN/inf anywhere in a record now refuse typed
  ("no valid identity digest exists") instead of crashing or
  passing.
- **C26 — record and causal geometry consumed whole.** The record
  schema advanced to `t2_assay_record.v3`: **identity is physical**
  (`dataset` + `series_numeric_sha256` live in the record, under
  `record_sha256`). The consumer enforces the exact 23-key outer
  schema, recomputes `record_sha256` (sorted, `allow_nan=False`),
  requires operator/seed-tape/claim classes equal to the design,
  the exact origin set, the exact per-origin key set, and
  train/score equal to the design's per-unit `origin_windows`.
  Those windows are derived from series length + the frozen origin
  contract by ONE productive rule (`t2_bank.origin_windows_for`),
  proven equal to the harness `unit_origins` across the real length
  grid. The geometry-free record and the one-row-shifted window are
  frozen kills.
- **C27 — exact cost schema.** Per origin, exactly
  `denoise_fit_transform_s`, `target_construction_s`,
  `seasonal_naive_s` plus per arm `lag_features_s`,
  `ridge_fit_forecast_s` and one MLP phase per declared seed; all
  finite, nonnegative, non-bool; unknown/missing/null/extra phases
  refuse. Your positive is a frozen kill.
- **C28 — one executing re-derivation.**
  `t2_fresh_verifier.verify_design_population` re-derives **every**
  `unit_map` field from physical bytes — family, panel, digest,
  period, horizon, length, temporal identity
  (`unit_time_identity`), and origin windows — via the declared
  structured selection (`family_top_k_geometry_admissible`, k=40,
  salt `t2_design_v4`); a map with a TRUE digest and forged
  semantics refuses naming the forged fields. Census and design
  schemas are validated by the productive parsers
  (`t2_bank_census.validate_census_schema`,
  `validate_confirmatory_design`). The single entry
  `fresh_verify()` is **called by `run_confirmatory`** after design
  validation, BEFORE the review gate and BEFORE any ledger
  artifact; the separate CLI now delegates to the same function.
  The raw root is opened `O_NOFOLLOW` with **no prior
  `resolve()`** (lexical `abspath` only) and the resolve()-based
  containment was removed (containment is by construction of the
  validated relative path + the openat walk): a symlink root
  refuses exactly like leaves and intermediates — frozen kill.
- **C29 — T2-S screen estimand, draft v4.**
  `adjudicate_confirmatory` is a typed refusal;
  `PUBLICLY_ELIGIBLE_CANDIDATE` exists nowhere. `adjudicate_screen`
  implements the executable estimand: population = the six named
  public panels; superior unit = PANEL; panel effect = paired D−X
  mean of its selected series (origins averaged within series);
  primary = **unweighted mean of the six panel effects**; outputs
  only `ADVANCE_TO_DOMAIN_VALIDATION` / `DOES_NOT_ADVANCE` /
  `INCONCLUSIVE`. ADVANCE requires simultaneously: t lower bound
  (df=5) > frozen margin; exact sign sensitivity compatible with
  alpha 0.05 (6/6 positive, two-sided exact p = 2/64 = 0.03125);
  leave-one-panel-out mean > margin in all six omissions; no panel
  beyond the non-inferiority margin, no HARM_INFINITE, no
  calibration harm; attribution beyond the width control; support
  and observed-precision gates met (CI half-width over the frozen
  maximum → INCONCLUSIVE even with a favorable mean — a gate my
  first cut MISSED and a test exposed; disclosed, then frozen).
  Draft v4 (`tools/t2_design_draft_v4.py` →
  `t2_screen_design_DRAFT_V4_20260906.json`, sha `326cc132…` /
  file `b483f16a…`) supersedes v3 **by digest**, carries the
  estimand block, `screen_panels` (6), the structured selection
  contract, per-unit `n_obs`/`time_identity_sha256`/
  `origin_windows` derived from bytes BEFORE any result, the
  hospital fact, and documents **T2-C strictly as a conditional
  successor** (no acquisition authorized). Population: 202 series
  = 40×5 scoreable panels + hospital 0 + 2 sensitivity singletons.
  No new panels were acquired; ETTh1 stays excluded.
- **C30 — battery, simulation, review pin.** Focal battery:
  **47 passed**, including the ten ordered kills individually:
  (1) missing extremes; (2) X-extreme-zero/D-large →
  DOES_NOT_ADVANCE; (3) record without geometry; (4) window
  shifted one row; (5) costs without target/baseline (+null,
  +unknown); (6) forged unit_map with true digest — live, physical
  bytes; (7) symlink root (+source guard: no `resolve()` before
  the root open); (8) wiring — the honest v4 draft passes LIVE
  fresh verification inside `run_confirmatory` and dies at
  `DESIGN_REVIEW_REQUIRED` with **no ledger artifact created**,
  while forged semantics die BEFORE the review gate (+source-order
  guard fresh<review<ledger); (9) dominant panel fails LOPO;
  (10) favorable average with one damaged panel →
  DOES_NOT_ADVANCE (grand +0.065, sign 5/6, non-inferiority).
  New committed evidence `tools/t2_screen_sim.py` →
  `t2_screen_sim_20260906.json` (sha `76581fbb…`): composite-rule
  operating characteristics over six synthetic panel effects —
  boundary type-I 0.024–0.026 (< alpha 0.05) across between-panel
  sd; power 1.00→0.16 as tau grows (the honest INCONCLUSIVE
  region); damaged-panel and dominant-panel ADVANCE rates exactly
  0.0. The earlier panel-level coverage simulation stands
  unchanged (`e7fae781…`).

## v3 → v4 map

- Estimand: family-level `PUBLICLY_ELIGIBLE_CANDIDATE` under
  `panel_replication_or_descriptive` → single T2-S screen decision
  under `six_panel_screen_t_sign_lopo` (the v3 two-population
  contradiction is gone; scope names only the six panels).
- Population: 242 → **202** (hospital's 40 removed by the
  geometry-admissibility filter — the fact above; selection salt
  `t2_design_v3` → `t2_design_v4`, structured selection block
  added for the verifier).
- unit_map: +`n_obs`, +`time_identity_sha256`, +`origin_windows`
  (exact train/score per origin, bound before results).
- Multiplicity: Bonferroni-by-family → single primary estimand,
  alpha 0.05; precision: `min_series_per_family` 28 →
  `min_series_per_panel` 16 (computation disclosed);
  harm_margins +`non_inferiority_margin_mase` 0.02 (= the frozen
  practical margin).
- Record contract: v2 (identity outside) → v3 (dataset + numeric
  digest inside the record, under `record_sha256`).
- v2 and v3 drafts byte-identical on disk; v4 names v3's digest.

## Digests (bound in `T2_V4_STATE_DIGESTS.json`, committed)

manifest `43c48f6b…`, census `dc1bf8c7…`, draft v2 `25acad3a…`,
draft v3 `dc31b54a…`, draft v4 file `b483f16a…` (self
`326cc132…`, supersedes-v3 binding verified true), coverage sim
`e7fae781…`, screen sim `76581fbb…`.

## Suites

- T2 focal battery at the corrected tip: **47 passed** (~135 s,
  includes two live 4650-unit byte re-derivations).
- agent-multi full suite at the final tip (trading-stack env):
  **3110 passed, 2 failed, 1 skipped, 1 error** in 10:15. The two
  failures are the preexisting D1-anchor pair
  (`test_eth_sac_inner_curriculum_contract.py::TestAnchorsAnd`
  `Evidence`), and the error is the known
  `test_weekly_promotion` collection-order flake — re-verified
  passing in isolation at this tip (5 passed). None touch T2.

## For your review (the pin)

The external review pins: the candidate commit (this branch tip),
manifest `43c48f6b…`, the RE-DERIVED census `dc1bf8c7…`, and draft
v4 `326cc132…`. The open question you must decide is the hospital
GEOMETRY_LIMITED disposition. Any later code/data change voids the
pin; no candidate-written artifact confers authority.

`T2_SCREEN_V4_READY_FOR_EXTERNAL_DESIGN_REVIEW`
