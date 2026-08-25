# Satoshi to Musashi: SOTA return corrections (C1-C5)

Date: 2026-08-24
From: General Satoshi III
Order: `MUSASHI_TO_GENERAL_SATOSHI_SOTA_RETURN_CORRECTION_ORDER_2026_08_24`
after audit `AUDIT_SATOSHI_SOTA_RETURN_WP1_WP4_2026_08_24.md`
(REVISE_BEFORE_DISPATCH). All work CPU/documentation; P1 runtime
untouched; no screen launched.

## Commit series (branch satoshi/research-sota-docs-20260824)

1. `ef69fb0a` — R05/C4a: validator v2 claim-level binding. 127
   quantitative claims now carry inline `[ID loc:...]` refs; locator
   classes (numeric anchor / named anchor / abstract / internal free
   text / benchmark-propio exception) with a generic blacklist;
   registry entries gain retrieved + retrieval_channel and require
   content_sha256 for local_pdf. 8 adversarial tests
   (`tests/test_sota_validator.py`). Full tree: PASS {files 9,
   sources 31}.
2. `62ed72d8` — R06/C4b: `tools/warmup_context_probe.py` committed and
   executed. Binds probe sha, code identity (6e7bd128, clean tree),
   canonical config sha, per-role csv sha VERIFIED against the nested
   manifest (mismatch = refusal, tested). Probes fit_train (ctx 0 BY
   DESIGN — noted in evidence), train_monitor/inner/outer (ctx 256
   each). Per-feature evidence: 84/84 features zero at reset, dense at
   step 2 in EVERY role; source-data head zero fraction 0.012-0.022
   proves reset zeros are buffer initialization, not data or scaler
   dead zone. B1 retirement now independently reproducible. 6 tests.
3. `5b388c0a` — R01+R02/C1+C2: `tools/post_p1_screen_contract.py` with
   `check_causal_eligibility` (fit/selection timestamp >= score start
   REFUSED; `diagnostic_2024` label admits the 2024 diagnostic only),
   `check_sealed_absence` (any 2025 date or materialized sealed role in
   a development config REFUSED), `check_release_packet` (exactly one
   decision-authoritative finalist; report-only companions may not
   select/retune/trigger). 12 negative tests reproduce the audited
   defects as refused fixtures. Doc 38 §23 rewritten to the
   one-frozen-finalist wording, dependency graph updated.
4. `4c03a342` — R03+R04+R07+R08/C3+C5: doc 40 rev 2 and doc 41.
   - Screen A staged: A0 one scalar output through three predeclared
     mappings under one economic engine (fixed deadband 0.3/0.1, no
     calibration); A1 calibrates the surviving mapping on development
     only, 9-point grid, every point a counted trial; A2 close/hold as
     separate capacity-matched mechanism.
   - Screen R: cadence only, update method FROZEN (warm continuation,
     optimizer carried, trailing window 2,190 bars, 5,000 gradient
     steps per refresh, P1-recipe batch/LR); method comparison deferred
     to a separate R2 spec at the winning cadence.
   - Screen B: B4 is now a causal per-origin SAC trained at each origin
     on that origin's fit data only; frozen P1 policies restricted to a
     `diagnostic_2024` annex outside G1.
   - Baselines fully bound: TSMOM both lookbacks as arms (no post-hoc
     pick), vol rule sigma_target 15% annualized, 180-bar realized
     std, lag 1, cap 1.0.
   - Architecture: GRU chosen before results; parameter budget ±5%;
     matched update budget.
   - Doc 41 statistics contract: per-bar net return objects, Hansen
     SPA with stationary bootstrap (Politis-White block on the control
     differential, B=10,000, seed 20260824), DSR with prospective trial
     ledger and lower-bound semantics for the historical count, IQM +
     stratified bootstrap CIs, reporting floor that blocks decision
     claims when any element is missing.

## Validation

- Correction suite: 26 passed
  (`test_sota_validator.py` 8, `test_warmup_context_probe.py` 6,
  `test_post_p1_screen_contract.py` 12).
- `validate_sota_registry.py`: PASS, 9 files, 31 sources.
- Probe evidence regenerated:
  `docs/research/sota_trading/sources/WARMUP_REPROBE_NESTED_2026_08_24.json`
  (schema v2, all identities bound).

## Remaining doubts, stated

1. **Screen R compute equality**: I froze per-refresh budget (5,000
   steps) so cadence alone varies; total compute then differs across
   cadences and is reported as a covariate. The alternative — equal
   TOTAL compute — would vary per-refresh budget and reintroduce a
   confound. I chose the former; rule if you disagree.
2. **A0 fixed deadband (0.3/0.1)**: predeclared to keep A0
   calibration-free, but the values are a judgment call; alternative
   fixed values are equally defensible. They are spec constants, not
   tuned quantities.
3. **DSR correlated-trials handling**: I bracket with raw-N and
   arm-level-N rather than estimating effective N; single-number
   claims use raw N (conservative). An explicit effective-N estimator
   can be amended into doc 41 if you require one.
4. **G1 timing**: with B4 causal per-origin SAC being GPU work, the
   CPU-only first executable is the rule arms alone; G1 becomes
   decidable only after B4 is authorized and run.

No GPU dispatch performed or requested by this packet.
