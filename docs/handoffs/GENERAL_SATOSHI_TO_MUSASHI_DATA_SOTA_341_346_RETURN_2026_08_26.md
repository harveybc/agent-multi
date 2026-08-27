# General Satoshi to Musashi: DATA-SOTA-341..346 Return

Date: 2026-08-26
Correction commit: `482ad91b`; evidence smoke committed after it.
Order: `MUSASHI_TO_GENERAL_SATOSHI_WP_PRETRAIN_M1_CORRECTION_ORDER_2026_08_26`

All six reproduced BEFORE editing (frozen in
`docs/audits/evidence/DATA_SOTA_341_346_REPRODUCTIONS.json`), then
corrected with permanent regressions
(`tests/unit/test_data_sota_341_346_regressions.py` + rewritten
`tests/unit/test_branch_pretraining.py`; 171 focused green incl. Tier-A).

## Order items

1. **v1 artifacts marked.** `WP_PRETRAIN_CPU_SMOKE_2026_08_26.json` now
   carries `status: MECHANICS_ONLY_NOT_TRANSFER_ELIGIBLE` bound to your
   audit; the v1 contract schema is refused by the v2 validator, so the
   violating configuration can no longer execute.

2. **Reproductions.** Concrete numbers: fit slice held 2,190 rows of
   monitor-2022 AND 2,190 of inner-validation-2023 (341); pretrain vs
   executing tensor max-abs divergence 4.21 at the probed step (342);
   perturbing only masked raw values changed visible normalized inputs
   (343); the runner call site passed no eps (344); final-epoch
   reconstruction/quantile ratios 26-92x under 1.0/1.0 weights (345);
   resume identity omitted runner/commit/torch/normalization and the
   checkpoint/manifest pair was tearable (346).

3. **341 — causal per-origin roles.** Contract v2 requires
   `score_origin` and refuses `fit_end >= score_start`; o2022 pretrains
   through 2021-12-31 only (fit slice 9,319 rows, later rows never
   loaded). Origins beyond o2022 refuse without an explicit
   `earlier_origin_decision_frozen` reference.

4. **342 — one shared transform.** Windows are emitted by the SAME
   `feature_window_preprocessor` the GymFxEnv calls (entry-point
   loaded, source-config bound, window cross-check, preprocessor module
   sha + scaling-config digest in the identity). Tier-A regression
   proves BITWISE equality between collector windows and real
   env-emitted observations at matched bars, whole-tensor and
   per-family.

5. **343/344 — mask-safe typed normalization.** Statistics for the
   window-zscore policy come from VISIBLE steps only; the frozen
   adversary proves masked raw perturbations leave visible inputs
   identical. Policies are typed per family
   ({identity_preprocessed, window_zscore_visible}), validated,
   digest-bound, and the declared eps reaches execution (sensitivity
   regression). `identity_preprocessed` for all five families is
   justified by measured evidence: the executing rolling-256 z-score
   already delivers ~unit scale (means 0.00-0.12, stds 1.00-1.17,
   |max| <= executing clip 10) —
   `PRETRAIN_NORMALIZATION_POLICY_EVIDENCE_2026_08_26.json`.

6. **345 — predeclared balancing + diagnostics.** Effective weight =
   declared / max(initial fit-tail monitor loss, floor), frozen before
   epoch 0 and recorded. Every epoch reports train AND fit-tail monitor
   losses per objective, per-objective gradient norms + pairwise
   cosines on a fixed monitor probe, and the quantile crossing rate.
   The quantile head is monotone BY CONSTRUCTION (base + cumulative
   softplus); crossing measured 0.0 throughout; the plain-linear
   crossing counterexample is frozen.

7. **346 — durable exact resume.** Resume binds EVERY identity field.
   Checkpoint + manifest + digest seal land as one atomic fsynced
   generation; a torn pair refuses (regression). The exactness test now
   compares the COMPLETE artifact set: encoders, heads, artifact
   digests and per-epoch loss records (wall-clock excluded). This
   deeper comparison caught a real bug — the resumed branch's fixed
   monitor mask was drawn from a fresh generator — fixed by
   checkpointing the mask.

## Bounded o2022 smoke (order item 7)

One origin, committed tree, 3 epochs, newest 4,000 fit windows, all
five branches: fit-tail monitor losses DECREASE for both objectives on
every branch; crossing 0.0 everywhere; weighted totals balanced (no
reconstruction domination). Evidence:
`docs/audits/evidence/WP_PRETRAIN_O2022_CPU_SMOKE_2026_08_26.json`
(sanitized; weights + generation seal digest-bound in the restricted
store). `transfer_eligibility: NOT_TRANSFER_ELIGIBLE` — nothing loads
into SAC.

## Boundaries

No GPU, no remaining-objective implementation, no SAC loading, no
economics. Collector code (Alpaca quotes / USDCOP TRM) continues in
parallel per your order. Disposition of 341-346 remains yours.

## Parallel collector code (per your order — independent of pretraining)

1. **Alpaca quote scheduler** — lts branch
   `satoshi/alpaca-quote-scheduler-20260826@cb0d604`:
   `tools/alpaca_quote_scheduler.py`, a thin bounded loop over the
   EXISTING read-only paper-lab plumbing (latest_crypto_quotes +
   record_quote dedup). `--max-samples` is required (no unbounded
   default), typed abort after K consecutive fetch failures, per-run
   lab session with honest status/counters. NOT activated by merge —
   operator invocation only; long-term operation goes behind the
   operator's scheduler after coordination with you. 8 injected-fake
   tests, zero network/credentials in tests.

2. **USDCOP TRM collector** — agent-multi `tools/collect_usdcop_trm.py`
   (declared in TOOL_DECLARATIONS): official datos.gov.co dataset
   32sa-8pi3 into a local reference store with provenance manifest;
   idempotent upsert; malformed/non-positive rows refuse;
   **REPORTING_ONLY authority stamped in every row and manifest** (the
   COP hurdle never enters fitness, per the owner's order). 7 tests
   (fetcher injected). Executed one real bounded `--latest` fetch as
   connectivity proof: TRM vigencia 2026-08-27 collected with
   provenance.
