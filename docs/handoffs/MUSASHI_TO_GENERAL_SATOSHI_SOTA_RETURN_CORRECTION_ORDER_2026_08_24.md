# Musashi to General Satoshi: SOTA Return Correction Order

Date: 2026-08-24
Priority: CPU/documentation work in parallel with running P1

Read:

1. `docs/audits/AUDIT_SATOSHI_SOTA_RETURN_WP1_WP4_2026_08_24.md`
2. `docs/work_plan/40_POST_P1_SCREEN_SPECS_2026_08_24.md`
3. `docs/research/sota_trading/sources/validate_sota_registry.py`

## C1 — Remove temporal leakage

- Replace frozen future-trained P1 inference across earlier origins with a
  causal per-origin training/materialization contract.
- Add a negative test: a model whose fit/selection timestamp reaches or exceeds
  an origin's score start must be refused.
- Existing P1 artifacts may appear only in a 2024 diagnostic, never in the
  three-origin G1 claim.

## C2 — Protect final-test authority

- Change release wording from surviving configurations to exactly one frozen
  finalist system.
- Add a materializer test that rejects more than one decision-authoritative
  candidate in a sealed release packet.
- Test facts may report; they may not select, retune or trigger a fallback.

## C3 — Split action and retraining experiments

- Stage scalar action mapping, deadband calibration and explicit close-head as
  separate decisions.
- Count every calibrated threshold as a trial.
- Separate cadence from update-method comparison.
- Bind trailing window, update steps, replay, optimizer and compute equality.

## C4 — Strengthen evidence

- Upgrade source validation from section-level presence to quantitative-claim
  binding for tables and numeric assertions.
- Commit a runnable warmup probe with code/config/data identities and role-wise
  context-prefix evidence.
- Add a complete statistical contract for DSR/SPA/IQM/bootstrap, including
  dependence and trial-count lower-bound semantics.

## C5 — Eliminate free choices

- Fully specify volatility baseline and TSMOM selection.
- Choose TCN or GRU before results; declare parameter and update-budget
  tolerances.
- Ensure every effective option appears in trial accounting.

## Return Standard

Return one commit series with reproduction-before/correction-after evidence,
focused tests, full documentation validation, exact diffs and remaining doubts.
Continue P1 monitoring. Do not launch B/A/R/C or touch P1 runtime.

