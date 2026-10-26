# Native wrapper: first sensitivity audit

Observed 2026-10-26, first snapshot completed at 11:51 UTC. This is a
diagnostic of frozen policies, not a feature release or architecture selection.
Parent agent code: `7bfb1047`; accounting engine: `cd6489f`.
Parent spec SHA256:
`05c06cb508b3e7bb9e1c33064c9dda4cd524671db62dd78d7defc5539da20f4d`.

## Findings

1. The positive zero-cost TCN control (training seed 7) is reproduced by an
   input-blind rule: open long when permitted, close at the next permitted
   decision. TRAIN and INNER_A trade files and daily-equity files are exactly
   identical to the learned policy's files; financial metrics match exactly.
   INNER_A: 118 orders, USD 654.13674 profit, Sharpe 3.13657545, maximum drawdown
   2.02163%. This does not establish learned conditional timing. The original
   policy's configured-cost cross-evaluation loses USD 825.41971, Sharpe
   -4.152105. Neither result is a new selected trading strategy.
2. The configured-cost TCN control (seed 7) is reproduced on INNER_A by an
   input-blind long/HOLD rule, retaining the same automatic SL/TP and calendar
   exits. Exact trades, equity and metrics: 65 long orders, USD -1265.28753,
   Sharpe -0.86499415. TRAIN is not identical; do not extend the equivalence
   beyond the measured period.
3. The TCN is NOT a dead or saturated encoder. Across the first 28 receipts,
   input sensitivity is measurable and weights differ between fits. TCN
   actor Tanh saturation fraction is zero on the sampled windows. For the
   configured seed-7 INNER_A flat probe, probabilities vary by up to 0.1863
   across rows, yet all 256 probes choose LONG. Reversing temporal order changes
   probabilities (max 0.0328) but changes no sampled flat argmax action. This is
   command collapse despite probability sensitivity, not evidence that time
   information was destroyed or that input features lack useful information.
4. The authenticated 28 receipts reconcile stored profits and costs; tested
   orders have SL/TP, positive volume and ordered entry/exit times. This is a
   bounded accounting check, not certification of every simulator behavior.
5. Synthetic mutations exposed two weaknesses in the new audit itself:
   infinite units and infinite TP could pass its first ledger check. Both now
   reject, along with invalid directions. No such values were observed in the
   first snapshot. The corrected audit uses a new evidence identity; the first
   snapshot is preserved. Focused tests: 24 passed, including sensitivity,
   negative controls, ledger mutations, path escapes and subset invariants.

## Operational decision

- Preserve the live experiment unchanged and continue its bounded sequence.
  Its existing business-coverage gate already rejects the one-direction
  controls. No positive zero-cost number bypasses that gate.
- Do not interpret tied one-out scores as evidence that removed features are
  useless, or publish them as a completed selected recipe. The behavior is
  presently a limit on the wrapper's ability to measure marginal utility.
- Run incremental authenticated CPU audits on the mirrored artifacts every
  15 minutes, one thread, with a 3 GiB memory ceiling. Cache only sealed audit
  results bound to the exact source receipt and audit source hashes. Errors
  remain visible as service failures; never convert them to passing results.
- Before attributing future gains to temporal pattern learning, require an
  input-blind comparator on the same calendar, support and execution engine.
  A separately identified paired categorical-sampling versus TRAIN-marginal
  blind diagnostic is the next inference experiment if command collapse
  persists. Do not silently change argmax evaluation inside this campaign.

## Limits and reproducibility

The sensitivity audit samples up to 256 fixed valid origins per period and
uses hypothetical flat/long/short masks. Gradients and perturbations are not
trading returns. The blind-rule replay traverses the complete native periods.
CPU/GPU forward bitwise parity is not asserted. Weights are authenticated
before and after every probe; no fitting or optimizer steps occur. No expert,
feedback feature, rolling scaler, clipping, or TEST access is introduced.

Private `IDENTITY.json`, per-unit `AUDIT.json`, replay trade/equity files and
`SUMMARY.json` retain the source receipt, code, checkpoint and artifact hashes.
Public documentation intentionally omits private paths and machine identities.
