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
   Sharpe -0.86499415. That original seed-7 TRAIN replay was not identical.
   Satoshi's later authenticated sample extends the result: 55/59 configured
   TCN receipts are exactly LONG_HOLD on TRAIN, 58/59 on INNER_A. Do not read
   the original narrow scope as evidence against widespread collapse. INNER_A
   is a reused selection block, not an independent confirmation period.
   See report91 for the larger follow-up snapshot and mandatory blind controls.
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

## Deployment verification

Audit implementation `99de55f5` is exported into an isolated runtime; the
training runtime remains unchanged. The enabled user timer completed its first
round on 29 receipts, then a manual second tick authenticated the cached
evidence and the next receipt: 30 audited, exit 0 at 12:02 UTC. Second-tick peak
memory was 1,422,274,560 bytes and CPU time 4.12 seconds. Its next tick is
scheduled 15 minutes after completion. The first snapshot's files remain
separate from this corrected audit identity.

At that check, training was active with no service restart, 30 completed fits
and eight complete subset summaries (including the full control). These are
progress counts, not a final selected feature/group count. The audit performs
no fits, accesses no TEST data, and writes only to its separate evidence tree.
