# Audit request: completed multiscale output-consumption validation

**WITHDRAWN AS A FINAL AUDIT REQUEST (owner clarification, 2026-10-26).**
This packet contains control-validation evidence, not the optimized final
strategy requested by the owner. Do not begin a final acceptance audit from
this request. The replacement requires completed DEAP results, controls,
finalist-wide scenario verification and the strategy justification together.
See [execution plan 101](101_FINAL_STRATEGY_OPTIMIZATION.md).

For Satoshi. Read [report 99](99_MULTISCALE_STRATEGY_EXECUTION_REPORT.md),
the frozen [design 98](98_MULTISCALE_STRATEGY_ACCEPTANCE_DESIGN.md), and the
[JSON axes](../../configs/validation/strategy_multiscale_matrix.json).
This supersedes request 97's instruction to merely propose the missing trend
strength / transition speed / decision-horizon extension: that extension is
implemented and executed. Do not audit an obsolete 5,832-cell run as this run.

## Pins and scope

- Full execution producer: agent-multi `d7ede10e`; engine `cd6489f`.
- Independent coverage verifier and mutation tests: `74fbded5`.
- Policy command decoder: real categorical `OrderDistribution`; the grid
  injects scores at that interface, not through a trained encoder.
- Full policy integration is exercised separately by the native integration
  tests and monotonic witnesses. No encoder fitting or financial data testing.
- Original failing run `8da62c24` is retained separately. Six recorded cells
  failed before the numeric correction and passed when replayed afterward.
- Private-index roles: `multiscale_matrix`, `multiscale_runtime`,
  `multiscale_verification`, `multiscale_tests`, `multiscale_failed_run`.

## Required independent checks

1. Reconstruct all 47,628 tuples without importing the generator. Check every
   cell, not just summary counts. Distinguish oscillation period, transition
   duration, and holding horizon; check ramps and prefix invariance.
2. Verify prescribed expected commands and actual fills for NOP, BUY, SELL,
   HOLD_LONG, HOLD_SHORT, CLOSE_LONG and CLOSE_SHORT. BUY and SELL both positive
   must not imply BUY-first execution. Invalid/wrong-side commands must fail
   before mutating balances, orders or the clock.
3. Attack the float32 correction. Distinct representable logits must not become
   a tie through softmax. Ordinal diagnostic scores must preserve the original
   counterfactual ordering, not pick a new desired winner. Exact ties retain
   their declared first-index behavior. Sampling/log-probability is unchanged.
4. Recompute TP/SL and unit sizing independently, separately by direction.
   Check next-open fills, the percentage-of-current-balance cap, below-minimum
   refusal without risk increase, q=0 and q=1, and crossed/gapped levels.
5. Independently recompute spread, commission, market/stop slippage and funding
   at the recorded clocks. Tampering with signed short P&L or removing fees
   must be caught even if the engine's balance agrees with its own wrong ledger.
6. Reproduce the two-sided broker witnesses: SL/TP, both-hit SL priority, gaps,
   early close before SL and before TP, forced-flat, daily cap, missing windows.
   Recalculate why some natural positioned states do not survive entry; never
   label those as successful open-position comparisons.
7. Kill deliberate mutants: SELL-to-BUY mapping, ignored discretionary close,
   ignored stop, calendar bypass and removed costs. Remove a cell or forge its
   state while retaining a zero-failure summary; the independent verifier must
   reject it. Record semantic failures separately from mere digest failures.
8. Check six-worker/serial cell parity and the immutable-code receipts. Replays
   go into a fresh directory, never into the closed evidence root.
9. Keep oracle scopes separate: literal prescribed commands and independent
   accounting versus simulator-derived comparative P&L. The latter tests
   arbitration, not an independent proof of economic optimality.
10. Return separate verdicts for the tested execution contract, finite coverage,
    optimization evidence and learning. This control run cannot rehabilitate
    the previously rejected DEAP decision or the failed planted-signal learner.

## Reproduction

Resolve paths privately. Use a clean export of `d7ede10e` and the same recipe,
engine and source spec. CPU only; six one-thread workers; memory cap 8 GiB.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON" -m tools.systematic_strategy_matrix \
  --recipe "$RECIPE" --source-spec "$SOURCE_SPEC" --engine-root "$ENGINE" \
  --strategy-config configs/validation/directional_strategy_midpoint.json \
  --design configs/validation/strategy_multiscale_matrix.json \
  --workers 6 --output "$NEW_REPLAY_ROOT"
```

The acceptance run has NO `--limit`. A limited replay is a smoke, never full
coverage. Run `app.strategy_matrix_verification.verify` from verifier pin
`74fbded5` on the closed root; compare to the published manifest. Mutations
belong only in disposable copies. Do not start training, new optimization,
TEST reads, deployment or active-worker changes during this audit.

Return a Markdown verdict with minimal counterexamples, affected requirements,
commands and exact evidence. Do not transfer missing implementation back to the
owner: name a precise correction for any finding. No universal continuous-space
coverage, optimality or predictive-performance claim is requested.
