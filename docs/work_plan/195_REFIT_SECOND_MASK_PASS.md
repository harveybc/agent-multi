# Second ordered mask pass after structural108 refit

Owner authorized unattended execution on 2026-11-04 after the two warmed-up
stopping fits completed. Both stopped at epoch48 and restored epoch28.
Validation-only fit: TRAIN return367.4318577%, VALIDATION -3.4537431%,
validation Sharpe -0.450882, drawdown8.7799195%,321 trades.
Guarded-mean fit: TRAIN541.5473353%, VALIDATION -16.46431855%,390 trades.
These are single-seed development results, not a paired multi-seed superiority claim.

## Frozen protocol

Use the validation-only bundle, seed7, model sha256
`e512f05a7f3f6bb747fa5200c63d8839097dc8f148f0801b1e6fa40a1450f71e`.
The first pass was120 ->108 ->108. This round begins with the108 physically
retained inputs and31 groups, not the old120-channel models. Previously removed
channels cannot be reintroduced into a narrower network without a separate fit.

1. Replay the exported probe and full unmasked account exactly on the original
   numerical backend. Authenticate source, package, data, scaler, calendar,
   support, native engine, checkpoint and parent completion receipts.
2. Visit each coding block once, in manifest order. Zero that whole temporal
   channel/block AFTER fixed TRAIN scaling, without changing any row or window.
   Compare to the CURRENT accepted subset. Remove when eligible and validation
   net return/week improves or is exactly equal. No rounding or loss tolerance.
3. After finishing all deletions, revisit accepted removals in removal order.
   Reinsert only if validation net return/week strictly improves. Keep paired
   sine/cosine and regime-probability blocks atomic. At least one feature remains.
4. Evaluate full TRAIN2020-2023 and full VALIDATION2024 for every candidate;
   both require strictly more than2 trades/week. No fitting, DEAP, expert input,
   feedback state, rolling normalization, clipping, window or support changes.
5. Replay the selected subset with trade ledgers and early-close diagnostics;
   repeat unmasked control after all interventions. Save counts, reasons,
   per-candidate native metrics, selected groups and an authenticated closure.

VALIDATION is development data used in this wrapper, not independent confirmation.
TEST2025 is never read. Masks do not establish the performance of a structurally
pruned/refitted model. Subsequent fresh structural confirmation is a separate step.

## Implementation and acceptance evidence

Generic traversal remains in feature_selector, version0.2.4, with the optional
`remove_on_tie` flag (default false for historical compatibility).
Native adapter: `tools/run_refit_mask_pass.py`. Resume recomputes the control and
authenticates cached evaluations, then reconstructs the same deterministic path.
One exclusive worker lock and a persistent bounded-restart service prevent duplicate
writers. No approvals are needed between removal, reinsertion and final replay.

Tests: exact ties removed/not reinserted; tiny losses rejected; current-state
comparison; coherent blocks; validation ranking not compensated by TRAIN;
activity boundary; nonfinite rejection; native-account parity; altered checkpoint
and incomplete-parent rejection. Real probe/control replay is mandatory before
the first deletion. The final count is measured, never targeted in advance.
