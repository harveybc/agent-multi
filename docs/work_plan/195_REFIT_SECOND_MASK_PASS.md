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

## Verified launch

Runtime `bbe89daf`, feature_selector `a53fbe0` /0.2.4. Native31 tests and
selector54 tests passed (4 legacy fixture-dependent skips). Actual selected
bundle probe, unmasked emissions and full native accounts all replay exactly.
Identity `b634709de2ff709847e7939ce79955dbe68e79a154c5b5f26cf8a7684876a61b`.
Persistent worker started2026-11-04T06:14:42Z on RTX5090;100 coding blocks imply
at most201 evaluation requests. Initial measured rate1.37 seconds/subset;
46 evaluations observed, no restarts, about1.7GB resident service memory.
Automatic collection retains SPEC, PREFLIGHT, PASSES, cache, RESULT, FEATURE_COUNTS,
final ledgers and CLOSE. No historical cache is reused with these new weights.

## Completion

Closed2026-11-04T06:17:33Z without errors or restarts.113 evaluations:
108 ->93 ->94 features,30 final groups;12 accepted block removals and one
single-column reinsertion (`cdl.close_minus_low_pips@asinh`). No accepted deletion
was an exact tie. Final control replay and selected account replay were exact.
Result digest `ee172eb0f2f5e692196129cd457889579840dca211f6067d81d495f2dce3e065`.

| Metric | Unmasked108 | Active94 |
|---|---:|---:|
| TRAIN return % |367.43186|210.18694|
| TRAIN Sharpe |4.67683|3.61645|
| VALIDATION return % |-3.45374|10.44706|
| VALIDATION Sharpe |-0.45088|1.44955|
| VALIDATION drawdown % |8.77992|4.72764|
| VALIDATION trades |321|298|
| VALIDATION wins % |50.77882|57.71812|
| VALIDATION trades/week |5.94444|5.51852|
| VALIDATION harmful early closes |36/85|33/64|

Early-close harm is versus the same position held to passive protection, not
account continuation. Its fraction worsened despite better account-level results.
These are selection-period, frozen-mask results; subsequent structural fit196 is
required to assess the94-channel network, and does not make2024 an untouched holdout.
