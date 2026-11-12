# COT identity repair: isolated model sensitivity

Owner approved2026-11-12 after310. Priority: verify the corrected COT series,
measure input/policy impact, then inspect critic reliability before another
phase2 pass. Retsu owns live source completion under the accompanying handoff.
No automatic deployment, checkpoint replacement or overwrite of sealed data.

## Frozen diagnostic protocol

NON_GOVERNING: replace only cot.eur.net_long, net_share, net_z156 in memory,
keeping the historical93-hour lag and14-day staleness rule solely to isolate
identity effects. Actual publication timing/vintages are NOT certified by
this probe. No fitting, retuning, rescaling, zero fill or changed support.
Recompute all three weekly series from Retsu's long/short counts; authenticate
the row digest and preserve source files. Source control must reproduce299
native accounts. Same71 inputs, weights, fixed TRAIN scaler and windows.
Replay TRAIN2020-2023 and VALIDATION2024 only. TEST stays excluded.

## Requirements and tests before execution

| Requirement | Test/evidence |
| --- | --- |
| Arithmetic and identity | Reject bad digest/market/order/duplicate; recalculate net/share/z156 |
| No future lookup or synthetic fill | Exact release boundary, stale/missing => NaN, future-prefix invariant |
| Same support and non-COT signals | Input copy; only3 channels differ; valid rows all finite; source unchanged |
| Native model/account | Replay original metrics; same weights; outputs and account on replacement |
| No false deployment claim | Explicit legacy-clock diagnostic, publication_verified=false, promoted=false |

The existing weekly z156 is a named feature from the original selection; it is
not the prohibited rolling normalization of all model inputs. This diagnostic
does not introduce another transformation or change its parameters.

Next: Retsu supplies actual publication/vintage ledger and live readers.
Musashi qualifies repaired data before new training/selection, while independent
critic diagnostics may continue with their historical-data limitation explicit.
Existing raw/seasonal31 queues stay paired and preserve their old data identity.

## Execution and results

Completed2026-11-12,23:03UTC,17.36seconds on CPU. Seven new unit tests and
15 existing frozen-input/signal regressions pass (22 total). Receipt identity
`70af56a585c8bd3efc45929bcc4b84092f1cf20663147f5366ed49299ccd7419`.
Owner-local complete run: `/var/tmp/sac71-cot-identity-20261112-r1`.
Compact evidence: [RESULT.json](generalization/cot_identity_20261112/RESULT.json).

| Inputs to identical frozen model | TRAIN return % | VAL return % | VAL Sharpe | VAL trades | VAL win % | VAL DD % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original mixed-contract COT |16.058224|18.369946|2.690712|207|59.9034|3.257290|
| EURO FX identity, legacy clock |13.859355|1.541746|0.348239|122|51.6393|3.934718|

TRAIN trades745 ->575; VAL buy/sell22/185 ->24/98. Both diagnostic accounts
remain above2 orders/week (TRAIN2.7512,VAL2.2593). Early closes TRAIN58 ->38
and VAL6 ->6. Full costs, sides, win counts, close reasons and native ledgers
are retained; no retuning or cherry-picked endpoint.

Changing only COT changes1202/24026 TRAIN greedy signals and527/6208 VAL
signals.438 VAL SELL signals become NOP;5 become BUY. These are raw model
decisions, not438 executed orders. Price path, controller, costs, support and
the other68 channels are identical. The original account metrics reproduce
the preserved299 result exactly. Weights are unchanged and source files intact.

The z156 input differs in every valid window; net/share differ in14923 TRAIN
and4419 VAL windows. This establishes a material dependency on COT identity;
it does not establish that all historical gains came from COT or forecast how
a model retrained on qualified repaired data will perform.

Independent arithmetic found104 extra warmup NaNs in Retsu's artifact: it
declares min52 but implements min156. Finite z values match the independently
recomputed sample std within2.23e-15; long/share match exactly. The diagnostic
recomputes min52, records the discrepancy and rejects arbitrary late gaps.
All104 cells precede TRAIN, so this discrepancy does not explain the results.
Initial fail-fast COT_ARITHMETIC caught it before any account ran.

Limit: Retsu's long/short payload and row digest are authenticated here, but the
annual archives are not independently re-extracted by this tool. Real publication
times/vintages are not yet qualified. These positive numbers are sensitivity
results under the old clock, NOT certification of corrected trading performance.

## Current order

1. Retsu executes [the updated handoff](../handoffs/MUSASHI_TO_RETSU_COT_AND_LIVE_COMPLETION_2026_11_12.md):
   raw reconstruction, actual publications and independent live readers.
2. Musashi audits frozen-Q coverage/policy drift, labeling historical-data
   evidence separately. No new actor reset merely because head16 fits Q better.
3. Once repaired data are qualified, build a versioned recipe/scaler on TRAIN,
   measure/refit paired controls, then resume cumulative phase2 with all49
   historical removed features. Do not silently reuse old selection scores.
4. Finish existing STL31 paired queues as historical-source diagnostics; no
   automatic live promotion. Do not interpret40 STL exclusions as proof that
   those inputs cannot be acquired in live trading.
