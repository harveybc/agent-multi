# Position-Value Surface Results

2026-11-10. Implements [design256](256_POSITION_VALUE_SURFACE_CONTRAST.md).
Twelve diagnostic fits completed: eight primary arms and four explicitly
post-primary constant-market controls. The source actor, critic encoders,
observations, simulator and selected-feature artifacts were not changed.
No new actor or trading strategy was promoted.

## What Was Tested

Each market-only head produces CLOSE-minus-HOLD values for both directions.
The external controller evaluates a field at the actual order's normalized
distances to SL and TP; position variables never enter a learned layer.
The source SL, TP, volume, costs and calendar are preserved.

All866 native same-direction banks from255 were retained:2342 position
instances,1781 fit and561 monitor. Fit uses2020-2022 and monitor2023; the frozen
source representation already saw2023. This is an internal diagnostic, not
an untouched holdout. No2024/TEST evaluation was performed in this experiment.

Fresh twin heads: embedding ->8 SiLU units ->2 directional fields. Scalar has
3380 trainable parameters; progress9 has3668; mean81 and surface81 have6260.
Seeds19/43,4096 updates,batch128,AdamW LR3e-5,WD0.01,SmoothL1 gap loss.
Surface81 and mean81 have exactly equal initial tensors and minibatch schedules.
Fixed final endpoints are compared, not a best checkpoint selected afterward.

The target is the difference between native discounted account continuations:
force CLOSE or HOLD once, then follow the same frozen source actor to the
quarter's end. It is not an optimal future policy or a price forecast.

## Learned Results

Regret below is in native discounted-reward units, not dollars, percentage
return or Sharpe. For target gap y and predicted close decision c, per-instance
regret is max(y,0)-c*y; average within banks, then across banks. Lower is better.
The effective closes and harmful fractions are counterfactual position
instances, not independent trades executed by a new portfolio policy.

| Arm | Seed | FIT regret | Monitor regret | Sign accuracy | Effective closes | Harmful closes |
|---|---:|---:|---:|---:|---:|---:|
|scalar|19|0.125548|0.141910|51.16%|443|223 (50.34%)|
|scalar|43|0.132313|0.142139|48.66%|445|235 (52.81%)|
|mean81|19|0.139485|0.139840|51.16%|509|256 (50.29%)|
|mean81|43|0.134139|0.140282|50.45%|505|256 (50.69%)|
|progress9|19|0.132736|0.135801|53.65%|383|183 (47.78%)|
|progress9|43|0.133881|0.131935|56.33%|378|174 (46.03%)|
|surface81|19|0.121509|0.138716|49.55%|280|146 (52.14%)|
|surface81|43|0.116957|0.125894|54.37%|329|160 (48.63%)|

Mean monitor regret: scalar0.142025, mean810.140061, progress90.133868,
surface810.132305. Surface81 is5.54% lower than its equal-size mean81 control,
but only1.17% lower than progress9 on the two-seed mean. These are descriptive
differences, not a statistically established winner.

Always CLOSE gives monitor regret0.135859 and285 harmful closes out of561;
always HOLD gives0.150536 and misses287 account-beneficial closes. The learned
surface19 is worse than always CLOSE; surface43 is better. Aggregate mean
improvement does not certify reliable early closing.

Harm is measured independently by replaying the original position under
CLOSE versus HOLD, including native costs, until that original position exits.
For44 fit and23 monitor instances, the sign of this local benefit differs from
the whole-account gap. Neither metric is substituted for the other, and these
counterfactual dollar differences are not summed into a portfolio return.

## Constant-Market Negative Control

Added after inspecting the primary endpoints and before executing the control.
Only the frozen market embedding is replaced by its fit-only bank-balanced
mean, at both fit and monitor time. Actual position queries remain. Same heads,
initialization, targets, axes, schedules, optimizer and update budget.

| Interface | Seed | Market monitor regret | Constant-market regret | Constant harmful / closes |
|---|---:|---:|---:|---:|
|progress9|19|0.135801|0.138424|188 /389|
|progress9|43|0.131935|0.134229|177 /379|
|surface81|19|0.138716|0.134419|142 /283|
|surface81|43|0.125894|0.123867|157 /328|

Surface81 is better with constant market in both seeds (mean0.129143 versus
0.132305). Its primary improvement cannot be attributed to useful learned
market patterns. Progress9 benefits slightly from the market embedding in both
seeds, but still makes many harmful closes. This isolates a geometry effect
and a learning problem; it does not establish that the inputs contain no signal.

## Oracle Audit And Correction Of Scope

The initial per-bank least-squares oracle reproduced the correct action signs
in all866 banks for both geometric interfaces, including the actual float32
GPU decoder. Scalar and mean81 fail139 banks. This does not imply one field
simultaneously fits different banks containing the same market observation.

The post-primary audit merges exact window bytes plus direction across all
banks:860 observation groups,6 merged groups,maximum6 positions,zero mixed-role
groups. No rows were removed. This stronger audit exposes two issues:

1. Progress9's least-squares gap fit makes one wrong decision in a merged group.
2. Surface81's exact-value fit in another merged group requires coefficients
   up to6.15e11 and a condition number4.61e13. Float64 preserves the action
   signs, but the float32 CPU decoder makes one wrong decision and value error
   reaches8192.47. All five targets there are negative: HOLD is easily
   expressible. The huge coefficients are an artifact of fitting exact gap
   magnitudes, not a requirement for making the correct decisions. These oracle
   coefficients were never used by any learned head.

To distinguish those questions, a separate linear program minimizes the
maximum absolute node coefficient subject to signed margin >=1 for every
nonzero target in each group. No knots or learned weights are modified.

| Interface | Infeasible sign groups /860 | Float32 errors in feasible groups | Largest required coefficient in this solve |
|---|---:|---:|---:|
|scalar|138|0|1.00|
|mean81|138|0|1.00|
|progress9|0|0|18.70|
|surface81|0|0|15.81|

This is a finite-panel output-interface witness using retrospective labels,
not training evidence. A separate field is solved per exact observation and
direction; no neural mapping between observations is fitted by this oracle.
The learned network may still fail to produce those fields. Numerical
least-squares failure must not be confused with action-sign infeasibility.
The merged sign audit was executed on CPU; primary per-bank decoding used GPU.

## Critical Opposed-Choice Witness

At2023Q3:71,lag4,three short positions share market input and direction.
Native gaps are [+0.538037,-0.291029,-0.288829]: required decisions are
CLOSE,HOLD,HOLD. Both geometric interfaces express this under the oracle.
The learned progress9 heads instead choose CLOSE,CLOSE,CLOSE in both seeds.
Surface81 chooses HOLD,CLOSE,CLOSE for19 and HOLD,HOLD,HOLD for43.
Thus neither learned interface resolves the critical witness at the endpoint.

Two monitor surface queries lie outside the fit-only axes. Both remain in all
metrics and are flagged in the CSV. Endpoint linear extrapolation is diagnostic
only, not permission to deploy extrapolated risk decisions.

## Evidence And Verification

- Primary runtime f2f437fa; identity
  `ca2e70da1a63daf22afc9eb37b4f3f6ad8ff681c81ab2ce60ff792307b4a9a4c`.
  Eight arms,56.24 seconds,89 collected files verified.
- Constant-market runtime176186ed; identity
  `ed82e61a7f88badfcf708a8913a06901e71411e951b424f944669167ae0353d2`.
  Four arms,30.48 seconds,37 collected files verified.
- Merged audit identity
  `1c35ef82c80d0821fe56b52aabf064b36ac5f9cba05e4e5f7b1e2e5ccb663480`.
  Four evidence files; no optimizer steps. The preceding least-squares-only
  audit remains archived under identity cc7c22c7, not overwritten.
-16 native first-action repeat controls: maximum discrepancy0.0. Parent state,
  reference traces/cash, source model hashes and full-TRAIN actor logits exact.
  All trained heads reload with identical predictions.
- Latest focused suite:13 tests passed, including duplicate-observation grouping,
  infeasible opposite signs at identical queries, stable same-sign fitting,
  native original-position accounting and no-monitor constant-embedding fit.
- Expanded run including the repository engineering-surface suite:29 passed,
  1 failed. The failing inventory assertion lists16 undeclared older tools;
  recomputation using HEAD's tracked tools and declarations yields the identical
  list. The new audit tool is declared and introduces no inventory difference.
  No unrelated tool declarations were guessed or changed to hide this failure.
- Both GPU worker services and automatic collectors exited successfully.

Compact copies: [primary CSV](generalization/position_surface_20261110/METRICS.csv),
[primary curves](generalization/position_surface_20261110/CURVES.png),
[constant-market CSV](generalization/surface_market_control_20261110/METRICS.csv),
[control curves](generalization/surface_market_control_20261110/CURVES.png),
[merged audit](generalization/surface_shared_audit_20261110/RESULT.json).
These are subsets of the complete sealed collections named by their DONE.json
receipts, not self-contained copies of every checkpoint and native record.

## Decision And Next Experiment

Keep the source actor and selected-feature artifacts unchanged. Do not transfer
these heads into the actor or use them to replace the phase2 evaluator.
The market-only field plus external geometry is an expressive candidate on this
panel, but learned decisions do not yet pass the harmful-close/witness checks.

Next focused contrast: absolute-gap SmoothL1 versus action-sign loss weighted
by absolute native gap, with a paired within-bank term for positions sharing
market inputs. Freeze the same encoder/actor, interfaces, labels, seeds and
constant-market controls. Compare actual decision regret and harmful closes,
not just value RMSE. This is a proposed separate experiment, not an unreported
extension already running. Actor transfer follows only after learned behavior
is qualified. Do not enlarge the network or blame the dataset from these results.
