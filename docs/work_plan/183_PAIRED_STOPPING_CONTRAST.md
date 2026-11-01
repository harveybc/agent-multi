# Paired Three-Signal Stopping Contrast

Owner authorizes continuation after182. The first campaign is closed:
seeds7/19 completed financial fits; seed43 exhausted131072 synthetic decisions
without qualifying and is retained as a failure. This contrast is conditional
on the two qualified seeds, not a three-seed success or robustness claim.

## Frozen Comparison

One fresh financial learning trajectory per qualified seed, with unchanged
phase1 inputs, TCN/SAC, risk profile, execution, costs, reward, replay, maximum
1048576 steps, minimum131072 and assessment every32768 decisions. No
synthetic weights transfer. Reuse authenticated qualification evidence, not
financial model weights. Same accelerator per seed as the original run.

Six independent shadow stopping rules observe the SAME periodic accounts:

- Original combined mean score, patience3/5/10.
- Combined score with internal-validation non-degradation, patience3/5/10.

Guarded acceptance requires both an improvement of the combined score by
min_delta1e-5 and internal return/week not below the currently accepted
checkpoint by more than that same numerical tolerance. This is a strict
engineering contrast, not an estimated noise threshold. First eligible
checkpoint initializes each rule. Activity remains mean>2 orders/week in
each half. An inactive rule stops explicitly after its patience beyond minimum.

Each rule freezes its best checkpoint and stop time permanently when its
patience expires. Continued learning for other rules cannot alter it. The
shared fit ends when every rule stops or reaches the original budget. Every
assessment checkpoint is saved. Full TRAIN/INNER_A accounts and early-close
diagnostics run once per distinct chosen checkpoint, with rules referencing
the same result when they select identical weights. No sixfold training cost.

## Test Design Before Code

SC01: original-rule decisions equal the existing Patience class on identical
eligible/ineligible sequences, including minimum and no-eligible outcomes.
SC02: guard rejects a combined-score gain purchased by internal deterioration;
accepts a genuine combined gain with non-decreasing internal score.
SC03: stopped rules never change checkpoint, counters or scores when presented
with future observations; a future suffix cannot affect an earlier selection.
SC04: observation bookkeeping and checkpoint saving do not change training RNG.
SC05: final manifests bind all six rule identities, selected weights, account
metrics and source/data identities; altered/missing artifacts are rejected.
SC06: previous qualification is authenticated; failed seed43 cannot be silently
admitted. Two-worker queue and CPU collector continue without approval prompts.

Chronological blocks remain the same. All2023H2 statistics are development;
upstream feature selection already saw it. Neither2024 nor TEST is touched.
No feature subset is changed during this contrast and no rule is declared
optimal from the two historical seeds. Original-control prefix agreement is
reported, not assumed across new executions.

## Outcome and Next Step

Publish per-rule stopping duration, selected step, matched monitor TRAIN/VAL,
full-period return/Sharpe/DD/activity/side wins/early closes, including the
failed third-seed qualification context. Compare cost saved by shorter patience
separately from changes in chosen weights. Do not select a rule merely because
it reduces the gap by destroying TRAIN performance or becoming inactive.

Use this evidence to freeze the real-model evaluator for the floating
feature-selector in feature_Selector. This bounded run does not automatically
launch that expensive search or claim the feature list is complete.

## Prelaunch Verification

35 focused tests pass; scoped Ruff and whitespace checks pass. The CPU smoke
executed128 financial transitions and29 SAC updates, assessed two checkpoints,
reloaded the chosen checkpoint and ran both complete native accounts with
early-close diagnostics. All18 receipt artifacts verified; maximum cash
discrepancy3.19e-12USD. This is engineering evidence, not candidate performance.
