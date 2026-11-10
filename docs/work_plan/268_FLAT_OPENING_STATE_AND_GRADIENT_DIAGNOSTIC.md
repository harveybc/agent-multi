# 268. Flat opening state and actor gradient diagnostic

Owner approved2026-11-10 following267. TRAIN only, zero optimizer steps.
Preserve all69 inputs, checkpoints, scaler, strategy and complete continuations.

## Frozen panel and controls

Use every distinct TRAIN timestamp in267 whose preserved greedy account is
flat and whose actor changes in any of the nine endpoints. Retain harmful,
helpful and neutral occurrences; do not select only errors. For each timestamp
add the nearest unused unchanged flat timestamp in the same calendar year and
with the same preserved action (earlier timestamp breaks a distance tie).
Use the archived position/action trace, never future profit, for matching.
All candidates must retain the old action at an unchanged control.

At each timestamp reconstruct TWO real native account histories: preserved
greedy and the seed7 stochastic trajectory used for critic calibration. No
synthetic position injection or fabricated balances. Save full state views,
including position terms, balance and entry history. Fork each state and force
each of NOP/BUY/SELL, then continue with the same three stochastic policy tapes
7/19/43. Reuse archived greedy-state suffixes only after numerical equality is
checked; retain native first-command effects and soft/reward/cash quantities.
This diagnoses state-distribution mismatch, not the isolated causal effect of
position: balances, prior fills and other state fields can differ together.

## Gradient diagnostic

On the full original TRAIN decision support, analytically differentiate the
exact categorical SAC expectation at the preserved deterministic mean head.
Separate Q and entropy logit gradients, and project both the local gradient
and the full-TRAIN shared weight/bias gradient onto each old/new boundary.
Verify formulas against torch autograd and finite differences in unit tests.
Compare directions with the saved endpoint displacement, not as a claim to
replay its256 AdamW steps. This first-order diagnostic excludes posterior
sampling, optimizer memory/preconditioning and subsequent gradients; the
mean-prior KL gradient is zero at initialization. Report those limitations.

## Requirements and tests before execution

| ID | Requirement | Test/evidence |
|---|---|---|
| FO1 | Deterministic full changed panel and matched controls | Permutation invariance, no outcome-based matching, no reused controls |
| FO2 | Same market, authentic distinct account states | Both full traces replay; native account checks; source hashes |
| FO3 | Same targets as Q | Three common-random-number soft suffixes per action/state; old267 parity |
| FO4 | Gradient signs/units correct | Autograd, finite difference, zero-gradient and shared/local fixtures |
| FO5 | Restart and distributed identity | Frozen panel/config digest, disjoint timestamp shards, per-cell receipts |
| FO6 | No hidden fitting | TRAIN-only panel, no optimizer, source hash unchanged, no promotion |

CPU-native suffix simulation dominates. Split the same frozen timestamp panel
into three disjoint deterministic shards; saved GPU surfaces eliminate the need
to recompute networks. Automatic collection authenticates all shards and writes
the report. No validation threshold fitting or new actor training is authorized
by a diagnostic result alone. Next correction must address a reproduced
mechanism and include unchanged/helpful controls, not only the failure examples.

## Follow-up decomposition, defined after native panel completion

The panel revealed many calendar-forbidden flat states. Before any new fit,
decompose the FULL original TRAIN gradient by its actual calibration-history
states: holding, flat/entry-allowed, flat/entry-forbidden. Each contribution
keeps denominator N (not its class size), so their sum must equal the original
gradient. Test this identity with deliberately unequal stratum sizes. Replay
the complete calibration history; do not infer legality from signals alone.
No rows are removed and this is not a training ablation. Report whether the
remaining projected direction still favors harmful changes without the
forbidden-flat contribution, without claiming that it predicts retraining.
