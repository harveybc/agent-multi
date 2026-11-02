# Simple SAC optimization before floating selection

Owner authorization: 2026-11-02. Optimize the selected three-signal discrete
SAC, then review its results before launching feature-selection phase 2.
Do not substitute another learner or optimize business rules simultaneously.

## Fixed contract

120 phase-1 columns, 31 groups, 13 observations with stride 2 observed rows.
Fixed TRAIN scaling; no rolling z-score, input clipping or agent-state inputs.
NOP/BUY/SELL; an opposite signal closes an existing position without opening
its replacement in the same decision. Native costs, protections and sizing
remain unchanged. Risk horizon 6 hours; leverage 50, maximum margin fraction
0.1 times balance, single position, weekly opening/closing exclusion 12 hours.
No newly imposed daily entry limit. No weekly model refit in this experiment.

TRAIN is 2020-01-01 through 2023-06-30; monitored development validation is
INNER_A, 2023H2. This is NOT calendar VALIDATION 2024. Inherited feature/risk
design saw 2023H2; results are development evidence, not an untouched holdout.
2024 and TEST remain unopened. Preserve the baseline's three fixed eight-week
TRAIN blocks and three fixed eight-week INNER_A blocks for monitoring.

## Search space

All genes are normalized to [0,1]. Choice decoding is
`choices[min(floor(g * len(choices)), len(choices)-1)]`; log decoding is
`10 ** (lo + g * (hi-lo))`. Continuous decoded values retain 12 significant
digits. Candidate identity includes the decoded configuration, not just genes.

| Effective parameter | Baseline | Search | Encoding |
|---|---:|---|---|
| Branch width, all groups | 8 | 4, 8, 16 | choice |
| Core width | 16 | 16, 32 | choice |
| Branch residual blocks | 2 | 1, 2 | choice |
| Core residual blocks | 2 | 1, 2, 3 | choice |
| Head hidden width | 8 | 4, 8, 16, 32 | choice |
| Learning rate | 0.0003 | 0.00001 to 0.001 | log10 [-5,-3] |
| Gamma per native hour | 0.99 | 0.95 to 0.9995 | linear |
| Target update tau | 0.005 | 0.001 to 0.02 | log10 |
| Entropy fraction of log(3) | 0.6 | 0.3 to 0.9 | linear |
| AdamW weight decay | 0.01 | 0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2 | choice |
| Batch size | 128 | 32, 64, 128, 256 | choice |
| Update every decisions | 4 | 2, 4, 8 | choice |

Actor and both critics use the same candidate architecture but independent
parameters. Each residual block has TWO causal convolutions, dilation 2**i.
Temporal groups use kernel 3; pointwise groups retain kernel 1. Fusion uses
kernel 1. No dropout/BatchNorm is added. Historical recipe dropout 0.3 and
legacy representation widths are NOT effective parameters of this SAC.

## Evaluation and stopping

Population 20; maximum 2000 generations; deterministic DEAP random seed.
Initial population includes the unchanged baseline AND an otherwise identical
batch-32 candidate (evaluated first), following the owner's correction. Two paired training seeds
(7 and 19) per candidate. New weights and replay for every candidate/seed.
Keep the existing synthetic learning qualification and a fresh financial fit.
Maximum financial decisions 1,048,576, minimum 131,072; checks every 32,768;
warmup 4096 and replay capacity 65,536. No shortened proxy training.

Let P be mean net percentage return per week across the fixed blocks.
At a check, M = (P_TRAIN + P_INNER_A)/2 on the SAME current weights.
Validation performs no gradient updates and never enters replay. Patience 10
and min_delta 1e-5 apply to M; restore the best eligible checkpoint. Eligibility
requires more than two orders/week on average in each monitored partition.
Store unrestricted scores too; no eligible checkpoint is an inadmissible
candidate, never a fabricated zero-return account.

DEAP fitness is mean selected M across the two seeds. Its separate generation
counter also watches improvement of that combined fitness, patience 10,
max 2000 generations. Both counters and their stop causes are logged separately.
Never replace M with TRAIN-only performance or an extra validation-veto rule.
Report TRAIN and INNER_A separately, their gap, activity, directions, win rates,
Sharpe, drawdown, costs and close reasons. Repeated INNER_A selection is explicit.

## Implementation and acceptance plan

| Requirement | Evidence required before launch |
|---|---|
| No inert genes | Every gene endpoint changes its decoded field; inspect live network dimensions, blocks, optimizer, TD discount, tau, entropy and replay/update settings |
| Baseline unchanged | Default network outputs and state keys equal the legacy construction under the same seed |
| Correct patience | Improvement/reset, degradation/increment, best-M restoration, explicit independent counters |
| Causal TCN | Future input perturbation leaves earlier encoded timesteps unchanged |
| Wavelet support | Recompute selected wavelet values from pinned source and check future perturbation and positive controls |
| Reproducibility | Effective config, source/data hashes, paired seeds, per-candidate receipts; reload equality |
| Automation | Exclusive coordinator lock, durable generation state, worker isolation, resumable completed evaluations, explicit failed task records |
| Phase boundary | No automatic full floating selection; notify owner with optimizer results first |

Only then launch the persistent queue. Runtime and ETA are measured from real
candidate fits; 2000 generations is a cap, not a promised completion time.
