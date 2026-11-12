# STL Availability Subset

2026-11-12. Owner authorizes removing the features blocking the STL experiment.
This is a separately identified lateral experiment, not removal from the
preserved selected71 actor, the live provider or the phase2 return universe.

## Frozen Decision

Use the completed original71 causal STL columns, without splicing the separate
source-repair candidates. Remove any column that loses a complete reference
TRAIN window after STL. Remove `act.p_hit` because its downstream fitted
context audit failed. Sine/cosine and probability-simplex blocks remain atomic.
Preserve column order. No trading metric or validation value chooses columns.

The joint support must exactly preserve the original raw reference in both
TRAIN and validation, not merely have observations in every month. Validation
does not choose the subset; it must pass this frozen support gate before launch.
Retain all calendar rows in the simulator, all four TRAIN years, the
validation year,13 observations and stride2. Invalid observations remain
unavailable; never fill, interpolate or silently treat them as learned NOP.

## Acceptance Before Launch

| Requirement | Evidence |
| --- | --- |
| Deterministic availability selection | Unit fixtures for partial loss, empty months, atomic blocks and forbidden activity |
| No validation-driven pruning | Changing validation masks cannot change selected columns |
| Same data in every comparison | Identical support hash, timestamps and column order in all four arms |
| Causal outputs | Authenticated original column receipts; endpoint prefix and resumed online parity |
| Structural removal | Actual subset recipes and group remapping, not zeroed channels |
| Training isolation | Same fresh seed, architecture, optimizer, risk and early-stop policy |
| Durable execution | Exclusive queue/worker locks, resume checkpoints, finite sequential queues |
| Source preservation | Original71 bundle, stores and receipts are read-only |

## Runs

Raw31 is the matched control. The primary GPU runs continuous seasonal then
ordinal seasonal. The secondary GPU runs raw then binary seasonal. No promotion
or claim of improvement is made before the matched results exist.

Retain the established full-epoch training contract: maximum2000 epochs,
patience20, minimum improvement1e-8, validation return per week, activity above
two trades/week, and20-epoch patience warmup while allowing early eligible
checkpoints. Seed7, fixed TRAIN scaling, fresh full-network training, no agent
state, no mobile z-score. Save checkpoints, recovery state and training plots.

This answers seasonal filtering on the available subset, not whether any
removed feature is predictively useless. Future source repairs may reintroduce
those inputs through separately identified paired experiments.

## Preparation Finding

The first53-column preparation used the old nonempty-month criterion. It left
only4 eligible January2020 windows. No training was launched from it. The
strict criterion was fixed before any training outcome and retains31 columns,
with24026 TRAIN and6208 validation windows, exactly the source support. All40
excluded columns and their TRAIN missing-window counts are recorded. The53
preparation remains historical evidence and is superseded, not silently edited.

## Executed Admission And Launch

Preparation identity:
`23a1826fbd6cbc1300cf22f5faf128901b6670350537d48b6042abbaea7c1af0`.
Source support and all four variants have the same digest:
`28d6088a92173592f39e221c1c2ab1efc9094e614eeccb11c9ae23cf14878c98`.
There are31 inputs in11 executable extractor groups. Tensor shape30859x13x31;
24026 valid TRAIN windows and6208 valid validation windows. Reference validity
is held fixed rather than admitting additional rows after removing columns.
All48 TRAIN months and12 validation months remain represented. This preserves
original support, not a claim that the source itself had no missing observations.

Five full-vector trailing-prefix/restarted-online probes reproduce every
selected seasonal output and every encoding exactly. Original per-column
tuning and receipts were authenticated; their input arrays were not changed,
so no repair-dependent retuning was necessary. No supervised activity or GMM
probability branch remains in this lateral model. Raw31 and the three seasonal
arms use the same remapped architecture, seed7, batch128, learning rate3e-5,
replay65536, update every4 decisions and4096 initial exploratory decisions.

Owner-local enabled finite services:

| Queue | Device | Sequential arms | State at publication |
| --- | --- | --- | --- |
| `am-stl31-primary-20261112.service` | RTX5090 |continuous -> ordinal |Continuous epoch1, SAC updates verified |
| `am-stl31-secondary-20261112.service` | RTX4090 |raw -> binary |Raw epoch1, SAC updates verified |

The secondary worker initially failed loading a cuDNN sublibrary during the
epoch-zero export, before training. Its private process environment now points
to the existing cuDNN/CUDA library directories. A CUDA forward/backward smoke
test passed, then the worker exported/reloaded and executed real SAC updates.
No shared package installation, model parameter or scientific input changed.
The initial failure receipt is retained; it is not a training result.

Both queues save recovery checkpoints and continue to their second arm without
an intermediate approval. Unexpected process failures can restart; explicit
Python errors stop with a failure receipt instead of silently skipping an arm.
No model is automatically promoted. The main selected71 actor remains intact.

Evidence: [PREPARATION.json](generalization/stl31_20261112/PREPARATION.json)
contains the full selected list, every removal reason and monthly coverage.
33 focused CPU tests pass, including the partial-coverage regression, paired
support refusal, causal STL, encoding, structural model reload and queue order.
Training returns are not yet available at publication.

Independent cross-device initialization check: authenticated epoch-zero bundles
from both workers cold-load on CPU with466/466 state tensors exactly identical.
Thus raw versus continuous does not start with a different random parameter
draw. Their recipe/normalization identities deliberately differ. This does not
claim identical future floating-point training trajectories across GPU models.
