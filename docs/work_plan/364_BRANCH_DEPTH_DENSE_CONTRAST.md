# 364 - Paired branch depth and dense pilot

## Predeclared scope

Owner approved A and D ahead of the running362 continuations. Preserve their
complete recoveries before suspension; never delete or replace the selected299
reference. No live deployment, STL, new features or DEAP in this experiment.

| Arm | Every group branch | Core | Attention |
| --- | --- | --- | --- |
| control | original two residual blocks | original two blocks | unchanged |
| A | one residual block | original two blocks | unchanged |
| D | Dense8 + SiLU independently at each timestamp | original two blocks | unchanged |

A residual block contains two convolutions. D is implemented as kernel1 Conv1d
plus SiLU, mathematically a shared Dense over channels; it does not collapse the
13 timestamps. A reduces the temporal branch receptive field from13 to5; D
has no branch temporal mixing. The downstream attention and flattened head
retain access to all13 timestamps. Pointwise branches are reduced too.

## Paired contract

- Same selected299 source, all71 inputs/28 groups, TRAIN2020-2023, VAL2024,
  fixed TRAIN scaling, window13/stride2, engine, costs and trading rules.
- Copy every retained actor, twin critic and target tensor exactly by name and
  shape. Copy retained Adam moments by parameter name, not positional indices.
  Refuse missing or mismatched tensors. No newly randomized trainable weights.
- All three arms use fresh replay, full online trainability and seed7. Fixed
  LR3e-6 for actor/critics/temperature in all arms; no scheduler in this pilot.
  Batch128, warmup4096, update every4 steps, replay65536 as in the parent.
- Full TRAIN traversal per epoch, full deterministic TRAIN/VAL evaluation once
  per epoch, validation return/week monitor, threshold1e-8, patience20,
  monitoring warmup20, minimum100 epochs, maximum2000. Preserve eligible early
  checkpoints as well as the best after warmup. An inactive epoch0 may begin
  learning but cannot become a selected checkpoint (>2 orders/week required
  in both periods). This is not authorization to change action thresholds.
- A on the fastest GPU, D on the intermediate GPU, matched control on the
  third GPU. Different hardware and one seed limit attribution; a measured
  advantage is a pilot result requiring paired replication, not promotion.
- Save initial/final cold-load probes, per-epoch weights, plots, native account
  metrics, full recovery, effective architecture and optimizer transfer receipt.
- Resume each paused362 queue after its364 arm has produced an authenticated
  terminal result. Failures retry from recovery, never silently skip an arm.

## Test Design / Traceability

| Requirement | Test / evidence |
| --- | --- |
| Existing defaults unchanged | legacy encoder tensor/output parity regression |
| D really is Dense per timestamp | direct F.linear equivalence and future perturbation |
| A/D are consumed | residual counts, resolved architecture, parameter counts |
| No lost state in retained components | exact tensor and named Adam moment comparison |
| Native learning works | finite SAC update, cold reload and next-update equality |
| Inactive initialization is not promoted | explicit opt-in epoch0 activity regression |
| Same dataset | sealed input files and shared support digest check before learning |
| Safe preemption | verified full recovery seals, paused services and restart receipts |
| Honest comparison | control epoch0 native parity and all-arm full account curves |

Implementation and deployment receipts will be appended after verification.
No winner, convergence result or speedup is asserted by this design.

## CPU Qualification (2026-11-16)

54 focused tests pass, including legacy parity, Dense equivalence, named Adam
transfer, finite updates, exact next-update cold reload, inactive epoch0 and
full recovery regressions. Touched Python files pass the repository Ruff rules.

Real selected299 source construction, not theoretical estimates:

| Arm | Actor parameters | All trainable parameters |
| --- | ---: | ---: |
| control | 30,733 | 92,200 |
| A | 20,301 | 60,904 |
| D | 9,869 | 29,608 |

Every retained tensor and moment matches in all three real-source transfers.
The362 full recoveries were sealed at epochs77/96/48 for seeds7/19/43 before
their services were stopped. Their original run directories remain intact.
The363 followup observer is paused with362, not discarded.

## Deployed / First Learning Verified

Immutable runtime `7002f85649f677755df7b7bdc8c18ce538da9fe9`, archive SHA256
`48a393473b76f59b33d710d0482cc4ba079482ffc509b2d84e89c07e84a68ba9`.
All three input/support identities and cold model/optimizer loads passed.
All three services are active, zero restarts; nonzero SAC updates observed.
The automated queue authenticates terminal result files before resuming362;
read-only collection every180s also restores the362/363 observers afterward.

These are **initial epoch0 accounts before new learning**, not experiment winners:

| Arm / GPU | TRAIN return | VAL return | TRAIN / VAL trades |
| --- | ---: | ---: | ---: |
| control / RTX4070 | +16.058224% | +18.369946% | 745 / 207 |
| A / RTX5090 | -24.317470% | +0.264061% | 1664 / 329 |
| D / RTX4090 | -43.175657% | -3.361762% | 2035 / 410 |

Control exactly reproduces the historical reference. Reduced architectures do
not preserve the source function despite exact surviving weights; their initial
account changes are measured explicitly. All three satisfy initial activity.
Training curves, absolute selected performance and improvement relative to each
own epoch0 must be reported separately. A favorable relative recovery alone
does not establish superiority over control. CPU tests and launch are complete;
the training comparison is running, not complete.
