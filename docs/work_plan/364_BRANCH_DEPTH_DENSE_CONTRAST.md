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
