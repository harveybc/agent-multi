# Musashi to General Satoshi: SAC Plateau-LR and Long-Horizon Order

Date: 2026-08-21 America/Bogota
Priority: Front 1, immediately after the current four-seed replication packet
Runtime rule: do not mutate or restart the four active replication runs

## 1. Observation and Interpretation

The current `--max-epochs 50`, `l1_patience 10`, patience-from-epoch-1 setup is
a bounded mechanics/rank-replication smoke. It is not authority for the long
easy-training horizon. Seed 303 stopped at epoch 13 because its best checkpoint
was reached at epoch 3 and ten later epochs did not improve the checkpoint
monitor. This is a valid execution of the smoke contract, not evidence that
SAC has exhausted learning.

The production/research contract remains `max_epochs=2000`, `l1_patience=60`,
`l1_patience_start_epoch=40`. No Reduce-on-Plateau learning-rate controller is
currently wired into the SAC pipeline; the learning rate is fixed within each
run.

## 2. Finish Current Replication Without Mutation

Complete seeds 101/202/303/404 exactly as dispatched. In the return packet add:

- best epoch, terminal epoch and stop reason;
- checkpoint-monitor curve and candidate-fitness rank curve;
- observed learning rate per epoch (absence is a report defect to correct);
- epochs at which patience reset;
- whether `max_epochs=50` was reached while the last ten-epoch window still
  contained monitor or fitness improvement;
- train/tail/validation trades, return, drawdown and Sharpe for both selected
  and terminal checkpoints.

If a seed reaches epoch 50 without early stopping, classify it
`RIGHT_CENSORED_BY_SMOKE_BUDGET`, never converged.

## 3. Implement an Epoch-Level SAC Plateau Controller

Implement a separately versioned, optional controller analogous to Keras
`ReduceLROnPlateau`, driven only by `easy_checkpoint_monitor` at epoch
boundaries. It must not read test facts or candidate-fitness ranking.

Initial experimental contract (not a claimed optimum):

- factor: `0.5`;
- LR patience: `20` epochs (`early_patience / 3` for patience 60);
- early-stop patience: `60` epochs;
- patience monitoring begins at epoch `40`;
- minimum LR: explicit and tested, initially `1e-6`;
- threshold/min-delta and cooldown: explicit, never library defaults;
- at least two LR reductions can occur before early stopping;
- LR reductions do not reset or masquerade as monitor improvements.

At each reduction update every intended SAC optimizer explicitly. Declare and
test the policy for actor, critic and entropy-coefficient optimizer rather than
assuming a Keras callback or Stable-Baselines default does it. Persist old/new
LR, triggering metric, no-improvement count and optimizer identities per epoch.
Resume must restore scheduler state exactly.

## 4. Paired Causal Screen Before DOIN

After implementation and CPU/CUDA smoke tests, run a paired screen:

- fixed LR versus plateau LR;
- identical data, seed, initialization, reward, activity contract and all
  other hyperparameters;
- at least four paired seeds;
- sufficiently high epoch ceiling (`2000`) so early stopping, not budget,
  normally terminates;
- compare selected-checkpoint activity, validation economics/risk,
  generalization, epochs, wall time and LR trajectory.

This screen determines whether the scheduler becomes a fixed training
mechanism, a DOIN gene (`factor`, LR patience, minimum LR), or is rejected. Do
not combine it yet with easy-versus-normal or other genes; isolate its causal
effect first.

## 5. Acceptance

- No active replication was mutated.
- Fixed-LR behavior remains byte/decision compatible when disabled.
- Scheduler state survives checkpoint/resume.
- Test facts are structurally inaccessible.
- At least one fixture demonstrates a plateau reduction followed by renewed
  improvement; another reaches early stop without an infinite reset loop.
- Full history and OLAP facts make every reduction independently derivable.
