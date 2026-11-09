# Critic Refinement Policy and Preserved-Policy Probe

Owner direction2026-11-09: retain finding229 in the findings/progress register
and apply head reinitialization systematically at NEW transferred SAC refinement
stages. This is not a reset on inference, frozen feature evaluation, each epoch,
or checkpoint recovery. Fresh-from-scratch fits already initialize their heads.
The rule concerns online twin CRITIC heads, never categorical actor heads.

## Required Sequence

Authenticate source and reproduce original accounts; preserve epoch0 fallback.
Reset online critic heads and optimizer memory once per new refinement stage.
Freeze actor, alpha and encoders. Calibrate against realized soft returns under
that SAME preserved policy on TRAIN only. Restore native bootstrap afterwards
and synchronize target critics to calibrated online critics before new learning.
No validation returns are training labels. Keep per-action calibration and input
support fixed; inactive checkpoints cannot replace the active reference.

The measured benefit in229 is a bounded calibration intervention, not a proof
that reset universally improves trading or prevents recurring bootstrap drift.
The policy is adopted for new transferred refinements; promotion still depends
on native account results, not on a lower critic loss alone. Other algorithms
and phase1 proxy selectors do not have this SAC head lifecycle.

## Acceptance Tests Before Execution

| ID | Requirement | Test |
|---|---|---|
| R01 | Reset only on a new stage | inference/resume/frozen selection no-op; duplicate new-stage invocation rejected |
| R02 | Preserve actor/encoder/alpha | effective tensors equal; only online head weights change |
| R03 | Own-policy labels | archived epoch0 checkpoint hash, source receipt, timestamp/reward/duration replay checks |
| R04 | Calibrate before actor learning | only critic heads train during4096 anchor updates; native target restored even on exception |
| R05 | Correct delayed critic | explicit online-to-target synchronization; exact equality immediately afterwards |
| R06 | Bounded actor sensitivity |1/8/32 updates,critic fixed,full TRAIN/VALIDATION accounts and preserved actor control |
| R07 | Bootstrap recurrence | separate256 critic-only native-target updates from calibrated copy; fixed actor |
| R08 | No false promotion | preserve69/70; diagnostic candidates never replace source or publish paper model |
| R09 | Durable automation | sequential arms,exclusive lock,terminal hashes,verified completed-stage reuse |

## Probe

Use preserved70 epoch0 because its on-policy TRAIN traces already exist and are
authenticated. Do not reuse epoch40-policy labels. Compare retained-head and
fresh-head calibration with identical empty optimizer state,4096 updates,
batch128 and all24025 transitions. Afterwards compare bounded actor updates
from each calibrated critic; freeze critics while testing actor changes.
Separately run a256-update bootstrap diagnostic from each calibrated checkpoint.
No data/window/strategy/feature changes or state feedback. No TEST access.

Implementation: `app/critic_refinement.py` provides the lifecycle and calibration
primitives. New `run_selected_warm_start.py prepare` specs include this policy;
the worker performs own-policy TRAIN calibration before `fit_epochs`. Completed
calibration receipts reload instead of resetting; RESUME.json defers directly
to the existing authenticated recovery. No modification to inference/load.
41 focused CPU tests pass before launch (one existing PyTorch performance warning).

Only the fresh-head arm is the newly adopted refinement policy; retained-head
is the explicitly named diagnostic control. This bounded probe precedes long
refinement. No phase2 cumulative readmission is claimed executed by this work.
