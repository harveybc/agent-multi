# 338: Paired Twin-Protection Replication

Status: IMPLEMENTED_TESTED_BEFORE_EXECUTION.

337 fixes a measured B43 nonsmooth protection defect:256/256 updates accepted
versus0/256 legacy, stable-added regret0.304704942 ->0.293662736. Internal TRAIN
stop/audit regret worsens. Do not infer generalization or actor/trading gains.

## Frozen Contrast

Use all four334 cost-only1024 endpoints (foldA/B, seeds19/43), same optimizer
moments,256 further scheduled attempts1025..1280 per arm. Eight runs in total.
Both arms use the SAME high-accuracy primal solver; the only treatment is
active-minimum versus both-chosen-twins constraints. This separates the337
combined solver/constraint change. Do not reuse mismatched dual controls.

No data, features, losses, temperature, batch, learning rate, original margins,
2x interior target,3 relinearizations,8 scales or4x norm-cap changes. Freeze each
stable-added mask from that source's unanimous five-view actions, intersected
with the previous label-free stable mask. Never select checkpoints using held
labels. Full original guards and actual weight motion remain mandatory.

## Execution And Verification Contract

- Authenticate parents, source checkpoint and support identity before fitting.
- Restore source model/AdamW/RNG separately for each paired arm.
- Save model/AdamW/RNG every32 attempts, endpoints64 and256. Resume from them.
- Guard each saved state across GPU/CPU/NumPy and batch/single execution.
- Assert optimizer step count equals source accepted count plus new accepts.
- Cold-reload both endpoints and reproduce three transitions after64 exactly.
- Seal each unit and root. Reuse completed units without overwriting evidence.
- Independent NumPy endpoint scoring/old-decision protection and paired support
  checks after completion. Existing analytic twin-crossing and derivative tests
  plus53 focused regressions must pass before launch.

Report every case at fixed256, including unchanged/worse cases. Primary outcome
is stable-added decision regret, alongside acceptance, full-FIT cost/MSE and
internal TRAIN stop/audit diagnostics. The encoder was pretrained on whole
TRAIN; these are not independent end-to-end generalization estimates.

No automatic budget extension,actor update,live promotion,2024/TEST read or
unchanged floating pass. Preserve299/71inputs/28groups and49 cumulative return
candidates;STL remains paused. At closure choose the next diagnostic from
measured attribution, not from loss improvement alone.

Prelaunch:53 focused CPU tests in3.87s, CLI smoke and scoped Ruff pass. The
runtime includes all imported plugin packages, following the337 deployment
correction. Each complete unit is sealed; restart reuses it, while partial
units recover full optimizer/RNG state with exact metric checks.
