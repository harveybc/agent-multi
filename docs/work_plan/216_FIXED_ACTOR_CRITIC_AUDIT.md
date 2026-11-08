# Fixed Actor Critic Audit

Accepted 2026-11-08 after [215](215_FROZEN_POLICY_RETURN_CALIBRATION_RESULTS.md).
Investigate before refinement and another deletion/addition feature pass.

## Frozen Protocol

Keep the selected epoch0 actor, temperature, inputs, encoders and account contract
unchanged. Reuse the authenticated 4096-transition TRAIN replay from diagnostic
213. This small, fixed replay is a mechanism test, not full training and not a
replacement for the four-year dataset. No validation-based choice or TEST read.

Five paired arms: no update, production critic-head update, core dropout disabled,
weight decay disabled, and target network fixed. Only the named factor changes
from production. Use 1024 updates, batch128, common seeded indices, fresh copies
of the same checkpoint and optimizer. Report at 0/1/64/256/1024 updates. Do not
change production code or select a new trading model from this test.

## Requirements and Tests

| Requirement | Evidence before interpretation |
|---|---|
| Correct Bellman target | independent probability-weighted twin minimum, entropy, duration and terminal formula |
| Terminal stops bootstrap | synthetic terminal rows with nonzero next Q |
| Correct critic loss | independently reconstructed loss from captured online Q and target |
| No actor-gradient leakage | actor loss gives no critic gradient; Q loss gives no actor gradient |
| Correct Polyak update | target after update equals (1-tau) old target + tau new online weights |
| Explicit mode changes | target and actor evaluations deterministic; dropout only online critic core |
| Single-factor contrasts | scope assertions for trainability, dropout and optimizer groups |
| No actor change | exact actor parameter and probe-logit equality throughout |
| Real observations | checkpoint, replay and calibration files authenticated by digest |
| No misleading proxy | label replay TD and fixed-policy Monte Carlo diagnostics separately |

Read-only snapshot inspection additionally compares epochs0/6/40 on the same
replay, including online train/eval Q, target Q and the train/eval gap. The
TRAIN-only frozen-policy sampled return trace provides a fixed diagnostic panel
of512 evenly spaced decisions over four years. It is not an independent fit set
or an IID statistical sample. Report bias and RMSE, no significance claim.

If an ablation isolates a problem, verify it in a second controlled test before
changing the learning recipe. Never silently remove dropout, change costs, add
agent-state observations or overwrite the selected checkpoint.

## Follow-up Arithmetic Inspection

After observing the snapshot twin disagreement, add a separately identified,
read-only decomposition of the TD residual into reward, entropy bonus, mean-twin
backup minus current online Q, and the penalty from taking the minimum twin.
This is an exploratory explanation of the existing snapshots, not an additional
optimization arm or proof that replacing the twin minimum is appropriate.
Its reconstruction must exactly match the native residual. The already running
five-arm export remains pinned and unchanged.

Predeclare a second, exploratory two-arm test starting from epoch40, where the
value drift already exists: production twin-minimum versus mean-twin bootstrap,
1024 identical TRAIN replay updates, actor40 and alpha frozen. Restore the same
epoch40 optimizer state in both arms. The return reference is now the frozen
epoch40 TRAIN sampling trace. No promotion: removing clipped-double-Q changes
the estimator and can introduce overestimation. This is a causal intervention
on the suspected pessimism feedback, not an endorsed SAC replacement. Current
actor logits must remain identical in both arms.
