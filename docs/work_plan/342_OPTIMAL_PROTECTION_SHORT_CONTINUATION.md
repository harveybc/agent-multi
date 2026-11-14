# 342: Matched Short Continuation After Optimal-Set Feasibility

Status: IMPLEMENTED_TESTED_BEFORE_EXECUTION.56 focused tests and scoped Ruff pass.

341's ten repaired proposals lower loss without changing any executed FIT
action. That demonstrates nonzero safe motion, not improved decisions. After
its independent verifier passes, run64 attempts for each of the four334
source checkpoints, with the SAME batch schedule, LR, AdamW state, source
weights and original guards as338. Reuse its authenticated complete-arm64
controls; do not retrain controls. No long fit, actor update or floating pass.

The sole new treatment is341's per-step expanded FIT optimal-set protection
and finite lower-cost/nonincreasing-regret gate. Recompute eligible sets from
the current state each step, so newly corrected predictions become protected
while wrong predictions remain free. Tied native target optima remain sets.

## Tests And Acceptance

- Require341 producer AND independent verifier gates; source/probe/coverage/
  control identities must agree. No continuation if either gate fails.
- Source replay, batch identity and saved baseline/control metrics must match.
- Preserve complete per-step records,16/32/48/64 checkpoints and optimizer count.
  Every accepted step must pass original and expanded finite guards. Rejected
  steps restore weights and optimizer via the tested transaction helper.
- Cold-reload all four endpoints; replay17/18 from16 exactly. All source
  seals and actor hashes remain unchanged. Four case units are resumable at
  completion; a partial failed case restarts from its authenticated source.
- Report full FIT loss, greedy regret, action changes, acceptance, protected
  counts and internal TRAIN stop/audit metrics for source/control/treatment.
  Held metrics are descriptive only and never choose an update or checkpoint.
- No automatic model promotion. Even four FIT gains would not establish
  end-to-end transfer or account profit. This finishes at64 attempts; do not
  silently extend the budget or retune thresholds.
