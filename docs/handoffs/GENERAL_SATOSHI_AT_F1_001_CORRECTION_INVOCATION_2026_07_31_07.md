# General Satoshi AT-F1-001 Correction Invocation 07

General Satoshi,

Your arithmetic was exact. Your unit conclusion was not. Reconstructing the
stored scalar proves implementation fidelity; it does not prove conformance
to the task's weekly-fraction contract.

## Mandatory Inputs

Read these files in order:

1. `docs/audits/AUDIT_GS_COUNTER_RESPONSE_AND_AT_F1_001_2026_07_31.md`
2. `docs/audits/CODEX_CROSS_REVIEW_OF_SATOSHI_AT_F1_001_2026_07_31.md`
3. `docs/audits/work_plan/01_AUDIT_BACKLOG_AND_SCHEDULE.md`, section 4.1
4. `pipeline_plugins/rl_pipeline_with_validation.py`, lines 226-270 and
   1197-1220
5. `docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md`
6. `docs/handoffs/CLAUDE_AUDIT_RECOVERY_PROMPT_2026_07_30.md`

Cross-review SHA-256:
`1f96988092abb331112f1e3bd2bfb086b99fe908549038f6174316ee5bc812c1`

## Required Work

1. Reproduce block-8 split evidence. Report `evaluation_weeks`,
   `risk_adjusted_total_return`, `mean_weekly_rap` and `trades_total` for both
   train tail and validation.
2. Reconstruct both scalars:
   - the configured full-period L2;
   - the task-specified mean-weekly L2.
3. Withdraw the unconditional `AT-F1-001 PASS`. Keep the valid lifecycle
   state `reported` with an open finding until Harvey selects the authoritative
   objective.
4. Publish a dated addendum for the impossible `04:40` chronology. Preserve
   the original report unchanged and separate evidence time from write time.
5. Confirm that findings 025-028 and the AT-F1-001 state are present in the
   source-of-record files. Update your recovery prompt's CURRENT STATE and
   name the next trigger.
6. State explicitly that no mid-chain fitness mutation is authorized.

## Output Boundary

Produce one bounded correction report and the required recovery-state update.
Do not create deadlines, escalation rules, runtime changes, new work packets
or additional findings unless new reproduced evidence requires one. Do not
repeat the already verified test suites or GitHub runs.

The acceptance criterion is not deference. It is a unit-correct result and a
consistent persistent handoff.
