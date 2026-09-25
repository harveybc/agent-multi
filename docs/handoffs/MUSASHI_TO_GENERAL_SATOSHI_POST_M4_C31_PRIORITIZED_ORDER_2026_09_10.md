# Musashi to General Satoshi: post-M4-C31 prioritized order

Date: 2026-09-10

Authority input:
`docs/audits/MUSASHI_AUDIT_M4_C31A_C31F_AND_EXTERNAL_CAMPAIGN_STATUS_2026_09_10.md`.

Execute the work packages below in order. Preserve separate worktrees and do
not rewrite any pushed history.

## P0 - Execute B4 C49-C56 now

The governing order remains:
`docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_B4_C49_C56_RUNTIME_STALL_RECOVERY_ORDER_2026_09_09.md`.

Additional correction to the return record:

1. Do not call B4 v7 completed because `systemd` reports `Result=success`.
2. Freeze the physical inventory: two completed and sealed cells, one
   `AMBIGUOUS_CLAIM`, nine pending, owner stop markers present.
3. Append a correction in the next packet; do not edit the pushed M4 packet.
4. Implement and test the external per-child watchdog, bounded escalation,
   reap, liveness and failed-time accounting exactly as C49-C56 orders.
5. Prepare but do not launch the recovery generation. CPU-only mechanics;
   no GPU and no B4 service.

Required disposition remains:

`B4_V7_QUARANTINED_AND_EXTERNAL_WATCHDOG_RECOVERY_READY_FOR_MUSASHI_REVIEW`

## P1 - Execute T2 C89-C94

After the B4 P0 return, execute:
`docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_T2_C89_C94_COMPLETION_AND_ADJUDICATION_ORDER_2026_09_09.md`.

Reconstruct all 242 completed units from physical arrays and records, rerun the
reviewed adjudicator, publish all family-level effects and stop before T4/T5.
Service success alone is not scientific acceptance.

Required disposition:

`T2_RESOURCE_SUCCESSOR_EXECUTION_AND_SCREEN_ADJUDICATION_READY_FOR_MUSASHI_REVIEW`

## P2 - M4 C32-C38 confirmation preparation

This package may begin only after the P0 and P1 returns are committed and
pushed. It authorizes CPU-only design, implementation, tests and DEVELOPMENT
mechanics. It does not authorize generating, loading or scoring CONFIRMATION
arrays.

### C32 - Bind the accepted calibration evidence

Consume the audit by exact bytes and bind:

- reviewed tip `5e7a8fd430c8231a049baf03f00e720ba24ec994`;
- design self-identity `d7280a92047d98898418fb7cd750b22c506a621eb381d9847e0fe926b7df69b9`;
- numeric amendment self-identity
  `43e0804e1e6e583b10ddbe46b7d4cd752838b0473ccbc6496f0e458c49aedd4b`;
- governing adjudication self-identity
  `b35b6fd969aa162047bdfb55b8f9fcce01aa76864c388d29a1c36642ab051ade`;
- exact re-derived facts: 21/28 eligible slots, two incomplete generators,
  zero calibration-incomplete cells, and M2 gain `-0.41982887`.

Any altered attempt, adjudication, rule or identity must refuse before a
confirmation ledger can exist.

### C33 - Freeze the calibration-derived policy honestly

Create an append-only confirmation successor that labels the selection rule
`CALIBRATION_DERIVED_AND_REVIEWED`, never predeclared. Freeze:

- eligible iff at least 12 of 16 CALIBRATION generators are
  `LEARNABLE_UNDER_FROZEN_BUDGET` and none is `NUMERICALLY_INVALID`;
- the exact 21 eligible family/noise/width slots and seven typed ineligible
  slots already present in the accepted adjudication;
- 48 CONFIRMATION generators per eligible slot;
- the existing 20-percent attrition allowance and minimum complete count;
- M2 as `DOES_NOT_ADVANCE_FROM_CALIBRATION`.

The successor must describe this as a calibration decision made before any
CONFIRMATION data. It is a scientific analysis freeze, not `scientific_change:
NONE`.

### C34 - Make the 16-contrast family executable

Materialize the exact confirmatory estimands and multiplicity behavior:

1. For each of the 14 family/noise intervention contrasts, compute the
   generator-level paired effect and average equally across widths that were
   frozen eligible by C33.
2. If exactly one width is eligible, use it and name that fact.
3. If no width is eligible, emit `NOT_EVALUABLE` with a non-rejecting `p=1`.
4. Publish width-specific effects and attrition as secondary heterogeneity
   results; never treat them as extra primary hypotheses.
5. Keep `checkpoint_effect::primary_pair` as the fifteenth contrast.
6. Keep `incremental_prediction::M2_vs_M1` as the sixteenth, non-rejecting
   `p=1` placeholder because M2 failed CALIBRATION.
7. Apply the frozen Holm procedure over all 16 slots, including placeholders.

The generator is the independent unit. Seeds and widths are nested or paired
repetitions, never independent observations.

### C35 - Build the confirmation runner and verifier

Implement a separate runner that is structurally unable to consume DEVELOPMENT
or CALIBRATION outcomes as confirmation observations. Before any execution it
must:

- materialize the exact generator/cell/seed/checkpoint census and update count;
- prove CONFIRMATION generator bytes are disjoint from prior roles;
- bind the accepted design, amendment, adjudication, C33 successor and code;
- create a complete pre-result ledger;
- enforce the existing CPU wall, RSS, nice, heartbeat and stop-file limits;
- preserve every incomplete or numerical state in the denominator.

The independent verifier must reconstruct generators, tapes, checkpoints,
restricted endpoints, paired effects, attrition, costs and all 16 contrasts
from raw records. Producer aggregates may never determine a verdict.

### C36 - External authority boundary

Add strict templates and consuming APIs for two separate external records:

1. a Musashi design-review record pinning the successor and executable analysis;
2. an owner execution record pinning the reviewed population and CPU limits.

Templates grant nothing. Candidate code may neither create nor install real
records. Without both records, planning may report counts but execution must
refuse before generating a CONFIRMATION array or ledger.

### C37 - Acceptance battery

At minimum freeze tests for:

1. substituted attempt-2 adjudication;
2. changed 12/16 threshold;
3. one changed eligible slot;
4. calling the policy predeclared;
5. treating two widths as independent hypotheses;
6. silently choosing the better width from CONFIRMATION;
7. dropping an ineligible contrast from Holm;
8. fitting or scoring M2 on CONFIRMATION;
9. missing one of three nested seeds;
10. attrition beyond the frozen allowance;
11. overlap between CALIBRATION and CONFIRMATION bytes;
12. producer aggregate differing from raw records;
13. absent, forged or transplanted external records;
14. any scientific change in the sealed successor after review;
15. a typed numerical failure disappearing from the denominator.

Include guard-removal mutation tests. Run a small DEVELOPMENT-only mechanics
probe through the real process boundary; prove that it creates no CONFIRMATION
array, score or ledger.

### C38 - Return and stop

Return one packet containing the PRE, successor diff, exact census, refusal
battery, mutation results, mechanics evidence, final-tip suite counts and the
two still-uninstalled record templates.

Required disposition:

`M4_CONFIRMATION_PROTOCOL_READY_FOR_EXTERNAL_MUSASHI_REVIEW`

Stop there. No CONFIRMATION execution, GPU, DOIN integration, financial data,
live action or production deployment.
