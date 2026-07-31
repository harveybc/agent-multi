# General Satoshi Objective-Contract and Curriculum Audit Invocation 08

Date: 2026-07-31
Task: `AT-F1-012`
Authority: Harvey, business owner
Technical counterpart: Musashi
Execution order: immediately after Invocation 07 is completed and persisted

General Satoshi,

This is a bounded independent audit of a decision that can change which model
the fleet declares champion. It is not a writing exercise and it does not
authorize any runtime mutation.

Act as an independent machine-learning scientist, quantitative-trading
researcher, distributed-systems auditor and reproducibility reviewer. Be
skeptical of both Musashi's implementation and your own previous conclusions.
Arithmetic fidelity, statistical meaning and operational compatibility must
all pass separately.

## 1. Preconditions

Before starting this task:

1. Complete
   `GENERAL_SATOSHI_AT_F1_001_CORRECTION_INVOCATION_2026_07_31_07.md`.
2. Persist its correction report and recovery-state update.
3. Confirm `AT-F1-001` remains `reported (finding open)` unless Harvey has
   explicitly selected an authoritative objective.
4. If Invocation 07 is not complete, stop after reporting that exact blocker.

## 2. Mandatory Inputs

Read only these inputs initially, in order:

1. `docs/audits/CODEX_CROSS_REVIEW_OF_SATOSHI_AT_F1_001_2026_07_31.md`
2. The correction report produced from Invocation 07
3. `docs/audits/work_plan/01_AUDIT_BACKLOG_AND_SCHEDULE.md`, section 4.5
4. `pipeline_plugins/rl_pipeline_with_validation.py`, especially the split
   metric construction and L2 aggregation paths
5. `examples/campaigns/phase_1_protected_execution_fleet_v2/campaign_plan.json`
   plus the job-0 worker configs and the job-1 base config it names
6. `examples/scripts/materialize_execution_curriculum_campaign.py`,
   `examples/scripts/materialize_execution_curriculum_followup.py` and every
   helper they directly invoke
7. `docs/work_plan/20_PROTECTED_EXECUTION_RECOVERY_AND_OPTIMIZATION.md`
8. `docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md`

Runtime evidence is read-only:

```text
http://127.0.0.1:8795/api/network
~/.local/state/agent-multi/doin-campaigns/phase-1-protected-execution-fleet-v2/
```

Use blockchain/OLAP candidate facts when available. Logs may fill a named gap,
but must not silently replace authoritative accepted-result evidence.

## 3. Questions to Resolve

### 3.1 Ranking sensitivity

For every accepted job-0 candidate with enough atomic split evidence,
reconstruct:

1. configured full-period L2;
2. mean-weekly-normalized L2;
3. train-tail and validation weeks, RAP, mean weekly RAP and trade counts;
4. eligibility-floor outcome under both interpretations.

Then report:

- Spearman and Kendall rank correlation;
- top-1 and top-5 overlap;
- every sign flip and every eligibility flip;
- whether the current champion remains champion;
- the magnitude and direction of unequal-horizon bias;
- sample coverage and the exact reason for every excluded candidate.

Do not impute missing candidate evidence. A coverage gap is a result.

### 3.2 Objective semantics

Classify each objective separately:

- what quantity it rewards;
- its unit and horizon dependence;
- whether it is comparable across split lengths;
- whether it is suitable as a DEAP fitness scalar;
- which failure modes it creates: inactivity, overtrading, horizon dominance,
  risk dilution or unstable ranking.

Do not call either objective correct merely because code and stored values
agree.

### 3.3 Curriculum inheritance

Trace the queued job-1 materialization path end to end. Determine whether it:

1. inherits job-0's champion genome, model weights, metric scalar or elite set;
2. recalculates fitness under weekly folds or merely carries the old ranking;
3. can repair job-0 selection bias, preserve it, or amplify it;
4. preserves mandatory SL and TP, minimum activity, test firewall and artifact
   provenance;
5. produces a comparable scalar across easy, nominal and stress difficulty.

Name exact files, symbols and config keys. Distinguish verified behavior from
inference.

### 3.4 Decision packet

Produce a decision matrix for Harvey with exactly these alternatives:

| Alternative | Meaning |
| --- | --- |
| A | Let job 0 finish unchanged; use it only as curriculum initialization and make job 1 authoritative |
| B | Stop only at a verified generation/job boundary, archive job 0 as a baseline, then launch a corrected v3 objective |
| C | Retain the current objective as authoritative because measured ranking sensitivity is immaterial |

For each alternative report expected information retained, compute cost,
comparability impact, artifact consequences and falsifiable selection rule.
Recommend one alternative, but do not execute it. Harvey owns the decision.

## 4. Academic and Innovation Consequence

Add a short research disposition, not a paper draft:

1. Is unequal-horizon objective bias merely an implementation defect, or does
   it motivate a publishable controlled study?
2. State one falsifiable hypothesis and the minimum experiment needed.
3. Identify likely prior-art collision categories without asserting novelty.
4. State whether this belongs in P1, P6, another registered line, or nowhere.

No new research line may be registered without checking the existing registry.

## 5. Required Outputs

Create exactly:

1. `docs/audits/AUDIT_OBJECTIVE_CONTRACT_AND_CURRICULUM_2026_07_31.md`
2. `docs/audits/evidence/AT_F1_012_OBJECTIVE_RANKING_2026_07_31.csv`
3. an update to `docs/handoffs/CLAUDE_AUDIT_RECOVERY_PROMPT_2026_07_30.md`
4. only the minimum backlog/finding updates justified by reproduced evidence

The CSV must contain one row per included accepted candidate and fields for
candidate identity, block/transaction provenance, split weeks, split RAP,
both reconstructed L2 values, both ranks and all eligibility flags.

## 6. Prohibitions

- Do not stop, restart, reconfigure or mutate the active swarm.
- Do not rewrite blockchain, OLAP or campaign artifacts.
- Do not change fitness code or queued-job configuration.
- Do not launch training, backtests or a replacement campaign.
- Do not create deadlines, command Hermes, close your own findings or convert
  missing evidence into a pass.
- Do not spend tokens re-auditing unrelated fronts.

## 7. Acceptance Criteria

The task passes only if:

1. Invocation 07 was completed first;
2. every numerical claim is reproducible from cited rows or code;
3. candidate coverage is explicit;
4. the current champion's status under both objectives is unambiguous;
5. job-1 inheritance is traced through executable code and config;
6. the decision recommendation is tied to a falsifiable threshold;
7. persistent audit and recovery state agree;
8. no live runtime changed.

Be concise where the evidence is simple and severe where it is not. The goal
is to protect the scientific meaning of the optimization without wasting the
valid compute already invested.
