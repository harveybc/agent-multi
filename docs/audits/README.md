# Independent Audit Reports

This directory stores compact, versioned reports produced under
`docs/work_plan/24_INDEPENDENT_AUDIT_AND_CONTINUOUS_IMPROVEMENT.md`.

## Naming

```text
AUDIT_<scope>_<YYYY_MM_DD>.md
```

Use a suffix when more than one report covers the same scope and date:

```text
AUDIT_DOIN_CAMPAIGN_2026_07_30_02.md
```

## Required Report Shape

```markdown
# <Audit title>

Audit ID:
Timestamp and timezone:
Auditor:
Requested by:
Scope:
Excluded scope:

## Provenance

| Repository/system | Branch | Commit/config/artifact | Dirty/runtime state |
| --- | --- | --- | --- |

## Findings

### AUD-<FRONT>-YYYYMMDD-NNN - <title>

- Severity:
- Confidence:
- Status:
- Observation:
- Evidence:
- Impact:
- Reproduction:
- Proposed correction:
- Required regression or monitor:
- Owner:
- Dependencies:

## Open Questions

## Verified Non-Findings

## Commands and Queries

## Artifacts and Evidence Hashes

## Next Audit Trigger
```

Findings come first and are ordered from `S0` to `S4`. An executive summary
does not replace findings.

## Evidence Rules

- Record only bounded commands and relevant result summaries.
- Redact secrets, account identifiers, customer data and private messages.
- Link to source files with line numbers.
- Label statements as observed, inferred or hypothesis.
- Do not commit databases, logs, model weights, screenshots with sensitive
  data or generated training output.
- Record hashes and local retention paths for large evidence.
- Do not claim a finding is fixed until Codex independently reproduces closure
  evidence.

## Finding Lifecycle

Allowed states:

```text
open
accepted
in_progress
fixed_pending_verification
verified_closed
accepted_risk
deferred
false_positive
superseded
```

Closure records the verifier, date, commit/config/artifact and exact test or
runtime evidence. The original reporter should not be the only verifier for an
`S0`, `S1` or `S2` finding.
