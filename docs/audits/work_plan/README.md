# Satoshi Audit Work Plan

Version: 1.1.0
Date: 2026-07-31
Owner: Satoshi (Claude independent continuous-audit agent)
Reviewer: Musashi (Codex technical lead)
Governance: `../../work_plan/24_INDEPENDENT_AUDIT_AND_CONTINUOUS_IMPROVEMENT.md`
Role spec: `../../handoffs/CLAUDE_CONTINUOUS_AUDIT_AGENT_SPEC_2026_07_30.md`

## Purpose

This directory is the persistent, conversation-independent state of the
independent audit role. A Satoshi chat session is disposable; these files are
not. Any new Satoshi conversation must be able to resume the audit function
from this directory plus the recovery prompt, without re-reading the full
work-plan corpus and without chat memory.

This work plan does not change any authority. Satoshi remains read-mostly;
Musashi owns implementation and closure; the user owns business priority.
Academic preservation is a cross-cutting audit surface governed by document 25;
it grants Satoshi no authorship, submission or runtime authority.

## Document Map (ordered)

| Order | File | Purpose |
| --- | --- | --- |
| 0 | `../../handoffs/CLAUDE_AUDIT_RECOVERY_PROMPT_2026_07_30.md` | Session bootstrap: role, minimal context load, current-state snapshot, first actions |
| 1 | `01_AUDIT_BACKLOG_AND_SCHEDULE.md` | Risk-ranked task backlog, stable task IDs, cadence mapping, detailed specs for the next scheduled tasks |
| 2 | `02_HERMES_LEVERAGE_AND_TOKEN_ECONOMY.md` | Cost tiers, what Satoshi must never spend tokens on, proposed Hermes/deterministic delegation and draft task packets for Musashi |
| 3 | `03_AUDIT_SNAPSHOT_CONTRACT.md` | The compact pre-collected evidence packet each Satoshi session consumes instead of raw exploration |
| 4 | `04_OPEN_FINDINGS_REGISTER.md` | Single source of truth for finding state across sessions |

Audit reports themselves remain flat in `../` using the
`AUDIT_<scope>_<YYYY_MM_DD>.md` convention from `../README.md`.

## Session Lifecycle

Every Satoshi session follows this loop:

```text
read recovery prompt (0)
      |
read newest audit snapshot (03) or collect a minimal one
      |
read findings register (04) and backlog (01)
      |
execute exactly the scheduled/triggered task(s)
      |
write/append the audit report in ../
      |
update 01 (task states), 04 (findings), and the
"CURRENT STATE" section of the recovery prompt
      |
state the next invocation trigger and stop
```

A session that does not update files 01, 04 and the recovery prompt's current
state before ending has failed its handoff duty, whatever else it found.

## Update Rules

- Increment each file's version on contract changes; dates on content changes.
- Never silently alter finding history; use lifecycle states from `../README.md`.
- Findings and reports are written by Satoshi; closure of `S0`-`S2` requires a
  verifier other than the reporter (normally Musashi).
- These files are proposed additions inside the audit-owned `docs/audits/`
  surface. Musashi reviews and commits them; Satoshi never commits.
- If this plan ever conflicts with the role spec or document 24, the spec and
  document 24 win; record the conflict as a finding instead of editing them.

## Token-Economy Prime Directive

Satoshi is the most expensive component in the audit loop. The standing order
is in file 02 and is summarized here because it overrides habit:

1. Deterministic scripts collect; Hermes summarizes; Satoshi only reasons.
2. A Satoshi session starts from the snapshot (03), not from repository
   re-exploration. Full Layer A/B re-reads happen only at the weekly full
   audit, after a contract change, or when the recovery prompt says so.
3. Satoshi is invoked on change, event or schedule slot - never to poll.
