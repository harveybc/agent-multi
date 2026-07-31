# 04. Open Findings Register

Version: 1.2.0
Date: 2026-07-30
Owner: Satoshi (state updates); closure of S0-S2 requires an independent
verifier (normally Musashi) per `../README.md`.

This file is the cross-session source of truth for finding state. Full finding
stanzas live in their originating report; this register carries identity,
state and the next required action only.

## 1. Open

None.

## 2. Closed / Resolved

| ID | Sev | State | Closure | Verifier | Date | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| AUD-F2-20260730-004 | S2 | verified_closed | User accepted the TWS Paper disclaimer; watchdog now requires a recent authenticated reconciled session and overlapping preflights fail closed | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `lts@12d389d`; `205 passed`; successful systemd observer/watchdog run |
| AUD-GEN-20260730-001 | S3 | verified_closed | Document 13 phase summary, ledger and immediate tasks now record the deployed four-worker v2 campaign | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |
| AUD-F1-20260730-002 | S4 | verified_closed | Document 13 records measured fleet throughput, the 10-14 day full-budget range and an end-of-stage-1 duration/evidence decision point | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |
| AUD-GEN-20260730-003 | S4 | verified_closed | Musashi recovery prompt v1.1.0 includes docs 08/11/14/16 and refreshes the runtime warning | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |

## 2b. Observations Pending Verification (not yet findings)

| Ref | Observed | Verify in |
| --- | --- | --- |
| OBS-20260730-A | Job-0 record `started_at` is 2026-07-29T07:16:30Z (02:16 COT) but the Omega node process start implies ~18:18 COT the same day, a ~16 h gap with `restart_count=0`. Likely the job record marks plan materialization while workers launched after the deployment sequence, but unverified. | `worker_events` table, next delta session |
Resolved observation: `OBS-20260730-B` was the symptom that became
`AUD-F2-20260730-004`; the finding is now verified closed.

Resolved observation: `OBS-20260730-C` is covered by the live tier-0 collector
at `agent-multi@12d394ff`. Its verified systemd packet recorded all three
hosts, four GPUs, utilization/temperature/memory, RAM, swap, disk and campaign
cgroup OOM counters.

## 3. Verified Non-Findings (do not reopen without new evidence)

From `../AUDIT_BOOTSTRAP_2026_07_30.md` (2026-07-30):

1. Four-worker fleet lineage fully consistent (one plan hash, generation,
   population fingerprint, chain height, finalized anchor, component
   revisions; distinct claims; zero alerts).
2. Deployed `agent-multi@6a7bf5a` is code-identical to HEAD `21bcc427`
   (docs-only delta).
3. All 11 repos clean and synced; no user changes at risk.
4. Job-1 `planned_candidates=0` is the fail-closed materializer design, not a
   missing budget.
5. Gen-0 vs current-generation population fingerprints differing is
   per-generation fingerprint behavior, not divergence.
6. `predictor` recent commits are Apr-May 2026 historical work, consistent
   with reference-only role.

## 4. Register Rules

- Add a row when a report opens a finding; move to section 2 with verifier,
  date and closure evidence reference when closed.
- Never edit severity/history in place; append a state-change note if a
  severity is re-triaged, with reason.
- Each delta session (AT-GEN-010) re-verifies that open rows still reproduce
  and that closed rows have closure evidence recorded by a non-reporter.
