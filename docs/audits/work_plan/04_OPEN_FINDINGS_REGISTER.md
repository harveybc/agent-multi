# 04. Open Findings Register

Version: 1.0.0
Date: 2026-07-30
Owner: Satoshi (state updates); closure of S0-S2 requires an independent
verifier (normally Musashi) per `../README.md`.

This file is the cross-session source of truth for finding state. Full finding
stanzas live in their originating report; this register carries identity,
state and the next required action only.

## 1. Open

| ID | Sev | State | Title | Source report | Next action | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| AUD-GEN-20260730-001 | S3 | open | Doc 13 reports live v2 campaign as "pending deploy" | `../AUDIT_BOOTSTRAP_2026_07_30.md` | Musashi refreshes doc 13 status/ledger with deployment evidence | Musashi |
| AUD-F1-20260730-002 | S4 | open (revised 2026-07-30 21:05) | Job-0 wall clock undeclared. REVISED: fleet throughput measured on all four workers is 1.73 cand/h (omega 0.280, dragon 0.539, gamma-5070ti 0.520, gamma-5090 0.393), not the 0.95-1.5 inferred from Omega alone. Remaining budget ~427 of 480 candidates -> ~10-14 days, subject to L2 patience. The bootstrap claim that protected brackets "roughly halved" throughput was an extrapolation error from the slowest worker; fleet rate is comparable to v1's 1.688 cand/h. | `../AUDIT_BOOTSTRAP_2026_07_30.md` | User/Musashi record an expected-duration decision point; auditor must not extrapolate fleet rates from one worker again | Musashi + user |
| AUD-F2-20260730-004 | **S2** | open | IBKR Paper observer failing every 5 min for ~4 h while the watchdog reports IBKR "available" from a TCP probe only | `../AUDIT_STATUS_2026_07_30.md` section 7b | (a) user accepts the IBKR paper API disclaimer; (b) Musashi makes the watchdog assert per-venue observer freshness instead of port reachability | Musashi + user |
| AUD-GEN-20260730-003 | S4 | open | Codex recovery prompt omits docs 08/11/14/16; snapshot warning stale | `../AUDIT_BOOTSTRAP_2026_07_30.md` | Musashi revises prompt at next version bump | Musashi |

## 2. Closed / Resolved

None yet.

## 2b. Observations Pending Verification (not yet findings)

| Ref | Observed | Verify in |
| --- | --- | --- |
| OBS-20260730-A | Job-0 record `started_at` is 2026-07-29T07:16:30Z (02:16 COT) but the Omega node process start implies ~18:18 COT the same day, a ~16 h gap with `restart_count=0`. Likely the job record marks plan materialization while workers launched after the deployment sequence, but unverified. | `worker_events` table, next delta session |
| OBS-20260730-B | `ibkr-paper-lab.sqlite` last written 18:12 COT while `alpaca-paper-lab.sqlite` writes every ~5 min; watchdog reports IBKR "available" from a TCP probe only, and `equity_market_open=false`. Possibly correct closed-market behavior, possibly a stalled writer. | AT-F2-002 |
| OBS-20260730-C | The campaign supervisor API carries no per-worker machine telemetry (GPU temp/util, RAM, swap). Remote host health is inferred from absence of watchdog alerts rather than measured in the audit path. | Snapshot collector packet (file 02 section 4.1) |

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
