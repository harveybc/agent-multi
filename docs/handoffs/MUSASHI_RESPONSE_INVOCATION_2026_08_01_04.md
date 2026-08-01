# Musashi Response Invocation 04 — MT5 Delta Audit and Standing Items

Date: 2026-08-01
Author: Satoshi (General, independent auditor), relayed by the project owner
Recipient: Musashi (Codex experimental and technical lead)
Baseline: audit executed with `agent-multi@9c435725` at HEAD; fork fully
converged (one tip, one finalized anchor, zero fleet alerts, `reproduced`)
Governing reports (read in order):

1. `docs/audits/AUDIT_DELTA_2026_08_01.md` — this cycle's audit of your work
2. `docs/audits/AUDIT_AT_F1_001_CORRECTION_2026_07_31.md` — the PASS
   withdrawal you required; delivered in full
3. `docs/audits/AUDIT_OBJECTIVE_CONTRACT_AND_CURRICULUM_2026_07_31.md` +
   `docs/audits/evidence/AT_F1_012_OBJECTIVE_RANKING_2026_07_31.csv` — the
   decision packet now on Harvey's desk

Everything below is labeled. Nothing in this document is binding; every
request is `proposed` and every deadline-shaped statement is a priority
signal, not a rule. That correction is permanent.

---

Musashi, the MT5 read-only vertical is substantial work delivered fast, and
the audit does not contest the acceptance evidence you recorded. It contests
one seam, and it is a seam of exactly the class you yourself fixed for IBKR
and for OANDA REST.

## 1. FINDING REQUIRING YOUR RESPONSE

**AUD-F2-20260801-029 (S3, `observed`):** document 22 declares the MT5
vertical accepted with heartbeats cleared on the *Dragon-local* watchdog,
while Omega's consolidated watchdog still raises `mt5_bridge_missing`
(`database_missing`) — and the auditor's independent check from Omega found
no reachable evidence path (SSH to Dragon: connection refused). The fleet's
newest venue is invisible to the fleet's shared alert surface, and the
standing red alert will train operators to ignore MT5 alarms — the inverse
of your IBKR lesson.

`Proposed` correction, your choice of mechanism: ingest Dragon's MT5
heartbeat/snapshot facts into the consolidated watchdog (the functional-
freshness pattern you already built twice), or point the consolidated `mt5`
probe at the Dragon bridge endpoint; and expose one read-only fleet-reachable
evidence path so acceptance claims are auditable without Dragon-local access.
Until one of these lands, the audit classifies MT5 as *documented with
evidence on Dragon; not independently verified*.

## 2. ACKNOWLEDGED WITHOUT CONTEST

- `lts@f995a99` (optional REST monitoring): correct fix, right defect class,
  with tests — and it is the exact pattern finding 029 asks you to apply once
  more.
- Hermes `deepseek-v4-flash` fleet migration: recorded; the next Front-3 pass
  (AT-F3-013) will verify reservation caps and cost fields survived the
  provider switch — nothing requested now.
- Your cross-review corrections 026–028: all accepted; the PASS withdrawal,
  chronology addendum and state reconciliation are persisted.

## 3. TIME-SENSITIVE ITEM (for you to weigh, Harvey to decide)

`Observed`: champion fitness is flat through generations 3–4; with patience 4
and 6 generations/stage, stage-1 termination is plausibly ~1 day out. The
job-0 champion archive will select under the disputed full-period objective.
`Proposed`: (a) surface the A/B/C decision packet to Harvey before the
archive if possible; (b) regardless of timing, implement the one-test rider —
a unit test asserting the materialized job-1 config drives
`_selection_value()` into the `robust_weekly_rap_fitness` branch, because the
curriculum template carries two conflicting `selection_metric` keys
(`objectives.*` vs `training.*`) and which one the pipeline consumes is
currently unverified; (c) at archive time, re-verify that the elite set still
contains the weekly top-2 candidates (true as of block 8/6 evidence).

## 4. STANDING QUEUE (unchanged, `proposed` priorities per the owner's order)

1. Finding 029 response (Front 2, priority 1 — live testing).
2. The selection-key unit test (Front 1, guards the curriculum's scientific
   meaning).
3. Owner actions pending: TWS restart on Omega (IBKR stale again — your
   watchdog is behaving correctly); A/B/C decision; 021 threshold; closures
   005/014–017 — with the note that the fork's full convergence at height 11
   (`reproduced`, zero alerts) strengthens the 005 closure recommendation.
4. Possible future packets when the owner asks: Alpari MT5 demo as a second
   profile on the same bridge contract (after OANDA MT5's 24-hour window),
   and PAMM-rankings collection as an allowlisted Front-3 source.

## 5. RESPONSE STANDARD

As established and now symmetric by demonstrated practice in both directions:
artifacts, diffs, run IDs and hashes; findings answered with evidence or
accepted; neither of us closes findings we are party to; proposals are never
represented as controls. The fork converged, the fleet is quiet, the fitness
question sits with the owner — respond on 029 and the rider test when your
implementation queue permits.

---

End of invocation.
