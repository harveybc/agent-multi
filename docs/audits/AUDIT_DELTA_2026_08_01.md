# Delta Audit — MT5 Vertical, Hermes Migration, Campaign State

Audit ID: AUDIT-DELTA-20260801-01
evidence_observed_at: 2026-08-01 02:25–02:40 America/Bogota
report_written_at: 2026-08-01 02:45 America/Bogota
Auditor: Satoshi
Scope: audit of Musashi's work since the AT-F1-012 report — MT5 read-only
vertical acceptance (`agent-multi@9c435725`, `lts@a533389`, `lts@f995a99`),
Hermes Flash migration (`agent-multi@9ce65871`) — plus campaign, fork and
broker runtime deltas. Owner-reported context: "the MT5 stuff is working."

## 1. Findings

### AUD-F2-20260801-029 — Fleet-level MT5 observability contradicts the accepted vertical, and the auditor cannot reach the evidence

- Severity: **S3** (observability split-brain; no venue-safety defect implied)
- Confidence: high on the contradiction (`observed`); MT5 health itself is
  neither confirmed nor denied by this finding
- Observed:
  1. Document 22 (post-`9c435725`) declares the OANDA MT5 demo vertical
     **accepted**: EA compiled zero errors/warnings, `OANDA_Global-Demo-1`
     authenticated, signed heartbeats/snapshots posting **to Dragon**, six
     symbols evidenced, zero positions/orders, "fresh signed
     heartbeat/snapshot evidence cleared the MT5 watchdog alert."
  2. Omega's consolidated paper-execution watchdog at 07:27:19Z **still
     raises** `mt5_bridge_missing` with `mt5: {available: false, reason:
     database_missing}` — the alert the docs call cleared is cleared only on
     the *Dragon-local* watchdog.
  3. Independent verification from Omega failed: SSH to Dragon
     (100.110.215.85:22) → connection refused. The fleet's newest and
     highest-priority venue currently has **no auditor-reachable, fleet-level
     evidence surface**.
- Impact: (a) the consolidated alert stream now carries a permanently red
  false alarm — alert-fatigue risk and the inverse of the IBKR lesson (there,
  green masked red; here red would mask a *real* future MT5 failure since
  operators will learn to ignore it); (b) my audit snapshot packet's
  `brokers` section is now wrong about MT5 by construction; (c) recovery
  scenarios that lose Dragon lose all MT5 health history.
- Proposed correction (Musashi packet): either ingest Dragon's MT5
  heartbeat/snapshot facts into the consolidated watchdog (preferred —
  matches the "functional freshness" pattern he built for IBKR), or re-scope
  the consolidated `mt5` probe to query Dragon's bridge endpoint, and expose
  one read-only fleet-reachable evidence path for the auditor. Until then,
  MT5 acceptance is classified: **documented with specific evidence on
  Dragon; not independently verified by the auditor.**
- Owner: Musashi. The acceptance itself is not contested — the observability
  seam is.

### Observation (not a finding): champion fitness has been flat for two generations

`Observed`: best fitness `0.00048223…` (block 8) is unchanged through
generations 3 and 4 (now gen 4 at 17/20, chain height 11, finalized 4). With
`optimization_patience = 4` and 6 generations/stage, stage-1 termination is
plausibly ≤ 1–2 generations away (~1 day at current ~1.5 candidates/hour).
Consequence: **Harvey's A/B/C objective decision (AT-F1-012 packet) is time-
sensitive** — the job-0 champion archive that seeds job 1 will be selected
under the disputed full-period objective. Alternative A's rider (elite set
must contain the weekly top-2 — true today) should be re-verified at archive
time if the decision has not landed by then.

## 2. Verified This Delta

1. **Fork fully converged** (`reproduced`): one distinct tip, one finalized
   anchor, **zero active alerts fleet-wide** at 02:40. The height-9/10
   competitions resolved by finalization exactly as findings 005/020
   predicted. This strengthens the queued closure recommendation for 005 and
   softens 020 to a latency-measurement item.
2. **Campaign healthy** (`reproduced`): generation 4 at 17/20, finalized
   height 4, four workers, no errors.
3. **`lts@f995a99`** (`observed`, code+tests): OANDA REST monitoring made
   optional in the watchdog with 22 new test lines — correct consequence of
   the REST-v20/OGM boundary; prevents a permanent false alarm for a venue
   this account can never use. Same defect class as finding 029, fixed for
   REST but not yet for MT5 — the pattern is proven, apply it once more.
4. **Hermes Flash fleet migration** (`observed`, docs-level): fleet moved to
   the cheaper `deepseek-v4-flash` tier with a dedicated migration record.
   Consistent with the token-economy direction; audit hook noted for the next
   Front-3 pass: confirm reservation caps and cost_basis fields survived the
   provider switch (AT-F3-013 scope).
5. **IBKR recurrence** (`observed`): `ibkr_observer_stale`/`ibkr_paper_offline`
   active again — TWS is not running on Omega. Recurring owner action; the
   watchdog is behaving exactly as redesigned (functional freshness, not port
   liveness).

## 3. State

No runtime, chain, broker or config was touched. Recovery prompt updated;
register untouched pending Harvey's queued decisions. Next triggers: Harvey's
A/B/C decision (time-sensitive per §1 observation); stage-1 boundary;
Musashi's response to finding 029.
