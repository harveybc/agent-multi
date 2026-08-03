# Musashi to Satoshi III: IBKR L1 Correction and Milestone-F Order

Date: 2026-08-03 America/Bogota
From: General Musashi, temporary independent auditor
To: Satoshi III / Mujuro Utsutsu, successor technical lead
Owner: Harvey, project owner
Priority: Front 2 P0
Broker write authority granted by this order: none

## 1. Read First

1. [Independent A-E audit](/home/harveybc/Documents/GitHub/agent-multi/docs/audits/AUDIT_SATOSHI_III_IBKR_L1_MILESTONES_A_E_2026_08_03.md)
2. [Independent reproducer](/home/harveybc/Documents/GitHub/agent-multi/docs/audits/evidence/IBKR_L1_MILESTONES_A_E_REPRO_2026_08_03.py)
3. [Codebase Memory operating specification](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/CODEBASE_MEMORY_MCP_OPERATING_SPEC_2026_08_03.md)
4. [MCP tooling research disposition](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/MUSASHI_DISPOSITION_SATOSHI_III_MCP_TOOLING_RESEARCH_2026_08_03.md)

Reproduce all five counterexamples before editing. Findings 069-074 are yours
to correct and General Musashi's to verify. Do not close them yourself.

## 2. Work Package F0: Correct the Execution Model

Implement in this order, with one bounded commit after each item:

1. **Durable immutable effect contract (072).** Persist canonical bracket
   plan, expected conId, authorized account binding and rounding inputs before
   any call. Resume from this record only.
2. **Proven no-call terminal recovery (073).** Resolve zero-attempt
   `journaled_pending` deterministically without releasing/reusing the consumed
   capability.
3. **Direct protection health and cumulative fills (069/071).** Model direct
   `filled`, `remaining`, status and execution identity. Re-verify SL/TP before
   accepting each cumulative fill and while exposure remains open. Missing or
   altered protection invokes executed recovery.
4. **Exact risk-reducing preflight (070).** Bind account, contract, position,
   quantity and side before flatten submission; prove it cannot increase or
   cross exposure through zero.
5. **Single L0 lifecycle path (074).** Make L0 intent-class-aware and remove
   direct lifecycle appends from L1.

Preserve the accepted L0 behavior for risk-increasing entries. Add schema
migration tests for existing SQLite ledgers; no destructive reset is accepted.

## 3. Required Deterministic Tests

At minimum add fixtures for:

- stop or TP disappears after initial acknowledgement and before fill;
- stop or TP changes after a partial fill while exposure remains;
- cumulative partial fills 5k -> 12k -> 20k, duplicate events and restart;
- partial fill followed by cancel, recovery and exact flat reconciliation;
- flatten delta larger, smaller, opposite-sign or stale relative to position;
- wrong connected account and multi-account position snapshots;
- same symbol/currency but wrong conId/account/secType position;
- restart with changed profile, rounding or connected account;
- restart preserves and enforces expected conId;
- zero-attempt crash resolves terminally with zero broker calls;
- hold and kill allow only exact risk reduction and never clear halt;
- L0 risk-reducing fill uses the accepted service API without recursive
  emergency flatten generation;
- recovery adapter raises non-I/O exceptions; runner journals and remains
  observable rather than crashing silently.

Add Hypothesis/property tests after these deterministic counterexamples pass:

- a risk-reducing order can never increase `abs(position)` or cross zero;
- accepted exposure never exists without directly verified SL and TP coverage;
- cumulative fills are monotone, bounded by requested quantity and conserved;
- replay/restart does not duplicate broker calls or exposure deltas.

## 4. Work Package F1: Real `ib_async` Client, Still Zero-Submit

Only after F0 is green:

1. implement the narrow real client behind `IbkrClientProtocol`;
2. map direct `Trade`, `OrderStatus`, executions, open orders and positions
   into the exact fact schema, including filled/remaining/conId/account;
3. add bounded acknowledgement polling with injected clock/sleeper and full
   observation journaling;
4. perform connected **read-only, zero-submit** preflight against TWS Paper;
5. reconcile the current account fingerprint discrepancy from direct evidence;
6. collect spread, quote age, request/response latency and commission schema
   availability into the existing OLAP, with no parallel database; and
7. update the heartbeat with evidence-derived fields whose names distinguish
   call attempts, broker acknowledgements, fills and open orders.

No order object may be submitted in F1. Return an audit packet with source
mapping, focused/full tests, graph-assisted impact trace, preflight hash and
`orders_submitted=0` derived from direct broker evidence.

## 5. Approved Side Jobs

These may run only while F0/F1 waits for independent review and may not delay
Front 2 P0:

- add read-only population-diversity telemetry to Front 1 status, derived from
  current population facts without campaign mutation;
- derive finalized-anchor history from existing DOIN chain/OLAP facts, without
  creating a second ledger; and
- document the Paper-canary decision-to-submit latency distribution once real
  read-only timestamps exist;
- implement `DEV-TOOLING-MCP-001`, the project-owned read-only SQLite evidence
  MCP, only after F0 is green or while F0 waits for independent review; and
- run `DEV-TOOLING-MCP-002`, the bounded Context7 measurement, only at F1 and
  only if installed `ib_async==2.1.0` source plus official documentation leave
  a named API question unresolved.

The two public SQLite MCP candidates reviewed on 2026-08-03 are rejected as
shipped. Do not install either one or connect any MCP to a live project ledger.

## 6. Stop Conditions

Stop and report immediately on any real broker submission, unknown account,
unreconciled position, missing protection, schema migration loss, secret
exposure, provider safeguard warning or test that touches a socket outside the
named read-only preflight.

Do not enable the systemd runner, mint a capability, request an activation
phrase or alter the running DOIN campaign.

## 7. Delivery Contract

Deliver:

- exact commits per milestone;
- changed-file inventory;
- focused and full test commands/results;
- mapping from 069-074 and every fixture above to tests;
- migration and restart evidence;
- zero-submit preflight facts for F1;
- remaining doubts stated directly; and
- an audit request that closes nothing.

Begin F0 immediately after reproducing the five auditor scenarios. Do not wait
for another planning round.
