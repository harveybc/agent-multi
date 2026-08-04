---
type: concept
id: paper-demo-roles
title: Current Paper and Demo venue roles
status: draft
producer: satoshi-iii
verified_by: none
created: 2026-08-04
updated: 2026-08-04
review_by: 2026-09-04
canonical_for: paper-demo-roles
supersedes: none
sources:
  - docs/work_plan/29_CONTINUOUS_DEMO_TRADING_OPERATIONS.md
  - docs/work_plan/13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md
tags:
  - front2
---
Three writable selected-model venues, all Paper/Demo, all hash-bound to
model manifests with drain-before-replacement and post-close equity
reseeding: Alpaca Paper (SPY daily), IBKR Paper (USD.CAD 4h through TWS on
the loopback Paper port), OANDA MT5 Demo (ETHUSD 4h through the signed
execution bridge on Dragon; both MT5 units run as systemd USER services,
so operational probes must use user scope). Read-only observers and the
consolidated watchdog provide independent reconciliation. No Live account
is authorized. Runtime state must be sampled fresh from heartbeats and
ledgers at use time.
