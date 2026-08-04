---
type: concept
id: recovery-runbooks
title: Failure recovery runbook pointers
status: draft
producer: satoshi-iii
verified_by: none
created: 2026-08-04
updated: 2026-08-04
review_by: 2026-09-04
canonical_for: recovery-runbooks
supersedes: none
sources:
  - docs/work_plan/29_CONTINUOUS_DEMO_TRADING_OPERATIONS.md
  - docs/audits/AUDIT_SATOSHI_III_MULTI_VENUE_CONTINUITY_VERIFICATION_2026_08_03.md
tags:
  - operations
---
Execution defects fail closed: unproven protection or ambiguous broker
facts trigger the deterministic hold/cancel/flatten/reconcile contract;
holds are never cleared by code, only by the owner. Runner stop is
event-driven; rollback of a runner is systemd user-service disable plus
the owner hold/kill command. Quote loss degrades to waiting_for_quote
without restart churn. Zero-attempt crashes terminalize with the burned
capability retained. Every recovery action is journaled before and after.
For exact procedures consult the sources; never operate from memory of
them.
