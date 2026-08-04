---
type: concept
id: repository-map
title: Repository map and primary roles
status: draft
producer: satoshi-iii
verified_by: none
created: 2026-08-04
updated: 2026-08-04
review_by: 2026-09-04
canonical_for: repository-map
supersedes: none
sources:
  - docs/work_plan/13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md
  - docs/work_plan/15_DISTRIBUTED_CAMPAIGN_LIFECYCLE.md
tags:
  - orientation
---
Eleven repositories under one workspace. `agent-multi`: agents, optimizers,
campaign supervisor, work plan and audits. `lts`: live/demo execution,
venue adapters, journals, runners and watchdog. `trading-contracts`:
canonical execution contracts. `doin-core`/`doin-node`/`doin-plugins`:
decentralized optimization runtime and chain. `gym-fx`: training and
simulation environment. `predictor`, `prediction_provider`, `feature-eng`,
`feature-extractor`: model and feature pipelines. Canonical work-plan and
findings live in `agent-multi/docs`; per the sources, the ledger (doc 13)
is the implementation status authority and doc 15 the campaign lifecycle
authority.
