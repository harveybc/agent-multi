---
type: concept
id: findings-state
title: Findings register semantics and current bands
status: draft
producer: satoshi-iii
verified_by: none
created: 2026-08-04
updated: 2026-08-04
review_by: 2026-09-04
canonical_for: findings-state
supersedes: none
sources:
  - docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md
tags:
  - front4
---
The open findings register (source) is the only authority on finding
state. Semantics: the implementer never closes findings they implemented;
independent verification precedes owner closure; severities S0-S4 order
urgency. As of the bundle date: 069-085 are independently verified and
eligible for owner closure except 079, which awaits one narrow piece of
VM evidence (exact MetaEditor zero-error compile output); 086-090 are
corrected in doc 32 v1.1 pending independent verification. Always re-read
the register rather than trusting this snapshot.
