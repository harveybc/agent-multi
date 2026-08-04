---
type: concept
id: metric-definitions
title: Metric definitions for promotion and operations
status: draft
producer: satoshi-iii
verified_by: none
created: 2026-08-04
updated: 2026-08-04
review_by: 2026-09-04
canonical_for: metric-definitions
supersedes: none
sources:
  - docs/work_plan/32_CHAMPION_SUCCESSION_AND_REGIME_RESEARCH.md
tags:
  - metrics
---
Primary decision metric: robust weekly RAP, compared as paired weekly
challenger-minus-incumbent differences on at least 26 common eligible
weeks with a one-sided 95% simultaneous lower bound from a paired
moving-block bootstrap (block four weeks, 10,000 deterministic resamples,
frozen max-statistic comparison family). DSR and PSR are Sharpe-specific
diagnostics only and are never applied to RAP. Daily observations trace
divergence mechanics (costs, action agreement, coverage, calibration) and
never establish alpha. Safety metrics have zero tolerance. Cost/drift
thresholds require a versioned baseline of at least 30 eligible fills or
four weeks before they exist.
