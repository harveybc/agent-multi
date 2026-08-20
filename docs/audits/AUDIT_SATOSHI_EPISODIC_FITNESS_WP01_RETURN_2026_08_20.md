# Audit: Satoshi Episodic Fitness WP0/WP1 Correction Return

Date: 2026-08-20 America/Bogota  
Auditor: General Musashi  
Subject: `agent-multi@93633229`  
Prior audit: `AUDIT_SATOSHI_EPISODIC_ACTIVITY_FITNESS_WP01_2026_08_20.md`  
Verdict: **substantial correction accepted in part; WP0/WP1 still blocked**

## 1. Independently Reproduced Good Work

- EAF-001's exact one-trade fixture now orders correctly for the tested values:
  the 40-trade learner beats the one-trade quasi-passive policy.
- Deep losses `-1`, `-10`, `-100` are distinct and locally monotone.
- Invalid branch weights used by the prior reproducer now refuse.
- `bars_per_year` now requires a positive non-boolean integer.
- Activity plateau bounds have no executable defaults.
- Handoff requires at least two declared crossings plus mapped action changes.
- Full project suite in `trading-stack`: **1738 passed, 2 unrelated sklearn
  convergence warnings, 140.66 seconds**.
- Four P1LR services remain inactive; no fleet GPU campaign was launched.

These are meaningful corrections and remain part of the accepted base.

## 2. Blocking Findings

### EAF-009 — S2 — Claimed non-aborting post-fix reproducer aborts immediately

Running the committed
`SATOSHI_EPISODIC_FITNESS_WP01_REPRO_2026_08_20.py` at the subject tip raises
`EpisodicFitnessError` on its first fixture because the new required plateau is
not passed. It therefore emits no per-case post-fix dispositions. The committed
`ZERO_REPRODUCED` JSON was not reproduced by the committed runner.

Required correction: update the reproducer with an explicit diagnostic
candidate contract, make it non-aborting per case, execute it from a clean
checkout, and persist stdout plus exit code. It must test the executing code,
not a manually reconstructed table.

### EAF-010 — S2 — Active finite loss can rank below zero trades

The replacement loss transform is strictly monotone but unbounded. At target
activity, independent values include:

```text
return -100:  -28.0269
return -1e6:  -74.0786
return -1e9: -108.6173
zero trades: -100.0000
```

Thus a finite active policy can again lose to the no-trade sentinel. The relaxed
solvency curriculum explicitly allows deeply negative balances, so this is not
an irrelevant mathematical tail.

The configurable sentinel also lacks a relational invariant: with sentinel
`-1`, an ordinary active `-20%` fixture scores about `-1.001`, making zero trades
win.

Required correction: use a bounded, strictly monotone loss transform over the
whole finite loss domain (for example `loss / (1 + loss)`), reserve a guaranteed
open scalar interval above the sentinel for every active-loss result, and
validate the sentinel/weight relationship. Property-test logarithmically spaced
losses through at least `1e300` and multiple accepted configs.

## 3. Prior Finding Dispositions

| Finding | Disposition |
| --- | --- |
| EAF-001 exact one-trade attractor | corrected for tested contract; retain until EAF-010 range proof |
| EAF-002 objective unwired | open, acknowledged by implementer |
| EAF-003 deep-loss alias | local alias corrected; global ordering blocked by EAF-010 |
| EAF-004 config inversion | tested ranges corrected; relational sentinel invariant still open |
| EAF-005 time-base typing | independently verified corrected |
| EAF-006 invented executable plateau | executable default removed; calibration remains unfinished |
| EAF-007 one-crossing handoff | mechanical one-crossing case corrected; calibrated split activity still pending |
| EAF-008 NOP integration | open, acknowledged by implementer |

## 4. Ordered Next Work

1. Correct EAF-009 and EAF-010 before wiring the objective anywhere.
2. Produce the source-referenced activity sensitivity dataset and select a
   plateau contract without outer/sealed-test tuning.
3. Implement WP2 reward arms and real environment NOP trajectories.
4. Implement WP3 wiring into the actual easy selector and stopping state, with
   call-path evidence.
5. Execute WP4 CPU smoke and return for independent audit.

WP5 local GPU smoke and every later fleet stage remain unauthorized.
