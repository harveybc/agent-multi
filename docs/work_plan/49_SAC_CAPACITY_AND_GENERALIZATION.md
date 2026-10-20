# SAC Capacity And Generalization Follow-Up

Owner authorization 2026-10-20: continue remaining generalization diagnostics
after the four critic-regularization contrasts closed. Those strengths/seed did
not improve VALIDATION. They do not exclude regularization families or prove
that the existing checkpoint objective prevents every form of overtraining.

## Requirements And Execution Order

1. D0: authenticate all completed checkpoint pairs and final traces. Extend
   existing read-only diagnostics with chronological quarterly daily-growth
   panels and paired contiguous-block bootstrap (5/10/20 observed days,
   1000 repetitions, seed7). Keep all eligible days, including zero returns.
   Derive only retained fields; missing telemetry is unavailable, never zero.
2. D3/GD01: sequential fresh SAC_FLAT capacity contrasts [8], [16], [16,16]
   against the existing [32,32] reference. These sizes apply to actor AND both
   critics, not the environment or feature windows. Count actual parameters.
   An improvement supports this capacity contrast, not a universal size law.
3. Preserve selected15 features,13 observations/stride2 observed bars,
   TRAIN2020-23, VALIDATION2024, seed7, costs, leverage50, mandatory SL/TP,
   sizing, one entry/day and session constraints. No regularizer, TEST,
   refit, DOIN or new temporal fit. Full inference pairs every5000 transitions,
   patience10, absolute min_delta1e-8, cap100000, unchanged negative mean of
   TRAIN/VALIDATION mean daily log growth. No validation-only selector.
4. Each success closes with existing warehouse outbox/readback and physical
   backup, then admits the next size automatically. One heavy process on5090;
   its shared-RAM5070Ti is not a second slot. Persist bounded failures and
   status/ETA; do not silently skip a failed scientific predecessor.
5. Use evidence to prioritize business-state/action mapping, regime shift,
   reward/ledger, replay/critic and seed diagnostics. Do not claim these are
   excluded because a model has low validation profit. New telemetry requires
   a separate tested identity; no fabricated historical gradient/Q readings.

## Test Design And Traceability

| ID | Structural and behavioral acceptance |
| --- | --- |
| CG01 | request_v6 accepts only declared capacities; all other scientific settings equal v4; regularization mixing and extra overrides rejected |
| CG02 | materialize/check/restart reject changed request or transport; source campaigns remain untouched |
| CG03 | three pinned predecessor-linked nodes and isolated OLAP experiment keys; no fits during prepare; idempotent identity |
| CG04 | rendered systemd units preserve RAM/GPU gates and finite attempts; publisher invokes capacity sequence, not regularization sequence |
| CG05 | synthetic native SAC smoke verifies both critic/actor layer sizes, finite updates, deterministic saved-policy reload and parameter counts before real admission |
| CG06 | daily panel includes first-day PnL/zero days; blocks never cross data support mismatch; bootstrap deterministic and typed missing fields |
| CG07 | authenticated existing curves retain original best checkpoint and exact patience; diagnostic output never writes inside source campaigns |
| CG08 | real launch check confirms fresh progress,assigned5090,exact contract and no second heavy process; existing outbox/collector retained |

Current stage: requirements through unit/integration test design frozen before
implementation. Runtime/launch evidence is added only after execution.

## Executed Evidence, 2026-10-20

D0 completed on all91 authenticated checkpoint pairs:20 reference,20 dropout,
20 L2,20 combination,11 BatchNorm. Five new panel reports were delivered and
read back (pending0,mismatch0); result digest
`c4148f52830ea60439868cbc94dc0746cc5e0f579cdeaf3b42c1cd7bc3e0d61e`.
Closed diagnostic JSON/outbox backup were copied to the independent bulk disk
and a second checksum comparison found no differences. No original checkpoint
or source campaign was modified. The bounded CPU job used98.5MiB peak and
12.253s wall time. Trade-level concentration and Q/gradient/replay telemetry
are explicitly unavailable from the authenticated pair trace, not imputed.

Reference VALIDATION quarterly returns:Q1 +1.82524%,Q2 -3.18103%,Q3 +1.13833%,
Q4 +1.04655%. Q2 was negative for all five arms, descriptive evidence to
investigate time dependence, not proof of a regime cause. Moving-block10-day
95% interval for reference mean daily log growth versus no-trade:
[-0.000593810,+0.000631361], point0.0000240857. These are311 observed NY
calendar dates including partial-session dates, not a relabeling as252 full
trading days. The interval is descriptive after adaptive checkpoint selection.

Capacity runtime pinned to `dad7294c`;132 focused tests passed in the actual
worker environment (525.3MiB peak,5.680s wall). Three native synthetic CPU
update/save-load tests and predecessor/publisher regressions are included.
Fresh GPU readiness returned GPU_READY. Real systemd parser verification
passed for generated units (an unrelated existing spice-vdagent warning is
not a campaign parser failure). A real request/contract comparison found the
only common scientific configuration difference was `net_arch`, for all3 arms.

All3 workers, warehouse publisher, post-closure panel analyzer and independent
closed-artifact collector are enabled on timers. The synthetic GPU smoke for
[8] passed16 updates and exact reload on its first attempt. At05:49UTC the
[8] task had5000 real transitions and was evaluating VALIDATION of its first
pair; no capacity outcome was yet complete. [16] and[16,16] reported
WAIT_PREDECESSOR. Service restart count0 and one assigned GPU process only.
Contracts:bb2a9f3c9072,d443f9dfd52b,edf777576a9c.

Status (single read, no log polling):

```bash
python -m tools.selected_rl_capacity_sequence status --spec <private-capacity-root>/SEQUENCE.json
```

Provisional budget from prior matched runs:2-3h per arm to cap,6-9h total;
patience can shorten this. Re-estimate from completed pairs, not bare training
speed (full TRAIN/VALIDATION inference dominates runtime). No next owner
approval is required between these declared arms. Temporal fits stay paused.
