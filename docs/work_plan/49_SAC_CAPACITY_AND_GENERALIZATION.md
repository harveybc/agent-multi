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
