# Musashi Note: Preserve M0 Focus and Observe IBKR Recovery

Date: 2026-08-07 America/Bogota
From: General Musashi, independent reviewer and orchestration lead
To: General Satoshi III, SAC inner-curriculum implementer
Authority: owner-approved M0 work order; this note changes no trading or DOIN
runtime state

## 1. Continue the Assigned SAC M0 Work

Do not switch to Front 3 implementation. Its social enrichment worker is now
an independent CPU-only service and does not need your intervention. Continue
the work in
`MUSASHI_TO_GENERAL_SATOSHI_III_SAC_INNER_CURRICULUM_ORDER_2026_08_07.md`.

Before editing, finish the two unresolved code-path facts required by that
order:

1. identify the exact function and line path that decides whether a SAC
   checkpoint is eligible for continuation or promotion; and
2. identify the exact diagnostics exported by the environment bridge for
   termination cause, margin state, equity, trade count and action activity.

Record unknown or absent diagnostics honestly. Do not invent an adapter field
to make the map look complete.

## 2. IBKR Recovery Fact

At 2026-08-07 10:37 America/Bogota, direct local evidence showed:

- TWS Paper listening on port 7497;
- account binding accepted and `read_only=false`;
- model runner fresh on the latest closed USD.CAD H4 bar;
- current model action `short`; and
- decision refused as `halted:hold`.

This proves connectivity, not permission to add exposure. The recovery hold
remains authoritative until the existing LTS reconciliation and owner-clear
path proves broker position/order state and clears it. Do not bypass, clear or
replace that hold, and do not submit an order merely to demonstrate liveness.
If the fact changes while you work, report the direct evidence and continue M0;
do not take ownership of the live runner.

## 3. Resource Boundary

The Front 3 worker is hourly, CPU-only, low priority and memory-capped. It must
not consume a GPU or interrupt SAC/DOIN work. If measured contention appears,
report the timestamp and process facts; do not disable unrelated services by
assumption.

## 4. Required Return

Return the M0 code-path map, proposed minimal implementation boundary, test
fixtures and explicit unknowns. Do not claim any curriculum result before an
actual paired measurement exists.
