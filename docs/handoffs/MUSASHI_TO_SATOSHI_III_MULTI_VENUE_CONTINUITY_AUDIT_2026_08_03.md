# Musashi to Satoshi III: Multi-Venue Continuity Audit

Date: 2026-08-03 America/Bogota
Role: independent read-mostly verification
Order authority: none; do not place, cancel or alter any broker order

Satoshi III, independently audit the selected-model Paper/Demo continuity
changes without repeating the implementation claims as proof.

## Required Reading

1. [audit report](/home/harveybc/Documents/GitHub/agent-multi/docs/audits/AUDIT_SATOSHI_III_IBKR_L1_F0_AND_LIVE_CONTINUITY_2026_08_03.md)
2. [continuous demo operations](/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/29_CONTINUOUS_DEMO_TRADING_OPERATIONS.md)
3. [implementation ledger](/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md)
4. [findings register](/home/harveybc/Documents/GitHub/agent-multi/docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md)
5. `lts@8b67235` (reconnect correction begins at `cffdc13`)
6. `prediction_provider@78f0af5`

## Required Verification

1. Reproduce finding 075 on the parent commit, then verify `cffdc13` joins a
   completed order only to an execution with the same broker permanent id.
   Missing or mismatched execution evidence must remain missing, never become
   a synthetic filled parent.
2. Reproduce the focused reconnect fixtures and the complete LTS suite.
3. Verify Alpaca, IBKR and MT5 runner heartbeats are written atomically and
   their stop paths do not wait for a full polling interval.
4. Verify model-manifest hot reload is fail-closed: invalid hashes cannot add
   exposure; a changed model drains prior exposure; actual post-close broker
   cash/equity starts the replacement session.
5. Verify the IBKR profile and mandate agree on 25,000 units, the IDEALPRO
   minimum, the stated exposure caps and a maximum 0.00625% risk at stop.
6. Audit the MT5 execution EA source and bridge contract before runtime
   acceptance. After the owner installs it, verify bridge version, bar flow,
   command acknowledgement, native SL+TP and restart idempotency directly.
7. Confirm no Live account, secret, LLM or Hermes path can mint an execution
   decision or bypass the selected-model and risk contracts.
8. Reproduce findings 076-078: temporary quote loss must not restart the
   runner or submit; near-terminal fills must satisfy exact conservation; and
   new L0 payloads must identify the actual venue without rewriting historical
   evidence.

## Deliverable

Write an append-only audit report with exact commits, tests, direct evidence,
new findings and explicit unknowns. Use clickable absolute file links. Do not
close finding 075 or any finding you helped implement; recommend disposition
to the owner instead.
