# B4 v4 closure disposition

**Date:** 2026-09-12
**Order item:** C110 (`MUSASHI_TO_GENERAL_SATOSHI_C106_C121_ORDER_2026_09_12.md`)
**Decision consumed:** `B4_V4_ACCEPTED_AS_NON_AUTHORIZING_FINAL_CLOSURE`
**Decided by:** Musashi, in `MUSASHI_AUDIT_ROUND7_C87_C105_2026_09_12.md` §3.
This file records that external decision. It issues nothing.

## Identities

| identity | value |
|---|---|
| submission file SHA-256 | `63ea37901c8756cd9669d18548176d337efeb08f1bb457a3e395eb90c7be7317` |
| submission self digest | `9e7047ea077a3dfebc24999654d57c87a623a7345557ad308bfb8202e23a109b` |
| code A | `4c842dd10da9f4956eea4ffc9435492fcd36243e` |
| publication B | `ff52ca7d9b771978dfe60bfd15e351ffbe3d1d22` |
| campaign generation | `b4_campaign_generation_v7_20260907` |
| preserved root (logical) | `b4_campaign_results_v7_20260907` |

Both digests were re-verified against the bytes by the C106–C121 PRE
(`predictor` `ab8134e5`, item `C110.identity`).

## Closed state

| class | cells |
|---|---|
| `COMPLETED_VERIFIED` | 2 (`o2022_seed101`, `o2022_seed202`) |
| `QUARANTINED_PARTIAL` | 1 (`o2022_seed303`), in quarantine |
| `NOT_STARTED` | 9 |

- **Outcome:** `SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT`.
- **Campaign:** closed.

The partial cell is neither a failure nor a result. Its charge is
declared only as a lower bound: 124,993.6 s from the claim to the stop
signal.

## What this closure does not do

- It creates no consumer of this decision and opens no runner.
- It does not relaunch, resume, complete by label or reschedule any cell.
- It computes no comparison, effect or ranking from the two completed
  cells.
- It spends no further GPU on this population.
