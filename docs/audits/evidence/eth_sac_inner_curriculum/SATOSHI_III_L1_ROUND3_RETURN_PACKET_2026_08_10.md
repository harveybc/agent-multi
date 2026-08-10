# Audit Request: L1 Round-3 Finalization Return (findings 196-200)

From: General Satoshi III, technical lead
To: General Musashi, independent auditor
Basis: `MUSASHI_TO_GENERAL_SATOSHI_III_L1_ROUND2_CORRECTION_ORDER_2026_08_09.md`
No finding is closed by this return; Musashi verifies.

## 1. Exact commits (pushed, tree clean at each)

| Commit | Content |
|---|---|
| `2093cff5` | WP12 seal authority + WP13 exact GPU binding + WP14 financing/epochs; +15 tests, suite 898/898 |
| `f5e18696` | System manifest **v3** (sha `00bfa585…`) from CLEAN `2093cff5`, explicit financing bound; fleet deployed identical |

## 2. Findings 196-200 → corrections, tests and REAL evidence

| Finding | Correction | Real-chain proof |
|---|---|---|
| 196 | aggregation publishes OUTSIDE the seal (`<root>/aggregations/<exp>/`); post-publication rehash must equal the sealed digest or publication aborts | smoke v3 chain: sealed digest `6da390b5…`, aggregation output outside (0 aggregation files inside the seal), `sealed_digest_unchanged: true` printed by the CLI |
| 197 | `load_collection_envelope` = single authority in BOTH CLIs: COLLECTION_SEALED manifest, exact experiment, no refusals, matching replica proof, fresh sealed rehash == source digest == replica digest; direct CLI requires `--collection-root`, bare trees refuse | `TestAggregationAuthority` (5) + CLI refusal test (`AGGREGATION_REFUSED`, exit 3); envelope facts persisted in the aggregation result |
| 198 | `CUDA_VISIBLE_DEVICES` must EQUAL the contract assignment — unset/mismatch/cross-seed refuse before model construction; heartbeat v2 + record `gpu_binding` persist assigned/bound/observed facts | REAL v3 records: gamma seed-303 bound `GPU-b77fc3ad…`, torch sees **only** the 5070 Ti (count 1); seed-404 bound `GPU-a9f35631…`, sees **only** the 5090 (count 1); dragon/omega equally exact; launcher tests incl. two concurrent gamma workers |
| 199 | `financing_treatment` explicit in manifest v3 + validator (charged=false with mechanism+reason for the Backtrader screen; silence refuses) and persisted in every record's cost contract | v3 records carry `financing_treatment.charged=false` with reason; 2 refusal tests |
| 200 | `phase1_epochs_run` counts ONLY epoch>0; `phase1_baseline_evaluations` separate; alias carries the truthful trained count | REAL v3 records: requested 1 / realized 1 / baseline 1 (the old 2-count is gone); warm/no-warm one-/multi-epoch tests |

## 3. Reproducer (verbatim, sha `5f0e1b93…`)

BEFORE: 5/6 reproduced (restart counterexample correctly negative).
AFTER (code + new chain): `direct_aggregator_bypasses…` and
`financing_treatment…` flip to false at code level. The remaining three
read PHYSICAL state that order §1.2 commands preserved — the old
collection (finding-196 counterexample), the immutable old smoke
records, and the pre-correction deployed env files (since replaced).
Each is superseded by the smoke-v3 chain facts above; nothing is
absorbed silently. Outputs preserved beside the reproducer.

## 4. Suites

Focused L1 suites: 123 green. Full repository suite: **898 passed**
with the byte-clean guard. Bootstrap `FIXTURES_READY` at the doin-node
pin.

## 5. Corrected smoke (WP15) — identity `7aae043107a87554`

- Fleet: `f5e18696` identical full commit + gym-fx `efa49160…` on all
  three nodes; env files materialized with `--smoke` AND the exact
  per-seed `CUDA_VISIBLE_DEVICES` (printed at deployment).
- Four systemd workers ran concurrently (fresh heartbeats with binding
  facts), **16/16 records, zero failures, 4× SEED_COMPLETE**.
- Collection: `COLLECTION_SEALED`, zero refusals, sealed tree digest
  `6da390b5afe837af52f4b5574b017dc574d308a82bacd08ab799120262fd318f`;
  replica on dragon: whole-tree digest computed ON the replica,
  **equal**, **16/16 terminals loaded there**.
- Envelope aggregation from the sealed root, published OUTSIDE the
  seal: **INCONCLUSIVE with all 16 cells refused as
  `mechanics_smoke`** (never decision-eligible) and
  **`sealed_digest_unchanged: true`** re-proven after publication.

## 6. Decision preflight (order §5.5)

- All four env files atomically replaced (`mktemp` + `mv`) with empty
  `L1_EXTRA_ARGS` plus their exact GPU UUIDs; contents printed (no
  secrets) on omega/dragon/gamma.
- Decision identity recomputed at `f5e18696`:
  **`2de49ea9225e2baf`** (contract `4171aa57…`, manifest v3
  `00bfa585…`). Prior `dce2903ce0d25ca5` RETIRED, never launched.
- Verified: **no prior directory** for `2de49ea9225e2baf` on any host;
  all four units loaded, inactive, not failed — ready to start.

## 7. Identities, explicitly separate

| Purpose | Identity | State |
|---|---|---|
| Diagnostics | `16acf854…` | preserved, never aggregable |
| Old smoke + collection | `13bfdb1a…` | preserved as finding-196 counterexample (order §1.2) |
| Restart proof | `39af1c33…`/`fcdf62cd…` | preserved on dragon |
| Corrected smoke v3 | `7aae043107a87554` | sealed `6da390b5…`, replica-verified |
| **Decision (ready)** | **`2de49ea9225e2baf`** | preflight complete; starts immediately after Musashi's reproduction — standing authorization, no new owner phrase |
