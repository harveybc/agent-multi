# M0→L1 Mechanism Ladder — Result (finding 220, order §3)

Date: 2026-08-11 · Diagnostic identity: `97c0bb29e82dfea3` · Seed: 101
only · Contract:
`examples/config/phase_3_eth_sac_dynamics/m0_l1_mechanism_ladder_v1.json`
(sha256 `c26fbed45581c2a2322162f30497b1be32fd631067953c4da5d618cf4ed4e604`)

## Question (from the contract)

Which single mechanism difference between the M0 recipe (screen
E1_N1_LR03, activity survived) and the L1 protocol (16/16 total
activity collapse, sealed result `2de49ea9225e2baf`, INCONCLUSIVE)
is sufficient to reproduce the collapse?

## Five-row contrast table (order §3.5)

Terminal facts are the as-run validation split of each arm's terminal
evaluation; every training arm ran the same seed-101 anchor
(`cb27375c…`, policy tensor `e747b893…`), the same data windows and
19000 phase-2 gradient updates.

| Arm | Only delta vs. its baseline | Handoff semantics | best_easy_epoch | raw action std | non-hold rate | trades (train/tail/val) | protected entries | Activity survived |
|---|---|---|---|---|---|---|---|---|
| D0_M0_EXACT | — (positive control, M0 recipe exact) | `m0_epoch0_eligible_v3` | 0 (anchor handed) | 0.0563 | 1.0 | 424 / 5 / 122 | 123 | **YES** |
| D1_EVALUATOR_ONLY | L1 activity DEFINITION applied to the D0 terminal (CPU, no training) | n/a | n/a | n/a | n/a | n/a | n/a | label `active` under BOTH definitions — `labels_agree: true` |
| D2_BOUNDARY_ONLY | boundary/reload semantics only: `phase1_handoff_semantics → l1_trained_epoch_v4` | `l1_trained_epoch_v4` | 1 (trained weights handed) | 0.0 | 0.0 | 0 / 0 / 0 | 0 | **NO** |
| D3_COST_PROTECTION | D2 + L1 normal cost & protected-entry contract | `l1_trained_epoch_v4` | 1 | 0.0 | 0.0 | 0 / 0 / 0 | 0 | **NO** |
| D4_FULL_L1 | D3 + L1 patience/stopping (full L1) | `l1_trained_epoch_v4` | 1 | 0.0 | 0.0 | 0 / 0 / 0 | 0 | **NO** |

Easy-phase behaviour was identical across all four training arms
(same seed, same anchor: 759 trades in easy epoch 0, 230 in easy
epoch 1) — the arms differ only in what crosses the phase boundary
and in the declared downstream deltas.

## Fail-closed interpretation (§3.4, applied rule by rule)

1. D0 reproduced M0 activity → the ladder is VALID.
2. D1 does NOT change the label (`active` under both the M0 and the
   L1 definition) → the activity definition is NOT the defect.
3. First active→inactive transition: **D2_BOUNDARY_ONLY**. The single
   proven delta — the easy→normal boundary handoff reloading the
   genuinely easy-TRAINED policy instead of the anchor — is
   sufficient for total activity collapse (std 0.0, zero non-hold
   actions, zero trades, zero protected entries) under otherwise
   exact M0 costs, floor and stopping.
4. D3 and D4 add the L1 cost/protection contract and the L1
   patience/stopping on top of an already-total collapse; they add no
   further measurable effect at this seed.

**Mechanism named:** the phase-1→phase-2 boundary handoff. M0's
reported survival is an artifact of the v3 `epoch0_eligible`
selection: with `best_easy_epoch=0` the boundary handed the pristine
anchor (proven: post-easy policy tensor hash equals the anchor tensor
hash `e747b893…` in the D0 record's `boundary_transfer_evidence`).
When the boundary hands weights actually trained on the easy phase
(v4, `best_easy_epoch=1`), the policy arrives at the normal phase
already collapsed toward hold and never recovers within the epoch
budget. The candidate for the next focused experiment (one delta,
subject to a new order — not launched): what easy-phase training does
to the policy/entropy that survives the boundary.

## What this result does and does not say

- Seed 101 only: it locates a deterministic mechanism; it supports no
  superiority decision and no protocol change without paired
  replication (contract rule 5).
- It NEVER relabels the sealed L1 factorial result
  `2de49ea9225e2baf`, which remains INCONCLUSIVE / total activity
  collapse under its exact protocol.
- D2's collapse carries M0 costs (no protection requirement): the
  collapse is not produced by the cost/protection contract.

## Custody

- Sealed collection root (operator-local, private):
  `~/.local/share/agent-multi/ladder_collection_97c0bb29_20260811/`
  — staged per assigned host (omega D0, dragon D2, gamma D3/D4),
  validated (one diagnostic identity, unique arm identities, uniform
  contract hash, terminal-artifact hashes), sealed with per-file
  manifest.
- Collection tree digest:
  `cdb6ef9947887992fc0a133a8c66adb76d64a4484cccb5cfc9f63fbea1c2ed8e`
- Replica: whole sealed tree on **dragon** (same path), digest
  recomputed ON the replica — equal; terminal artifacts rehashed and
  really loaded there.
- Published table (OUTSIDE the seal, post-write digest re-proof
  equal):
  `docs/audits/evidence/eth_sac_inner_curriculum/M0_L1_MECHANISM_LADDER_CONTRAST_2026_08_11.json`
- D1 evaluator record sealed inside the D0 subtree
  (`d1_evaluator_record.json`).
- Prior attempt preserved read-only under identity
  `177c32c6a75bee0d` (same output root): D0 ACTIVE there too —
  reproduced twice — D4 inactive-typed; D2/D3 ARM_FAILED under the
  legacy no-eligible-checkpoint raise that motivated the uniform
  `inactive_terminal_is_typed_result` fix (commit `6b8e2bab`); the
  unified rerun executed at commit `8fd05bcb` on every host.

## Reproduction

```
# per arm, on its contract-assigned host/GPU
python tools/m0_l1_mechanism_ladder.py --arm <ARM>
# CPU relabel of the D0 terminal
python tools/m0_l1_d1_evaluator.py --d0-record <root>/<id>/D0_M0_EXACT/ladder_arm_record.json
# sealed collection + replica + published table
python tools/m0_l1_ladder_collect.py --diagnostic-identity <id> \
  --collection-root <fresh root> --replica-host dragon \
  --publish-table docs/audits/evidence/eth_sac_inner_curriculum/M0_L1_MECHANISM_LADDER_CONTRAST_2026_08_11.json
```
