# 37. Mechanism Falsification Track: M1 Factorial + M0-X Cross-System Probe

Status: v2.0.0, 2026-08-08 — **owner accepted option C; Musashi's six
mandatory corrections APPLIED and implemented.** v1's proposal survives
in spirit; its design errors are corrected below and named.
Author: General Satoshi III. Implementation branch:
`satoshi/m0-aggregation-hardening` (worktree; active checkouts untouched).

## 0. Origin, for the record

The mechanism under test descends from an accidental discovery:
removing the margin call in the old gym-fx under a NEAT agent showed a
policy can learn to ACT before learning to survive. M0 carried the
principle into SAC weight learning. This document now encodes Musashi's
correction of my inference error before it could become a gene.

## 1. The confound Musashi caught (v1's central defect)

M0's arms were {normal-only@1e-4, easy→normal@1e-4, easy→normal@3e-5,
easy→normal@1e-5}. **There was no normal-only reduced-LR arm.** M0
therefore proved that reduced LR preserves activity AFTER easy — it did
NOT prove easy contributes causally. The entire effect may be the LR.
v1 of this document inherited that ambiguity and so did the original M1
design (order §10). Freezing R3 genes on that evidence could have
encoded folklore as a gene.

## 2. Corrected sequence (Musashi's ruling, verbatim in structure)

1. **Musashi completes the independent M0 audit.**
2. **M1 as a 2×2 factorial, equal compute (14 epochs/arm):**
   {N14, E4_N10} × {LR mult 1.0, winner mult} — measures the easy main
   effect, the LR main effect and their interaction directly.
3. **M0-X on USDCAD:** if M1 shows easy contributes → test curriculum
   transfer; if only LR matters → test gentle-fine-tune dynamics and
   attribute nothing to easy.
4. **Only then freeze R3 gene bounds.**

## 3. Implemented artifacts (this branch, tested)

- **Generic schema** `agent_multi.inner_curriculum_screen_contract.v2`
  in `tools/eth_sac_inner_curriculum_screen.py` (correction 5): ONE
  schema, per-asset contract instances; **LR multipliers over the
  anchor's own baseline** (correction 1), resolved to absolute rates at
  load; typed refusals before model construction (bool-as-number, NaN,
  nonpositive, negative epochs, unequal compute); `launch_eligible:
  false` blocks execution until the audit selects the winner. The
  frozen M0 v1 contract loads unchanged through the v1 path.
- **M1 instances:** `m1_factorial_contract_M03.json` / `_M01.json`
  (winner 0.3 / 0.1 over ETH baseline 1e-4) — both materialized,
  neither launch-eligible; the audit picks one.
- **M0-X instances:** `m0x_usdcad_contract_M03.json` / `_M01.json` —
  2×2 at M0 scale (2 epochs/arm) over USDCAD baseline
  **3.55936e-4** (the anchor's own training rate, exact), arms at
  multipliers {1.0, winner}.
- **Immutable anchor manifest** (correction 2):
  `docs/audits/evidence/eth_sac_inner_curriculum/USDCAD_SEED2703_ANCHOR_MANIFEST.json`
  — artifact sha `f40dfca1…`, full lineage (48 obs columns, window 1,
  1,604 train rows, entropy `auto`, dataset sha prefix `98a3cc73`),
  activity facts attributed to Musashi's audit (15 validation trades,
  −0.0720%, Sharpe −1.69) with **mandatory re-proof at the launch
  gate**, and the declared limitation: **one verified anchor reused
  across all four seeds** — seed variation measures fine-tune
  stochasticity, not anchor diversity. The `full_genome` smoke champion
  is REJECTED (zero trades).
- **INCONCLUSIVE** (correction 3) is a first-class outcome in both
  contracts' interpretation tables: gate failures, unverifiable
  records, anchor re-proof failures, or mixed cells meeting no rule
  produce NO conclusion and escalate — nothing is forced into
  "replicates/fails".
- Tests: 32 green in `tests/test_eth_sac_inner_curriculum_contract.py`
  including factorial shape, multiplier resolution, USDCAD-baseline
  distinctness, single-anchor declaration, launch blocking, and v1
  back-compatibility.

## 4. Corrected budget (correction 4)

M0 measured: ~81 aggregate GPU-minutes (~30–40 wall-clock minutes on
four parallel GPUs). M1 factorial is 7× M0's per-arm epochs (14 vs 2):
~9.4 GPU-hours aggregate, ~2.4 h wall-clock on four GPUs. M0-X mirrors
M0's scale: ~81 GPU-minutes aggregate, ~30–40 min wall-clock.

## 5. Epistemic limits (correction to v1's §3 framing)

ETH and USDCAD differ in features (83/48), window (32/1), training
observations (13,699/1,604) and entropy mode (fixed/auto). M0-X
therefore measures transfer between two COMPLETE systems and cannot
isolate the asset factor. A replication concludes **"not falsified in
a second system"** — never "universal mechanism". A failure concludes
"system-conditioned", which converts R3 bounds into per-asset
calibration parameters.

## 6. What stays deferred (correction 6)

anchor → easy → gentle-normal is NOT portfolio onboarding doctrine. It
is a hypothesis with one supporting system and, pending M1, an
unresolved attribution. The portfolio milestone consumes whatever M1 +
M0-X establish, at that time, with Musashi's verdict attached.

## 7. Standing principle (unchanged from v1, restated)

Mechanism-level findings earn cheap cross-system falsification BEFORE
parameterizing optimization domains; decision-level findings stay
ETH-only until the portfolio milestone. Transfer mechanisms after
falsification; never transfer decisions.
