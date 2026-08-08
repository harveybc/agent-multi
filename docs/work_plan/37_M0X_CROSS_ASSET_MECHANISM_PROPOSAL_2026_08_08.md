# 37. M0-X: Cross-Asset Replication of the Inner-Curriculum Mechanism

Status: PROPOSAL for owner–Musashi discussion — v1.0.0, 2026-08-08.
Author: General Satoshi III. Decision authority: owner, with Musashi's
experimental-lead verdict. Nothing here is scheduled or launched.

## 0. Origin, stated for the record

The mechanism M0 just confirmed descends from an accidental discovery:
removing the margin call in the old gym-fx under a NEAT agent showed
that early solvency relaxation lets a policy LEARN TO ACT before
learning to survive. M0 proved the same principle operates one level
down — inside SAC weight learning — provided the return to reality is
gentle (reduced normal LR). This proposal addresses the owner's
question: where does cross-asset validation of that mechanism belong
in a work plan that deliberately committed to ETH?

## 1. What M0 established — and its exact scope limits

Established (16/16 verified, 21/21 replicated, pending Musashi audit):
equal-compute normal-only fine-tuning at LR 1e-4 kills activity in the
first epoch; easy pretraining alone does not save it; easy plus
reduced-LR normal fine-tuning (3e-5, 1e-5) retains an active,
weight-changed, loadable terminal in 4/4 seeds.

NOT established: anything beyond ETHUSDT 4h, one observation contract
(83 features), one anchor family, fixed entropy 0.2, one replay
regime, activity-not-profit. If any of those conditions is load-
bearing, the R3 SAC-dynamics genes we are about to design would
inherit an ETH-shaped constant as if it were physics.

## 2. The strategic question

The ETH-focus discipline exists to prevent scope explosion, and it
worked. But the M0 result is a MECHANISM claim, and mechanism claims
have a property decisions like D1 do not: they are cheaply
falsifiable on a second system. One more asset answers the question
"is this SAC-under-distribution-shift physics, or ETH folklore?" for
about half an hour of fleet time. That answer changes what R3's genes
mean and what the portfolio milestone can assume.

## 3. Three options, one recommendation

**A — ETH-only, defer cross-asset to roadmap §12 (status quo).**
Cheapest now. Risk: R3 gene bounds (easy_epochs,
normal_finetune_lr_multiplier) get calibrated on one asset; if the
mechanism is conditioned, we discover it AFTER building the
optimization domain on it, at maximum rework cost.

**B — Multi-asset pivot now.** Rejected by me. It dissolves the ETH
discipline, multiplies every audit surface, and starts portfolio work
we have no mandate for.

**C — RECOMMENDED: M0-X, a bounded cross-asset mechanism replication
that never blocks the ETH critical path.** One additional asset, the
SAME frozen 4-arm screen, run when the fleet would otherwise idle
between ETH jobs (the no-idle mandate and this proposal are natural
allies). ETH remains the only DECISION track; M0-X is a falsification
probe with exactly two outcomes, both valuable:

- **Replicates** → the mechanism is general. R3 genes enter with
  cross-asset priors; §12 multi-asset onboarding gains a doctrine:
  per-asset recipe = anchor → easy → gentle normal. The portfolio
  milestone inherits a de-risked training primitive.
- **Fails to replicate** → the mechanism is conditioned. We learn it
  BEFORE R3 gene design instead of after, and §12 learns that
  curriculum constants are per-asset calibration parameters, not
  portable defaults. A negative here is cheap insurance, not a loss.

## 4. Concrete M0-X design (bounded, reuse-only)

- **Asset: USDCAD 4h.** Lowest cost by inventory: phase-1 SAC
  champions exist as anchors (`full_genome/usdcad_4h`,
  `evidence-e4/long_horizon__usdcad__4h__seed2703`), the dataset
  lineage and observation contract exist from phase-1, and it is
  maximally different from ETH where it matters (FX vs crypto,
  volatility regime, session structure) — a strong generalization
  test. BTC would need dataset+anchor creation: deferred.
- **Contract:** a NEW frozen `m0x_usdcad_contract.json` — same four
  arms, same equal-compute rule, same easy LR pin, USDCAD's OWN
  observation/feature contract (the mechanism test requires internal
  consistency per asset, not feature identity across assets — this is
  itself part of what M0-X measures).
- **Precondition gate (cheap, CPU):** verify anchor loadability,
  anchor activity on USDCAD validation (an anchor that never trades
  cannot measure activity collapse), dataset manifest hashes, and a
  no-future-leak split mirror of the ETH rules. If the gate fails,
  M0-X is refused, not improvised.
- **Runner:** `eth_sac_inner_curriculum_screen.py` generalized to take
  `--contract` (asset-agnostic); the D1-helper reuse and the hardened
  16-record aggregator apply unchanged. No new architecture, no new
  pipeline, no gym-fx changes.
- **Budget:** 4 seeds x 4 arms x ~7 min ≈ 30-40 min of fleet time,
  measured from M0's real durations.
- **Discipline:** same 8-class verification, replica topology,
  no-test-year evaluation, typed absences, no positive-profit gate.

## 5. Proposed work-plan modification (for the discussion)

1. Insert **M0-X** as a parallel item beside M1 (not on the critical
   path): `M1 (ETH confirmation) -> R3 gene design` proceeds exactly
   as ordered; M0-X runs opportunistically on idle fleet windows and
   must COMPLETE before R3 gene bounds are FROZEN (that is its only
   coupling to the critical path).
2. Amend roadmap §12 (multi-asset) from "later exploration" to
   "consumes M0-X": if replicated, §12 starts from the onboarding
   recipe; if not, §12 starts from per-asset calibration.
3. Add a standing principle to the roadmap: **mechanism-level findings
   (level-1 training dynamics) are candidates for cheap cross-asset
   falsification BEFORE they parameterize optimization domains;
   decision-level findings (which config wins) remain ETH-only until
   the portfolio milestone.** This is the general answer to "does the
   rest of the ETH roadmap transfer?" — transfer the mechanisms after
   falsification tests; never transfer the decisions.
4. Portfolio note (no work now): the recipe anchor→easy→gentle-normal
   becomes the hypothesized per-asset onboarding primitive for the
   portfolio milestone; M0-X is its first evidence either way.

## 6. What I ask Musashi to rule on

1. Approve/reject M0-X as scoped in §4 (asset, gate, budget, coupling
   rule "before R3 bounds freeze").
2. Whether M0-X requires its own D1-style anchor provenance (phase-1
   champions carry full lineage) or the anchor-manifest gate (finding
   158 discipline) suffices.
3. Whether the LR winner for M0-X arms should mirror ETH's (3e-5,
   1e-5) exactly for comparability, or scale with the asset's observed
   gradient magnitudes (I propose: mirror exactly — comparability
   first, calibration only if it fails).
4. Sequencing relative to M1 and the M0 audit: my proposal is M0
   audit -> M1 launch -> M0-X in the first idle window.
5. Ownership of the generalized screen contract schema (one schema,
   per-asset instances) under the tool-declaration discipline.

## 7. What this proposal does NOT do

No portfolio implementation. No BTC/new datasets. No change to D-track
ETH decisions. No launch before the M0 audit verdict and the owner's
explicit approval of this document. No modification of any frozen M0
evidence or contract.
