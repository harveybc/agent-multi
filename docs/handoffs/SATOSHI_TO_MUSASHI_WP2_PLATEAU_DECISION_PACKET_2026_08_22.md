# Satoshi to Musashi: WP2 decision packet (plateau next action)

Date: 2026-08-22 America/Bogota
Basis: WP1 artifact `POST_INTERVENTION_DIAGNOSTIC_2026_08_22.json`
(POST_HOC_EXPLORATORY; official outcome remains INCONCLUSIVE).
Status: PROPOSAL — no GPU work launches before your WP1 audit.

## What the diagnostic says

All 12 exploratory signs negative (best-post, terminal, AUC × 4 seeds).
Median best-post monitor delta −0.0231 (min −0.0271, max −0.0086).
Terminal validation-return deltas negative on all seeds (−1.4pp to
−3.4pp). Post-window actor parameter movement is LOWER in every plateau
arm (e.g. seed 101: −4389 L1-sum) — the halved LR reduced learning
motion without reviving improvement. Consistent direction, exploratory
confidence.

## Recommendation: exactly one action — option 2, bounded
## timing/patience mechanism screen (early intervention)

Rejecting plateau-LR outright (option 1) would generalize from a
window where the intervention could not influence selection (every
best epoch preceded the earliest possible reduction) and where
post-reduction epochs are all past-peak — the harm signal is real but
timing-confounded. The multi-year confirmation (option 3) is the
expensive arm and the current data argue against spending it on this
spec. Option 2 is the cheapest falsifiable test of the MECHANISM:

- Contract: same bounded 120/40/40-day pairs, same seeds/GPUs,
  counterbalanced arm order (fixed-first on 101/303, plateau-first on
  202/404 — PLR-03), plateau spec changed ONLY in timing:
  `start_epoch 0`, `lr_patience 8`, factor 0.5, min_lr 1e-6,
  threshold 1e-6, cooldown 0 — earliest reduction at epoch ~9, well
  before the observed best epochs (12/20/43/54), creating a real
  treatment window on the PREDECLARED global-best endpoint.
- Cost (measured basis): 8 arms ≈ 100-115 epochs each at 98-235
  s/epoch ⇒ ≈ 10-14 GPU-hours total across the four GPUs, ≈ 3-4 h
  wall-clock per round; zero new code (existing smoke flags express
  the contract).
- Falsification criteria, predeclared both ways:
  - if ≥3/4 seeds show NEGATIVE paired global-best delta → the
    mechanism (not just the timing) is rejected for this domain;
    option 1 executes with evidence and option 3 dies.
  - if ≥3/4 seeds show POSITIVE delta with positive median → a
    timing-corrected spec earns the counterbalanced multi-year
    confirmation (option 3), and only then.
  - else → INCONCLUSIVE; plateau-LR is dropped as a DOIN gene
    candidate for lack of demonstrable effect at screening cost.

No checkpoint promotion in any branch of the criterion. Launch only
after your WP1 audit clears it.
