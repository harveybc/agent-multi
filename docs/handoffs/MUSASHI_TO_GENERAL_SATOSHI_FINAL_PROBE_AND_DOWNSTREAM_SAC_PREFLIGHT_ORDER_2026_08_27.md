# Order: Final Neutral Probe and Downstream SAC Preflight

Date: 2026-08-27 America/Bogota
From: General Musashi
To: General Satoshi
Priority: P0; CPU first, then return for GPU dispatch

## P0 -- Permanent adversarial tests (373)

Add tests for five-way ordering/purge, target boundary crossing, frozen cached
encoders, adapter-only gradients, all five target formulas, missing/degenerate
probes, cardinality-invariant ranking, deterministic replay and monitor/2022/
outer/sealed inaccessibility. Mutating probe-score must never alter adapter
training; mutating monitor must not alter any routing result.

## P1 -- Valid adapter fitting (371)

Split the existing probe-fit block causally into adapter-train and adapter-val,
with purge. Keep probe-score untouched until final scoring. For each task:

- predeclare max 2,000 steps, minimum 200, validation cadence 20 and patience
  200 steps;
- checkpoint and restore the best adapter by adapter-val loss;
- require finite learning curves and a declared minimum improvement over the
  initial adapter;
- use three fixed adapter seeds and report median plus dispersion;
- reject material seed instability rather than selecting the best seed.

These numbers are probe-training mechanics, not economic hyperparameters. Commit
them before execution and do not change them after seeing probe-score.

## P2 -- Neutral baseline and specialist ceiling (372)

Evaluate the same common probes for:

- random strong-architecture encoder, the neutral floor;
- solo specialist encoders, diagnostic ceilings;
- the four existing routes.

For loss probes compute normalized skill with random=0 and solo=1:

`skill = (loss_random - loss_route) / (loss_random - loss_solo)`.

Refuse an ill-ordered or near-zero denominator. Preserve raw losses. The solo is
no longer a hard 1.2 eligibility gate. Predeclare route ranking per family:

1. no predictive probe materially worse than random across adapter seeds;
2. maximize median normalized skill across quantile, volatility and barrier;
3. then median skill across all five probes;
4. ties within the frozen tolerance are `INCONCLUSIVE`.

Reconstruction and contrastive remain representation diagnostics; the three
forward probes lead because the downstream business task is trading.

## P3 -- One final CPU routing run

Execute the existing 20 routes with the corrected common probe protocol. Do not
retrain routes merely because an old ratio looked bad; preserve identical route
training identities. Return raw/normalized distributions and the selected or
inconclusive route per family. If any family is worse than random on a predictive
probe, use full5 as the conservative candidate for that family and label the
choice diagnostic, not proven optimal.

## P4 -- Materialize the downstream test, do not launch

Create one candidate routed pretraining generation from P3 and bind its real
seal. Materialize the smallest informative SAC comparison:

- same strong architecture and all contracts;
- random initialization control versus pretrained-finetuned treatment;
- four paired seeds, counterbalanced order;
- full 260k-step budget, activity/dead-policy refusals and Alpaca cost envelope;
- primary endpoint and INCONCLUSIVE rule frozen before execution;
- frozen-encoder arm deferred unless fine-tuned pretraining shows signal.

Estimate GPU-hours and prepare the executable driver with CPU dry-run and
identity tests, but mark every GPU command `NOT_LAUNCHED`. After independent
audit, Musashi will dispatch this downstream screen. No further proxy-method
cycle will intervene unless P1/P2 themselves fail mechanically.

Live Alpaca and MT5 remain untouched.
