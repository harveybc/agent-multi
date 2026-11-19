# 384: directional supervised probes, not four SAC agents

Conditional diagnostic, not a release. Use the frozen 299 actor's exact
pre-head representation from 71 market channels, window 13, stride 2.
One 6-hour profile first. ZigZag is offline label data only. Per split and
uninterrupted session, use the verified 6h hindsight pivots. At the next
executable opening bar, valley/peak membership has inclusive left/right
fractions of the adjoining leg lengths (default .25/.50, floor in ticks).
Score only the region between the second and penultimate pivots, never
silently label unresolved edges as negative. This is not an order command.
The pivot center has zero timing offset, not zero predictive uncertainty.

Flat, admissible: paired BUY and SELL classifiers for window membership.
Also regress each action's net marked-equity advantage against waiting
(24 valid decisions, same source continuation, native re-entry and costs,
basis points of current marked equity). Occupied: only the corresponding
close head is eligible. Binary target = closing has strictly positive
account advantage vs HOLD; regression target = signed advantage in bp.
External lookup of market-produced grids at SL/TP distance and time to
force-flat; position coordinates never enter the market encoder. Use the
source trajectory, no synthetic positions, no extra NOP expert.

Annual TRAIN episodes split at each boundary: 2020-21 fit; 2022 H1 monitor,
2022 H2 calibration diagnostics; 2023 internal evaluation. The inherited
encoder, feature selection and pivot parameters have seen these periods;
this is a conditional head experiment only. Fixed defaults (not tuned):
32 batch, 8 hidden units, variational Gaussian final layer, seed 19.
Binary log loss + standardized advantage squared error + KL per fitting
row. Monitor predictive log loss, also track regression MAE independently.
Maximum 2000 epochs; patience 80; restore best model and Adam on plateau
every 20 misses and divide current LR by 5, minimum improvement 1e-8.
Report per-epoch TRAIN/monitor, frozen fit prevalence Brier/log loss,
fit median absolute-error baseline, reliability bins, posterior decision
dispersion, counts and per-role curves. These alone cannot pass the account
gate. No claim of calibrated profit probability from a pivot classifier.
