# Critic Coverage And Small-Set Fit Results

## Verified Execution

Design246 completed 2026-11-09T22:39:10.867959Z (17:39 Bogota), in 472.96
seconds. Runtime `04c2d215`. Four arms completed; 26 terminal files authenticated.
Worker and automatic collector exited0. Identity:
`fe5e5cb33a14cb201e7aa83e8c7b2ca3c7b958928b49e7c495f5c05ee270d1a6`.

Actor updates: zero. Full TRAIN actor logits and frozen parameters remained
exact in all four arms. Original69/70 checkpoint hashes remain unchanged.
No model promoted, no new trading return inferred from critic fit metrics.

Full evidence: `/var/tmp/sac70-critic-small-fit-20261109`.
Public subset: [curves](generalization/critic_small_fit_20261109/CURVES.csv),
[plot](generalization/critic_small_fit_20261109/FIT_CURVES.png).
SOURCE_DONE authenticates the full source collection, not just this subset.

## Label And Coverage Audit

Both replicas have 284 fit anchors and 95 monitor anchors within TRAIN.
The original labels use native greedy-parent continuations. Replaying one
anchor per year and all three actions gave 24 suffix simulations across both
replicas, with maximum discrepancy 0.0 in returns and cash. Reference account
replays also matched. No exact duplicate input windows or contradictory exact
duplicates were found in these panels. NOP/same-direction holding labels are
equivalent as required; greedy soft-return equals reward-return.

| Fit-anchor class | r19 | r43 |
|---|---:|---:|
| Flat, BUY and SELL permitted | 80 | 74 |
| Flat, entries restricted by calendar | 16 | 22 |
| Open long | 92 | 92 |
| Open short | 96 | 96 |

Restricted flat states have no informative action pair in these panels.
They are valid examples of action equivalence, not corrupt labels. Keeping
them separate prevents their count from overstating entry-decision coverage.
The audit checks this panel, not every label in every historical experiment.

## Fixed 4096-Update Endpoint

Each small set contains 32 permitted flat-account states, eight from each of
the four TRAIN years, selected without looking at model performance. The
architecture, initial checkpoint, optimizer configuration and objective are
identical across scopes. Only critic trainability changes.

Metrics below use the minimum of the two critics. Accuracy means correct
pairwise ordering of action returns, NOT trade win rate or prediction accuracy.

| Replica | Trainable scope | Small-set pair accuracy | Small-set gap RMSE | Small-set max regret | Monitor pair accuracy |
|---|---|---:|---:|---:|---:|
| r19 | Heads only | 86.46% | 0.338638 | 0.433598 | 53.57% |
| r19 | Critic representation + heads | 100.00% | 0.000021 | 0.000000 | 54.08% |
| r43 | Heads only | 85.42% | 0.356522 | 0.948321 | 60.70% |
| r43 | Critic representation + heads | 100.00% | 0.001334 | 0.000000 | 54.23% |

Gap RMSE and regret are in discounted-reward units. Both individual twins
also satisfy the predeclared small-set criterion in the representation arm;
neither heads-only arm satisfies it. Four parameter tensors change in each
heads-only arm versus 311 in each representation arm.

The heads-only twins differ substantially: twin0 pair accuracy is 97.92% and
96.88%, versus twin1 at 77.08% and 72.92%. Taking their minimum does not remove
these calibration differences. This is a diagnostic observation, not grounds
to discard a twin or replace clipped double-Q without a separate contrast.

## Interpretation And Limits

The current architecture can fit these real-label examples nearly exactly
when its critic representation can adapt. Freezing that representation
restricts fitting under this budget; increasing network size is not needed
to pass this particular small-set test. This does not prove heads could never
fit with another budget, optimizer or initialization, or identify which encoder
component accounts for the difference.

The successful small-set fit is intentional memorization. It is not a
generalization or profitability result: monitor performance remains mixed
and deteriorates against heads-only in r43. Moreover the monitor includes
open-position states while fitting used only flat states, and its return
trajectories overlap with other anchors. Do not interpret its aggregate as
an independent same-distribution generalization estimate. Subsequent reports
must separate flat entries, open-long and open-short decisions.

## Next Contrast

Use a broader, explicitly stratified TRAIN label panel and compare critic
heads-only versus trainable critic representation under the same budget and
loss. Report held-out diagnostics by operational state and keep the actor
frozen until the critic comparison is complete. Any temporal separation must
also prevent label continuations from crossing the fit/monitor boundary.
Then, separately, test whether calibrated critics improve the same bounded
actor refinement. Do not adopt these 32-example critic checkpoints as trading
models or start another feature-removal pass on that assumption.

This report launches no further training; the bounded queue has completed.
