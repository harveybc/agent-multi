# 408. Frozen comparison: opening versus waiting with the actual closer

Fixed classifier, four rolling real historical years, 61-channel 406 encoder,
seeds 7/19/43, 3/6/12/24 h; do not promote. Head-level chronological OOF only:
the frozen upstream features, scaling, risk and encoder are retrospective.
For score year Y=2022/2023, train the 406 fine recipe on [Y-4,Y), monitor
only the preceding July-December; purge labels ending at/beyond each split.
Keep the fine endpoint 20 epochs after the second actual champion rollback;
do not train an exploratory branch. For 2024 use the existing 406 fine quartet.

For each legal flat origin replay BUY and SELL at every horizon with native
fills, costs, broker protections and the frozen same-seed directional closer.
NOP is worth the same continuation: once per session at first legal observation
at least 36 h after operational start, if available after the branch take the
sign of the preceding three bars with the SAME 6h profile and its learned
closer. No reentry otherwise. If the opportunity is occupied it is lost.
Outcomes at a truncated session/split are ineligible. Target = terminal equity
difference versus NOP / initial balance in bp. These overlapping hypothetical
positions do not sum to a tradable account. Report missed gains / avoided losses
only on matched outcomes; final account equity is the economic measure.

Frozen calibration rule: monotone affine least squares of managed advantage on
the raw opening probability independently for each side and horizon, with no
additional threshold, no validation-fitted hyperparameters. Clamp probability
to training range; if constant or fitted slope nonpositive use training mean.
Use 2022 to predict 2023 first; pool the two OOF blocks for the frozen 2024 map.
Compare exact current router vs calibrated router at identical openings,
closers, tie handling, calendar and costs. Score abstentions and positive
opportunities rejected as well as losses avoided. Report width of common
support, activity (>2/week required, no forced trades), net, costs included,
DD, Sharpe, side counts, seed-level paired changes; do not select by 2024.

Acceptance: hashes for weights, cache, labels and receipts; no head-fit or
monitor outcomes reaching its score year; 2024 baseline reproduces 406 native
accounts; no year-2024 labels in fitting; exact ties abstain; check the common
continuation on both sides of the anchor and single-trade cash reconciliation.
On failure stop and retain evidence, never silently discard a seed or side.

## Reproduction and custody

Independent CPU experiment under `/var/tmp/role408-managed-v1`; isolated
runtime `/var/tmp/role408-runtime`. No modification to predictor, its running
workers, 404/406 exports or feature selection. Fixed dates: the 2022 heads
use 2018-21, the 2023 heads use 2019-22, with the last six months in each
historical window reserved for monitoring; 2024 reuses 406 (2020-23).
The head-level refits use the same AdamW/plateau and stopping contract as
the 406 fine endpoint. The sampler and optimizer are not seeded from 2024.
The upstream frozen encoder was not re-selected separately for these folds.

| Requirement | Implementation | Evidence |
|---|---|---|
| Disjoint head fit / monitor / score dates | fold_end in role404_fit; role408_folds | per-role CONTRACT/RESULT, label hashes, max end of monitor strictly before fold |
| Same economic action alternatives | Panel.isolated, targets, terminal | cash identity for every fill; cloned same state; common 6h continuation |
| Calibration has no current-year outcomes | calibrate and evaluate_seed | frozen MAP.json before label construction; exact prior-fold file digests |
| No trading-account substitution | Panel.account | real 2024 baseline checked against immutable 406 returns and both side counts |
| No silent missing opportunity / fake activity | opportunity, calibration_report | matched offer coverage, win/loss opportunity counters, native trades/week |
| Degeneracy and look-ahead tests | tests/unit/test_role408_calibration.py | affine/constant fits, negative slope, clipping, fold boundary, NOP, missing labels, occupied continuation |

Sequential entry points from the pinned runtime and its existing CPU project
environment: `python -m tools.role408_folds` then `python -m tools.role408_router`.
Alternatively the pair of bounded user services overlaps scoring a completed
seed with training its successor. Head fitting resumes from committed epochs;
scoring verifies sealed label shards before reuse. For the final read-only
summary, run `tools/role408_summary.py` from agent-multi with the pinned
runtime on PYTHONPATH after DONE. Three targeted CPU tests passed. The
2024 raw baseline matched +0.811268/-2.748394/-10.358957 percent and
112/182/447 trades on the same 4,912 eligible origins; this parity by
itself says nothing about the effectiveness of the calibrated alternative.

## Completed contrast: do not promote this opening rule

All 24 chronological head fits and all three account replays completed.
Returns below are net percentages. The comparison holds the managed closer,
costs and account execution fixed.

| Seed | 2023 raw / calibrated | 2023 calibrated opens (buy/sell) | 2024 raw / calibrated | 2024 trades raw / calibrated | Highest calibrated bid anywhere on TRAIN support (bp) |
|---:|---:|---:|---:|---:|---:|
| 7 | -33.05 / -8.10 | 0/107 | +0.81 / 0.00 | 112 / 0 | -2.02 |
| 19 | -36.11 / -17.12 | 0/477 | -2.75 / 0.00 | 182 / 0 | -2.09 |
| 43 | -37.72 / -3.17 | 0/53 | -10.36 / 0.00 | 447 / 0 | -1.06 |

In 2024 each seed had 4,912 complete hypothetical decision origins. The
best bid of the eight fitted curves was negative even at its best supported
probability, so none could beat waiting (zero incremental value). Sharpe
is undefined and the activity gate fails for all three. On matched
hypothetical offers the veto discarded 80/119/307 profitable alternatives
and avoided 161/259/653 losing alternatives respectively. These alternatives
can overlap: they are not independent trades and do not sum to an account
return. The activity gate uses native market sessions, not civil-week counts.

For no seed did any side/horizon show positive mean managed advantage in
its TRAIN high-score decile in both chronological years. This finding is
about this scalar ranking and payoff definition, not the information content
of the 208-dimensional representation. The encoder was selected retrospectively.
Next controlled comparison: keep the encoder, labels, closer, and folds
identical; estimate signed managed advantage directly from the frozen features
with regularization, versus this scalar and a historical-mean control.
Require positive held-forward advantage and activity before promotion; do
not lower the threshold using the observed 2024 replay.

Local evidence: `/var/tmp/role408-managed-v1/INTERPRETATION.json` (account
and opportunity tables), `COMPARISON.json`, `FOLDS_DONE.json`, and `DONE.json`
in the same directory.
