# 406: one paired exploration cycle

Fixed historical classification cohort: 2020-23, fit ending before July 2023,
H2 internal monitor, seeds 7/19/43, four independent role heads, untouched
61-input frozen market encoder. Compare paired branches on the same backend.
404 fixed-LR is a historical GPU reference, not a paired CPU rerun.
No search grid, live account, feature change or promotion.

| Acceptance criterion | Evidence |
|---|---|
| Only the second actual 405 drop opens the fork | Reduction counter, anchor checkpoint with optimizer and ordered role/seed identity |
| Same restored weights and moments, same minibatches | Deep copy after restoration; both consume the very same index tensor |
| Only exploratory LR differs | 20 complete epochs, cosine from 3e-4 to 1.2e-5 inclusive versus 405's 1.2e-5; evaluate before any subsequent rollback |
| Independent trajectories | Unit test mutates explorer only; checkpoint-resume roundtrip |
| No endpoint cherry-picking | Save both fixed endpoints, validation loss, sign mistakes corrected/introduced; separately retain 405's ES-selected champion |
| No repeated prefix, no fake trigger | Shared prefix; interruption resumes at committed epoch; if second drop absent report NO_PAIR |
| Account confirmation | Compare both endpoint quartets on real 2024 on identical native engine, costs and rows per seed; 2024 never drives scheduling |

Primary paired diagnostic: endpoint internal loss (by role, within seed).
Economic readout: 2024 return, activity, sides, Sharpe and drawdown. Do not
infer discovery of another basin from a change in parameter norm. Three
seeds on one calendar year do not constitute independent market years.

Execution receipt (2026-11-20): CPU-only unit `role406-cls4-paired.service`,
three single-thread workers, isolated runtime `/var/tmp/role406-runtime`,
results `/var/tmp/role406-cls4-paired/{champion,fine,explore}`. Each seed's
completed account block is atomically recorded in `COMPARISON.json`; `DONE.json`
is written only after all seeds have finished. Each endpoint records the SHA256
of the shared anchor. Global ES and scheduler counters continue in the control;
the explorer is discarded after its endpoint is exported. The coordinator
resumes from committed fit epochs on retry. No 404 results are overwritten.
The 405 and 406 CPU test suites passed (5 tests including matched permutation
and endpoint replay after interruption). Initial real-data evidence: for both
opening roles of seed 7, the second drop triggered at epoch 60, fixed endpoints
at 80. OPEN_BUY fine/explore monitor BCE 0.689559/0.691619; OPEN_SELL
0.691671/0.695160. These are head losses, not returns. No account promotion.

## Completed results and weight-space interventions

All 12 normal continuations stopped by patience at epochs 179-256, with
four actual champion restorations and final LR 4.8e-7. All 12 fixed
exploration endpoints had worse internal BCE than their fine counterparts.
All three seed quartets finished. 2024 returns below are historical
validation, not three independent years. No weights were promoted.

| Opening / closing heads | Seed 7 | Seed 19 | Seed 43 | Mean net % | Trades across replays |
|---|---:|---:|---:|---:|---:|
| Fine / fine | +0.81 | -2.75 | -10.36 | -4.10 | 741 |
| Explore / fine | -19.69 | -12.79 | -16.81 | -16.43 | 1931 |
| Fine / explore | -1.81 | -2.46 | -9.72 | -4.66 | 764 |
| Explore / explore | -22.27 | -14.48 | -21.01 | -19.25 | 2049 |

The first and last rows were independently replayed and matched the original
account dictionaries exactly. Endpoints are taken before the next rollback;
the separate ES-selected champions returned -4.10% on average. Switching the
openers at fixed fine closers cost 12.33 percentage points; changing only
closers at fixed fine openers cost 0.56 points. Effects are not additive.

Using the full H2-2023 monitor, verify endpoint BCE against the training
receipts, then interchange hidden and output layers (with biases) within
each matched head. Relative L2 displacement from the authenticated common
anchor (medians across the six heads of each role family, hidden/output):
OPEN fine 8.99/3.32%, explore 88.66/38.43%; CLOSE fine 21.55/11.20%,
explore 170.83/116.51%. Mean fine-vs-explore sign changes: 12.70% OPEN,
14.04% CLOSE. For closers the mean BCE increase is 0.04648; averaging the
two orders of layer substitution attributes 0.03208 of that to the final
layer. The largest local loss shift is not the main account-level loss.
These results apply to this 20-epoch excursion; they do not establish
that the optimizer is trapped, nor justify changing the frozen encoder.

Reproduction: `tools/role406_forensics.py OUT.json` in the pinned CPU 406
runtime (sibling tools from that runtime on PYTHONPATH). Source receipts:
`/var/tmp/role406-cls4-paired/{DONE,COMPARISON,WEIGHT_FORENSICS}.json`.
