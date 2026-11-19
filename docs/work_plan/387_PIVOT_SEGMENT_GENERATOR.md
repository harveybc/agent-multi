# 387: conditional fused TRAIN augmentation

Generate four separate paths on the original four-year TRAIN calendar. Use
paths 1, 1-2, 1-4 for 4/8/16 synthetic market-years; never trade across
path boundaries. Do not extend, generate or tune on validation.

Acceptance before comparing training arms:

| Requirement | Test |
|---|---|
| Real segments | Consecutive hindsight pivots on contiguous TRAIN H1 data only; preserve gap transitions, warm-up and censored edges. |
| Control response | Match mean log return (slope), standard deviation of log returns, and linear trend of returns (discrete curvature). Perturb slope and curvature in isolation; reject impossible combinations. |
| Alternative paths | Reproduced seed is identical; different seeds change at least 25% of covered closes. Count short constrained segments and copied bars. |
| Market shape independent of controlled moments | Redetect pivots on generated OHLC. Require at least 80% same-direction pivots within one bar in BOTH directions (one-to-one); report exact matches too. Require native seven-series distribution/correlation screen. |
| Usable by champion | Recompute all 71 features, fixed scaler and actual post-fusion activation; reconstruct logits from saved fusion tensors; check timestamp alignment with OHLC. |
| No synthetic performance claims | These are conditional scenarios given the historical pivot regime and release calendar, not independent histories or a validation gain. Preserve rejected draws; no selecting favorable seeds. |

## Measured results

All four prespecified seeds passed. Each path has 24,895 OHLC bars, 4,122
complete segment templates after warm-up, 23,853 covered bars, 71 rebuilt
features and 24,052 valid actor fusion tensors (16 channels by 13 time slots,
before attention). Three focused unit tests pass. CPU native replay verifies
output hashes and first reproduces 745 trades and +16.058224% real TRAIN.

| Seed | Pivot recall / precision within one bar | Covered closes differing by at least 1 pip | Frozen actor trades | Entries absent from real (time, side) | Frozen actor return |
|---|---|---:|---:|---:|---:|
| 19 | 92.02% / 87.79% | 35.05% | 714 | 165 | +10.52% |
| 7 | 91.61% / 87.74% | 35.34% | 719 | 171 | +11.40% |
| 43 | 92.57% / 88.33% | 34.24% | 736 | 173 | +10.54% |
| 71 | 92.18% / 87.96% | 34.59% | 718 | 168 | +3.83% |

Worst control discrepancy: 1.39e-17 on log-return scale. Ledger reconciliation
error below 2e-11 USD. The 4/8/16-year cohorts contain 24,052/48,104/96,208
valid fused windows. Never concatenate separate paths into one account.
The 1,475 segments of length three or less per path have too few free interior
returns under these constraints. The generator does not create new macro
releases, regime durations, price endpoints or unprecedented crises.

Tools: tools/generate_pivot387.py and tools/replay_pivot387.py. Per-seed
files: OHLC.parquet, FEATURES71.parquet, FUSED.npz (availability timestamps),
SEGMENTS.npy (start/end bar offsets, target and measured mean, standard
deviation and curvature), RESULT.json. Cohort identity: COHORTS.json.
NATIVE_TRAIN_REPLAY.json contains frozen accounts, not trained-arm results.
Next: real only versus real plus 4/8/16 synthetic years, matched model,
task, updates, real validation dates and training seeds. No model promoted.
