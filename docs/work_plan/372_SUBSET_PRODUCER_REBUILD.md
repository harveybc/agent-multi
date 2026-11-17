# 372: subset-fitted producers and fixed scalers

## Contract before implementation

Build6/12/24/48-month stores ending2024-01-01. Retain all71 selected columns,
28 groups,13 observations/stride2, fixed transformations and business rules.
No source file, checkpoint or running worker changes. No neural training.

Refit FX PCA, Kalman, GMM and activity on each subset. Fixed filters may retain
causal prehistory as warmup, not fit samples. Activity uses the existing five
purged cross-fit folds,96-bar embargo and full input/label-resolution exclusion.
This is offline cross-fitting, not a forward-only historical deployment claim.
Feature selection and transformation choices remain fixed; conclusions are
conditional on this common selected representation.

## Tests and evidence

| Requirement | Acceptance |
| --- | --- |
| No validation/out-of-subset fit | Recorded fit indices contained in subset; activity entire label/input support allowed |
| Fit insensitive to excluded data | Repeat fits after changing excluded parent values/labels; same parameters and fixed-probe activity predictions |
| Apply causal with fixed parameters | Perturb suffix and compare prefix for new price-derived columns |
| Scaler subset-only | Future/validation perturbation leaves mean/std identical; zero-variance recorded, never drop a column |
| Persistence | Saved/reloaded producer predictions exact; recipe/store/manifest digests validated |
| Source preservation | Input pins rechecked after work; unchanged63 selected columns compare exactly |
| All arms progress without babysitting | Sequential finite four-arm driver; per-arm receipts, resume skips verified arms; explicit failure |

Original factor code uses factor_fit_start2015-01-01 and rows before grid start
2020-01-01. Its archived weights path is no longer present. We do not need those
weights for a fresh subset fit: use the pinned original PCA implementation and
six pinned FX series, recording that this is a new estimator, not old-weight
recovery. Original snapshot/source hashes remain unchanged.

## Execution results

All four arms completed sequentially on CPU. Combined370/371/372 unit tests:
26 passed; scoped lint passed. Cold rerun verifies pins and reuses completed
arms without fitting again. All71 columns and28 groups remain present.

| Months | TRAIN windows | Validation windows | Activity full fit rows | CPU arm seconds |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 2781 | 6208 | 2341 | 7.93 |
| 12 | 5892 | 6208 | 4811 | 8.22 |
| 24 | 11842 | 6208 | 9733 | 9.97 |
| 48 | 24026 | 6208 | 19564 | 18.84 |

Validation timestamps have the same digest for all arms. Values intentionally
differ because each arm's producers/scalers are fitted independently.
All63 untouched selected columns compare exactly; the other8 are rebuilt.
Excluded-data perturbations preserve Kalman/GMM/PCA fit parameters and all six
activity estimators' fixed-input predictions. Future suffix perturbations leave
the fixed-state Kalman/GMM/factor prefix unchanged. Scalers are invariant to
excluded-row perturbations. Reload predictions are exact.

One relevant observation: the12-month Kalman fit converges at q_slope/r near
zero (r approximately6.34e-26), so level_minus_close is numerically constant.
It is retained, recorded in constant_columns, and its standard-deviation divisor
is1 under the existing fixed-scaler convention. No fabricated feature or column
removal. This is a measured consequence of the smaller fitting interval, not a
neural training result or evidence of insufficient market signal.

Outputs per arm: store.parquet, manifest.json, RECIPE.json, PRODUCERS.joblib,
FIT_ROWS.npz, RESULT.json and authenticated DONE.json. Existing source store,
champion weights and active GPU jobs remain unchanged. Public documents contain
only aggregate evidence, not raw market data or trained producer weights.

Next: a fresh-network matched-update learning-curve runner using these recipes;
validate model/optimizer initialization and raw-versus-transformed input parity
before dispatch. No neural training for this new comparison has started yet.
