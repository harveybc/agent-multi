# STL source coverage diagnosis

## Scope and decision

The 69-column preprocessing run completed; its four training arms have NOT
started. The monthly availability gate rejected the prepared tensors. This is
not an STL trading result and does not replace the selected69 champion or the
main phase2 critic work.

Identity: `04fb822af81df2708dae1fbf7bc1105e152760244d757e5d76a734b3ac6996b8`.
Evidence lives in the run's `operations/SOURCE_SUPPORT_AUDIT.json` and
`SUPPORT_AND_ENCODING.json`. `tools/audit_stl_source_support.py` reproduces the
source comparison and authenticates preprocessing receipts. It does not change
any observation, model, scaler, selected feature or window.

## Verified findings

| Period | Original eligible windows | Common STL windows | Difference |
| --- | ---: | ---: | ---: |
| TRAIN | 24026 | 21865 | -2161 |
| VALIDATION | 6208 | 5460 | -748 |

January 2020, June 2022 and December 2023 have zero common complete windows.
The 13-observation, stride2 model window was not shortened. STL requires up to
480 consecutive finite feature observations; upstream wavelet/Hilbert windows
and rolling factor sums propagate a missing price observation before STL adds
its own history requirement.

The cross-asset store has only five rows with any missing one-step return among
the six FX legs: its first row and two consecutive rows for each of two incidents.
Those return rows trace to these absent source-price observations:

1. USDCHF, 2022-05-19 21:00 UTC bar start. Absent in the recovered H1 and M30
   sources and the clock-adjusted lake H1, M15 and M5 observations examined.
   Adjacent H1 closes agree between recovered data and lake. A neighboring close
   cannot be substituted for the absent observation.
2. All six FX pairs, 2023-11-29 13:00 UTC bar start. Absent in recovered H1;
   present in the clock-adjusted lake H1. The previous hour differs between
   sources for all six pairs, whereas the following hour matches. Presence alone
   therefore does not qualify an automatic source splice. Source boundary and
   clock checks must precede use of those real candidate observations.

Affected selected channels include `fac.fx_f1.ret_4`, `fac.fx_f1.ret_24`,
`wv.usd.A4` and `hb.eur.inst_freq`.

The engineered store starts at 2020-01-02 01:00 UTC availability, while the
recovered FX histories start in 2010 and contain more than 60000 observations
per pair before TRAIN. Real raw warm-up history exists for these FX legs; this
does NOT establish availability of all 69 engineered feature histories.

## Repair contract

- Obtain and qualify the real missing USDCHF observation; do not interpolate,
  forward-fill, replace with zero, infer an hourly close from a coarser bar, or
  call an unavailable observation synthetic evidence.
- Qualify the November lake observations and preserve per-row source lineage.
- Reconstruct pre-TRAIN feature history using the frozen production definitions
  and fitted states. Extending the input history must not silently refit PCA,
  regimes, transformations or standardization, or alter the training interval.
- Any repaired store receives a new digest and a new STL experiment identity.
  Existing sealed outputs remain unchanged. Reuse requires explicit parity and
  lineage checks; a repaired input is not the same original experiment.
- Re-run causal-prefix, future-perturbation, streaming parity and monthly support
  checks before releasing the four matched GPU training arms.

## Operational correction

The collector previously raised its coverage error before publishing status,
leaving a stale count of 68 columns. The private supervisor now publishes
`INPUT_COVERAGE_REPAIR_REQUIRED`, 69 completed columns and the three unavailable
months before refusing release. The gate remains intact. No training is claimed
while the services are waiting for qualified data.

Four focused tests cover non-wrapping window indices, missing observations,
preserved neighboring source values and duplicate-timestamp rejection.

## Main work remains separate

The eight operational critic contrasts finished. The full-critic absolute-loss
arm reduced internal mean regret from 0.103982 to 0.048127 with actor logits
unchanged. That is a critic-ranking diagnostic, not a new trading return.
The next main-lane gate is bounded actor refinement with native account and
activity checks, before another selection pass on changed weights.
