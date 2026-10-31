# Economic Multihorizon Arbitration

Owner approved2026-10-31 after177: consult all horizons each available hourly
decision; compare economic CLOSE versus continued HOLD on the same live position.

## Frozen First Implementation

Keep all12 discrete-SAC models, selected inputs, fixed scalers,39-logit surfaces,
native costs/calendar and horizon risk profiles unchanged. These policies differ
by risk profile; this is not a new downsampled-world training experiment.
The network still receives market features only. Position routing stays outside.

An external economic calibrator consumes each model's opening logits/twin-Q
reports, or externally queried HOLD/CLOSE logits/twin-Q reports. It never treats
soft Q or softmax as money or success probability. Fixed StandardScaler plus
multi-output Ridge(alpha1) is the preregistered simple calibration baseline;
fit constants on calibration TRAIN only, no rolling scale or input clipping.
Fit separate opening-action models and close models by original order horizon,
reporting horizon and action. No balance, age or raw position state is added
to learned network inputs. Position-dependent interpolation is the existing
approved external interface. Limited information in these curves is a measured
model limitation, not a sufficiency claim.

## Comparable Targets and Reference

At each labelled state, every alternative has the identical native account
state and deadline: that week's forced liquidation. Targets are return percent
of equity at the origin and future maximum drawdown percent. Both include native
costs and later opportunities. Opening alternatives are NOP and both directions
at3/6/12/24h. Occupied alternatives are CLOSE and HOLD; all four reporting models
are queried on that identical position. No reopening in the same tick.

The frozen continuation reference nominates an opening horizon by UTC-hour
round-robin, independent of prices/outcomes, and applies its categorical policy.
While occupied it uses the original opening horizon's close curve. This is an
explicit reference policy for labels, not the new optimized selector. After a
position exits, subsequent reference entries are allowed. Thus target values
include opportunity cost instead of assuming permanent inactivity after exit.

Only TRAIN creates targets. Use two calendar-selected eligible origins per
session, separated by48 observed bars; open both directions with each native
risk profile and inspect surviving positions at ages1/3/6 bars. These initial
conditions deliberately cover weak/strong early-close cases without requiring
the learned policy to visit them. Replay each alternative to the same boundary.
Discard any label whose complete session crosses a calibration split boundary.
Fit calibration on2020-2022; diagnose on2023H1. Underlying SAC weights saw2023H1,
so this is NOT fully nested held-out model evidence. Upstream feature/risk
selection also saw2023H2.2024/TEST remain closed.

## Runtime Mechanism

For flat accounts compare eight economic opening bids and a NOP value averaged
across the four reporters. For occupied accounts average the four calibrated
estimates separately for CLOSE and HOLD; do not add redundant estimates as
independent profits. Select the estimated return/drawdown Pareto frontier and
maximize return minus drawdown, reusing the existing account risk weight1.
Tie between directions -> NOP; close/hold tie -> HOLD. Negative predicted risk
is projected to its physical zero bound, recorded explicitly; input signals are
never clipped. Missing unsupported reports -> NOP/HOLD, never fabricated values.
The original SL/TP/volume remain fixed throughout an open order.

This is one-step improvement relative to a frozen continuation, NOT an exact
value of the subsequently adaptive joint policy and NOT a globally certified
Pareto optimum. Measure the forecast-versus-executed continuation gap explicitly.
No tuning on INNER_A, no promotion and no floating selection from this pilot.

## Tests Before Implementation

- Numeric NumPy routing equals the pinned torch codec, including endpoint extension.
- Ties, all-negative opportunities and close-versus-hold examples behave symmetrically.
- Same-position queries consult all four reporters, without changing protections.
- Counterfactual replay leaves the original native state untouched; both directions.
- Calibration rejects split-crossing outcomes and uses only fit-side scaler constants.
- Native positive controls demonstrate useful HOLD, protective CLOSE and NOP.
- Resume authenticates source/config/data/weights and preserves finished receipts.
- Runtime decisions access only the current sealed report; future prices are
  available exclusively to TRAIN label construction and retrospective scoring.

## Automation

Three seed lanes run sequentially on a bounded CPU worker: generate authenticated
per-session target batches -> fit calibration -> report chronological calibration
errors against constant targets -> joint TRAIN/INNER_A replay -> exact ledger and
activity checks -> output report/audit packet. CPU execution is intentional:
native counterfactual accounts dominate work, not GPU tensor training.
All scientific outcomes, including no activity and negative returns, are retained.

After the complete joint account is recorded, replay its actions exactly and
audit every close proposal: fixed-position passive protection and intermediate
exit optima, plus a one-tick CLOSE/HOLD intervention followed by the same adaptive
joint controller to the common weekly boundary. Publish both local and account
harms, broker-preempted proposals and censored terminal weeks. These retrospective
diagnostics cannot affect already-recorded actions or calibration.

## Preflight Evidence

33 focused CPU tests pass, including the inherited native account and frozen
follow-up regressions. Both-direction full-account fixtures reproduce the
original ledger exactly during posthoc close attribution. Same-position checks
query all four reporters; changing future reports cannot change the current
opening choice. Completed-closure mutation and identity tests reject tampering.
The real-data one-session smoke produced328 finite target rows covering all76
role/action models; it is a wiring check, not a trading performance result.
Ruff correctness checks pass for the three added Python files.

Runtime resumes authenticated completed sessions and seed receipts. A completed
closure is verified without rewriting it. Typed contract/integrity failures exit78
and are not retried by the service; process failures have bounded retries.
The immutable runtime, CPU/memory limits and actual launch are recorded separately
from this design. No running predecessor is modified.
