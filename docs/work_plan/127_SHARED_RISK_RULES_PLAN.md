# Shared risk rules: parameterization before account optimization

## Goal and placement

Continue plans 125-126 before mechanism qualification and phase 2 floating
selection. Share TRAIN findings with Satoshi without imposing Musashi's pure
RL action/credit definitions on the expert-guided lane. No new financial
model fitting, historical rerun, TEST access or modification of the pinned
engine. This step implements and verifies order construction, not a completed
entry/close strategy or a claim that its coefficients are optimal.

## Top-down contract

- RR01: JSON references originate from the authenticated published profile
  report. Preserve original ZigZag genes, source digest, horizon and side.
- RR02: common business constants remain leverage 50, rel_volume 0.1, one
  position, no daily cap, 12h reopening delay and 12h pre-close liquidation.
- RR03: at a known fill quote, SL distance is a_sl * MAE_reference +
  b_sl * sigma + buffer_pips * pip. TP distance is a_tp * TP_reference -
  b_tp * sigma. No signal clipping, rolling scaling or new model inputs.
  sigma is in price units and supplied from already observed data; choosing
  its estimator/window is a later paired experiment. c=1 is the control.
- RR04: round the SL away from entry and TP toward entry on the declared
  price tick; reject wrong-side, too-close or nonpositive levels instead of
  extending a TP whose opportunity has disappeared. Check both entry and
  liquidation-side quotes. Raw MAE never becomes an unbuffered stop silently.
- RR05: account currency USD, instrument EURUSD in this adapter. The native
  Sizing class gives integer EUR units at q=1. Compute loss per EUR from the
  rounded SL distance plus explicit market/stop slippage, both commissions,
  and an explicit nonnegative funding reserve. Spread is already in the two
  quotes; do not debit it again. General currency conversion is not claimed.
- RR06: R=rho*balance; q=c*min(1,R/L1). rel_volume applies once. Use native
  sizing with minimum rounding UP disabled. Actual rounded margin and modeled
  stop loss must stay under their caps. A gap can exceed modeled stop loss.
- RR07: reject nonfinite inputs, contract mismatch and invalid genes. Keep
  all coefficients and source identities in the result. No assumed calibrated
  uncertainty; c is an explicit scenario input, with c=1 as the baseline.
- RR08: native protection witnesses must agree with independent intended
  outcomes for mirrored TP, SL, simultaneous touches, adverse gap and no
  touch. These test protection only, not the early-close decision policy.

## Configurable candidates, not selected trading parameters

References: median and p75 from positive calendar-compatible TRAIN paths.
Baseline: a_sl=a_tp=1; b_sl=b_tp=0; cushion=1 pip; c=1. These are engineering
controls, NOT optimized settings. Sensitivity grid before DEAP:

| Axis | Values |
|---|---|
| horizon | 3, 6, 12, 24h |
| direction | buy, sell |
| reference quantile | median, p75 |
| sigma (synthetic, pips) | 5, 15, 30 |
| spread (synthetic, pips) | 0.5, 2, 5 |
| rho (simulation candidates) | 0.001, 0.005, 0.01 |
| a_sl | 0.75, 1.25 |
| a_tp | 0.75, 1.25 |
| buffer pips | 0.5, 2 |

This is 3,456 order proposals, not 3,456 account backtests. b_sl=0.25 and
b_tp=0.1 are explicit sensitivity-probe coefficients to exercise the sigma
path, not winners. Separate tests prove each coefficient actually reaches its
formula. Bounds for future DEAP are configurable candidates, not inferred
confidence intervals: a_sl/a_tp [0.5,2], b_sl/b_tp [0,1], buffer [0,5] pips,
rho [0.001,0.01]. Search budget remains at least population 20 / 300
generations when account optimization is admissible; no premature DEAP on
isolated order geometry or overlapping oracle profit.

## Test design and implementation structure

Pure `app.horizon_risk_rules` consumes profiles, coefficients, current quotes,
costs and an injected native Sizing adapter. `tools.verify_horizon_risk_rules`
pins the source report and execution module, emits config and deterministic
matrix evidence. Unit tests cover coefficient effects, tick symmetry,
SL widening reducing q when risk binds, minimum units, exact single
rel_volume use, spread handling, invalid values and empty opportunity NOP.
Integration invokes actual native Sizing and protection on synthetic quotes.
Each case has an expected outcome fixed before simulation. No training,
rewards, credit assignment or close thresholds are invented by this step.

## Existing consumer incompatibility

The pinned BarEnv._levels clips multipliers to tp_bounds/sl_bounds and may
clip distances again to effectiveness_c * sigma_rule; check_order also
enforces that upper distance. Do not route the new absolute levels through
that API and pretend they survived unchanged. This step verifies native
sizing/protection primitives only. A separately tested absolute-level entry
adapter and next-open revalidation must precede full account execution.

## Completion evidence and next action

Publish source references, numerical parameter tables, findings, config,
matrix results and tests in a handoff for Satoshi. Then qualify the full
position-dependent early-close and absolute-level execution interfaces; only
after that compare account-level objectives/risk budgets and optimize joint
strategy coefficients. Business restrictions cannot become DEAP genes.
