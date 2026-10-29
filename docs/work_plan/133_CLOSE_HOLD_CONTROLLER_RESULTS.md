# Position-conditioned close/hold controller

2026-10-29. Implements the controller component in plan132 after the metric and
execution qualification in131. This is a report-to-action implementation, not
a trained financial reporter and not a fully optimized trading mechanism.

## What changed

The control in131 which always closed at the first opportunity was deliberately
bad:576 of its1728 early exits sacrificed gains. It remains an immutable
negative control. The new controller does not close merely because exiting is
better than reaching SL. It compares the estimated net value of closing with
the estimated net value of executable waiting alternatives for the actual
position, after spread, commission, slippage and financing.

`app/close_hold_controller.py` owns valuation and decision; the native engine
still owns execution and automatic exits. `app/close_hold_reports.py` contains
only a simple prefix-based diagnostic producer. The future-aware ideal/noisy
producers live separately in `app/close_hold_verification.py`, a test harness.

## Explicit report contract

| Field | Dimensions / meaning |
|---|---|
| paths | [S,H,4], joint mid-price OPEN/HIGH/LOW/CLOSE samples |
| spreads | [S,H], full bid/ask spread for each sampled bar |
| weights | [S], nonnegative, sum1 |
| issued_at / observed_through | Decision time / latest available observation |
| first_fill / step / deadline | First executable boundary, spacing and forced-flat boundary |
| provenance | causal, synthetic_ideal or synthetic_noisy |

This is a versioned auxiliary market-report candidate, not a requirement that
the existing RL actor directly predict an entire OHLC tensor. No learned model
was altered. Reports may eventually be produced by a validated adapter, but
that integration and report quality are not established by these tests.

The controller receives a separate immutable position view: original entry,
side, units, SL/TP, fixed monetary risk and costs. This is controller state,
not a model observation or agent-state feedback feature. Known rollover
charges are supplied per unit at each prospective open/close time.

The report covers the known forced-flat deadline. Short/incomplete support,
malformed OHLC, wrong sample weights, NaN, stale/future timestamps and duplicate
decision ticks are refused. Synthetic report tags are refused by default;
only the test harness explicitly enables them. Metadata cannot prove that an
arbitrary external producer is causal; that producer requires its own audit.

## Valuation and decision

For each sample, evaluate native-equivalent first passage separately, then
calculate net cash for CLOSE at next open, each possible later open, and
passive protection/forced-flat. Automatic protection at an open wins over a
discretionary exit; otherwise an agent close precedes intrabar touches.

Do NOT average price curves before evaluating barriers. Do NOT choose the
best future action separately for each sample and average those hindsight
profits. Waiting action j means the same prospective closing time in all
samples; after forced SL/TP its outcome is already settled.

For each wait action j, with net outcomes C_s and W_sj, compute:

```text
D_sj = C_s - W_sj
mu_j = sum_s(weight_s * D_sj)
sd_j = sqrt(sum_s(weight_s * (D_sj - mu_j)^2))
advantage = min_j(mu_j - uncertainty_penalty * sd_j) / initial_risk
```

Close only when advantage exceeds the configured margin and numeric epsilon,
and either enough consecutive valid reports agree or the urgency threshold
is reached. Ties HOLD. A gap in reports or a different position resets the
streak. State can be serialized and restored with configuration validation.

The dispersion penalty is not a calibrated probability/confidence bound.
This finite waiting-action, receding-horizon rule is not a global solution to
stochastic optimal stopping or the full shared-capital game. It is an explicit,
testable decision component in that larger mechanism.

## Configuration and controls

`CloseConfig(**parameters)` accepts JSON dictionaries with these fields:

```json
{
  "uncertainty_penalty": 1.0,
  "margin_risk": 0.02,
  "persistence": 2,
  "urgent_risk": 0.25
}
```

These are declared engineering controls, not selected optimal values. The
immediate control uses uncertainty0, margin0, persistence1, urgency null.
No input clipping, rolling scaling or new feature selection is introduced.

Five frozen arms run on every position: ideal/immediate, ideal/guarded,
noisy-ideal/immediate, noisy-ideal/guarded and prefix/guarded. Noise is three
antithetic sampled paths with cumulative innovations of standard deviation
0.0001 in synthetic price units, preserving the mean price curve but not
barrier outcomes. It is a declared stress, not measured market-model error.

The prefix diagnostic sees only the last8 completed OHLC bars and current
spread; median increments and increment dispersion generate three paths.
This cheap extrapolator is not the selected feature model or a trained RL head.
Its causal implementation can be correct while its predictions are poor.

## Required interpretation

Ideal reports test that the strategy uses informative outputs correctly.
Noisy ideal reports test sensitivity to this specific corruption.
Prefix reports test causality and a weak forecasting baseline. None establishes
financial profitability or justifies launching feature selection on an
unqualified full mechanism. No financial training, DEAP or TEST was run here.

## Completed measurements

Each arm executed all1,744 original positions through the native engine:
8,720 rollouts, not a selection of favorable examples. First execution took
248.05seconds on CPU. The ideal controller is checked against the independent
oracle on every case, with no parameter adjustment after the results.

| Report / controller | Early closes | Destructive closes | Missed optimal exits | Mean regret/R | Captured improvement |
|---|---:|---:|---:|---:|---:|
| Ideal / immediate | 1,152 | 0 | 0 | <1e-15 | 100% |
| Ideal / guarded | 1,152 | 0 | 552 | 0.063218 | 94.1054% |
| Antithetic noisy ideal / immediate | 1,152 | 0 | 0 | <1e-15 | 100% |
| Antithetic noisy ideal / guarded | 1,152 | 0 | 582 | 0.071463 | 93.3279% |
| Causal prefix / guarded | 0 | 0 | 1,152 | 1.071048 | 0% |

The576 destructive TP closes of the first-close control in131 disappear with
informative reports: the implemented controller holds those positions instead.
The ideal controller closes the1152 rescue/preservation positions at the
executable optimum. Gap protection still wins in16 cases. This verifies the
report-to-action rule, not that a learned reporter can supply those reports.

Persistence/uncertainty are not automatically improvements: this guarded
configuration delays exits and sacrifices5.89-6.67% of available improvement.
It is not promoted as a winning default. Perfect results under this antithetic
noise do not establish robustness to biased forecasts, omitted modes, incorrect
spreads or a different noise distribution. The mean was deliberately preserved.

The prefix baseline failed: zero early closes. A trace audit explains why and
does not blame feature quality or RL. In864 episodes its advantage would
trigger a close if persistence were removed, but only at a fill already lost
to broker protection; none is still in the
independently enumerated executable close menu. Removing persistence alone,
or both uncertainty penalty and persistence, from its frozen full traces yields
the same864 late requests and zero executable early exits. The other880 cases
have no positive request. This is evidence against that deliberately weak
extrapolator on this panel, not against the selected data or a trained model.

25 future-perturbation witnesses preserve prefix reports and decisions exactly.
134 focused unit/component tests pass, including same-market/different-position,
per-path barriers, uncertainty, persistence/urgency, state restore and report
rejection. Native code did not change from the74-test qualified engine of131.

## Evidence and reproduction

`docs/audits/evidence/close_hold_controller_20261029/` contains the source and
arm manifest, full compressed decision traces, aggregate results, replay receipt
and the prefix trace audit. Neither raw financial series nor model weights are
in this synthetic evidence.

Two fresh full executions produced byte-identical manifest, report and
compressed cases. The trace audit is reproducible with the read-only tool
`tools/audit_close_hold_prefix_trace.py --cases <CASES.json.gz> --reference
<131-CASES.json.gz>`; it explicitly refuses truncated early-close traces.

Identity:`e490dbeb63e09b493e65cd85bc07ae6dd58f69bbb9338a4b66c98a6e48fbbc94`.
Cases SHA256:`8e0d495ea658ccf493dd3d4630b18512a543a79cb7dc211bc0a5677145879d35`.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/verify_close_hold_controller.py \
  --engine-root <heuristic-strategy-checkout> \
  --output <new-output-directory>
```

The sequential runner authenticates131's support/config/engine and never
overwrites output. It records progress every64 positions and a terminal state.
The learned-report adapter, broader signal-error qualification, multi-expert
entry arbitration and joint account optimization remain separate work. A
component which correctly uses ideal reports is not the finished RL system.
