# Report-driven close/hold controller: predeclared implementation plan

2026-10-29, owner requests the actual close/hold rule after metric qualification.
This extends131, before joint mechanism optimization and floating selection.

## Contract and architecture

Implement a pure position-conditioned controller in agent-multi, not a second
market simulator. It accepts a market-only report of weighted joint OHLC paths
and spreads, an external position snapshot and known fee/calendar schedules.
This is a versioned auxiliary report candidate from disposition113, not a change
to learned model inputs or a claim that a learned reporter already exists.
The controller receives no engine object, realized future or scenario label.

Report paths preserve first-passage order; never take the mean price path
before applying SL/TP. Reports cover through the known forced-flat deadline;
shorter reports are rejected rather than inventing continuation. Each possible
delayed close is one common action across samples, not a different hindsight
optimum per sample. This is receding-horizon finite-action control, not a proof
of a globally optimal stochastic stopping policy.

For each path compute net liquidation at next open, each later open and passive
protection/deadline. Existing protection takes priority at an open; discretionary
close takes priority over later intrabar touches. Use identical fixed position,
costs and known rollover schedule. Position state stays outside the neural model.

For each wait action j define D_j = net_close_now - net_wait_j. Close advantage:
`min_j(mean(D_j) - uncertainty_penalty * std(D_j)) / initial_risk`.
This paired dispersion penalty is a control heuristic, NOT a calibrated confidence
interval. Close only above a configurable nonnegative margin and numeric epsilon;
ties hold. Uncertainty penalty zero is an explicit control. Consecutive-report
persistence resets per position and after a missing tick; an urgency threshold
may bypass persistence. Broker exits always remain active.

## Acceptance tests before implementation

1. Both directions: known imminent loss favors CLOSE; rising-to-TP favors HOLD.
2. Same market with tight versus wide stop requires different decisions.
3. Evaluate barriers per path: no Jensen/first-passage shortcut through mean path.
4. Uncertainty can block a marginal close; no sample-specific future optimum.
5. A one-tick false reversal is suppressed by persistence; urgent deterioration
   can bypass it. A longer false reversal remains an explicit limitation.
6. Reset on position change/gap, reject duplicate/stale/future/malformed reports;
   controller state is JSON replayable. No agent-state feedback is introduced.
7. Perfect synthetic reports through the actual native engine reproduce the
   1744-case oracle and eliminate the576 destructive first-close outcomes.
8. Noisy ideal reports are labeled retrospective stress, never causal evidence.
9. A deliberately simple prefix-only extrapolation reporter is a diagnostic
   control, not a selected model. Perturbing future bars cannot change its report
   or the controller's decision at the cut. Keep poor results if it fails.

## Frozen evaluation arms

Use all1744 positions from130, same costs/sizing and source identity. Report
ideal/immediate(z=0,margin=0,persistence1), ideal/guarded(z=1,margin=.02,
persistence2,urgent=.25R), noisy-ideal/immediate and noisy-ideal/guarded,
and causal-prefix/guarded. All coefficients are engineering controls, not optima.
Noise uses deterministic seeded path innovations, not rounded action labels.
Use the same metrics from131, native realized exits and replay both runs.

The causal prefix baseline uses only the last8 completed bars, median close
increments and their standard deviation for three extrapolation paths. It is
an interface/causality diagnostic, not a replacement for selected features or RL.
Do not select parameters on this panel or launch training/DEAP/feature selection.
Publish source pins, traces, failure counts and an explicit component verdict.
