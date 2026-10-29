# Audit request: implemented position-conditioned close/hold component

Satoshi, please review plan132, implementation/report133 and evidence directory
`docs/audits/evidence/close_hold_controller_20261029/`. This requests a component
verdict, not acceptance of a final strategy, learned model or optimized mechanism.
Native engine remains heuristic-strategy e0a1c2b; the manifest pins all sources.

## What is implemented and measured

- Joint market-path reports evaluated separately against the actual position;
  no position state added to the model, no neural SL/TP/volume estimator.
- Net close-versus-common-wait alternatives; no mean-path first-passage shortcut,
  no average of independently clairvoyant sample-specific exit choices.
- Paired dispersion penalty, normalized margin, persistence, urgent bypass,
  per-position reset and explicit synthetic/causal report separation.
- 5 frozen arms x1744 positions =8720 native executions. Ideal/immediate
  reproduces the independent optimum, with1152 helpful closes and zero
  destructive closes, versus576 destructive closes for the old first-close
  control. Guarded variants lose about6% of available improvement by delaying.
- The prefix-only diagnostic closes nothing and fails. Its traces are included;
  no claim of successful learned/causal financial performance is made.
- 25 future perturbations preserve prefix reports/decisions. 134 tests pass.
  Two fresh executions reproduce all report, manifest and case bytes exactly.

## Attack these points first

1. The auxiliary report is a candidate interface, not an existing trained head.
   Is it sufficient for the stated finite-action problem? Do not certify
   learnability, calibration or actor compatibility from these simulations.
2. The paired standard-deviation penalty is a heuristic, not a confidence
   interval. Does its minimum across common wait alternatives match the formula?
3. Native priority at a next-open gap and before intrabar barriers: verify
   timing, quote sides, fees and known financing schedules independently.
4. Distinguish choosing one common delayed action from sample-specific foresight;
   check the two-position fixture and the nonlinear mean-path counterexample.
5. Persistence suppresses isolated signals but can miss urgent exits; longer
   false reversals are not solved by this finite test. No guarded arm was selected
   after observing this table.
6. Antithetic noise preserves the mean and may be an easy stress. Its100% result
   is not evidence of general noise robustness. Recommend prospectively defined
   biased/correlated/missing-mode controls without changing these records.
7. Verify the prefix trace diagnosis:864 first requests are already unexecutable,
   even without the guard filters. This says nothing about the predictive
   information in the selected financial features or about SAC's capability.
8. Controller timestamps reject invalid metadata, but cannot independently
   authenticate an arbitrary external producer's causal computation. The real
   report adapter needs its own prefix-invariance and source-identity tests.

Please separate execution/metric correctness, decision-rule correctness under
the supplied reports, signal sufficiency and full-mechanism eligibility. Only
the first two are candidates for acceptance in this delivery. Do not launch
training, reselect features or change the shared business contract for this audit.
