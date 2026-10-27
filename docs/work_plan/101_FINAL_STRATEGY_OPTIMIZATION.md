# Final strategy optimization and audit delivery

Owner clarification: the audit is of the FINAL optimized strategy and its
experimental justification. Request 100 was premature and is withdrawn for
that purpose. Control run d7ede10e remains evidence, not final acceptance.

## Frozen procedure before executing search

1. Fixed synthetic output tapes, independent of the candidate's risk genes.
   Use the actual categorical output distribution and native business engine.
   No predictor fitting, financial rows, feedback features or rolling scaling.
2. Calibration: 24 paths, eight before/after direction patterns x three drift
   strengths; distribute volatility, period, transition duration and horizon
   deterministically. No selection of scenarios based on observed results.
   Confirmation: 24 separate paths with different phase, strengths, amplitudes,
   periods and horizons, read after all searches; never used by the fitness.
3. Optimize all six named directional fractions in [0,1] using the existing
   DEAP implementation: population 20, maximum 100 generations, patience 10,
   relative improvement 1e-8, seeds 7/19/43. Midpoint and prior grid controls
   enter every initial population. Elitism must never lose an eligible control.
4. Compare S, S+R and S+R+A, where S is pooled daily-return Sharpe, R is mean
   net return / initial balance, A mean orders / business day. Scale each by
   a fixed positive scale determined from the initial controls BEFORE search.
   Record raw and normalized components, not just the composite fitness.
5. Business correctness is a constraint, never a term an optimizer may trade
   away. Aggregate activity >=2 orders/week, both directions, both discretionary
   closes and finite non-ruined accounting qualify a candidate. No forced trades.
   Report every failure. NOP controls have undefined Sharpe, not fabricated zero.
6. Measure one-gene endpoint sensitivity, stop/target/agent-close counts, order
   counts, effective RR, realized costs per trade and barrier break-even rate.
   Never label a numerically inactive gene or constant objective term identified.
7. Finish all nine searches; compare calibration winners to controls. Freeze
   finalists BEFORE confirmation. Report confirmation without retuning.
8. Run every distinct finalist through the complete 47,628-case acceptance
   matrix with that finalist's JSON (not the midpoint JSON), broker witnesses,
   endpoint controls and independent receipt verification. A finalist failing
   a contract is rejected, not patched into a passing result.
9. Generate the final report and audit request ONLY after all stages have
   terminal receipts. A negative result is a completed study, not acceptance of
   a failed strategy. No financial selection/training release follows unaudited.

## Tests before implementation

F01 panels deterministic, disjoint, every directional pattern included.
F02 tape invariant under risk-gene changes; actual categorical command output.
F03 mean returns, pooled Sharpe, activity and costs reconcile with raw ledgers.
F04 null/degenerate metrics cannot become a favourable zero.
F05 objective scales fixed and all three formulas recompute independently.
F06 control elitism, exact resumed DEAP state, cache/input/code identity checks.
F07 final audit exporter refuses incomplete search/confirmation/matrix evidence.
F08 winner configurations supplied to the matrix; no midpoint substitution.

Sequential stages are resumable and keep full ledgers. CPU workers may parallelize
independent calibration scenarios; searches and finalist matrices remain ordered.
The new run must not modify previous closed evidence or active financial workers.
