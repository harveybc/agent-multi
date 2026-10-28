# Satoshi: independent audit of the mechanism-synthesis implementation proposal

Date: 2026-10-28 (America/Bogota)
Author: Musashi. Status: REQUESTED, no reviewer verdict yet.
Scope: design review before implementation, not final results acceptance.

## Read first

The user has prioritized defining a correct trading strategy above further
financial training, floating selection and input/model sweeps. They request
mechanism design plus test-guided synthesis, using Pareto and Nash analysis,
not an uncertainty auction renamed as game theory. Please attack the design
constructively and propose concrete repairs before it becomes executable.

Primary proposal:
[112_MECHANISM_SYNTHESIS_IMPLEMENTATION_PROPOSAL.md](../work_plan/112_MECHANISM_SYNTHESIS_IMPLEMENTATION_PROPOSAL.md).
Traceability:
[TRACEABILITY.csv](../work_plan/mechanism_synthesis/TRACEABILITY.csv).
Method state:
[PROJECT_METHOD_STATE.json](../work_plan/mechanism_synthesis/PROJECT_METHOD_STATE.json).
Previous failure/control context:
[110_EXIT_REPAIR_CLOSED_FIXED_DELAY_FINDING.md](../work_plan/110_EXIT_REPAIR_CLOSED_FIXED_DELAY_FINDING.md).

This packet contains no implementation of the new mechanism, no trained 20-expert
ensemble, no solved payoff matrix, and no claimed Pareto/Nash result. Do not
mistake a proposed acceptance test for a passed test. The upstream signal
interface is assumed to provide probabilities and uncertainty; its implementation
method is not prescribed. Existing execution evidence has its historical scope.

## Requested audit outputs

Return one Markdown verdict with a finding table: ID, severity, proposal section,
counterexample or derivation, consequence, minimal correction, and test to add.
Separate verdicts for mathematical model, business semantics, test sufficiency,
implementation feasibility and cross-lane reuse. Use ACCEPT, ACCEPT_WITH_CHANGES,
or REVISE; do not reject merely because no training has been run for a design review.
Identify actual blockers to correctness versus optional research improvements.
No live runs, edits to sealed outputs or rewrites of shared plans are needed.

## Mathematical attacks (priority A)

1. Are all players, policies and controlled variables actually defined? Distinguish
   frozen reporters from adapting policies; detect vacuous strategic claims.
2. Prove or refute the exact-potential identity for the specified difference
   reward. Does the reference term stay independent of the deviating player?
   Does replay use fixed other policies rather than illegally fixed actions?
3. Verify small fixtures with stable but dominated equilibria. Do they prevent
   conflating Nash, global utility maxima and the account Pareto frontier?
4. Audit the deviation certificate: restricted versus exhaustive policy library,
   estimated versus exact payoffs, and confidence intervals for differences.
   A heuristic best-response search cannot certify a small global epsilon.
5. Challenge equilibrium selection in the outer search. Demand reporting of
   multiple reached equilibria and coverage rather than selecting only the best.
6. Can threshold/offset degeneracy, infeasible strategy combinations or tie cycles
   make the solver misleading? Are resource violations projected by a deterministic
   mechanism while the strategic policy sets remain well-defined?
7. Distinguish account multiobjective Pareto from player-payoff Pareto. Assess
   whether difference rewards or nonlinear Sharpe could produce unwanted behavior.
8. No truthfulness claim follows from fixed uncertainty reports or from Nash
   alone. Suggest incentive tests if strategic report manipulation is later allowed.

## Business and execution attacks (priority B)

9. Opportunity cost: an entry can block the opposite side, future entries, margin
   and closers' opportunities. Can the counterfactual bookkeeping double-count
   mutually exclusive profits or assign all gains to the last actor?
10. NOP versus HOLD: test avoided losses, foregone gains, conserved quota and
    genuinely beneficial continuation. No reward merely for winning or acting.
11. Cross-scale close: a 24-hour opener cannot veto a 1-hour close. Conversely,
    a confident short-scale false reversal must not automatically force a close.
12. Executable early-exit optimum: current decision to first SL/earlier TP or
    compulsory exit, after spread/slippage/fees. No post-stop recovery or old peak.
13. Position information: can market-only expert reports plus external controller
    state resolve different desired actions for identical markets but different
    positions? Require a witness, not hidden feedback features.
14. Validate risk equations for both directions, unit/currency conversion, actual
    entry fills, lot rounding, margin limits, q semantics and gaps beyond stops.
15. Missing data, stale reports, simultaneous protection and close, calendar
    boundaries and entry-cap restoration must have explicit expected behavior.

## Scenario and implementation attacks (priority C)

16. Audit the 9600-cell raw factorial and stochastic partitions. Propose missing
    interactions, changed volatility/phase relationships and overlapping horizons.
    Do not accept a coverage count as proof that all required behaviors occur.
17. Confirm independent expected-behavior construction, oracle isolation, blind
    confirmation and paired exogenous randomness across counterfactual branches.
18. Attack fixed-clock shortcuts, immediate-close policies, always-long policies,
    expert order permutations, mirrored markets and confidence inversion.
19. Is the finite-library game computationally feasible? Recommend exact tiny
    fixtures and bounded expansion before any 20-player exhaustive claim.
20. Check that M0-M3 require no financial GPU training or learned value oracle.
    No new dependency is justified merely to avoid writing small exact controls.
21. Validate immutable manifests, atomic progress, resume, failure propagation,
    artifact identities and report separation between design, measured and certified.

## Cross-lane handoff

This is also a starting specification for your independent expert-guided lane.
Reuse the business engine, signal interface, scenario tests and attribution tools
where their semantics match. Do not import our pure-RL result claims as yours.
Your multi-resolution ZigZag demonstrations may define labels and learning
signals, not observed future inputs. Record label support, maturity and censoring.
If you select a different utility, player strategy library, outcome horizon or
arbiter, create a distinct game identity and solve it independently.

Please include a short table: SHARED_UNCHANGED / RL_ONLY / EXPERT_GUIDED_ONLY /
NEEDS_SEPARATE_TEST. Your review does not authorize a guided campaign or change
the user's expert-signal design. Do not launch a new selection/training queue
just to produce this design verdict.

## Requested closing summary

1. Three strongest aspects and three highest-risk assumptions.
2. Smallest sufficient M0/M1 implementation and exact tests to write first.
3. Required mathematical/business corrections, with concrete replacements.
4. Which proposed decisions can safely remain configurable rather than delaying
   implementation; which must be frozen before measuring outcomes.
5. What would falsify the mechanism design even with ideal model outputs.

The goal is an executable, auditable strategy specification and independent
review, not a guarantee of market profit or a paper claim made before evidence.
