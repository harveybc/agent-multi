# Mechanism theory, cross-audit disposition and M0 delivery

Date: 2026-10-28 (America/Bogota).
This addendum governs over proposal 112 where they differ. The current business
contract is the owner's no-daily-cap clarification in commit ba2af54a.
Independent review received: Satoshi 3326f898 (REVISE before M1, allow M0).
Our guided-lane audit: 660511a0, linked below. Historical documents remain intact.

## 1. The mathematical object being implemented

Let Omega be a frozen finite manifest of exogenous scenarios, each with exact
weight w_omega summing to one. Let theta describe the arbiter, risk profiles,
signal interpretation, business contract and objective preference. Player i
chooses a full causal policy pi_i from finite library Pi_i. A policy may use
its admitted report history; controller state and filter memory are explicit
parts of the transition state, not hidden new neural features.

For fixed theta and omega, replay the entire episode:

    s[t+1] = T_theta(s[t], M_theta(s[t], reports[t], pi), world[omega,t])
    Phi_theta(pi) = sum_omega w_omega * J_theta(trajectory(theta,pi,omega))

Only available observations can enter reports. The synthetic label/oracle plane
can inspect the world for evaluation, never for an online causal report.
Fixed exogenous noise is keyed by world, time and event; changed actions do not
shift random streams. Counterfactuals rerun other policies from their initial
memory on their changed states; they do not reuse their factual actions.

The resources are simultaneous slot, margin, planned risk capacity and time.
There is NO daily entry quota. Profit/loss updates account equity once. A
close releases the slot and margin, making a later legal entry possible.
The engine retains external broker restrictions and weekly 12h/12h boundaries.

## 2. Potential identity, exactly scoped

Choose fixed reference pi_i^0, normally abstention. Define

    U_i(pi) = Phi(pi) - Phi(pi_i^0, pi_-i).

For an alternative pi_i', with other policies fixed:

    U_i(pi_i',pi_-i) - U_i(pi_i,pi_-i)
       = Phi(pi_i',pi_-i) - Phi(pi_i,pi_-i).

The second term cancels because its reference trajectory is independent of
the player's actual policy. This holds per scenario and therefore for the
frozen weighted manifest. It is an identical-interest/team optimization in
incentive differences, not a newly proved truth-telling auction.

For a finite policy library, any global maximizer of Phi is pure Nash. A pure
Nash profile only excludes improvements in one player's complete policy at a
time; joint changes may improve it. Strict better responses terminate because
there are finitely many profiles and the potential strictly increases. Do not
move on equal-payoff ties or assume neural gradient training is best response.

The identity is algebraic. Statistical intervals concern generalization beyond
the manifest, not uncertainty about that identity. M0 uses Fraction arithmetic
to eliminate rounding ambiguity. Later floating-point rollouts require declared
numerical tolerances and retained differences, not a claim of rational exactness.

## 3. Computation and certificate meanings

At a fixed profile, exhaustive unilateral certification evaluates

    1 + sum_i (|Pi_i| - 1)

distinct policy profiles, each across the chosen scenario manifest. For 20
players and 27 policies each this is 521 profiles (520 alternative deviations).
Finding all joint profiles/equilibria instead involves up to 27**20 profiles.
This distinction corrects the original proposal's loose complexity language.

    epsilon(pi) = max(0, max_i,pi_i' [Phi(pi_i',pi_-i) - Phi(pi)])

Epsilon zero is exhaustive only for the declared policy library AND manifest.
A stratified subset may certify that subset exactly; it cannot certify omitted
scenarios. Heuristic best-response search finds profitable deviations but cannot
upper-bound undiscovered ones. Evidence must name both axes of coverage.

M0 fixtures enumerate all equilibria in 2x2 and 3x3 games; a separate 20x27
fixture certifies one profile in 521 calls without enumerating the joint space.
Budget exhaustion returns a non-accepted state, not a Nash certificate.

## 4. Pareto and preference selection

Account objective vector: net return, Sharpe under a fixed sampling convention,
negative drawdown. Undefined metrics stay undefined outside explicitly finite
solver fixtures. First apply hard business/coverage constraints. Then compare
non-dominated outcomes. A finite archive is not a certified global frontier.

To analyze Nash, freeze one scalar preference Phi; repeat for declared preference
variants if needed. The account Pareto frontier and per-player credit table are
different objects. Do not spend on a separate player-credit Pareto solver.

Report all found equilibria, global best when enumerated, best found otherwise,
and basin/outcome frequencies for the actual update rule. Worst-equilibrium
selection is an optional stress criterion, not the assumed behavior of PPO or
of supervised per-head training. Re-solve after changing theta or preference.
Sharpe alone does not automatically imply bad behavior, but it cannot replace
net return, exposure, turnover and risk reporting or choose its own winning
objective family. No net-floor/90% preference is silently imposed on the owner.

## 5. Credit is not automatically the learning reward

With five identical proposers, removing any one may leave the same account
trajectory: every difference reward is zero. This is correct marginal credit
at that joint profile, not an accounting bug. Group Shapley averages marginal
contributions over all coalitions with outsiders fixed. For five identical
closers of value 10 it allocates 2 each, using 32 coalition evaluations.

It is diagnostic attribution, not an incentive-preserving reward replacement.
Counterexample, player 1 rows 0/1/2, player 2 columns 0/1:

| Phi | 0 | 1 |
|---|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 4 |
| 2 | 10 | 3 |

Changing (1,1) to (2,1) reduces account value 4->3 while increasing player 1's
Shapley 2->6.5. Difference reward correctly falls 4->3. Thus M5 must either
train under the declared team objective or identify a different learning game
and analyze it. Shapley does not automatically cure sparse-learning problems.
Role-conditional Shapley values need not sum across roles to account PnL.

## 6. Position-dependent close semantics before M1

Satoshi's core observation is accepted: an action scalar from market-only
features plus a position-insensitive threshold cannot in general be sufficient.
Proximity to SL/TP alone does not prove two different decisions, so use a fully
specified separating fixture, not a verbal assertion:

* current executable quote 100, future path 98 then 104, no costs for this
  analytical witness; long A has SL99/TP106, long B SL97/TP104;
* A: closing now gives 100, continuation hits 99 first, so CLOSE is optimal;
* B: continuation reaches 104 without hitting 97, so HOLD is optimal.

These are absolute exit proceeds comparisons with the same unit position;
entry price is sunk for the current close-versus-hold comparison. The native
engine version must add quotes, costs, deadlines and execution order.

The controller can access position state, but no neural state vector is added.
Candidate signal contracts: (a) a market-only function/surface over remaining
barriers with uncertainty; (b) richer market signals plus an explicit external
position-conditioned decision model. Merely changing thresholds by distance
does not establish sufficiency. Preserve the 20 binary reports as proposals;
any auxiliary close representation is a versioned extension and must pass the
separating fixture before adoption. Curves are not yet mandated for our lane.

A future pointwise mean curve is not necessarily sufficient for expected
first passage or optimal stopping. Joint event order, spread, financing,
execution gaps, horizon censoring and boundary extrapolation require tests.
No access to next-open price is permitted before that quote is observable;
only the execution layer may calculate fills from a later realized quote.
No interpolation through discontinuous barrier outcomes without a declared
error certificate. This is design work, not an implemented live closer.

## 7. Policies, cross-scale arbitration and oracle validity

Extend the policy library with immediate response, k-consecutive valid ticks,
and explicit EWMA/Kalman candidates. Filter the report history, not the raw
market input or its TRAIN-fixed scaler. Reset by position; measure elapsed
time across missing reports. Include a real urgent reversal that persistence
misses and a false reversal that filtering suppresses. Correlated noise is a
test axis, not automatically a measured financial property.

Cross-scale coupling is a candidate matrix with a zero-coupling control. No
scale gets an absolute veto. Long and short horizon scores require common
units and declared continuation beyond a short report's support. Evaluate
parameter complexity before introducing all 25 matrix genes.

An oracle failure diagnoses a conflict; it does not automatically authorize
relaxing the witness. EMPTY, NO_SUPPORT, UNVALIDATED and CONTROL_CONFLICT cannot
pass acceptance. Correct independently demonstrated test defects prospectively,
retain old verdicts and use fresh confirmation. Separate ideal-position exit
regret from account utility: closing earlier can improve subsequent opportunity
allocation even when it is not the isolated-position optimum.

Expand scenarios in named blocks: decreasing volatility, trend magnitude,
superposed horizons, phase shifts, spread/cost shocks, gaps with/against the
position, calendar boundaries and missing observations. A scenario's dominant
horizon is not the specialist identity. No mandatory 80% scale-winning quota
or guaranteed monotone profit degradation with noise is justified in general.

## 8. Disposition of Satoshi's 21 findings on our proposal

| Finding | Disposition |
|---|---|
| F01 | Accept: exact potential/team interpretation and per-scenario identity |
| F02 | Accept: Nash is coordinate-wise; worst-equilibrium choice must name learner model |
| F03 | Accept: linear unilateral certificate, exponential joint search |
| F04 | Accept: exact frozen measure versus out-of-manifest statistical uncertainty |
| F05 | Accept scalar insufficiency; prove with explicit separating paths; curves are candidate, not automatically sufficient |
| F06 | Accept redundancy; amend recommendation: group Shapley diagnostic, not guaranteed compatible learning reward |
| F07 | Accept filter/persistence contrasts; reported percentages are not revalidated for our lane |
| F08 | Accept cross-scale false/true reversal tests; matrix is an ablation, not a proven remedy |
| F09 | Accept joint metrics and reentry controls; Sharpe claims conditional on accounting and objective |
| F10 | Accept: no separate player-credit Pareto search |
| F11 | Resolved by owner clarification: 50, no daily cap, weekly 12h/12h; shared contract ba2af54a |
| F12 | Accept oracle validity requirement; reject automatic corrected-PASS on any oracle failure/vacuity |
| F13 | Accept explicit additional scenario blocks and per-block coverage |
| F14 | Amend: neither mirror-loss nor n*planned-risk/complementary-win-rate is universal under gaps/costs |
| F15 | Accept: isolated exit optimum and account allocation objective reported separately |
| F16 | Accept explicit conversion/lot equations; proposed simple formula needs account currency, fees and step size |
| F17 | Accept every proposal accounted for, verdict-based summaries, seed and mirror tests |
| F18 | Accept per-worker memory admission, lifecycle-aware progress and typed failure |
| F19 | Accept cost measurement; subset certificate never relabeled global by being final |
| F20 | Accept proper scoring and calibration; reject requirement that synthetic inflation can never improve Phi |
| F21 | Accept dominant-horizon axis; reject arbitrary 80% matching-scale winner as universal correctness |

Detailed cross-lane counterexamples and remedies:
[guided-lane audit](../audits/MUSASHI_VERDICT_SATOSHI_GUIDED_MECHANISM_2026_10_28.md).

## 9. Delivery and next executable boundary

M0 implementation: `app/mechanism_games.py`, exact certificates, budgeted joint
enumeration, strict best response, Pareto dominance witnesses and role Shapley.
Sequential reproduction: `python tools/mechanism_m0.py` (stdout only). It closes
all its bounded tasks immediately; no watcher or GPU needed. Tests:
`tests/unit/test_mechanism_games.py`, `tests/unit/test_mechanism_m0_runner.py`.

M0 cannot authorize M1 automatically: next work is the position-dependent signal
and oracle-validity contract plus analytical execution tests. It cannot authorize
financial fits, feature selection or a claimed optimized 20-expert strategy.
All new source and contract identities are included in its exact fixture report.
