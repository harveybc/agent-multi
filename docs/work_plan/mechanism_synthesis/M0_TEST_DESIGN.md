# M0 test design, before implementation

Date: 2026-10-28. Applies after audit 3326f898 and owner contract ba2af54a.
CPU only, standard-library rational arithmetic; no trading performance claim.

## Exact mathematical requirements

1. Two-player coordination table: AA=0, AB=BA=-1, BB=2. Both AA/BB are
   Nash; only BB is efficient for scalar account utility. Three-choice extension
   adds CC=3 with off-diagonals -1: three equilibria, one global maximum.
2. Three-player finite full-policy replay with a state-changing abstention
   reference: difference-reward deviations equal potential changes per scenario.
3. At one profile, evaluate exactly 1 + sum(k_i-1) distinct payoff profiles to
   certify all unilateral deviations. Twenty players with 27 policies require
   521 profiles, not 27**20. This does not enumerate all equilibria.
4. Every coordinate is checked, including the last library member; exact
   rational arithmetic detects improvements smaller than float tolerances.
5. A global maximum has epsilon=0. Nash may be dominated. Full enumeration is
   budgeted and refuses an oversized joint search before calling the payoff.
6. Pareto maximizes all signed objectives, keeps duplicate non-dominated
   outcomes, and rejects mismatched dimensions and non-exact values.
7. Five redundant closers: common value 10 iff any acts. Leave-one-out credit
   at all-act is zero for each; exact role Shapley is 2 each over 32 coalitions.
8. Shapley is descriptive, not an interchangeable reward: demonstrate a
   unilateral change increasing Shapley while decreasing account utility.
9. Frozen reference policies and cached exact payoffs; reject invalid player,
   profile, reference, dimensions, sizes and non-rational outputs explicitly.

## Independent audit witnesses

At heuristic-strategy commit b4b19ed, extract only the inspected `validity`
function using Python AST; do not import the repository or execute training.
Record its source SHA and execute crafted dictionaries demonstrating:
empty PASS, vacuous-control corrected PASS, and one-control-fails/one-passes
choosing the corrected rule. These are code behavior witnesses, not replays of
Satoshi's reported market matrices.

Also calculate exact counterexamples for negative-optimum 90% retention,
translation dependence, zero redundant difference rewards versus Shapley,
and a gap loss exceeding planned stop risk. Preserve input/output pairs.

## Boundaries

No finite-game fixture validates a native execution engine, signal curves,
business calendars, stochastic learning, market returns or a global equilibrium
outside the declared library. M1 remains design-revision work, not qualified.
Do not change a test to obtain a pass. New mathematical counterexamples augment
the suite with a documented revision. All proposed M1-M6 tests retain NOT_RUN.
