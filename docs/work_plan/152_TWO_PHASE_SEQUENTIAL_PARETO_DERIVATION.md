# Two-phase sequential account Pareto derivation

Date: 2026-10-30. Owner: Musashi. Priority: before phase 2 of floating
feature selection. This consolidates the owner's existing specification;
it does not introduce a new trading contract or replace a running search.

## What existed and what was missing

Plans 118/122/123 define the retrospective opening reference: at each TRAIN
origin, evaluate BOTH directions against EACH of the next K=3 chronological
ZigZag pivots, independent of peak/valley ordering. Measure the intervening
path, costs, adverse excursion and duration. The horizon label does not impose
an artificial holding-time cap. Plans 125/126 measured the resulting profiles.
Plans 130/131/132 separately define fixed-position close/HOLD counterfactuals.
Plan 134 defines whole-account opportunities and the return/drawdown frontier.

The missing connection was an explicit continuation-value calculation in the
decision rule. The running DEAP searches coefficients of a LOCAL bid scorer
and evaluates its resulting COMPLETE accounts. That is an empirical frontier
of the searched policy library, not an exact sequential Pareto planner.
`interval_best` alone does not fix this: its additive fixed trade gains omit
balance-dependent sizing, early close and path-dependent account drawdown.

## Fixed business and information boundaries

- One position at a time; leverage 50; margin at most `q * 0.1 * balance`.
  The 0.1 is applied ONCE. Broker lot rounding may reduce actual margin.
- No strategy daily-entry cap; opening begins 12 hours after market reopening;
  all positions close 12 hours before market close. Broker protections prevail.
- When flat: NOP or BUY/SELL at h in {3,6,12,24}.
- When occupied: HOLD or CLOSE the EXISTING direction. No opposite opening
  in the same decision. A close releases the slot for the next legal opening.
- All four horizon closers may advise on an existing position, but they refer
  to the SAME entry, direction, SL, TP, units and costs. They cannot replace
  those by their own ideal initial conditions to claim a better exit.
- Future pivots and known future paths belong ONLY to TRAIN design/oracle
  evaluation. Neither is an observation or an executable causal forecast.

## Numerical references, not invented optimum coefficients

Source: `docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json`.
SHA256: `99d7d4dc92f6437c84b8b36de4c29a1ba474489a7c07444899699c00203c3c0e`.
Units below are pips; these are TRAIN medians, not individual future labels.

| h | BUY TP reference | BUY MAE reference | SELL TP reference | SELL MAE reference |
|---|---:|---:|---:|---:|
| 3 | 20.0 | 7.4 | 19.6 | 7.7 |
| 6 | 24.0 | 8.5 | 23.4 | 8.9 |
| 12 | 35.3 | 11.9 | 35.8 | 12.7 |
| 24 | 42.1 | 16.0 | 43.1 | 16.2 |

The native proposal computes:

```
dSL = aSL * MAE_reference + bSL * sigma + buffer
dTP = aTP * TP_reference - bTP * sigma
SL = entry - direction * dSL
TP = entry + direction * dTP
q = quality * min(1, rho * balance / loss_at_q1)
margin <= q * rel_volume * balance
```

Tick rounding is outward for SL and inward for TP, followed by executable
quote checks and integer unit sizing. `loss_at_q1` includes loss to the stop,
execution costs and conservative positive funding reserve. Current controls
are aSL=aTP=1, bSL=bTP=0, buffer=1 pip, rho=0.005, quality=1. Their source is
explicitly `ENGINEERING_CONTROLS_NOT_OPTIMIZED`: the optimized ZigZag parameters
do NOT make these risk coefficients optimal. q is recomputed from actual
balance, price, costs and stop geometry, not fixed by horizon or counted twice.

## Exact finite formulation

Let x be a complete native account state: time, balance, existing position,
execution/calendar state, peak marked-to-market equity and largest drawdown
so far. Any controller history that affects legal transitions also belongs
to x. On a fixed design path w, the native transition is T_w(x,a).

For terminal equity E_T, initial equity B0, peak M_t and marked equity E_t:

```
R = 100 * (E_T / B0 - 1)
D = 100 * max_t (1 - E_t / max_{u<=t} E_u)
F_T(x) = {(R(x), -D(x))}
F_t(x) = ND( union_{a in A(x)} F_{t+1}(T_w(x,a)) )
```

ND retains vectors not strictly dominated componentwise (both maximized).
This is a set, not a unique optimum. A declared risk preference chooses a
member, e.g. maximize R subject to D <= d. Without specifying d or a utility
there is no mathematically unique 'best Pareto' parameter vector.

Proof by backward induction: every feasible continuation has a legal first
action and a feasible suffix. Their union enumerates all feasible outcomes;
removing dominated complete-account outcomes preserves the nondominated set.
Do NOT prune two partial paths merely because current profit/DD is worse:
their balance, occupied slot, remaining SL/TP and future opportunities differ.
Do NOT sum weekly drawdowns; carry the peak and drawdown state across time.

SL/TP/volume alone cannot produce numerical payoff matrices: a path, timing,
costs, legal actions and terminal rule are also required. The finite native
calculation uses those saved profiles with explicit synthetic paths. It is
an exact enumeration of the DECLARED finite action set, not the full TRAIN
frontier, a continuous-volume optimum, or a causal deployment policy.

## Opening and closing are connected but have different diagnostics

For a chosen utility U on complete account outcomes, define:

```
Q(x,a) = max_{feasible suffixes} U(account after a and suffix)
opportunity_cost(x,a) = max_b Q(x,b) - Q(x,a)
close_account_advantage = Q(x,CLOSE) - Q(x,HOLD)
```

A short trade can beat a larger long trade by enabling better later entries.
A long trade can also beat repeated short trades. NOP can preserve an even
better future opportunity. None of these wins is hardcoded by regime name.
Without choosing U, compare the corresponding conditional Pareto sets instead.

Separately, for the SAME already-open position, enumerate executable exits
before its passive SL/TP/deadline and include passive HOLD as a control:

```
close_regret_cash = best_feasible_exit_net - executed_exit_net
close_regret_risk = close_regret_cash / original_cash_risk
close_vs_hold = executed_exit_net - passive_exit_net
```

For a path reaching SL before TP, the optimum may lie at an intermediate
extremum before SL, not at an opening-reference pivot. Native gap/protection
preemption removes impossible exits. A locally inferior close can still improve
the account by releasing the slot; publish BOTH diagnostics, never substitute
one for the other. No requirement forces a preset percentage of early closes.

## Competition, opportunity cost, Nash

Blocked agents' individual hypothetical profits are diagnostic alternatives,
not additive lost account cash: many use the same time slot. Evaluate feasible
sequences jointly. With fixed scalar account utility Phi and a fixed baseline
policy per participant, difference utility is:

`u_i(pi) = Phi(pi) - Phi(pi_i_baseline, pi_-i)`.

A unilateral utility change equals the account utility change, so a pure Nash
equilibrium is coordinatewise stable in that FINITE policy game. It is not a
certificate of a global optimum or the full Pareto frontier. The exact account
enumeration below uses a cooperative controller and reports no fictitious Nash
certificate. DEAP is useful later for larger policy/parameter spaces; its
approximation can be measured against these small exact references.

## Test design frozen before implementation

| ID | Expected property | Evidence |
|---|---|---|
| SP01 | two short feasible trades can dominate one longer trade | rational finite tree |
| SP02 | converse case and NOP-first optimum remain representable | rational finite tree |
| SP03 | incomparable return/DD outcomes both survive | exact nondominance test |
| SP04 | CLOSE and HOLD both exist only in occupied phase | native tree and trace checks |
| SP05 | all 8 opening alternatives exist, fixed q recipe, no overlap | native trace checks |
| SP06 | native cash, weekly flat, costs and mirror paths | independent terminal replay |
| SP07 | exhausted node budget never emits a complete frontier | failure test |
| SP08 | repeat produces identical calculation and source hashes | cold rerun |

Native fixtures: smooth rise/fall, oscillating rise/fall, reversal and mirror;
each starts flat or with an existing BUY/SELL. The bounded decision window
ends at the actual weekly force-flat, not an invented favorable liquidation.
Native outcome rankings are measured, not rewritten to force a desired winner.
No claim of exhaustive market-scenario coverage follows from this small proof.

## Work order

1. Implement exact finite tree calculation and its independent tests.
2. Emit conditional payoff tables, terminal frontiers and native replay proof.
3. Preserve the current DEAP runtime as the local-policy comparator.
4. Construct the opportunity graph from the existing TRAIN K=3 alternatives,
   keeping nonwinning endpoints and executable early exits. First qualify
   bounded exact subproblems on those data; a small synthetic proof must not be
   presented as the four-year numerical frontier. The full graph also needs
   balance and equity-peak state, not additive independent trade scores.
5. Extend the planner to a bounded causal scenario tree, with one shared action
   for paths having the same observed history (non-anticipativity), then compare
   it with exact references. Optimizing each future independently and averaging
   its optimum would falsely grant the causal policy future knowledge.
6. Qualify learned reports and sequential allocation before model-specific
   floating selection. The small oracle calculation does not authorize it.
