# Musashi Order: Episodic Activity Fitness and Easy-to-Normal Continuity Correction

Date: 2026-08-20 America/Bogota  
From: General Musashi, independent auditor  
To: General Satoshi, technical lead  
Owner priority: establish a learnable active policy before any further DOIN optimization  
Runtime authority: none. This document launches no GPU job and changes no running venue.

## 1. Immediate Disposition

The P1LR decision identity `ac0941e7bdb1a163` is stopped on all four workers.
Its records remain immutable diagnostic evidence, but none may be called a
champion or used for live succession. Do not restart that identity.

No DOIN optimization, long factorial, L2 search or fleet GPU campaign may run
until the local acceptance sequence in section 8 passes. CPU implementation and
bounded local smoke work are the only Front-1 work authorized by this order.

## 2. Reproduced Failure

For seed 404 / `P1E_LR3E5`:

- easy stopped at epoch 100 although its declared maximum was 1000;
- patience was 60 after epoch 40 and the selected easy checkpoint was epoch 1;
- epoch 1 closed 1722 trades on the 5.3-year fit interval, but produced only 7
  monitor trades and 3 inner-validation trades under normal action semantics;
- only 5 of 100 easy epochs remained viable under the normal threshold;
- the maximum easy fit equity was 12426.09 at epoch 2 (+24.26% over 5.3 years),
  not an exceptional converged easy result;
- the maximum monitor return was +4.07% at epoch 15 while fit equity was
  8592.24, so it was not a stable generalizing optimum;
- normal ran 1000 epochs, ended with zero train and validation trades, and the
  final outer-validation checkpoint closed only 5 trades during all of 2024.

The current step reward is normalized equity delta. A flat policy earns zero,
while an active policy that is still learning commonly earns a negative value.
The paired checkpoint comparator uses weekly risk-adjusted utility and requires
only one trade per split. Trade count does not improve its score. The resulting
ordering makes passive collapse locally attractive.

## 3. Historical Reference, Not a Blind Port

Recover and cite the final relevant historical implementation from `gym-fx`
Git history. At commit `2a94cb3`, `environment_plugin_automation.py` used:

```python
if num_orders < 1:
    fitness = -200
elif margin_call:
    fitness = final_reward
elif 0 <= sharpe_ratio <= 1:
    fitness = final_reward + (profit_factor * num_orders) * (
        sqrt(num_orders) + sharpe_ratio)
elif sharpe_ratio < 0:
    fitness = final_reward + (profit_factor * num_orders) * sqrt(num_orders)
else:
    fitness = final_reward + (profit_factor * num_orders) * (
        sqrt(num_orders) + sharpe_ratio**2)
```

The invariant to preserve is episodic, not per-step: `NOP` is a valid action
and carries no penalty merely because the agent waits. Only an episode that
finishes with zero closed trades receives the inactivity sentinel.

Do not copy the historical formula without dimensional normalization and
counterexample tests. It is motivating evidence, not the new contract.

## 4. Required Objective Contract

Implement a typed, versioned `episodic_activity_economic_fitness` plugin. It
must consume independently recorded facts and publish every component:

- total return;
- maximum drawdown fraction;
- Sharpe (nullable when undefined);
- closed trade count;
- scored rows and scored years;
- annualized trade rate;
- activity utility;
- economic utility;
- final scalar selection value;
- branch/reason used by the piecewise function.

Required semantics:

1. `trades == 0`: return an explicit configurable sentinel, initially `-100.0`.
   The sentinel applies once at full-episode evaluation, never per bar.
2. `trades > 0` and return <= 0: rank first by movement toward zero loss while
   retaining a bounded activity contribution. Every finite active diagnostic
   fixture must rank above the zero-trade sentinel, but catastrophic loss must
   still remain economically bad and separately visible.
3. return > 0 and Sharpe <= 0 or unavailable: positive return multiplied by a
   bounded activity utility, with drawdown penalty.
4. return > 0 and Sharpe > 0: same base plus a bounded Sharpe bonus.
5. Trade activity uses an annualized rate so differently sized splits remain
   comparable. Use a declared bounded curve: steep rise from zero, target
   plateau, gradual overtrading decay. Do not invent the target from the one
   observed outer result. Materialize candidate ranges from historical ETH
   train/monitor traces and run a cheap sensitivity table before choosing it.
6. No multiplication may reverse ordering because a return is negative. No
   NaN, infinity, nullable Sharpe or zero variance may become an accidental
   champion.
7. Train-monitor and inner-validation are paired. The sealed test remains
   unopened and outer validation is not used to tune the activity curve.

## 5. SAC Learning Signal

Do not add a per-step penalty for `NOP`, flat exposure or elapsed bars. Waiting
is part of the policy.

Implement and compare, locally, two compatible learning signals:

- A: existing economic step reward plus the episodic terminal objective;
- B: existing economic step reward plus a bounded reward emitted only when a
  trade closes, derived from realized economic outcome, plus the same terminal
  objective.

An opening or closing event alone must not earn a positive reward. Churn cannot
manufacture fitness. The terminal zero-trade sentinel is primarily checkpoint
selection authority; if injected into the SAC terminal transition, its scale
must be normalized relative to cumulative episode reward and tested for critic
instability.

## 6. Easy Training and Handoff

- easy maximum: 5000 epochs;
- easy patience: 80 eligible checkpoint evaluations;
- do not start patience until a declared minimum-learning floor has elapsed;
- record best epoch, terminal epoch, all component metrics and patience state;
- preserve actor, critic, target critic and topology byte-for-byte at handoff;
- preserve optimizer state in the primary continuity arm;
- fixed entropy configuration must remain identical; learned entropy state, if
  introduced later, must also transfer;
- do not silently transfer or discard replay. Compare two bounded arms:
  optimizer continuity with clean replay, and fully labeled replay continuity;
- normal must start from the selected easy checkpoint, never genesis;
- verify tensor hashes and component L1 distances (`0.0`) before the first
  normal gradient update.

The current discontinuity (`easy action threshold = 0.0`, normal threshold =
`0.1`) is a separate causal factor. First run easy and normal with identical
action semantics while relaxing only solvency/cost dynamics. Then compare one
declared annealed-threshold arm. Never mix threshold changes into the primary
solvency claim.

## 7. Mandatory Counterexamples

Unit/property tests must prove all of these strict orderings:

1. zero trades loses to each finite active learning fixture;
2. NOP on many individual bars receives no penalty;
3. with equal activity/risk, -5% return beats -20%;
4. with equal return/risk, target activity beats insufficient activity;
5. excessive activity loses to target activity but beats zero activity in the
   declared easy-learning fixture;
6. with equal return/activity, lower drawdown wins;
7. with equal positive return/activity/risk, positive Sharpe beats negative or
   unavailable Sharpe;
8. negative-return multiplication cannot reward a larger loss;
9. one trade cannot satisfy a production-promotion contract merely because it
   satisfies experimental mechanical validity;
10. an easy checkpoint whose actions do not survive normal semantics cannot be
    selected for handoff;
11. changing difficulty does not change any model tensor or topology before
    the first normal update;
12. no-trade, active-loss, active-profit and overtrading cases serialize all
    raw components into OLAP evidence.

## 8. Ordered Execution

WP0: reproduce the current ordering with a minimal deterministic fixture and
preserve before evidence.

WP1: implement the typed episodic objective and tests in sections 4 and 7.

WP2: add terminal/closed-trade SAC reward arms without penalizing NOP.

WP3: implement complete handoff identity and the action-threshold isolation.

WP4: CPU smoke on short data. It must show that the objective orders all cases
correctly and that gradients remain finite.

WP5: one local GPU smoke, not fleet-wide. Acceptance requires substantial fit
activity, nonzero monitor and inner-validation activity, improvement beyond the
initial checkpoint, and exact easy-to-normal tensor continuity.

WP6: bounded seed replication on one second GPU only after Musashi independently
reproduces WP0-WP5.

WP7: only after two seeds pass may a new experiment identity and fleet campaign
be proposed. The stopped identity can never be resumed.

## 9. Return Packet

Return one document containing commits, exact configs, before/after reproducer,
test commands, scalar-ordering table, per-epoch activity/economic curves, model
hash continuity evidence, runtime process inventory and explicit confirmation
that no long optimization was launched. State doubts directly. Close no audit
finding yourself.
