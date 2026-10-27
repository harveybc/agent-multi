# Priority: optimal early exit of an existing position

Date: 2026-10-27. This is the current priority before financial selection or
strategy promotion. It refines plans 105-106; it does not certify the rejected
strategy or confuse an entry optimum with an exit optimum.

## Contract

An order already exists, with fixed side, entry, size, SL and TP. Entry is an
artificial test precondition, not an optimization variable. At decision t the
exit target is the best net liquidation still executable before the FIRST SL
event. An earlier TP, margin protection or compulsory close ends that order and
also bounds the search. A recovery after termination and a peak before t are
not admissible action targets. Native H1 decisions execute at the following
open; this is not an intrabar tick simulator.

For executable net closing values C_i and automatic terminal value T:

- H_i = max(T, C_j for j > i), the best continuation still available.
- V_i = max(C_i, H_i), the remaining exit optimum.
- A_i = C_i - H_i: positive favors CLOSE; negative favors HOLD.
- The local error of closing now is V_i - C_i, not an entry/pivot error.
- At episode end report V_0 - realized_net as a separate whole-trajectory error.
  This measures missed earlier opportunities; it is NOT the current action label.
- Earliest executable exit wins an exact tie. Costs include spread, commission,
  slippage and financing. No later recovery beyond SL/TP/calendar is counted.

Initial monetary SL risk R is fixed for the episode. The dense diagnostic
reward is (Q_action - V_i) / R, where Q_close=C_i and Q_hold=H_i. It penalizes
premature and delayed exits. With gamma=1, its sum equals the negative realized
normalized exit error. Compare it against terminal-only negative normalized
error, not against a different economic objective. These future-aware rewards
and labels are SYNTHETIC DIAGNOSTICS ONLY; they never enter observations or a
live financial policy. This is not a claim that a market oracle exists.

## Executed engineering evidence

- 30 focused tests passed, including literal peak timing, losses, ties,
  telescoping rewards, future-label isolation, deterministic DEAP replay and
  two-sided native-engine comparisons.
- Full corpus construction checked 208 cases: 144 training/calibration and 64
  confirmation, with disjoint peak durations/amplitudes/initial-mark offsets.
- All 6,840 executable close alternatives were executed on native engine copies
  and reconciled against independent quote/fee arithmetic. The full-position
  reference's commands match the optimum of those executed alternatives.
- 193 cases have an optimal early exit; 15 retain the automatic exit as optimal.
  This is designed position-management coverage, not a natural trade-frequency
  estimate. No 72% quota is imposed.
- Corpus paths use controlled curved reversals, both directions, initial
  profitable/neutral/losing marks and normal/tripled spreads. They are positive
  learning controls, not proof for every stochastic market regime.

The original quote grid completed 1,312,200 full episodes, zero failures in its
declared checks, at 2026-10-27 17:16:04 UTC (12:16:04 Bogota). Its receipt is
ENGINEERING_GRID_COMPLETE_NOT_STRATEGY_RELEASE. Many cells have no close
opportunity; that result cannot replace this exit-learning qualification.

## Automatic bounded sequence

Configuration: `configs/validation/optimal_exit_study.json`.
Runner: `tools/optimal_exit_study.py`, CPU-only, exported frozen code.

1. Build/authenticate the native exit menus and causal observations. Retain
   every executable exit and automatic terminal, with hashes and receipts.
2. Report immediate-close, passive-to-protection and ideal-optimum baselines
   on both partitions and directions. A constant policy must not pass by merely
   producing many closes.
3. Nine DEAP searches: three seeds x negative normalized exit error, synthetic
   episode Sharpe, and episode Sharpe + mean return/initial capital. Population
   20, generation cap 100, patience 10, relative increment 1e-8. Record all unique
   genomes, generations and boundary hits; no cross-objective winner chosen by
   one objective's own score. Activity is constant (one seeded order per case),
   so no unidentifiable activity reward is added.
4. In this FIRST exit-only calibration, optimize two real routing coefficients:
   long/short CLOSE log-odds biases, each [-2,2]. The five-command categorical
   space remains intact; no continuous output is quantized into action classes.
   Perfect +/-6 synthetic log-odds must retain every ideal decision across the
   bounds. Fixed noise/delay output panels exercise imperfect heads. This is
   output calibration, not a claim of calibrated learned-model probabilities.
5. Six sequential PPO controls: dense versus terminal regret reward x seeds
   7/19/43, 32,768 steps each, no early stopping. Use the grouped TCN policy
   class and 13 observations at stride 2. Gamma=1 is explicit for finite exit
   regret; this is not a reproduction of C4's gamma=0.99 experiment.
6. Evaluate all training and confirmation cases, reload weights, and report
   regret mean/median/p90, exact-optimum share, early-exit share and direction.
   A learned control passes only if it beats BOTH blind baselines in BOTH
   partitions and BOTH directions on normalized regret. Negative results remain.
7. Close and retain artifacts automatically. No financial selector, TEST reader,
   live trader or additional experiment is enabled by this diagnostic closure.

The synthetic policy observes three causal price channels, all other channels
zero, under the full grouped architecture. These are NOT the selected financial
features. No order-state feedback, rolling z-score or clipping is introduced.
An observation-prefix/future-label independence test guards this separation.

## Broader management matrix and later joint optimization

`configs/validation/optimal_position_grid.json` declares 874,800 long/short
position probes across the existing stochastic trend/volatility/mirror/gap axes.
Its exit oracle searches the full remaining position lifetime, not just terminal
direction or a four-bar return. The retained horizon field affects subsequent
entry diagnostics only. The earlier 2,624,400-case finite-horizon extension in
plan 106 was not deployed; this full-lifetime design supersedes it.

TP/SL/volume remain fixed during the first exit-only study so DEAP cannot erase
the difficult close cases by changing entry/protection. The subsequent joint
strategy optimization must include those six parameters AND the two close
parameters, retain this fixed-position acceptance panel, and use fresh paired
confirmation. It is not declared complete or automatically qualified by the
two-parameter study. Further representation/financial experiments remain behind
the strategy and learned-control checks. Entry selection and exit targets must
remain separately named in reports and configuration.

## Documentation map

- 102: audit rejection and correction requirements.
- 103: quote engine and first full-episode grid.
- 104: failed learned-policy trade lifetimes and fixed risk limitations.
- 105: explicit two-sided CLOSE/HOLD counterfactuals.
- 106: opportunity-capture coverage, no-witness accounting.
- 107 (this document): current optimum-before-SL definition and priority queue.

Keep these versioned GitHub documents as the current source of truth. A wiki or
documentation site can later index them without creating a conflicting second
specification. Final audit must retain known failures and distinguish engineering
execution, exit calibration, synthetic learning and financial generalization.

## Deployment receipt

Exported code `61c3e88e`, quote engine `842320a`. The frozen producer completed
its own smoke at 18:59:04 UTC on 2026-10-27, then the full corpus and baselines.
At 18:59:12 UTC (13:59:12 Bogota) the complete study was executing DEAP objective
negative_exit_regret, seed 7. Smoke had only 512 learning steps and no learned
control pass; it certifies wiring, not learning. The full budgets above remain.

Three enabled user services provide the study, the dependent management grid
and its read-only producer follower. The study is a sequential oneshot job;
the grid start job waits for its successful completion. Failed engineering
execution does not silently admit a successor. Negative scientific results are
retained as completed diagnostic results, never financial promotion.

Study CPU quota 2 cores, memory soft/hard 4/6 GiB; subsequent grid quota 6 cores,
memory 6/8 GiB; auditor quota 1 core, memory 1/2 GiB. All are CPU-only synthetic
engineering work. The study and large grid do not run simultaneously. Original
sealed experiments and the completed earlier grid remain unchanged.

No measured full-study ETA was available at first launch. Its current stage,
training progress and per-unit elapsed times are retained automatically. Do not
extrapolate the small smoke as if it were a full-budget learned result.
