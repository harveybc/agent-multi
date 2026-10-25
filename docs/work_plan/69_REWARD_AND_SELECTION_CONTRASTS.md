# Learning rewards and strategy selection contrasts

Owner request: 2026-10-25. Status: planned sequence, not executable or deployed.
Depends on output contract 67 and its engine/learner acceptance tests. Do not
change action encoding, representation, reward and selection simultaneously.

## Current implementation

`RecipeNativeEnv.step` supplies, for each engine bar:

`r_t = 100 * (E_(t+1) - E_t) / E_initial`.

E is marked equity, including entry costs; exit costs and financing settle at
closure. Missing observation bars retain native accounting and aggregate
discounted rewards; the next bootstrap uses gamma**elapsed_bars. The original
engine tests establish undiscounted ledger equality, not discounted equivalence
of different booking conventions. Reward timing is a separate prerequisite.

SAC optimizes expected discounted reward plus entropy, not Sharpe. Actor and
critic losses are distinct from both the reward and the experiment ranking.
The completed activity grid's `app.recipe_activity.choose` selects eligible
final100k arms by mean TRAIN/INNER_A daily log return, then smaller gap, then
smaller INNER drawdown. Both periods must exceed two closed trades per scored
week with the existing daily cap and validity constraints. OBS_2024 does not
select. The generic SAC plugin also has a return-minus-drawdown fitness method,
but that is NOT the objective used by this queue. No DEAP runs in this queue.

## Sequential experiments

First establish the categorical control using the old equity reward. Freeze
the accepted action/risk mode, data, masks, capital, leverage, costs, learning
budget, seed set, evaluation checkpoints and scaling policy for the following
reward contrast. Compare fixed-risk and learned-risk modes separately in plan
67, not as confounded reward arms. No new feature transforms or expert labels.

Let g_t = log(E_(t+1)/E_t), d_t = 1-E_t/max_(s<=t) E_s and
m_t = max_(s<=t) d_s. Equity must be positive for the logarithm; bankruptcy
handling is explicit and tested, never hidden with an arbitrary epsilon.

| Arm | Learning reward | Purpose |
| --- | --- | --- |
| R0 | Existing 100*delta_equity/E_initial | Reproduce the accepted categorical control |
| R1 | 100*g_t | Test proportional compounded growth |
| R2 | 100*(g_t-lambda*(m_(t+1)-m_t)) | Penalize additional maximum drawdown, not every bar spent at the same drawdown |
| R3 | Differential Sharpe of causal net equity returns | Test online risk-adjusted learning rather than repeated full-period Sharpe |
| R4 | Scaled mixture of R1 and R3 | Follow-up only after individual components pass engineering and learning checks |

R2 coarse lambda candidates are 0.25, 1, 4, plus lambda=0 via R1. These are
declared experimental preference weights, not theoretically optimal numbers.
At gamma=1, R2 sums to 100*(log(E_final/E_initial)-lambda*MDD) if initial MDD
is zero. With discounting this identity changes; report it rather than claim
equivalence. Incremental drawdown prevents repeated occupancy penalties, but
still introduces path-dependent reward state. No account-feedback inputs are
silently added to the actor. Carry these limitations into critic diagnostics.

R3 follows the differential Sharpe definition of Moody and Saffell. Its running
moments update only after each observed reward. Predeclare smoothing, warmup,
variance-floor handling and reset/resume state, and check against the primary
formula before admission. Zero volatility and NOP-only episodes must not yield
infinity or spurious performance. This is reward bookkeeping, NOT rolling
z-score normalization of market inputs. No lookahead Sharpe from the final
episode is distributed into earlier observations or decisions.

Keep global reward magnitude/advantage normalization and entropy coefficient
explicit. Reward scaling is not generally neutral to SAC's entropy trade-off
or critic optimization; a PPO successor also has value-loss and entropy scale
interactions. Do not normalize online differently for each candidate without
declaring that as an additional treatment. Any scale calibration uses a common
TRAIN-only pilot set and freezes before the comparison. A floor/fallback must
be justified by measured numerical range, recorded and tested.

## Fitness after evaluation

These objectives rank frozen candidates; changing their ranking alone does
not retrain policies. Compute every component for every candidate once.
S = Sharpe from native daily equity; R = net profit/E_initial; D = MDD as a
fraction; A = entries/all scheduled business days in the evaluated period,
including zero-entry days. Also report closed trades/week separately. Never
change denominator to active days or multiply returns from unequal periods
and call them the same total return. Fix Sharpe annualization/calendar convention.

| Fitness | Formula | Initial use |
| --- | --- | --- |
| F0 | S | Sharpe reference |
| F1 | wS*S/sS + wR*R/sR | Owner's Sharpe plus return |
| F2 | F1 + wA*A/sA | Owner's activity composite, diagnostic contrast |
| F3 | R-lambda*D | Transparent return/drawdown preference |
| F4 | F1-wD*D/sD | Mixed performance with explicit drawdown cost |
| F5 | Calmar and dependence-aware Sharpe uncertainty | Secondary diagnostics; undefined denominator is not a winner |

Eligibility precedes ranking: unchanged business rules, no ruin, finite required
metrics and activity strictly >2 closed trades/week. NOP-only remains a valid
behavioral comparator but not an activity-qualified candidate. F2 is bounded
by the daily cap but can still encourage unnecessary trades; it is not the
default per-step reward. Compare net costs, duration and NOP behavior explicitly.
A saturating activity-preference variant can be tested separately if needed.

Scales sS,sR,sA,sD are positive and shared across candidates, derived from one
declared TRAIN reference/pilot set and frozen. Never use each arm's own scales
to compare arms. Start equal weights after scaling (not claimed optimal), then
coarse two-term ratios and local refinement internally; use nonnegative weights
with fixed sum to remove redundant global scaling. Calibrate/refine on TRAIN
internal blocks only; report performance on later fixed blocks without changing
years. Previously inspected years remain development evidence, not fresh tests.
Keep the complete native metric vector and Pareto comparisons: different
fitness values are different preferences, not directly comparable scores.

## Transfer from the expert-guided lane

Reviewed `app/short_horizon/expert_guided/action_contract.py` at checkout
87104b652d4e6a5a545b006be37985c666c3bc32. It implements Sharpe lower bound,
Sharpe+return, +activity, trade t-statistic, Calmar, return-minus-DD and a balanced
composite. Reuse ideas and validated metric definitions, not supervised labels
or a claim that these objectives trained our actor. Its return term is simple
annualized block return, distinct from our R on a fixed equal-duration period.

Two points for a shared implementation review:

- Its scales are fitted per arm/fold on a calibration frontier. That does not
  define a common score scale for comparisons between arms.
- Its NSGA-II frontier is nondominated only in (Sharpe, DD). A candidate removed
  by those objectives can still win on profit or activity. Rescore all valid
  archived candidates as a diagnostic; a genuine search for another fitness
  must optimize that fitness (or include its objectives). Frontier reranking
  alone is not an optimization of every composite.

## Tests and reporting

| ID | Acceptance evidence |
| --- | --- |
| RF01 | Gamma=1 equity/log reward telescopes to the matching ledger quantity; all costs once |
| RF02 | Entry, exit, financing and forced-flat timing; skipped bars use the correct time discount |
| RF03 | R2 lambda=0 equals R1; unchanged drawdown adds no MDD penalty; worse peak loss penalized |
| RF04 | Differential Sharpe primary-formula parity, prefix invariance, flat paths and restart moments |
| RF05 | Zero trades, no-loss variance and ruin never win by NaN, infinity or artificial zero |
| RF06 | Scale identity shared; changing later evaluation values cannot change scales/weights |
| RF07 | F2 churn counterexample; counts distinguish entry, close, refusal and missing-observation HOLD |
| RF08 | Fitness-reranking versus learning-reward change recorded separately; no TEST read |
| RF09 | Same support/seeds/architecture/budget except declared reward; arbitrary reward scaling observable |

Report TRAIN and internal validation separately, plus fixed development-year
results: profit/return, Sharpe, MDD, entries, closed trades, win rate, average
orders/day and per-week activity, costs, occupancy, seed spread and generalization
gap. Learning-speed evidence: common-budget curves and time to an internally
defined criterion; final checkpoint remains fixed, no hidden early stopping.
Automate sequential dispatch only after executable reward contracts and these
tests exist. This document schedules the work; it does not claim GPU jobs exist.

References:
- [Moody and Saffell, Reinforcement Learning for Trading](https://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf)
- [SAC reward and entropy objective](https://spinningup.openai.com/en/latest/algorithms/sac.html)
