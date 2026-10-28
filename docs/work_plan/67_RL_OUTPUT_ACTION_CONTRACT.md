# Categorical decisions and conditional order parameters

Owner direction: 2026-10-25. Original design below; implementation and deployed
scope are recorded in `70_HYBRID_AUTOMATION_EXECUTION.md`. Existing pinned
producers and results stay unchanged.

## Mandatory discretionary early exit

Owner clarification: voluntary early closing in either direction is indispensable
business behavior, not an optional strategy enhancement. SL/TP and mandatory
session flattening are safety/execution rules, not substitutes for this behavior.
The policy must be able to close when its causal observations support closing
rather than retaining exposure, including an unfavorable SL-before-TP outlook.
No future barrier outcome is available to the live policy; an inferred risk is
not certainty, and the raw probability alone is not a calibrated net-value rule.

For an existing long/short, CLOSE_LONG/CLOSE_SHORT competes with HOLD at every
valid observation. An exhausted daily entry allowance or a closed opening window
must not suppress exits. Execution remains at the next permitted engine fill,
with actual configured costs and without a hindsight fill. The current RL lane
does not add forecasting outputs or expert labels to accomplish this.

Three separate claims require separate evidence:

1. Mechanical capability: both close commands execute before SL/TP on a neutral
   fixture, including after the entry quota is exhausted, with correct costs.
2. Learning capability: paired synthetic long/short trajectories where closing
   avoids subsequent adverse movement, plus favorable continuation controls
   where unnecessary closing loses value. Compare net executable outcomes with
   the same entry and random seed, not just a close-action counter.
3. Financial behavior: voluntary close counts/rates, side, time-to-barrier and
   same-entry close-versus-hold counterfactuals, with costs and coverage declared.
   Such counterfactuals are retrospective diagnostics, never policy inputs.

The active batch proves neither (2) nor (3) merely by offering a close action.
The initial completed policies had zero AGENT_CLOSE events. They are incomplete
business-behavior candidates, even where the activity threshold passes. Do not
promote them as complete strategies. There is no arbitrary mandatory close quota
and no reward bonus for closing merely to make this metric nonzero.

Priority after the frozen batch: qualify learning on the paired close/hold
scenarios, investigate missing position-dependent information under the existing
no-feedback constraint, then compare close-policy specialization if needed.
Any new input/state representation requires an explicit contrast and identity;
do not silently reintroduce account feedback or mutate a running experiment.

## Finding and scope

`env_plugins/order_contract.py` implements COMMANDS5 using five intervals of
one continuous coordinate. Incompatible commands become HOLD. This introduces
arbitrary distances between nominal commands and permits probability mass on
commands that cannot execute. A continuous distribution's deterministic mean
is not a categorical mode. These are structural limitations, not proof that
they alone caused inactivity or that a replacement will be profitable.

Do not patch plots, tune the interval boundaries, or turn five logits into a
scalar to keep the old SAC trainer. Preserve the admitted input recipe, fixed
TRAIN scaling, selected temporal representation, periods, native execution
accounting and business restrictions for the initial structural contrast.

## Typed output contract

| Module | Output | Legal use |
| --- | --- | --- |
| Flat decision | Categorical NOP, OPEN_LONG, OPEN_SHORT | Flat inventory; opening additionally requires business eligibility |
| Long decision | Categorical HOLD, CLOSE_LONG | Existing long only |
| Short decision | Categorical HOLD, CLOSE_SHORT | Existing short only |
| Long risk | Conditional continuous SL distance, TP distance, volume fraction | Selected OPEN_LONG only |
| Short risk | Conditional continuous SL distance, TP distance, volume fraction | Selected OPEN_SHORT only |

The three routed categorical heads share the admitted temporal representation
initially. They are specialized heads, not five independently trained agents.
NOP is an explicit competing outcome, not an arbitrary interval or a fallback
for a contradictory action. A five-command masked categorical head is an
equivalent interface alternative, not an additional initial experiment.

## NOP decision and learning contract

Owner clarification on 2026-10-25: abstention is essential, a separate NOP
network is optional. Decision: keep NOP as one outcome of the flat categorical
head, without a separate classifier or supervised regime labels. HOLD while
positioned is a different decision: it retains exposure and must not be counted
as risk-free abstention.

This RL lane learns from sampled trajectories and advantage estimates, not
from marking historical sections as "should not trade". Under a policy-gradient
learner, positive estimated advantage increases the selected action's relative
likelihood and negative advantage decreases it (subject to the learner's update
constraints). Waiting can preserve capital and the daily entry allowance for a
later opportunity. Its immediate flat-market reward can be zero without its
continuation value being zero. A policy's P(NOP) is an action probability, NOT
the calibrated probability that no profitable trade exists.

Current reward in RecipeNativeEnv is 100 * change in marked equity / initial
capital, discounted over skipped decision bars. Entry costs affect marked
equity; exit costs and financing are settled by the native engine at closure.
There is no explicit variance, drawdown or tail-risk penalty in this reward.
Do not claim that maximizing it necessarily learns to avoid all high-volatility
or high-tail-risk opportunities. Business limits remain hard constraints;
risk-sensitive rewards/constraints require an explicit subsequent paired arm.
High volatility alone is not a label for NOP. Cost-only and net-return learning
must be separated from learning an additional risk preference.

No reward for placing an order, no penalty for NOP, and no forced entry to pass
the activity threshold. Distinguish policy-selected NOP from business-mandated
ineligibility and missing-observation HOLD in all reports. If all policies fail
activity, diagnose that result; never relabel it as a valid trading winner.

Regime estimates may be used for causal diagnostic stratification, not as
ground-truth labels. Any later feature use must preserve decision-time
availability: filtered state probabilities, not retrospective smoothing or
full-series regime assignment. Report behavior around regime transitions;
do not assume instantaneous detection or alter the frozen input recipe here.

Additional acceptance cases: flat price with positive round-trip costs;
gross favorable move smaller than all costs; clear legal opportunity above
costs in either direction; wait-then-enter opportunity under the daily quota;
and legal versus forbidden openings near session boundaries. Compare full
executable trajectories with the same starting state, not isolated trades
that ignore the quota or future exit policy. These cases establish mechanics
and synthetic learning ability separately, not profitability on real data.

ZigZag may describe hindsight opportunities or serve as an executable oracle
diagnostic, with explicit confirmation/support and next-bar fills. It is not
an input, reward component, pretraining label or validation selector in the
initial expert-free RL comparison. Using it in any such learning role creates
a separately identified expert-assisted arm. Absence of a ZigZag opportunity
does not prove absence of a profitable strategy outside that oracle's search.

Opening distributions use direction-specific bounded parameters. Preserve
the existing sigma units and declared SL/TP bounds initially. Volume fraction
q is in (0,1]; requested margin budget is q * rel_volume * balance(t).
The broker contract maps margin through leverage, price, contract size and lot
step to units. Round down, never round above the budget to satisfy a minimum.
Reject infeasible orders with a reason; never silently enlarge their size.
SL, TP and quantity must be finite, valid and attached before an order opens.

HOLD does not modify SL, TP or size. CLOSE closes the matching position only;
no implicit reversal, new order or enlargement. Only opening branches sample
and score risk parameters. Unused risk heads must not contribute artificial
entropy or likelihood terms. A fixed-risk categorical control can isolate
the decision interface before enabling learned risk, with its values recorded.

## Routing and observability

The native harness routes by actual inventory and masks commands using only
information available at decision time: calendar, entry quota, resources and
business constraints. Apply identical routing and masks during collection,
training likelihood evaluation, deterministic evaluation and replay.
Forced flat, SL and TP remain mandatory engine events, not policy permissions.
Record the action requested, mask, chosen branch, execution and refusal reason.

No account feedback vector is added to the market tensor. Routing nevertheless
conveys position category. It does NOT expose entry price, floating P&L, age,
SL or TP to the learned closer. Without those or policy memory this is a
partially observed policy; equal market windows can require different exits
for different positions. Keep this limitation explicit. A later minimal-state
or recurrent-memory contrast requires a separate design, not a silent input.

## Learning algorithm

For command a and opening parameters u, use the joint policy:

`log pi(a,u|x,m) = log pi_cat(a|x,m) + I_open(a)*log pi_risk(u|a,x)`.

The bounded risk distribution needs its correct density transformation, not
post-sampling clipping. The same joint log probability must drive optimization
and replay checks; retain collection-time masks and old log probabilities.

Stock Stable-Baselines3 SAC supports Box actions, not categorical or hybrid
actions. Replacing its last layer alone does not implement this contract.
Recommend a conditional categorical/continuous PPO candidate using a maintained
library with a tested custom distribution/module, before considering a custom
hybrid SAC. RLlib documents autoregressive action modules, but is not yet an
admitted runtime here. First verify it in an isolated environment. Default
independent Tuple distributions are insufficient for conditional risk heads.

For PPO, the critic predicts scalar state value; it does not choose an action.
If later retaining SAC, its twin Q critics and Bellman/entropy objectives must
explicitly handle discrete commands and active continuous parameters. That is
a separate implementation, not the existing SAC renamed.

Independent specialist agents and an optimized arbiter are a later contrast.
Do not combine uncalibrated independent probabilities or add a post-training
threshold that changes the deployed policy without evaluating that policy.
The initial arbiter is the routed categorical policy. Tunable arbitration,
if introduced, is trained/selected internally with its own identity. Calendar,
mandatory exits and exposure limits are not optimizer genes.

## Required tests before a financial fit

| ID | Evidence required |
| --- | --- |
| OA01 | Integer command identity; no scalar intervals; relabeling commands consistently does not change behavior |
| OA02 | Illegal commands have zero probability; flat/long/short cases; mandatory engine exits retained |
| OA03 | Risk bounds, minimum lot, leverage and budget; long/short SL/TP geometry; no silent enlargement |
| OA04 | Joint log probabilities checked analytically; gradients only through active risk branch; bounded density correct |
| OA05 | Same masks/routing in rollout, optimization and evaluation; requested action matches recorded action |
| OA06 | Save/reload output parity and RNG-controlled sample parity; effective config and distribution identity retained |
| OA07 | Native ledger parity for known actions including fees, financing, gaps, SL/TP and forced closing |
| OA08 | Known-signal synthetic scenarios and feasible oracle controls; no future oracle values in observations |
| OA09 | No-action, minimum-activity and valid-action controls; invalid commands cannot masquerade as useful exploration |
| OA10 | Existing input tensor/scale/support parity; no TEST access or expert demonstrations in this RL lane |

## Execution order and reporting

1. Implement and test the typed interface and native adapter on CPU.
2. Verify the conditional policy and learner in an isolated, pinned runtime.
3. Exercise known-signal engine scenarios and a small synthetic learning test.
4. Run the fixed-risk categorical control, then conditional learned-risk arm,
   keeping inputs/support/business accounting identical and costs recorded.
5. Report TRAIN/INNER/2024 separately: profit, return, Sharpe, max drawdown,
   closed trades, win rate, all-weeks activity, legal/illegal command counts,
   command probabilities/entropy and realized sizing. No inactive winner.

Activity admission remains strictly more than two closed trades per scored
week on average, with zero-trade weeks included and daily entry cap unchanged.
Report entry counts separately. This is eligibility, not a reward for churning;
no forced trades to meet quota. It is not a guarantee that every week trades.
Estimate GPU budget only after measuring the replacement learner. Do not claim
the old continuous policy checkpoints are trained categorical checkpoints.

After the accepted categorical control, follow
[plan 69](69_REWARD_AND_SELECTION_CONTRASTS.md) for isolated learning-reward
and fitness-ranking contrasts. Do not change both simultaneously.

## Primary references

- [SB3 SAC action-space support](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [RLlib conditional action distributions](https://docs.ray.io/en/latest/rllib/rl-modules.html#auto-regressive-action-distributions)
- [Policy learning from return and advantage](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
