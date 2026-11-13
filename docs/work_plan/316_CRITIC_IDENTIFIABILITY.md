# Critic Identifiability And Native Coverage

Owner2026-11-13 authorizes the next diagnosis after315. No model training,
feature changes, STL, VALIDATION/TEST inference or live release. Source299
historical mixed-COT restriction remains. Preserve71 inputs and49 absent
phase2 candidates. This is not a claim that the data have no useful signal.

## Frozen Questions

1. On the same55 held-out anchors, do critic rankings beat simple FIT-only
   mean-return and nearest-window baselines? Report per-state/year, not only
   the aggregate. Use k=1/5 fixed before measurement, no tuning. Include
   always-NOP/BUY/SELL and the unchanged actor for action-regret context.
2. Compare a global FIT mean with a FIT mean grouped by position/legality.
   The second is diagnostic only: position is NOT added to the network.
   Report missing strata; never fill them with an invented outcome.
3. For16 target-blind origins (four per year) selected from the held-out
   executable-flat anchors, fork the native account with first action
   NOP/BUY/SELL. At the next decision, compare all three actions in each
   branch under the unchanged source continuation. These are native reachable
   counterfactual states, not fabricated positions or synthetic market prices.

The three branches must have identical next decision row, timestamp and
market tensor. Record their actual direction, balance, SL/TP and age through
native state receipts; if protection closes an opening immediately, retain
that fact and do not invent an open order. Terminal or unavailable branches
are reported, not silently discarded. Compare action-optimal set intersection,
pair-order conflicts, and the minimum equally weighted regret achievable by
a SINGLE shared action over the reachable states. This finite counterexample
tests a memoryless market-only policy on those states; it is not a statement
about all states or a lower bound on market-wide trading profit.
All action comparisons keep the subsequent source policy fixed. A conflicting
optimum therefore concerns this one-step Q target contract, not a proof that
every alternative continuation policy must have the same conflict.

Labels are full-interval discounted native rewards, exactly as300/312.
The baseline holdout already appeared in earlier diagnostics. Same-origin
branches and overlapping windows are dependent; no independent-sample claim.
Nearest-window distance is descriptive, not an estimate of signal/noise.
Native repeat witnesses must reproduce returns and first effects exactly.

## Tests And Automation

| Requirement | Evidence |
| --- | --- |
| No monitor fitting | Pure FIT indices for means/neighbors; tests perturb monitor outcomes without changing predictions |
| Honest support | Disjoint indices; valid TRAIN rows; same315/314/312/source hashes |
| State distinction | Known conflicting/compatible action tables and invariant shifts test regret calculation |
| Native branches | Real engine NOP/BUY/SELL fork, identical next observation, parent unchanged, cash identity and repeat witness |
| Fixed design | Seed316, four per year, no outcome-dependent selection or resampling |
| Zero mutation | No optimizer; source checkpoint hash unchanged; no agent-state features |
| Durable result | Locked finite CPU service, sealed per-origin resumable receipts, final table/JSON/plot |

After measurement, distinguish a coverage error from an observation/target
contract mismatch. Do not promote a critic, silently reintroduce feedback, or
repeat an unchanged floating pass based on this diagnostic alone.

## Completed Evidence

Runtime `7b96e61e`,14 focused tests. CPU service completed2026-11-13T15:07:52Z,
19.12seconds including report generation, no errors/restarts.16/16 origins,
144 native suffixes plus9 exact repeat witnesses; original reference account
replays and parent-state preservation pass. Source checkpoint is unchanged.
Receipt `84f89d0d79431852dd9cc71422d519c310f6a88f337b3eb014d0012189e2872d`
verified after completion. Output `/var/tmp/sac71-critic316-20261113-r1` holds
`BASELINES.json`, per-origin native receipts, `RESULT.json`, `RESULTS.md` and
`STATE_AMBIGUITY.png`.

### Baselines On The Same55

| Method | Pair accuracy | Mean native regret | Value RMSE |
| --- | ---: | ---: | ---: |
| FIT mean (always chooses NOP here) |0.546154|0.207532|0.631007|
| FIT mean per observed state, diagnostic only |0.500000|0.244567|0.623856|
| Nearest window, k1 |0.484615|0.255701|0.806224|
| Nearest windows, k5 |0.438462|0.299397|0.659512|
| Preserved actor |not a Q ranking|0.200629|not a Q estimate|
| Expanded146 heads, average two seeds |0.511538|0.263956|0.599231|
| Expanded146 representation, average two seeds |0.492308|0.275048|0.724213|

The preserved actor has lower sampled regret than the fitted critics. Lower
absolute Q error does not suffice to improve action selection. Nearest-window
distance alone does not supply a better decision rule here. No missing state
baseline; no exact held/FIT input duplicates.15/55 held windows overlap a FIT
window in time, reinforcing that this is not independent temporal confirmation.
Neither these55 points nor the counterfactual forks measure full-account return.

### Same Input, Different Required Action

14/16 origins retain actual flat/long/short positions at the next observation.
One immediately closes both new positions through native protection; another
retains only the long. These cases are kept, not replaced with artificial orders.
Three origins have no common optimal signal across the three reachable states;
all three include actual flat/long/short states. Next decision times UTC:

| Timestamp | Flat optimum | Long optimum | Short optimum | Minimum shared-action mean regret |
| --- | --- | --- | --- | ---: |
|2021-08-18T15:00:00Z|NOP|SELL (close)|NOP or SELL (hold)|0.090254|
|2022-01-11T09:00:00Z|SELL (open)|NOP or BUY (hold)|NOP or SELL (hold)|0.053771|
|2023-07-05T11:00:00Z|BUY (open)|SELL (close)|NOP or SELL (hold)|0.049390|

For the last witness, native discounted reward vectors in NOP/BUY/SELL order:
flat `[-0.295655,0.085315,-0.797177]`, long
`[-0.385307,-0.385307,-0.305937]`, short
`[-0.240411,-0.309209,-0.240411]`. These are reward units, not USD/profit%.
Market tensor and observation time are identical; only the preceding action
and resulting native account state differ. Equal-weight mean minimum regret
over16 origins is0.012088. This is not an estimate of population regret.

### Interpretation And Next Action

This falsifies exact position-specific Q reconstruction from a memoryless
market-only observation at these witnesses under the fixed continuation.
It does NOT invalidate market-only SAC in general: such a critic can estimate
the return marginalized over account states under the policy's visitation
distribution. Equal-weight counterfactual states are not that distribution.
It also does not establish that state ambiguity explains the whole refinement
loss, or that adding feedback alone would solve it (the state-mean baseline
does not beat the global mean here).

Next define the critic's estimand explicitly: policy-visitation marginal value
versus a position-specific value selected outside the market encoder. Compare
the associated target construction and action ranking before another actor
update. A changed output/router interface needs its own explicit contract and
parity controls; no agent-state inputs were introduced by this diagnostic.
Preserve299 and the cumulative49 return candidates. Source qualification
remains necessary before corrected-data training or live promotion.
