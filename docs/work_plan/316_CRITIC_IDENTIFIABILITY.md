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
