# Same-Market Action-Value Compatibility

## Purpose

Owner authorized the next diagnosis after251. Test whether the same market
window can require incompatible action values or choices across feasible
flat/long/short account states. Do not assume this explains all critic errors.
No optimization, training, new observations or trading-model replacement.

## Frozen Construction

Use the512 permitted-flat origin anchors already selected in248, across all16
TRAIN quarterly episodes. No new outcome-based origin selection. Reconstruct
each reference account and fork three native histories at each origin:
NOP, BUY, SELL. Thereafter each fork holds with NOP. Inspect after1,4,12
decision steps; record actual elapsed hours separately. A step can span a gap.

Flat history remains flat. Include a long/short history in a comparison only
while its original native position survives. Record barriers, refusals, calendar
closures and terminal exclusions; never assign position fields, invent fills,
force a position to survive, or select replacement origins. Compare all surviving
pairs as well as complete triplets, with explicit denominators. A comparison
with fewer than two distinct position states is uninformative and is recorded.

For each comparison assert identical timestamps, row IDs, feature bytes and
observation masks. Preserve70 features,13 slots,stride2, original scaler and
all costs, SL/TP, volume and calendar rules. For each valid state evaluate all
three forced first actions, then follow the same frozen parent to that quarter's
terminal settlement. Future values are diagnostic labels only, never inputs.
Replay an identical-state negative control in each quarter; it must yield zero
discrepancy. Parent account and native cash accounting must remain exact.

## Distinct Questions

For a bank with state/action return matrix Y, equally weight its surviving
states as an explicit diagnostic convention (not estimated deployment weights).

1. Shared-vector squared-error floor: RMSE of Y about its column means.
2. Remove state-common value offsets; compute centered and action-gap floors.
   Pure value-offset differences must not be called preference conflicts.
3. Count action-pair strict ordering reversals and intersect optimal-action
   sets using1e-8 tolerance. Reversed lower-ranked pairs need not prevent a
   common optimal action.
4. Best common-action mean regret: mean of row maxima minus maximum of column
   means. This is also the minimum mean regret of a state-blind randomized
   one-step choice under these equal weights and fixed subsequent policy.
5. Deterministic minimax regret, explicitly NOT the randomized minimax bound.
6. Frozen actor and seven archived critic choices: common action, per-state
   regret and excess over the best common-action regret. No checkpoint selection.

The label policy and terminal horizon are fixed. These finite-panel lower
bounds are not real-trading profit bounds or a proof against every market-only
policy, recurrent architecture, alternative action encoding or controller.

## Test Matrix

| Requirement | Test |
|---|---|
| Correct algebra | Identical rows, constant offsets, strict reversals, common optimal action despite value conflicts |
| No invented account state | Real native opens/holds, survivor reporting, source fork unchanged |
| Same observation | Exact feature/mask/time equality across histories; repeated-input model canary |
| Label integrity | Hold equivalence, native cash reconciliation, identical-state replay |
| Temporal limits | Quarter terminal only; monitor context stays in2023; no2024/TEST scoring |
| Reuse authentication |248/250 receipts, source and checkpoint hashes, fixed-code identity |
| Durable execution | Per-quarter sealed results, resumable queue, automatic verified collection |

## Interpretation And Outputs

Report origin coverage, surviving banks by lag/role, strict conflicts, compatible
optima, regret floors and model residual regret. Banks overlap in price paths
and may share observation timestamps. Percentages describe this constructed
panel, not an independent sample or the natural frequency of positions.
The2023 role was used in source pretraining; it is not an untouched holdout.

If conflicts exist, separate exact-Q fitting incompatibility from actual
action-choice incompatibility before proposing a remedy. This experiment does
not authorize adding account-state feedback to the learned inputs. If none are
observed, report that result without manufacturing more difficult states.

## Launch Verification

Runtime `0e3502c1`; worker started2026-11-10T08:47:33Z. Focused tests11 passed
locally and11 on the worker, including native-history isolation and compatibility
algebra. GPU launch gate and real-checkpoint repeated-input equality passed
before native histories started. No gradient or optimizer step is executed.

Persistent sequential worker and automatic collector are active. Completed
quarters are sealed and verified on resume; unfinished origins have per-record
digests. Abnormal-exit restart is bounded and accounting/identity failures stop
explicitly. Worker memory cap6GiB, swap disabled. Artifact root:
`/var/tmp/sac70-state-compat-20261110`.
