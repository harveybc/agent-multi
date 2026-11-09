# Paired Action Ranking Contrast

Owner approved 2026-11-09, following [233](233_CHANGED_DECISION_ATTRIBUTION_RESULTS.md).
This is a bounded diagnostic repair, not a replacement champion or a new full
feature-selection pass. No observation/state feedback, rolling scaling, changed
features, costs, windows, position rules or trading restrictions.

## Frozen Design

Use the preserved70 actor/representation and the same fresh calibrated twin
critic checkpoint as231. The four-year TRAIN market and full TRAIN actor replay
remain unchanged. A new **counterfactual calibration panel** selects up to32
decision states per year and position category (flat/long/short), before seeing
any counterfactual outcome. Fixed RNG seed41; no sampling of the harmful dates
identified in233. At most384 states;24 per stratum fit,8 diagnose. Positions are
simulator state for interventions/stratification only, never network inputs.

Diagnostic anchors are disjoint from correction-fit anchors, but are NOT an
independent market holdout: the inherited representation already saw TRAIN,
paths can overlap and nearby features are correlated. Generalization is not
inferred from this monitor. Validation is only evaluated after fixed updates,
not used to build labels, select anchors or choose calibration checkpoints.

At each state force NOP, BUY and SELL from the exact same position, cash,
calendar and protection state, then continue to the native TRAIN endpoint:

* greedy frozen reference actor: reward-only discounted return;
* stochastic reference actor: four common-random seeds7,19,43,73, soft returns
  excluding forced-current-action entropy and including future entropy;
* additionally record reward-only stochastic returns to expose entropy's
  contribution without conflating it with changed policy actions.

## Arms

| Arm | Additional critic fit | Meaning |
|---|---|---|
| control | None | Exact previous fresh-head calibrated control |
| chosen_stochastic | One frozen sampled action per fit state | Additional-fit control with the same states, target convention and update budget |
| paired_stochastic | All three actions, mean of four soft continuations | Tests alternative-action coverage |
| paired_greedy | All three actions, deterministic reward-only continuation | Tests policy-evaluation target aligned with greedy deployment |

All fitting arms:4096 updates,batch128 states,identical state batch schedule,
SmoothL1 averaged across active entries and twins,same LR/weight decay,empty
optimizer memory at correction start. Only online critic heads are trainable.
Frozen representation is evaluated without dropout in all correction arms;
this is inference-surface calibration,not new stochastic encoder training.
The greedy arm intentionally changes both continuation policy and its entropy
convention; it is not claimed to be an unchanged native SAC critic objective.

After calibration sync targets and apply identical original full-TRAIN replay
actor batches, with empty actor optimizer memory, at1,8,32 updates. Do not tune
alpha,critic,encoder or strategy during these actor updates. Restore source for
each arm. The uncorrected control must reproduce231 native accounts exactly.
Record every arm regardless of sign/activity. No automatic model promotion.

## Acceptance Tests And Traceability

| ID | Requirement / evidence |
|---|---|
| R01 | Source recipe, checkpoints, support, surfaces and full replay authenticated |
| R02 | Anchor selection deterministic, TRAIN-only, no repeated rows, fit/monitor disjoint, state counts reported |
| R03 | Three actions share an immutable snapshot; no future features in observation; all suffix cash conserved |
| R04 | Greedy original first action restores reference final account; stochastic current entropy excluded; reward-only return independently checked |
| R05 | Two CPU shards cover manifest exactly once; cached rows and shard receipts verified before fitting |
| R06 | Paired loss uses all actions; chosen control uses only chosen action; malformed/nonfinite shapes rejected |
| R07 | Only critic heads change during calibration; actor/encoder/alpha exact and source digests unchanged |
| R08 | Ranking, regret and RMSE measured on fit and disjoint diagnostic anchors, with per-position summaries and sampling disagreement |
| R09 | Fixed calibration/actor budgets; no monitor-based checkpoint selection, no validation labels, no TEST |
| R10 | Control exactly reproduces archived231 trading accounts after identical actor updates |
| R11 | Native activity, trades, win rate, direction, early closes, costs, Sharpe, DD and returns saved for all arms |
| R12 | Resumable row receipts, exclusive worker locks, immutable runtime, automatic collection and terminal status |

Implement unit tests for selection boundaries, loss gradients, ranking ties,
entropy convention and frozen-parameter scope before launch. Existing native
same-state cloning and account tests remain mandatory. The runner uses two CPU
workers for counterfactual generation, then sequential GPU fitting on RTX5090.
No decisions from the owner are needed between stages.

Success means measured improvement in action ordering and native account
behavior, not merely smaller fitting loss. Report conflicting metrics as such.
This one bounded contrast cannot establish a universal repair or eliminate
hidden-account-state ambiguity in the deliberately market-only critic.

## Prelaunch Verification

37 focused tests pass, including real native account interventions and updates
that must move only critic-head tensors. The new correction initially attempted
to use the inspection-only `q_surfaces` API for gradients; its no-weight-movement
test failed. Correction now uses TorchRL's native functional Q execution, with
numerical parity and actual optimizer-parameter movement both checked. This was
caught before any contrast launch; it is not a finding about historical runs.

The runtime pins source hashes and exports its complete effective design.
The dedicated Ruff executable is absent in this checkout, so lint has not been
claimed. CLI import and the focused pytest suite are the executable preflight.
