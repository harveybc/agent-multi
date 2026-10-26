# C4: actual learner qualification

## Scope and frozen design

Owner requested execution after the independent audit rejected decision use of
the existing selection and DEAP results. This is a diagnostic, not a financial
experiment or permission to resume feature selection. No expert demonstration,
agent-state vector, rolling z-score or input clipping is added.

Use the actual categorical PPO and GroupedTCN, native trading simulator,
configured costs, business calendar, required SL/TP and fixed risk outputs.
Keep the parent budget: 102400 environment steps, rollout 512, minibatch 256,
5 optimizer epochs per rollout, learning rate 0.0003, entropy 0.01, discount
0.99. Early stopping is off. Shape remains 13 x 120, with 5718 parameters.

Nine sequential units: seeds 7, 19, 43, each with these three arms:

1. PLANTED: replace only tensor channel zero with
   `close[b+4] - open[b+1]`, normalized by fixed TRAIN constants.
2. PERMUTED: the same channel, shuffled within each period with a fixed seed.
3. ORIGINAL: unchanged channels, restricted to the identical support.

The planted channel deliberately contains future information. Four means
observed bars, not elapsed hours. All its source bars must remain within the
same split. It is tagged non-financial in the recipe, checkpoint and result;
its store column does not exist in the financial store. No financial result
schema is emitted. The loader is truncated before 2024: neither OBS_2024 nor
TEST can be evaluated here.

## Acceptance and interpretation

Every planted seed must have positive TRAIN and INNER_A Sharpe and pass the
strict executed-trade behavior checks. Mean planted INNER_A Sharpe must exceed
both controls. Undefined comparisons fail qualification. All nine units run
even if a completed seed fails the scientific criterion. INNER_A is selection,
not independent confirmation. Passing establishes bounded learning evidence,
not financial profitability; failing does not establish absence of market
signal. A four-bar directional oracle is not an oracle for every SL/TP path or
optimal closing decision, so any failure needs behavioral diagnosis.

## Automation

`tools/c4_learning_queue.py` implements prepare, CPU preflight, isolated CUDA
workers, and a locked sequential supervisor. Code, software, support and
artifact hashes are checked. Completed units are adopted without another fit;
failed process attempts are retained, with at most two fresh attempts per unit.
Each worker has a 2700-second timeout. Resource admission requires 5 GiB
available host RAM and an idle assigned GPU. A resource wait launches nothing.
The private service has a memory cap and bounded restart policy. No existing
campaign is modified. STATUS/PROGRESS report unit, steps and heartbeat; the
final table includes TRAIN/INNER_A Sharpe, profit, orders by side, win rate,
drawdown and strict eligibility. Closure hashes bind every retained artifact.

## Traceability and verification

| Requirement | Evidence |
|---|---|
| Same learner, budget and support | CPU preflight builds all three actual policies; common support hash and parameter count asserted |
| Controlled one-channel intervention | test_c4_learning_control: unchanged other channels and immutable inputs |
| No scaler fit on later period | TRAIN constants invariant to INNER_A mutation test |
| Split-safe diagnostic labels | future_move boundary and alignment test |
| Actual optimization and reload | worker asserts 102400 steps, changed weights, identical reload weights and action |
| Executed trading, not requested closes | ledger_checks plus executed_behavior on actual trade ledger |
| Explicit non-financial interpretation | separate receipt schema, recipe marker, diagnostic checkpoint flag and qualification output |

Local verification: 26 focused tests passed, including altered-artifact
rejection and RAM/GPU admission refusal. Real-data CPU preflight passed:
21217 TRAIN and 2777 INNER_A common valid origins; all three arms have 5718
parameters and support SHA256
`eec922b5f42dc38559678c326c23088c3a70b3bc1e4b3155bff13301ed8eb8cd`.
This preflight performed zero training steps. Deployment and measured progress
are recorded separately; no launch or success is inferred from this document.
