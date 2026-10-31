# Real-Output Probe

Owner authorization: 2026-10-31. This follows the frozen synthetic reference
rollout (172), before phase 2 floating feature selection. Synthetic noise levels
are stress controls, not measurements of a trained model's errors.

## Scope and Interfaces

Implement the common market encoder first: two residual causal TCN blocks per
temporal group (8 channels), time-preserving pointwise fusion, two residual TCN
core blocks (16 channels), and small output heads. Preserve selected pointwise
groups, column ordering, and the declared window/stride. No rolling scaler,
clipping, imputation, agent-state input, expert demonstrations or future data.
Each residual block has two convolutions; temporal dilations are 1 then 2.
The combined temporal receptive field is 25 sampled observations. Pointwise
groups retain their measured non-temporal extraction before the temporal core.

Two interfaces must not be conflated: the current pragmatic controller consumes
path ensembles, whereas the existing RL policy emits categorical actions.
Neither a scalar close probability nor a fabricated path conversion resolves
that mismatch. The owner has been asked whether to use market-only close/HOLD
curves evaluated externally against the actual position, or a path predictor.
The owner has also been asked to confirm the existing categorical PPO algorithm
versus a new discrete SAC implementation. No PPO result may be called SAC.

Owner answers received in this session: **discrete SAC**, not PPO, and
**categorical entry plus CLOSE/HOLD curves**, not a path predictor. This
supersedes the two interface questions above. The existing pragmatic path
controller is preserved as a different control, not silently claimed compatible.

Per horizon, actor output is 39 market-only logits: 3 for NOP/BUY/SELL plus
2 directions x 9 position-coordinate knots x 2 logits (HOLD/CLOSE). Each of
two independent critics emits the analogous 39 soft Q values. The controller
queries z=(current liquidation quote-SL)/(TP-SL), interpolating adjacent knots
0,1/8,...,1. Beyond a protection due to a quote/spread change, the endpoint
segment is linearly extended, without clipping inputs; native protection at
the following open takes precedence. No learned layer receives z, direction,
account balance, volume, or the action mask. These only route complete emitted
surfaces outside the encoder/heads. Flat actions are 0 NOP,1 BUY,2 SELL;
occupied actions are 0 HOLD,1 CLOSE, with action2 masked. This is categorical
sampling, never quantization of a continuous command.

Use the official TorchRL0.14.0 DiscreteSACLoss, TensorDict0.14.2, twin critics,
target tau0.005, gamma0.99, automatic entropy with initial alpha1, AdamW lr0.0003 and
the inherited recipe weight decay. No dropout or BatchNorm is added. These
are initial probe constants, not selected optima. Reference:
[DiscreteSACLoss](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.DiscreteSACLoss.html).
The entropy target is0.6 times the batch mean log number of legal actions;
forced NOP therefore contributes zero, not log3. The official temperature
loss is optimized along with actor and twin-critic losses. All these settings
are explicitly bound from the pinned design to the constructor.

## Automated Prerequisites

1. Verify encoder causality, two blocks at each level, input invariance under
   routing changes, exact save/reload, categorical codec and signal learning.
2. Qualify each seed's network using a contextual planted-reward fixture with
   lags0/4/8/12, both close directions and uniform position coordinates over
   the whole0..1 interval (every interpolation segment receives learning). These
   are not financial horizons or demonstrations. Uniform legal-action behavior
   covers every Q action independently of early actor collapse; rewards, not
   supervised imitation loss, train the official off-policy SAC objective.
   Run2048 SAC updates, batch96;
   all216 conditional cases must select the expected action with probability>0.6.
   Reuse only authenticated qualification evidence on the same host, not weights.
3. Replay those learned outputs through native accounting on two mirrored
   planted-price paths, including both directions and early closes. Both must
   produce positive net profit and exact ledger reconciliation. A failed
   qualification is terminal evidence, not justification to change thresholds.
4. Authenticate the real recipe, scaler provenance, selected columns, calendar,
   temporal support and inner-fold fit boundaries before real-data fits.

Distributed jobs are partitioned, never duplicated: final revision2 allocation
is six/five/one cells after measured small-batch throughput (see174). One process per host bounds RAM. A
persisted service serializes that host's cells, authenticates completed receipts,
and retries process failures at most twice. Scientific failures remain failures.
Host addresses and data paths stay in private deployment configuration.

## Real Probe After Interface Resolution

Four horizon profiles (3/6/12/24 h), three seeds, fixed32768 decisions, no weekly refit
or expensive early stopping. Fit and emission characterization use chronological
inner TRAIN blocks; the 2024 validation and 2025 TEST files are not used by the
prerequisite queue. A scaler fitted on a later TRAIN block must not be described
as out-of-fit for an earlier probe. Feature-selection exposure is also declared
separately from model-weight exposure.

Record emissions before reading outcomes. Report distribution, saturation,
switching, error and calibration against their explicitly defined estimands.
Softmax policy probability is not automatically a probability of profitable
execution, and ensemble disagreement is not automatically calibrated error.
Then replay through the native engine with the frozen horizon risk profiles,
1:50 leverage, rel_volume 0.1, one simultaneous position, no daily entry cap,
entry at least 12 h after weekly open and forced flat 12 h before weekly close.
Report TRAIN and internal held-out trading: return, profit, Sharpe, drawdown,
trade counts, direction-specific win rates, activity and close attribution.
Only afterwards measure the combined-horizon controller and empirical-noise
robustness. This stage does not start floating selection or declare a winner.

The initial queue evaluates individual horizons; it does not fabricate a
combined-horizon arbitration rule. Raw surfaces and twin-critic surfaces are
sealed for every valid TRAIN/INNER_A window before account replay. Replay is
uniform, capacity8192, batch64, warmup1024, one update every4 decisions.
Duration-aware targets use gamma raised to actual automatically advanced bars.
Reward is100 times net equity change divided by initial capital, including
spread, commission, slippage and financing. No trade-count bonus is included.
SL/TP/q use the frozen shared-risk median references and baseline coefficients,
quality1: uncalibrated policy probabilities do not shrink exposure. Native BID
quotes remain BID; ASK=BID+declared spread. Equity metrics use direct native
equity marks, avoiding the generic MID accounting report for BID bars.

Current source recipe:120 columns in31 groups, window13, stride2 observed H1
bars. Fixed scaler fit2020-01-01 through2023-07-01. Model weights use that same
interval; INNER_A is2023H2. Both phase-1 selection and fixed risk profiles have
seen2023H2: results diagnose weight generalization, not fully nested selection
or untouched external validation. Existing phase-2 results are not inherited.

## Execution Verification

27 focused tests pass, including real native fixtures, two-direction planted
controls, source receipt mutations and duration-discount tests. A separate
engineering smoke executed128 real transitions, eight replay updates, model
reload, sealed24,026 market emissions and an entire INNER_A account. It is not
a scientific candidate. Direct native cash reconciled within4.55e-12 USD.
Peak allocated CUDA memory was87,970,816 bytes; measured update time on the
smallest device was0.704s with batch32 (not a full-campaign ETA).

No mid-fit continuation is claimed: a process failure restarts that unit from
the same seed in a new attempt, preserving failed artifacts. Completed units
are reused only with matching identity and every artifact digest. Qualification
evidence is cached by seed on each host; synthetic-trained weights and optimizer
are discarded before every real fit. The collector periodically mirrors receipts
and writes a consolidated Markdown table, full JSON metrics and final closure.

## Tests Before Implementation

| Requirement | Test/evidence |
|---|---|
| Two small TCN blocks per temporal group and core | structural module counts |
| No future influence within a window | suffix perturbation preserves prefix |
| No state feedback into learned encoder | routing-mask perturbation invariance |
| No learned SL/TP/volume in qualification | absent trainable risk head |
| Honest discrete action codec | integral command, legal masked distribution |
| Reproducible checkpoint | identical pre/post reload probabilities |
| Resume never trusts stale outputs | config/source/model digest mutation rejection |
| No false full-chain qualification | receipt scope SYNTHETIC_ENCODER_ONLY |
| No scientific failure hidden by retries | completed FAIL is terminal evidence |

The complete strategy positive control remains distinct from encoder learning.
Passing the latter alone must never be reported as passing the former.

Pre-financial correction: the first qualification draft sampled only two
position coordinates, leaving interior curve knots without direct coverage.
The three qualification workers were stopped before any financial fit. Revision2
samples the entire coordinate interval and tests six interior coordinates, with
a regression proving coverage of all eight interpolation segments on each side.
No financial result from the interrupted engineering queue is reused.

Revision3 corrects a second issue exposed by qualification: fixed alpha0.01
and policy-only sampling saturated incorrect opening logits. The failed
revision2 evidence is retained, and its workers were stopped before financial
fits. Full legal-action coverage and mask-aware automatic entropy replace
those engineering settings, without changing qualification expectations,
financial data, reward, risk profiles or evaluation periods. See175.
