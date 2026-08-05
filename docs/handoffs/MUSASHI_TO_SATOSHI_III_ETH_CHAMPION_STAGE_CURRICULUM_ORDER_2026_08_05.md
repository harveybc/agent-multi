# Musashi to Satoshi III: ETH Champion, Stage-Integrated Solvency Curriculum, and Paper Execution Order

Date: 2026-08-05 America/Bogota
From: General Musashi, temporary independent auditor, relaying the owner's directive
To: Satoshi III (Mujuro Utsutsu), temporary technical lead
Priority: P0, supersedes the current USDCAD optimization priority
Scope: Paper/Demo execution and research only; no Live capital

## 1. Owner Intent, Without Interpretation Drift

The owner ordered the system to do all of the following now:

1. Stop spending the fleet's primary optimization capacity on USDCAD. Preserve
   its valid state and artifacts, but do not let it delay ETH work.
2. Use `ETHUSD@4h`, the asset/model line with the strongest relevant historical
   evidence, to perfect the training, optimization, simulation-to-Paper, and
   champion-succession path.
3. Put the best available **trained trading policy** in control of ETH Paper/Demo
   trading. The current linear direction model is not the requested champion.
4. Implement and test the owner's solvency-relaxation observation rather than
   continuing to describe it as future research.
5. Integrate `easy -> normal` training **inside every incremental optimization
   stage**. A candidate learns under relaxed train-only solvency dynamics, then
   continues from those learned weights under realistic dynamics before the
   optimizer scores it and advances to the next stage.
6. Compare that curriculum against `normal-only` using identical data, seeds,
   initial population, search space, and candidate budget.
7. Report raw, same-scale business metrics. Do not present an opaque weighted
   fitness proxy as profit, return, or champion quality.

Do not introduce a new owner-decision gate for work already authorized here.
Normal safety invariants remain mandatory: Paper/Demo only, exact account and
symbol binding, native or independently reconciled SL and TP on every entry,
bounded size, and fail-closed recovery.

## 2. Required Expertise and Working Method

Act simultaneously as:

- a senior reinforcement-learning and evolutionary-computation researcher;
- a DOIN distributed-optimization engineer who understands one shared domain,
  candidate pooling, migration, blockchain anchoring, artifacts, and OLAP;
- a trading-simulation engineer who understands margin, liquidation, fees,
  financing, position sizing, and chronology;
- an MLOps engineer responsible for reproducible model artifacts and exact
  train/inference parity; and
- a Paper/Demo execution engineer responsible for uninterrupted model-controlled
  trading and broker reconciliation.

Read source before changing it. Use the codebase-memory graph first for code
discovery, then inspect exact configs, ledgers, and non-code artifacts directly.
Do not rewrite DOIN coordination. Extend the external `agent-multi` optimizer
and simulation contracts consumed by the already-working DOIN network.

## 3. Facts That Must Appear in Your Takeover Report

These facts have been independently read from the current artifacts:

### 3.1 The MT5 incumbent is not the requested champion

- Model: `ethusdt-4h-linear-live-v1`
- Artifact SHA-256:
  `539f946071a1870672c2d2c1ce7b1ce0f0d4b4317a15b2d082b59b32c98d10bf`
- It is a supervised linear classifier selected from 32 grid candidates.
- Its own manifest says `live_inference_eligible=false` and
  `live_execution_eligible=false`.
- Its validation mean weekly net log return is `-0.13337328963884815`
  (`-13.3373%`), over 831 reported trades. The manifest's
  `annualized_return=-6.935411061220104` is only `weekly_mean * 52`; it is
  not a compounded equity return and must not be displayed as such.

It may remain as a frozen **shadow/control**. It must not remain the execution
authority after a compatible SAC policy is available.

### 3.2 The strongest-looking historical ETH run was in-sample only

The Stage-A ETH SAC run reported:

- initial/final equity: `10000 -> 11512.164417774693`;
- total return: `+15.12164417774693%`;
- max drawdown: `11.113226854446845%`;
- trades: `426`.

It has no validation/test result and no saved policy artifact in that run
directory. Do not call it an all-time deployable champion.

### 3.3 The only immediately loadable historical SAC candidate is modest

Artifact:
`examples/results/project3_ethusdt_4h_sac_train_val_test_v2/policy.zip`

- Artifact SHA-256:
  `6b73f26f57ad4aa8bb34e0d9bd0b8641f1823f0dc43563cf2c99693bb71a0df7`
- Train: `+21.1069%`, `15.3801%` max DD, `21` trades.
- Validation: `-1.38248%`, `7.22119%` max DD, `129` trades.
- Test: `+0.669806%`, `2.11568%` max DD, `106` trades.

This is a temporary SAC incumbent candidate, not proof of an optimized ETH
champion. It may control Paper/Demo only after exact observation, preprocessing,
action, simulator, and runtime feature parity is reproduced. If it cannot pass
that test, state the incompatibility and produce the first new ETH champion;
do not silently keep the linear model in authority.

### 3.4 The requested experiment does not exist yet

There is currently no completed ETH DOIN comparison of:

- `normal-only`; versus
- stage-integrated `easy -> normal` warm continuation.

There is also no archived ETH DOIN all-time champion that can honestly be
reported. State this directly. This order exists to correct it.

## 4. Work Package A: Preserve USDCAD and Transfer the Fleet

1. Prevent the supervisor from materializing another USDCAD job.
2. At the first coherent candidate/generation boundary, pause the active USDCAD
   campaign across all workers.
3. Preserve and independently reload:
   - chain/genesis/domain identity;
   - finalized block and population state;
   - pending/claimed candidates and lease state;
   - champion policy artifact;
   - decoded genome/config;
   - raw split metrics;
   - all hashes and commit identities.
4. Mark it `paused_resumable`, never `complete`, unless its own termination
   contract actually completed.
5. Prove no worker remains on USDCAD before ETH materialization. A reachable
   idle worker is an error during the transfer; an unreachable worker is an
   explicit fleet alert, not permission to start an independent chain.

No valid USDCAD artifact, OLAP result, or blockchain state may be deleted.

## 5. Work Package B: Immutable ETH Data and Split Contract

Use the causal model-ready ETH data file as the initial common contract:

`/home/harveybc/Documents/GitHub/predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`

Current SHA-256:
`1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`

Materialize explicit, identical boundaries for every paired arm:

| Split | Inclusive start | Exclusive end | Current row count |
| --- | --- | --- | ---: |
| Train | `2017-09-28 04:00:00` | `2024-01-01 00:00:00` | 13,699 |
| Validation | `2024-01-01 00:00:00` | `2025-01-01 00:00:00` | 2,196 |
| Protected test | `2025-01-01 00:00:00` | `2026-01-01 00:00:00` | 2,190 |

Before launch, recompute the hash, boundaries, monotonicity, duplicate count,
NaN/inf counts after warm-up, feature provenance, and row counts. If the file
changed, stop materialization, explain the exact data delta, and freeze one new
hash; never let paired arms use different data.

Rules:

- Features and scaling are fitted/derived causally from past data only.
- Validation selects candidates. Protected test never selects a genome,
  threshold, stage, seed, or curriculum arm.
- Both optimization arms receive the same train/validation/test rows.
- All four workers verify the same data/config/domain hashes before joining.

## 6. Work Package C: Implement Train-Only Solvency Continuation

The existing `gym-fx` behavior terminates when broker termination occurs or
equity falls below `min_equity`. A negative cash balance also drives current
relative-volume sizing toward zero, so merely setting a very negative floor
does not reproduce the owner's successful NEAT experiment.

Implement two explicit simulation modes:

### C1. `normal_realistic`

- Realistic nominal fees, spread/slippage, financing, leverage, margin, and
  margin-call/liquidation behavior.
- Margin call terminates or resets exactly as the declared realistic simulator
  contract specifies.
- This is mandatory for train-tail, validation, protected test, and every
  Paper/Demo execution check.

### C2. `easy_chronological_continuation`

- Available only while training.
- A would-be margin call is recorded with timestamp, position, equity, loss,
  and cause.
- Positions are liquidated and their complete economic loss is retained.
- The chronological episode continues instead of entering an absorbing terminal
  state.
- Maintain two explicit ledgers:
  1. **economic equity/debt**, which accumulates every real loss and may become
     negative for reward and reporting; and
  2. **operational training capital**, deterministically recapitalized only to
     permit continued action and credit assignment.
- Every recapitalization is recorded as debt, never profit. Fitness and raw
  metrics subtract the debt, so continuation cannot manufacture performance.
- The environment must remain capable of opening subsequent bounded positions;
  an inert negative-cash state fails acceptance.
- `data_end`, `would_margin_call`, safety stop, simulator error, and external
  stop are separate termination/event causes in OLAP.

Add deterministic unit/property tests proving:

- normal mode still terminates/handles margin exactly as before;
- easy mode continues to later bars after would-be ruin;
- loss and recapitalization debt are conserved;
- no recapitalization improves economic equity;
- the agent can act after recapitalization;
- validation/test reject easy dynamics;
- Paper/Demo code cannot enable easy dynamics.

## 7. Work Package D: Put `easy -> normal` Inside Every DOIN Stage

Retain the established outer stage order unless source evidence proves a code
dependency requires a repair:

1. `data_observation`;
2. `model_training`;
3. `execution_risk`;
4. `joint_refinement`.

For **every candidate in every stage**, execute this inner curriculum:

1. Decode/freeze the candidate's full feature, preprocessing, architecture,
   training, and execution contract.
2. Train under `easy_chronological_continuation` with early stopping and a
   declared maximum budget.
3. Save an immutable `post_easy` model, optimizer metadata, decoded genome,
   event counters, and hashes.
4. Initialize the normal phase from the learned actor/critic weights. Do not
   reset learned weights.
5. Reset the SAC replay buffer at the dynamics boundary unless transitions are
   explicitly mode-tagged and the normal learner demonstrably excludes easy
   transitions. Unrealistic easy transitions must not silently contaminate
   normal updates.
6. Continue training under `normal_realistic`.
7. Save immutable `post_normal` artifacts and evaluate train-tail and validation
   only under `normal_realistic`.
8. Return the transparent normal-validation selection result to DOIN. Never
   select a candidate from its easy-mode score.

Within a candidate, easy and normal use the same decoded observation dimensions
and network architecture, so weight continuation is mandatory. Across candidates
whose feature dimensions or architectures differ, do not force incompatible
weight loading. At the end of each outer stage, the `post_normal` stage champion
becomes the base/seed for the next stage; its paired `post_easy` artifact remains
immutable for ablation and recovery.

## 8. Work Package E: Paired DOIN Control

The owner asked whether the same data optimized through DOIN actually performs
better with the solvency curriculum. Answer that experimentally.

Create two immutable ETH optimization domains:

- `ETH-N`: normal-only training in every outer stage.
- `ETH-EN`: `easy -> normal` training in every outer stage as specified above.

They must share:

- dataset and split hashes;
- feature/preprocessing/model/execution genome schema;
- DEAP seed and initial population genomes;
- population, crossover, mutation, generations, patience, and candidate budget;
- normal simulation costs and validation contract;
- software commit and environment lock; and
- four-worker topology.

Run them sequentially through the same coordinated swarm, not as parallel
independent blockchains. Queue `ETH-EN` first so it can produce the requested
live candidate sooner; queue `ETH-N` immediately after it with no idle interval.
Both remain independently resumable. Do not migrate champions between the two
comparison domains.

Before the full paired campaigns, run one fixed-genome, paired-seed acceptance
fixture for:

1. normal-only;
2. easy-only training evaluated under normal conditions; and
3. easy -> normal continuation evaluated under normal conditions.

This fixture validates the mechanism; it does not replace the full DOIN
comparison.

## 9. Transparent Selection and Reporting Contract

Do not use `train_validation_l1_score`, the old dimensionless proxy, or an
undocumented weighted sum in owner-facing status or champion claims.

Use a transparent constrained/lexicographic selection contract on validation:

1. valid observation/action/protection contract;
2. no simulator or evidence failure;
3. declared minimum activity, without a positive-profit gate;
4. maximize **mean weekly net simple return**;
5. tie-break by lower maximum drawdown, then higher total net return.

If DEAP requires a scalar compatibility field, persist the lexicographic tuple
and its ordered comparison components; any encoded scalar is transport only and
must never be shown as return or profit.

For train, train-tail, validation, and protected test, and for both curricula,
persist and display at minimum:

- initial and final equity;
- total net return (%);
- each weekly net simple return and mean/median weekly return (%);
- geometrically annualized net return (%) with its formula and observed weeks;
- maximum drawdown (% and money);
- weekly return standard deviation;
- trade count, long/short count, win rate, profit factor, average trade;
- gross profit/loss, fees, spread/slippage, financing, and net P&L;
- exposure/time-in-market and turnover;
- margin-call/would-margin-call/recapitalization counts and economic debt;
- SL, TP, early-close, forced-liquidation, and data-end exit counts;
- action distribution/collapse diagnostics; and
- seed, exact date range, data/config/code hashes, artifact hash, and stage.

Show weekly metrics as weekly and annual metrics as annual. Do not multiply a
weekly mean by 52 and label it a compounded annual return.

## 10. Work Package F: Put the SAC/DOIN Champion in Paper/Demo Control

### F1. Immediate transition

1. Freeze the linear artifact as a shadow/control only.
2. Implement the exact Stable-Baselines3 SAC inference adapter required to load
   the current temporary SAC incumbent and future DOIN `policy.zip` artifacts.
3. Reproduce a golden observation vector and action from the training pipeline
   in the live runner. Exact feature order, rolling state, window, agent state,
   action threshold, sizing, SL, and TP parameters are part of the artifact
   contract.
4. If the historical v2 artifact passes exact parity, make it the temporary
   Paper/Demo SAC incumbent while `ETH-EN` runs. Label it honestly with its raw
   metrics; do not call it DOIN-optimized.
5. If it cannot pass parity, keep the venue safe and complete the first
   `ETH-EN` stage champion immediately; do not silently leave the linear model
   as the requested champion.

### F2. Champion succession

At every completed outer stage, the normal-validation stage champion is eligible
to replace the current Paper/Demo policy after deterministic load/parity checks.
The system itself does not stop between policies:

1. incumbent remains active until successor is loadable;
2. stop new incumbent entries;
3. close/reconcile any incumbent-owned position under its existing protection;
4. record the broker's actual post-close balance/equity;
5. atomically switch artifact ID/hash and observation contract;
6. successor begins with that real Paper/Demo account balance, never a reset
   balance; and
7. incumbent continues as a shadow comparator.

Use the ETH champion on each Paper/Demo venue that has the exact ETH instrument
and a supported protected-order path. Current priority is OANDA MT5 Demo
`ETHUSD@4h`; add Alpaca Paper `ETH/USD` if its direct instrument/order semantics
pass the same contract. Do not substitute a proxy asset at IBKR merely to claim
three-venue coverage. Record exact venue unavailability instead.

Every submitted entry must carry or immediately establish both SL and TP under
the accepted venue-specific atomic/reconciled protection contract. No naked
entry is allowed. No Live-capital account is authorized.

## 11. Continuous Operation and Evidence

- All workers join one active domain/genesis at a time.
- The supervisor automatically queues the next ETH comparison arm when the
  current arm ends.
- Worker commits, environment lock, data hash, config hash, domain hash, chain
  tip, stage, generation, candidate, GPU assignment, and lease are visible.
- Telegram alerts only unresolved actionable conditions: fork/parallel domain,
  duplicate evaluation, stalled candidate, idle reachable worker, stale model
  runner, venue disconnect, protection loss, reconciliation failure, GPU over
  78 C, and failed champion handover.
- Completion messages include raw champion metrics, artifact/config hashes,
  curriculum arm, stage, and next queued job.
- Persist simulator-to-Paper divergence for signals, fills, spread, fees,
  slippage, exits, P&L, and drawdown. This is business evidence, not a gate that
  postpones Paper/Demo execution.

## 12. Acceptance Gates for This Order

Return one evidence packet only after all of the following are directly proven:

1. USDCAD is paused/resumable and independently reloadable; no worker is still
   evaluating it.
2. ETH data/split/config/environment/domain hashes match on all participating
   workers.
3. Easy continuation passes the conservation, chronology, action-after-ruin,
   and no-easy-evaluation tests.
4. The fixed-genome N/E/EN fixture completed with raw metrics.
5. `ETH-EN` is running collaboratively on the full swarm in one blockchain,
   and `ETH-N` is queued next without a parallel chain.
6. The web/status surface shows the active outer stage and inner phase
   (`easy` or `normal`) for every worker and candidate.
7. A loadable SAC artifact, decoded JSON, raw metrics, and all hashes exist for
   the current ETH incumbent/champion.
8. MT5 Demo reports the SAC model ID and artifact hash as execution authority;
   the linear model is marked shadow/control.
9. At least one model-originated protected ETH Paper/Demo decision has traversed
   inference -> intent -> order -> direct SL/TP evidence -> reconciliation, or
   the policy has produced no entry signal and the complete signed decision
   trace proves that fact. A manual canary cannot be reported as model trading.
10. Full focused and repository test suites pass, with exact commands/results.

## 13. Required Return Packet

Create `SATOSHI_III_ETH_CHAMPION_STAGE_CURRICULUM_DELIVERY_2026_08_05.md` with:

- exact commits per repository and clean/pushed state;
- source paths and codebase-memory traces inspected before implementation;
- USDCAD archive/checkpoint evidence;
- ETH dataset/split/hash evidence;
- solvency accounting specification and tests;
- fixed-genome N/E/EN raw comparison table;
- `ETH-N` and `ETH-EN` DOIN domain/config/node files and queue state;
- per-worker chain/commit/environment/GPU evidence;
- current SAC incumbent and latest DOIN champion raw split metrics;
- model/config/decoded-genome hashes and load tests;
- MT5/Alpaca Paper inference and protected-order evidence;
- unresolved defects or unknowns stated plainly; and
- an audit request that maps every acceptance item above to durable evidence.

Do not close your own findings or declare this order accepted. Musashi will
independently reproduce the delivery. Do not wait for another audit to begin the
authorized implementation and Paper/Demo work.

