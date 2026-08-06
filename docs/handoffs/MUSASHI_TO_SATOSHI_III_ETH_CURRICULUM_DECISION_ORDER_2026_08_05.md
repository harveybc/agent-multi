# Correction and Execution Order: ETH Easy/Normal Decision Packet

Date: 2026-08-05 America/Bogota
From: General Musashi, temporary independent auditor
To: Satoshi III (Mujuro Utsutsu), successor technical lead
Owner priority: decide whether easy-first training should be used for the
remaining asset/model optimizations

Act as a senior machine-learning scientist, reinforcement-learning engineer,
evolutionary-computation engineer, distributed-systems engineer and
reproducible-experiment engineer. Be direct about null/negative evidence. Do
not optimize presentation. Do not mutate a running chain, inspect the disclosed
2025 outcome, or represent a transport order key as profit.

Read in order:

1. `docs/handoffs/SATOSHI_III_ETH_CORRECTION_DELIVERY_2026_08_05.md`
2. `docs/audits/AUDIT_SATOSHI_III_ETH_CORRECTION_DELIVERY_2026_08_05.md`
3. `docs/work_plan/19_EXECUTION_CURRICULUM_AND_ORDER_ROUTING.md`
4. `docs/audits/evidence/ETH_ANCHORED_SWARM_RECOVERY_2026_08_05.md`
5. `pipeline_plugins/rl_pipeline_with_solvency_curriculum.py`
6. `tools/eth_curriculum_fixture.py`

This order does not close findings. General Musashi independently verifies the
result. The current `full-v2` workers remain productive while you prepare the
packet; do not touch their units, profiles, PIDs, state or chain until every
pause/resume and local-job preflight below is ready.

## 1. Priority and Scientific Question

The primary question is not whether easy-only can make trades. It is:

> For the same current-stack ETH/SAC policy, initialization, data, causal
> features, validation contract and total update budget, does `easy -> normal`
> produce a better realistic-normal policy than `normal-only`, consistently
> enough to justify carrying that curriculum into later asset/model work?

`easy-only` is a diagnostic arm. The operational decision is
`normal-only` versus `easy -> normal`.

Do not infer a universal answer from one seed, the best candidate, raw
positive profit, or the current ETH-EN chain alone.

## 2. WP-A: Complete Findings 113 and 114

### 2.1 Executable genome validity

The current `forbid_value` object is ignored by
`project3_full_genome_optimizer._apply_repair_rules()`, which implements only
`if` + `set` rules.

Choose one coherent correction:

1. implement and schema-validate `forbid_value`/categorical resampling at the
   encoding boundary; or
2. remove the unsupported declaration and enforce the forbidden value through
   one typed schema validator used by fresh generation, resume, migration and
   network-champion import.

Required tests:

- fresh genomes never decode to `none`;
- an injected legacy/resume/network genome carrying `none` is repaired or
  rejected before environment or GPU construction;
- unknown repair-rule kinds fail materialization;
- an empty cosmetic rule cannot satisfy validation;
- rejected invalid candidates cannot become initial/local/remote champions or
  blocks.

### 2.2 Complete evidence packet

For every arm/seed preserve:

- fully resolved config JSON and SHA-256;
- source anchor model hash and candidate genome hash;
- model artifact hash and a retrievable artifact location with at least two
  replicas or an explicitly verified GitHub-compatible artifact mechanism;
- complete equity/return trace and SHA-256 for train, train-tail and validation;
- per-epoch learning/activity trace;
- exact code/data/environment lineage;
- raw same-scale metrics with units;
- test split disabled and absent from every result payload.

A manifest pointing only to ignored local ZIP files does not satisfy
cross-workstation recovery.

## 3. WP-B: Make Pause Reversible and Profile Installation Safe

Correct findings 115, 119 and 121 before interrupting `full-v2`.

### 3.1 Pause verification

- unavailable `nvidia-smi` evidence means pause verification failed;
- inspect every owned process group, API port and GPU compute PID;
- record plan/profile/domain/genesis/tip/population hashes before stopping;
- preserve incomplete candidate/lease facts without editing the blockchain or
  candidate pool.

### 3.2 Same-chain resume

Implement one authenticated, idempotent resume API/CLI that:

- is legal only from a verified `paused` state;
- requires exact plan, profile, config semantic hash, domain, genesis,
  population fingerprint and component revisions;
- rejoins the existing chain and shared pool; it never creates genesis;
- refuses a changed tip unless the normal peer convergence contract accepts it;
- leaves an append-only pause/resume event and direct four-worker evidence;
- cannot resurrect invalid `full-v1`, old smoke or archived chains.

### 3.3 Profile transaction

Installing a different systemd profile while a campaign is active must fail.
Persist the expected profile hash and compare it to systemd `ExecStart` on
every supervisor tick. Any drift is a high-priority alert and blocks restart.
Install profile changes only after verified pause and as an atomic fleet
transaction.

## 4. WP-C: Paired Four-GPU Curriculum Experiment

Materialize a fresh local-only result root and four new seeds. Do not write
DOIN blocks, seed populations, champion archives or live-trading succession.

### 4.1 Fixed contracts

Hold constant across arms:

- ETHUSD 4h frozen dataset and data SHA;
- current 83-feature causal observation contract, window 32 and rolling
  normalization 256;
- current-stack anchor artifact and exact initial policy weights;
- model hyperparameters, execution router, order types, SL/TP, rel-volume,
  random seed and split boundaries;
- 2017-2023 train, 2024 validation, disclosed 2025 disabled;
- realistic-normal validation and `lexicographic_weekly_v1` comparison;
- fresh replay buffer at every easy-to-normal dynamics boundary.

Use four fresh seeds, one per GPU. Each GPU executes all arms for its seed so
host/GPU is a blocking factor rather than an arm confound.

### 4.2 Arms and equal compute

Primary arms:

1. `N14`: 14 normal epochs;
2. `EN4_10`: 4 easy epochs followed by 10 normal epochs;
3. `E4`: 4 easy epochs, then inference-only evaluation under normal conditions
   as a mechanism diagnostic.

Use 20,000 timesteps per training epoch and disable early stopping for this
bounded packet. `N14` and `EN4_10` therefore receive equal total training
timesteps. If resource preflight shows that this budget cannot finish within a
bounded operational window, propose one smaller equal-budget packet before
running it; never silently reduce only one arm.

### 4.3 Measurements

Persist per arm/seed/split:

- mean weekly net simple return, annualized return and total return;
- maximum drawdown as fraction and percent;
- trades, wins/losses, turnover and cost drag;
- long/short/hold action counts and raw action distribution;
- entry actions, submitted market/limit/stop entries, cancellations, fills,
  risk-reducing closes, SL and TP outcomes;
- entropy coefficient trajectory, actor/critic deltas and collapse epoch;
- `would_margin_call_count`, actual termination cause, recapitalization count
  and debt;
- normal-handoff activity and policy-weight hashes before/after normal;
- wall time, GPU time/energy when available and peak memory;
- eligibility plus the complete ordered tuple; the order key is reported only
  as transport evidence.

Never replace the raw table with one composite score.

### 4.4 Interpretation

Report each paired seed and median paired differences. Four seeds do not
justify a theatrical p-value; show direction consistency and effect sizes.

Explicitly answer:

1. Did EN beat equal-compute N under realistic validation?
2. Did easy produce behavior that survived the normal threshold/cost/margin
   handoff?
3. Did any arm collapse to hold, one-sided action or entropy zero?
4. Did any episode actually cross the would-margin-call boundary?
5. If no margin event occurred, was any observed difference caused only by
   lower costs or the zero deadband rather than solvency relaxation?
6. Is the answer consistent across seeds or candidate-luck dominated?

The result may support ETH/SAC adoption, rejection or an explicit ambiguous
state. It cannot by itself authorize all model families.

## 5. WP-D: Mechanism Ablation Only If Needed

The current easy phase changes more than solvency: it also changes costs and
the action deadband. If WP-C finds an EN effect, run a bounded follow-up that
isolates:

1. solvency relaxation only with normal costs/deadband;
2. low costs only with normal solvency/deadband;
3. zero deadband only with normal solvency/costs;
4. combined easy.

If all production-like runs have zero would-margin-call events, add a clearly
labelled mechanism stress probe using a preregistered higher-risk configuration
to exercise the termination boundary. That probe diagnoses mechanism; it does
not select a production champion.

## 6. WP-E: Fleet Execution Sequence

1. Prepare and test all code/configs locally while `full-v2` continues.
2. Return the preflight packet to Musashi before touching runtime.
3. After independent acceptance, invoke the verified fleet pause and prove all
   four workers absent.
4. Run one seed on each GPU, all arms sequentially, with hourly temperature
   monitoring and the existing 78 C alert.
5. Materialize the cross-seed packet and recommendation.
6. If EN is retained, use the authenticated same-chain resume and prove the
   exact old domain/tip/pool was rejoined. If N is selected, preserve `full-v2`
   as a stopped EN experiment and materialize a fresh N domain/genesis; never
   rewrite its chain.

No GPU should be idle between accepted preflight and experiment launch. No
profile may be switched independently on one host.

## 7. Other Important Aspects After the Immediate Decision

These are ordered follow-ups, not excuses to delay WP-C:

1. **Model-family transfer:** repeat a reduced paired packet for a second SAC
   asset before treating the result as SAC-wide; repeat separately for a
   different model family before making it system-wide.
2. **Entropy stability:** current candidates can drive `ent_coef` close to zero;
   compare collapse rate and consider an entropy floor/range only from paired
   evidence.
3. **Regime sensitivity:** report causal 2024 subperiod/regime slices after
   global evaluation; do not select on hindsight labels.
4. **Execution realism:** after curriculum choice, retain nominal/stress cost
   evaluation and order-type/protection metrics; easy success never replaces
   realistic validation.
5. **Live divergence:** the selected Paper/Demo challenger must be compared to
   deterministic simulation from the same due-bar decisions for at least the
   existing rolling 24-hour/7-day evidence windows.
6. **Compute efficiency:** preserve quality-per-GPU-hour as an operational
   metric, not the alpha fitness, so an expensive curriculum is adopted with
   eyes open.

## 8. Delivery Contract

Return one audit request containing:

1. corrections and tests for 113/114/115/119/121;
2. exact clean/pushed commits and environment lock;
3. pause/resume/profile-drift adversarial evidence;
4. local experiment plan and hashes before runtime mutation;
5. direct four-worker pre-pause snapshot;
6. per-seed/per-arm raw result table and paired differences;
7. configs, traces and retrievable artifact manifest;
8. a plain recommendation limited to the evidence domain;
9. doubts and anomalies stated directly;
10. if a runtime switch occurred, exact post-action chain/pool/GPU snapshot.

Do not close your own findings. Do not ask the owner to choose before the raw
comparison table exists unless a genuinely external action blocks execution.
