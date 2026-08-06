# Audit: Satoshi III J0/J4 Delivery and Retraining Frequency

Date: 2026-08-06 America/Bogota
Auditor: General Musashi, independent verifier
Delivery under review:
`docs/handoffs/SATOSHI_III_J0_J4_PACKET_AND_DISSENT_2026_08_06.md`
Runtime mutation by this audit: none
Broker submissions: zero
Network use by the reproducer: false

## 1. Verdict

The delivery is **not accepted as a curriculum-decision or exact-controller
packet**. Corrections 123, 124 and 127 pass their focused mechanical tests;
they still need a bounded runtime smoke before operational acceptance.
Corrections 113, 122, 125 and 126 remain incomplete. J4 does not inventory the
controller actually running on every host and cannot establish that an eligible
SAC artifact is authoritative.

The active `phase-2-eth-anchored-full-fleet-v2` campaign remains untouched.
This audit does not authorize pausing it, resuming it under changed semantics,
or launching N14/EN4_10/E4. The correction packet must first pass independent
reproduction.

At 2026-08-06 04:48 COT, Omega's old-semantics worker was actively computing
epoch 295/2000 at 100% process CPU, 18% GPU utilization and 50 C, while still
reporting zero train/validation trades. It is not hung; it is direct evidence of
the already-open activity-patience defect. Its status ETA of zero after elapsed
time exceeded a one-sample median is not treated as a valid completion estimate.

Canonical reproducer:

```text
docs/audits/evidence/SATOSHI_III_J0_J4_RETRAINING_REPRO_2026_08_06.py
```

It reproduces every finding below without sockets and publishes the exact data
contract used by the current ETH runner.

## 2. Findings

### AUD-F1-20260806-128 - S2 - empty lineage can pass rejoin proof

`CampaignSupervisor.verify_rejoin()` compares domain, genesis and population
only when the corresponding bound value exists. It also permits a missing
observed domain. A paused state with all three identity fields absent and a
worker with no chain evidence returns `rejoin_proven=true`; the fleet resume
tool consequently accepts it.

Evidence:

- `app/campaign_supervisor.py:3493`
- `app/campaign_supervisor.py:3578`
- reproducer field `empty_lineage_rejoin`

Impact: correction 122 is rejected. Absence of identity is not proof of the
same chain. Resume must require complete bound identity and complete post-start
evidence from every expected worker before success.

### AUD-F1-20260806-129 - S2 - terminal policy is not preserved or evaluated

The real RL pipeline reloads the best checkpoint before returning and returns
only metrics plus `best_model_path`; it does not expose the policy that existed
at the terminal training step. The decision runner looks for `model`,
`best_model`, or `terminal_model_path`, finds none, catches the resulting gap,
and can still write the note that both weight sets were evaluated. Its `final`
artifact label points to the configured best-checkpoint path.

Evidence:

- `pipeline_plugins/rl_pipeline_with_validation.py:1364`
- `pipeline_plugins/rl_pipeline_with_validation.py:1373`
- `tools/eth_curriculum_decision_experiment.py:243`
- `tools/eth_curriculum_decision_experiment.py:249`
- reproducer field `terminal_artifact_gap`

Impact: correction 125 is rejected. The pipeline must save the terminal policy
before reloading best weights, hash both artifacts, evaluate both under the same
validation contract, and fail the arm if either is absent or invalid.

### AUD-F1-20260806-130 - S2 - stale arm records are reusable

`run_arm()` reuses an existing `arm_record.json` after checking only arm name,
seed and a nonempty split map. It does not bind the result to data hash, code
lineage, base contract, resolved config, epoch budget, shared anchor or artifact
hashes. The reproducer changes the requested timesteps and removes the new
anchor; the old record is still accepted.

Evidence:

- `tools/eth_curriculum_decision_experiment.py:180`
- reproducer field `stale_arm_reuse`

Impact: correction 126 is rejected. Idempotence requires a content-addressed
execution identity, not a filename plus seed.

### AUD-F1-20260806-131 - S2 - incomplete and incompatible packets can promote

The aggregator treats any truthy validation mapping as complete. It does not
require finite decision metrics, terminal evaluation, margin telemetry,
artifacts, traces, common data/base/code contracts, or unique packet identities.
Four malformed packets with different lineage and only
`validation={"garbage": 1}` yield exit zero and
`promotion_eligible=true`, while paired weekly return is null.

Evidence:

- `tools/aggregate_curriculum_decision.py:97`
- reproducer field `empty_packet_promotion`

Impact: corrections 125 and 126 are rejected. The aggregator must validate a
versioned packet schema and common experiment identity before computing any
decision.

### AUD-F1-20260806-132 - S3 - repair-rule schema is incomplete and biased

Repair validation accepts a rule for a nonexistent gene and does not prove
that the target is categorical or that values belong to its declared choices.
`resample_categorical` deterministically chooses the first allowed value. That
repairs validity but creates an ordering-dependent evolutionary prior.

Evidence:

- `tools/materialize_eth_curriculum_configs.py:311`
- reproducer field `repair_schema_gap`

Impact: correction 113 is only partial. Repair must validate against the typed
gene schema and make a deterministic seeded draw over allowed choices, recording
the chosen value and repair reason.

### AUD-F2-20260806-133 - S3 - J4 is host-blind and cannot prove SAC authority

The inventory classifies a controller from whether its `model_id` contains the
word `linear` and hard-codes `sac_champion_authoritative=false` for every
nonempty model id. It never joins the heartbeat's artifact/config/input hashes
to an eligible selected-model manifest. It also reads all seats as if they were
Omega-local.

Direct evidence contradicts the delivered J4 conclusion: the MT5 runner on
Dragon was active and fresh during the audit, controlling
`ethusdt-4h-linear-live-v1`, with one direct position fact. Omega-local
inspection reported that same seat inactive.

Evidence:

- `lts/tools/controller_inventory.py:172`
- `lts/tools/controller_inventory.py:185`
- reproducer field `sac_classifier_false_negative`

Impact: J4 is rejected as an exact fleet inventory. Each venue seat needs an
explicit evidence host/path and an exact manifest-to-heartbeat hash join.

### AUD-F1-20260806-134 - S2 - current evidence does not match the adaptation business contract

The current experiment chooses one policy on a complete 2024 validation year.
The business contract instead permits periodic fine-tuning and needs the policy
to work over the next deployment interval or week. No rolling-origin,
test-then-train experiment currently estimates the value or feasible cadence of
that adaptation. Existing use of `window_size=32` and rolling scaling window
`256` proves only that those values ran before; it is not evidence that either
is optimal or should be frozen under the no-default contract.

Impact: the typed parameter registry and restricted joint integration cannot
freeze these values or a retraining cadence yet. This does not invalidate the
mechanical N14/EN4_10 question; it prevents relabeling its annual static result
as the final deployment policy.

## 3. Independently Verified Data and Compute Contract

Dataset:

```text
/home/harveybc/Documents/GitHub/predictor/examples/data/project3/
ethusdt_4h_tech_stat_full_model_ready.csv
sha256: 1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f
```

| Partition / quantity | Exact value |
| --- | ---: |
| Total rows | 18,085 |
| Train, 2017-09-28 04:00 through 2023-12-31 | 13,699 bars |
| Validation, 2024 | 2,196 bars / 366 days |
| Disclosed test, 2025 | 2,190 bars / 365 days; disabled in N/EN/E |
| Bar interval | 4 hours |
| Input features | 83 |
| Observation window | 32 bars = 128 h = 5 d 8 h |
| Rolling scaling window | 256 bars = 1,024 h = 42 d 16 h |
| Flattened observation | 2,724 values: 83x32 features + 32 prices + 32 returns + 4 agent-state |

The base config still contains `train_years=4`, but the runner's explicit
dates override it with approximately 6.25 years and 13,699 unique training
bars. That override must be explicit in every packet; it cannot remain a
misleading dormant field.

Training compute per seed is:

- N14: 14 x 20,000 = 280,000 environment steps;
- EN4_10: 4 x 20,000 + 10 x 20,000 = 280,000 steps;
- E4 diagnostic: 4 x 20,000 = 80,000 steps.

Across four seeds this is 2.56 million environment steps. Repeating epochs does
not create additional historical observations.

## 4. Retraining/Fine-Tuning Frequency Program

### 4.1 Two clocks

The deployment contract separates:

1. **fast adaptation**: fixed architecture, feature contract and objective;
   update weights/replay state from newly available causal data;
2. **slow structural optimization**: DOIN changes architecture,
   hyperparameters, feature/preprocessing masks or curriculum and produces a
   successor artifact.

The current fleet cannot yet promise a complete DOIN structural campaign every
6 or 12 hours. A single observed candidate duration is about 78 minutes and a
current candidate has exceeded four hours; those sparse samples are not enough
for a deadline guarantee. Weight-only adaptation may fit the interval, but RT0
must measure it.

### 4.2 Cadences are bar-aligned

With 4-hour bars, the first candidate set is:

| New bars | Cadence | Role |
| ---: | ---: | --- |
| 1 | 4 h | feasibility-only stress case |
| 2 | 8 h | high-frequency candidate |
| 3 | 12 h | recommended first operational candidate |
| 6 | 24 h | daily control |
| 18 | 72 h | medium control |
| 42 | 168 h | weekly incumbent |

Six hours is not directly bar-aligned and would alternate between one and two
new closed bars. It is excluded unless a future causal 1-hour input contract is
selected. Twelve hours is a candidate, not a default.

### 4.3 RT0 - deadline and orchestration feasibility

Use one frozen 28-day 2024 block, the current fixed SAC/config, one seed and
lookbacks of one and two years. Run 8/12/24/72/168-hour cadences, adding 4 h
only if the first benchmark can finish safely. Measure:

- p50/p95 end-to-end update time, GPU time, peak RAM/VRAM and temperature;
- deadline misses, failed updates, rollback and artifact-switch time;
- number of new bars per update, model age and deployment coverage; and
- exact pre/post-switch account state in simulation.

RT0 may reject infeasible cadences. It cannot promote a profit/risk winner.
Proposed operational acceptance is p95 update time no greater than two thirds
of the cadence, with zero unreconciled switches; the owner must ratify that
budget before it becomes binding.

### 4.4 RT1 - bounded performance screen

Use four preregistered, non-overlapping 28-day blocks across 2024, two paired
seeds and lookbacks of one, two and four years/expanding. Keep model topology
fixed. At each origin:

1. fit or fine-tune using only information available through time `t`;
2. deploy and score `(t, t+h]` before using those rows for an update;
3. preserve the ordered interval and weekly metric series; and
4. continue test-then-train through the block.

This rolling-origin/prequential design estimates the next-interval behavior
the business sells. A single month remains useful for plumbing and runtime; it
is too sparse to select profit/risk by itself.

### 4.5 RT2 - dedicated DOIN adaptation-schedule domain

Start RT0/RT1 after the SAC topology/learning domain. Finalize RT2 after every
admitted interface-changing component line and before restricted joint
integration. Its typed genome contains only evidence-bounded adaptation genes:

- `retrain_interval_bars`;
- rolling/expanding lookback and `lookback_bars`;
- warm-start, reset or bounded full-refit mode;
- update/gradient budget per new bar;
- replay retention/reset and recency weighting;
- encoder freeze/fine-tune choice; and
- handover policy: next-flat or bounded-delay activation.

Use successive fidelity: one block/one seed, then four blocks/two seeds, then
full 2024/four seeds for elites. Fitness is the ordered next-interval and weekly
validation series under a fixed compute/deadline constraint. Cadence cannot buy
fitness by consuming unlimited compute.

### 4.6 RT3 - frozen confirmation

Freeze the schedule and all adaptation hyperparameters before final
prequential evaluation. The repeatedly inspected 2025 period is a disclosed
secondary benchmark, not a pristine sole test. The cleanest confirmation is
prospective 2026 Paper/live-shadow data collected after freeze.

Every handover preserves account continuity:

```text
stop new risk -> close/reconcile protected exposure -> record exact post-close
balance -> activate hash-bound successor -> resume
```

No live capital is authorized. Mandatory SL/TP and fail-closed venue controls
remain unchanged.

## 5. Required Metrics

Report every deployment interval and an ordered weekly aggregate in the same
units:

- net simple return and return in percent;
- annualized return derived only from the ordered weekly series;
- maximum drawdown, trades/actions and long/short/hold counts;
- commission, spread, slippage, financing and forced-handover cost;
- update duration, deadline miss, failure/rollback and deployment coverage;
- model age, new bars seen, lookback, update budget and seed;
- before/after account balance and open-exposure reconciliation; and
- model/config/data/code/input/decision hashes.

Rolling-origin evaluation follows the standard requirement that every test
window be evaluated using only preceding observations. The implementation may
cite Hyndman and Athanasopoulos' time-series cross-validation procedure and the
MOA prequential evaluation framework as methodological foundations:

- https://otexts.com/fpp3/tscv.html
- https://www.jmlr.org/papers/volume11/bifet10a/bifet10a.pdf

## 6. Test Evidence

```text
independent reproducer: all_counterexamples_reproduced=true; network_used=false
agent-multi focused corrections: 36 passed
agent-multi full suite (trading-stack): 571 passed, 2 warnings
lts full suite (trading-stack): 652 passed, 1 warning
```

Passing suites do not negate the independently reproduced contract failures;
they establish that the defects are test gaps rather than unrelated suite
breakage.

## 7. Disposition

- 123, 124 and 127: mechanically corrected; runtime smoke pending.
- 113: partially corrected; finding 132 supersedes the remaining schema/bias
  gap.
- 122: rejected; finding 128 remains blocking.
- 125: rejected; findings 129 and 131 remain blocking.
- 126: rejected; findings 130 and 131 remain blocking.
- J4: rejected as exact fleet/controller evidence; finding 133 remains open.
- retraining cadence: new decision-bearing domain; finding 134 blocks freezing
  an arbitrary cadence or treating annual static selection as deployment truth.
