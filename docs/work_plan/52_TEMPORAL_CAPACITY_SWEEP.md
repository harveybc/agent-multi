# Autonomous Temporal Capacity Ladder

Owner authorization2026-10-20: run the approved gradual capacity study end to
end,sequentially on the preferred5090,without intermediate owner approvals or
agent polling. Reuse the closed fixed-budget result; never retrain its seed7
baseline or alter its artifacts. No DOIN,TEST,window/stride change,regularizer,
pretraining,refit or change to SAC updates/reward/cost/business constraints.

## Frozen Scientific Rules

15 selected features,13 observed open-market hourly bars,stride2,includes t;
TRAIN2020-23,VALIDATION2024,100000 transitions,observations at25k/50k/75k/100k,
final100k selected regardless of validation. Same costs,leverage50,SL/TP,
volume budget,entry quota and session rules. Native actor and critic extractors
remain separate; twin Qs retain their native shared critic extractor.

Stages,always run both widths even if the first is worse:
1. Core filters8 then16,branches2 and hidden[8].
2. Hidden[16] then[32],using the stage1 incumbent architecture.
3. Branch filters4 then8,using the stage2 incumbent architecture.
4. Add one further causalConv1D in the core,same core width;preserve all13
   positions and t. Depth1 remains exactly the original registered structure.

Every stage also includes its previous incumbent. The provisional promotion
rule is deliberately conservative: validation mean daily log growth must
improve by more than1e-8;validation maxDD must not increase beyond1e-8 numeric
tolerance;positive TRAIN-minus-VALIDATION daily-growth gap must not exceed
max(incumbent gap,0)+1e-8;and validation must close at least one trade.
Both candidates are compared against the same frozen stage-start incumbent;
eligible candidates rank by higher validation daily growth,then lower maxDD.
These are development ranking rules,not a new business risk limit or evidence
of statistical significance. All outcomes,including rejected candidates and
tradeoffs,remain recorded. No favorable-year replacement or seed picking.

At most7 new seed7 fits. If the final incumbent differs from baseline,run
baseline and finalist with shared seeds17 and29 (four further fits),report all
three shared seeds including retained seed7. Do not declare a confirmed winner
from the best seed. Report mean daily-growth differences,return/risk/activity;
mixed replication is INCONCLUSIVE. This is a diagnostic coordinate ladder,
not proof its stagewise choices optimize interacting parameters jointly.

## Execution And Recovery Contract

One resource-capped systemd worker with exclusive sweep lock;one bounded
publisher on a timer;one second-machine collector on a timer. Native queue
owns all SAC training/inference,outbox and warehouse readback. No alternative
learning loop. The next candidate is materialized with pinned JSON only when
its parents are authenticated closed or explicitly failed after finite retry.
Publication outage waits without refitting;completed native TRAIN is adopted
when only terminal validation needs retry. Interrupted/failed TRAIN is retained
as partial and at most one fresh-root scientific retry is allowed,because
policy-only checkpoints cannot resume exact replay/RNG. Do not merge old/new
monitor attempts or overwrite historical failures. Exhausted candidates are
recorded and skipped;an integrity/data/version contradiction fails closed,
never produces fabricated success. Resource waits preserve the pending work.
No implementation can promise progress through permanent hardware failure.

All complete pairs go to existing OLAP with effective model/config/data/seed
identities;curves and native terminal reports remain authenticated. Sweep-level
JSON/CSV/Markdown summarize every trial. Closed source files stay immutable.
Status at sweep and child level includes completed/current/planned work and
provisional ETA from measured completed runs;blocked ETA is null. Independent
physical collection has a distinct completion state. At most11 new successful
fits,excluding bounded retries;nominal duration starts at roughly70min each,
not a measured promise for larger architectures.

## Tests Designed Before Implementation

| ID | Structural and behavioral evidence |
| --- | --- |
| TC01 | strict new request permits only declared widths/depth/seeds; legacy v7/v8 unchanged; common config differs only declared architecture and seed |
| TC02 | core depth1 preserves state_dict names,construction and outputs;depth2 preserves shape,prefix causality,current-tick influence and native save/reload |
| TC03 | all7 architectures materialize in order;incumbent promotion/gap/risk/ties deterministic;seed replication uses paired baseline/finalist |
| TC04 | one lock/worker;resource waits;publication waits do not refit;bounded fresh-root TRAIN retry;validation-only retry adopts TRAIN;exhausted arm advances |
| TC05 | exact native SAC synthetic updates on every shape;real monitor retains declared encoder/seed/depth and parent policy/RNG/replay invariants |
| TC06 | plot/publish/close then readback before advancement;collector discovers new candidates and verifies closed proofs without copying live SQLite |
| TC07 | staged config/code/plan pins survive restart;tampering,unknown shape,changing support or seed are refused;status/results preserve scope and units |
| TC08 | actual worker CPU tests,GPU admission,systemd parser,real initial progress;all sequence units and physical collector enabled before handoff |

## Verified Deployment

Runtime `ee35cb3b174cac1941c703ba0cad6dd3374a21c4`;private spec digest
`b7cca8ea21c326fc51bd788c3816eba0a8cc97385d5c2ac323cf6b73beb2c3f9`.
219 focused local regressions passed,including both native FIXED and new
CAPACITY real-engine monitors;79 collector/orchestration regressions and19
final status cases passed after their narrow follow-ups. Actual-worker
acceptance35/35 passed in30.13s,1.5GiB peak,no swap. Depth2/seed17 propagates
through the real monitor with parent policy/RNG/replay invariance and native
roundtrip;legacy depth1 state keys and deterministic forward remain unchanged.

All generated services parsed by systemd. Worker and publisher timers are
enabled;second-machine discovery/collection timer is enabled independently.
The first candidate,branches2/core8/hidden[8]/depth1/seed7,passed native GPU
admission16 updates and exact reload on attempt1;contract
`446e378c77130de164b7994a7706305e2675b2ed53ba9f14ec959a535aa3ec53`.
Scientific TRAIN began2026-10-20T09:50:37UTC and reached5000 transitions in
104.9758s. No previous weights were imported. The next width is core16;
later heads/branches/depth and conditional shared-seed confirmation are
selected and dispatched by the frozen script,not by conversational turns.
Startup/progress is not a completed financial result.

## Runbook

The operator provisions one pinned runtime and supplies private paths:

```bash
python -m tools.temporal_capacity_sweep prepare --root "$ROOT" \
  --base-request "$BASE_REQUEST" --baseline "$BASELINE" \
  --baseline-sha256 "$BASELINE_SHA256"
python -m tools.temporal_capacity_sweep render --root "$ROOT" \
  --units-dir "$UNITS" --environment-file "$ENVIRONMENT_FILE"
```

Install those four rendered files into the user systemd directory,reload and
enable `rl-temporal-capacity-worker.timer` and
`rl-temporal-capacity-publication.timer`. Start the worker service once for
immediate launch. Do not launch individual configs manually while it owns the
sweep lock. Enable the independent collector with its authenticated SSH alias,
remote runtime/interpreter,private state directory and bulk archive. Do not
print EnvironmentFile contents or copy live SQLite WAL files to NTFS.

```bash
python -m tools.temporal_capacity_sweep status --root "$ROOT"
```

Read-only status includes the active child/native monitor,stage incumbent,
completed trials,full closed metrics and provisional whole-trial ETA bounds.
`SUMMARY.json`,`RESULTS.csv`,`RESULTS.md` refresh after complete trials.
`SWEEP_COMPLETE.json` ends computation/publication;the collector's own
`COLLECTION_COMPLETE.json` separately certifies physical copies. Failed partial
run folders remain at source with explicit failure ledgers;the collector's
scope does not falsely claim nonterminal roots as closed models.

The collector initially reclaimed memory under its inherited192MiB soft cap:
no OOM occurred. The dedicated collection unit was corrected to768MiB soft,
1500MiB hard because importing the existing sequence helpers also imports the
native policy library. This operational correction changes no training cap,
runtime,contract or metric. Closed-model physical collection is checked below
through its generated status,not inferred merely from the timer being enabled.
After correction it exited successfully at468893696 bytes peak. The baseline
closed-model copy is verified,pending0;the collector now waits for newly
completed trials. Scientific progress subsequently reached10000 transitions
in221.296s,worker restart count0,no BLOCKED marker. No new complete financial
pair had been issued at that startup cut.

Stage: OPERATION_GRADUAL_TEMPORAL_CAPACITY_RUNNING.
