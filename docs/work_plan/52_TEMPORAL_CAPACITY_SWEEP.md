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

Stage: REQUIREMENTS_AND_UNIT_TEST_DESIGN_BEFORE_IMPLEMENTATION.
