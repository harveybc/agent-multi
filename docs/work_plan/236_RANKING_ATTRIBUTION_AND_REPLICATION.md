# Ranking Improvement Attribution And Replication

Owner approved 2026-11-09 following235. Preserve all source checkpoints and
the successful original candidate; do not promote or replace any live policy.

## Fixed Sequence

1. CPU native attribution of original paired-greedy/32 versus both the unchanged
   preserved70 actor and matched control/32, in TRAIN and VALIDATION. Use exact
   timestamp support, same-state isolated interventions and chronological
   contributions that sum to the full account difference. No fitted labels or
   thresholds are derived from these dated outcomes.
2. Two prespecified replication sets, run sequentially. In each, retain the
   four234 arms and add `paired_stochastic_reward`, using the same stochastic
   trajectories but excluding their entropy contribution. This isolates entropy
   convention from stochastic action continuation. Models, source replay, raw
   data, risk rules, input windows and representation remain fixed.
3. Automatically authenticate, collect and generate per-run CSV/PNG plus a
   combined paired-difference table. Report negative or inactive arms too.

| Replication | Anchor seed | Update schedule/RNG seed | Sampled-action seed | Continuation seeds |
|---|---:|---:|---:|---|
| r19 |101|19|751|101,103,107,109|
| r43 |149|43|857|211,223,227,229|

Both begin from the identical frozen parent actor and previously calibrated
fresh critic. These are replications of **correction sampling and optimization**,
not independent market samples or independently trained parent networks. The
inherited full TRAIN replay remains the same stochastic7 trajectory. Correction
labels come only from TRAIN; validation is evaluated after the frozen budgets.

Budgets remain4096 critic updates,batch128 states,actor checkpoints1/8/32.
The deterministic counterfactual panel still takes at most32 states per
year/position stratum, with24 fitting and8 diagnostic anchors. No failed
states are omitted. Native controls verify original epoch-zero cash before
new actor updates. The exact231 actor-update control check is only applicable
to update seed7; new seeds have their own matched control, not a fabricated
claim of reproducing seed7 endpoints.

## Resource Allocation

Measured prior shard runtimes support a5:2 interleaved CPU allocation to the
primary and secondary workers. This changes ownership only, never cohort or
targets. Each row has exactly one producer and both shards must cover the
manifest exactly once. After CPU collection, all five arms use RTX5090
sequentially; no GPU job competes with another arm. Reports and replication2
start without intermediate approval or dependence on favorable metrics.

## Test And Evidence Contract

| ID | Required check |
|---|---|
| P01 | Replication seeds, arms and allocation are explicit in effective config and forwarded to all workers |
| P02 | Default design retains original seed semantics; invalid replication IDs and allocations rejected |
| P03 | Weighted shards have no overlaps or missing IDs; labels and chronological support do not depend on shard ownership |
| P04 | Plain stochastic targets equal reward-only terms from the identical stochastic continuations, not resampled or greedy trajectories |
| P05 | Source/producer/config hashes verified on every resumed stage; no writes under old artifact roots |
| P06 | Attribution authenticates both tapes, replays both accounts and reconciles chronological cash exactly; isolated effects are not summed |
| P07 | Same-state clones preserve parent state, all three signals remain correctly routed and protections unchanged |
| P08 | Only critic heads change during correction; actor/encoder/alpha exact, original checkpoints unchanged |
| P09 | All5 arms and both replications complete or emit explicit failure; ranking/returns/activity reported even if worse |
| P10 | Combined report includes paired differences per seed and fixed endpoint; no selection of favorable seeds/checkpoints |

Use mean, range and paired differences descriptively. Two new seeds do not
support a broad robustness claim or independence assumption. Success requires
looking at ranking and account behavior separately; neither alone proves the
other. Retain reports of early-close counts and helpful/harmful outcomes.

## Execution Receipt

Runtime `c3fb4a69` published to master. Focused CPU verification:44 passed,
including native attribution reconciliation, altered support rejection,
seed forwarding, weighted ownership, critic gradient scope, ordered supervisor
execution, negative-result retention and resume/config identity checks.
Primary GPU launch gate passed with Torch CUDA and explicit RTX5090 identity.
The persistent supervisor started2026-11-09T16:11:19Z; native attribution is
the first stage. This is a launch receipt, not a replication completion claim.
Both source champion file hashes remain unchanged. No successor policy is
promoted automatically. Full private artifacts are under
`/var/tmp/sac70-ranking-followup-20261109`; its PROGRESS/DONE/FAILED records
are the current execution authority, with per-replica shard and fit receipts.

Pre-measurement estimate for the entire queue:35-50 minutes from launch,
using measured prior shard runtimes,5:2 ownership and the additional fifth arm.
This estimate is replaced by actual progress, not treated as an execution result.
