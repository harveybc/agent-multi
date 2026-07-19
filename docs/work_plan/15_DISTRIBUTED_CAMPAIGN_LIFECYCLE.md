# Distributed DOIN Campaign Lifecycle

Status: deterministic bootstrap hardening implemented; feature-aware fleet v5 pending redeploy
Date: 2026-07-19
Owner: `agent-multi`

## 0. Active Campaign Revision

The earlier queue/history descriptions below document the deployed lifecycle
evolution. The active next plan is now:

```text
examples/campaigns/phase_1_asset_policy_fleet_v5/campaign_plan.json
```

It contains fresh shared-v3 domains for BTCUSDT 1h, ADAUSDT 1h, EURUSD 4h and
DOGEUSDT 4h. The BTC shared-v2 domain in `fleet_v4` was stopped and frozen
after direct artifact replay proved an always-short saturated actor caused by
a stale price-only preprocessor. No v2 chain or log was removed. The exact
cause, historical SOL disposition and new action-collapse gate are recorded in
`16_FLAT_FITNESS_ROOT_CAUSE_2026_07_19.md`.

## 1. Scope

DOIN already coordinates one shared-population optimization. It owns candidate
creation and claiming, duplicate prevention, deterministic reproduction,
champion migration, controlled flooding, blockchain consensus, and evaluation.
It deliberately keeps the node HTTP process alive after a domain converges.

The campaign lifecycle supervisor owns only the boundary between complete DOIN
domains:

1. observe convergence on every required worker;
2. require one synchronized chain tip and exact component versions;
3. archive and verify the accepted champion and trained model bytes;
4. stop every local DOIN process and prove that its PID and API port are gone;
5. wait for the same durable acknowledgement from every physical host;
6. preflight the next immutable job on every host;
7. start one deterministic bootstrap worker;
8. join every remaining worker in a verified order.

It does not replace DOIN's internal shared-population protocol. During the
fleet rollout, that protocol was hardened in `doin-node` without changing its
population, migration, flooding, blockchain, or reproduction semantics; see
the claim-safety notes below.

## 2. Decentralization Model

Each physical machine runs one supervisor:

| Supervisor | Logical workers |
| --- | --- |
| Omega | `omega` |
| Dragon | `dragon` |
| Gamma | `gamma-5070ti`, `gamma-5090` |

Every supervisor has the same versioned campaign-plan JSON. The plan fixes:

- participant identities and supervisor endpoints;
- globally unique logical worker identities;
- ordered and unique job identities;
- one DOIN node config per worker and job;
- domain semantic hash and metric direction.

There is no permanent scheduler leader, mutable central queue, central SQL
database, or Omega-only campaign truth. Every participant derives the same next
ordinal from the same plan. A plan-hash mismatch blocks progression and
produces an alert.

There is one deliberately narrow bootstrap role at the start of each job. The
first worker in the immutable participant order (`omega`) is the only worker
allowed to create the generation-zero shared population. The remaining workers
join in this order:

```text
omega -> dragon -> gamma-5070ti -> gamma-5090
```

This is not a permanent central coordinator. It is a short startup mutex that
prevents simultaneous generation-zero transactions and independent equal-height
chains. After the join barrier, all four nodes resume normal decentralized
claiming, flooding, consensus and block production. If Omega is unavailable,
the strict startup barrier waits and raises an alert rather than risking a
second swarm. The plan and completed history remain replicated on every host.

The complete immutable six-cell plan is:

```text
examples/campaigns/phase_1_asset_policy_fleet_v3/campaign_plan.json
```

Jobs 0-3 preserve the deployed `fleet_v2` prefix exactly. Jobs 4-5 add
`EURUSD@4h/fx_full` and `DOGEUSDT@4h/kitchen_sink_guarded`, completing the
initial six model/timeframe components:

| Ordinal | Component | Operational state at plan extension |
| ---: | --- | --- |
| 0 | `SOLUSDT@4h` | completed and archived |
| 1 | `SOLUSDT@1h` | running |
| 2 | `BTCUSDT@1h` | queued |
| 3 | `ADAUSDT@1h` | queued |
| 4 | `EURUSD@4h` | queued |
| 5 | `DOGEUSDT@4h` | queued |

Per-host profiles are adjacent to the plan. Runtime state and champion bytes
remain under the existing `fleet_v2` state directory so the append-only plan
migration preserves worker adoption, history and the current job. The migration
utility refuses changed prefixes, participant changes, active supervisor locks,
non-running phases and mismatched plan hashes:

```text
examples/scripts/migrate_doin_campaign_plan.py
```

The supervisor API and dashboard expose the complete ordered queue, including
completed, running and queued jobs, so continuity can be audited before the
current optimization reaches its stop barrier.

Gamma does not set `CUDA_VISIBLE_DEVICES`: PyTorch 2.13.0 enumerates the external
5090 as `cuda:0` and the internal 5070 Ti as `cuda:1`, the opposite of the
physical indices displayed by `nvidia-smi`. Its versioned runtime overlays use
the PyTorch order consumed by Stable Baselines3. Masking either device would
renumber it and make the overlay's explicit ordinal invalid.

Dragon and Gamma currently report `linger=no`, which requires root privileges
to correct. Until the operator runs `sudo loginctl enable-linger harveybc` on
both, Omega's lingering user manager holds one enabled, persistent systemd user
SSH session to each host. Both units use `Restart=always` and survive an Omega
user-manager restart. This bridge prevents `logind` from stopping the remote
enabled user services when an administrative SSH command ends; it is an
explicit temporary availability dependency, not a scheduler or source of
campaign truth. The installed units are versioned under `examples/systemd/`.

## 3. Lifecycle State Machine

```text
starting
   -> preflight(all supervisors)
   -> bootstrap(omega)
   -> join(dragon)
   -> join(gamma-5070ti)
   -> join(gamma-5090)
   -> running
   -> archiving
   -> stopping
   -> stopped
   -> starting(next ordinal)
   -> complete(no remaining ordinal)
```

Any incompatible config, exhausted restart budget, corrupt model artifact, or
unverifiable stop transitions to `blocked`. A blocked state is visible through
the API/dashboard and never silently skips a campaign.

### 3.1 Startup contract and ordered join

Before any worker starts, every supervisor independently computes and publishes
one content-addressed startup contract containing:

- plan hash, ordinal, job ID and domain ID;
- domain semantic hash;
- explicit shared-population seed;
- population size;
- input dataset SHA-256;
- exact component revision map;
- bootstrap worker and global worker join order.

All required supervisors must be online, on the same ordinal, and report the
same contract hash. `require_deterministic_seed=true` is mandatory.
Shared-population jobs cannot use per-node seed offsets.

`doin-node` now uses the configured `shared_population_seed`, falling back to
`ga_seed`, for generation zero. The seed is embedded in the population state.
The population receives a canonical SHA-256 fingerprint and its transaction ID
is derived from `(domain, generation, fingerprint)`.

The bootstrap worker must expose:

- the canonical genesis block;
- a generation-zero population block;
- the configured seed;
- the expected population size and fingerprint;
- the exact component revisions.

Each follower is launched only after every predecessor reports `join_ready`.
The follower must then prove the same genesis hash, generation-zero population
block hash and population fingerprint. A mismatch waits through a bounded sync
grace period and restarts only that worker. Repeated failure trips the circuit
breaker and blocks the campaign instead of starting independent work.

### 3.2 Convergence barrier

Advancement requires all expected logical workers to report:

- the same plan hash and job ID;
- `converged=true` for the planned domain;
- a non-empty chain height and tip hash;
- exactly the same chain height;
- exactly the same component-version map.

The normal case also requires the same tip. DOIN can produce an equal-height
terminal fork when two generators seal the last unfinalized block almost
simultaneously; its existing fork choice only reorganizes toward a strictly
longer chain. In that bounded case the barrier accepts one identical finalized
`(height, hash)` anchor. It still cannot stop until every participant decodes
its local chain and proves the exact same content-addressed champion model.
A different model, component revision, chain height or finalized anchor fails
closed. The condition must remain stable for the configured interval.

### 3.3 Champion archive

Before stopping a worker, every supervisor reads its local synchronized chain,
selects the best full `optimae_accepted` record for the active domain, and:

1. decodes `parameters._model_b64`;
2. verifies its declared byte count and SHA-256;
3. writes a content-addressed `.zip`, `.keras`, `.pt`, or `.bin` file;
4. writes `champion_manifest.json` with plan, domain, block, transaction,
   peer, fitness, parameters, metrics, versions, and artifact lineage;
5. records compact campaign evidence in local SQLite.

This retains the exact trained model that produced the accepted statistics.
It does not retrain from the same hyperparameters.

### 3.4 Stop and advancement barrier

The supervisor sends `SIGTERM`, waits, escalates to `SIGKILL` only after the
configured timeout, reaps local child processes, and verifies both process
identity and API-port closure. The next job cannot start while any local worker
has an unverified stop.

All hosts then exchange stopped/archive acknowledgements. An early participant
may advance seconds before another participant reads its transient `stopped`
state, so campaign completion is also exposed as a durable SQLite-backed
acknowledgement. This prevents a distributed barrier deadlock without electing
a permanent leader. The durable acknowledgement path now includes every
already-advanced participant in artifact, height, tip and finalized-anchor
consistency checks.

### 3.5 Shared-candidate lease and duplicate prevention

The original shared pool used a fixed 600-second claim timeout and expired a
claim after four other results. The trading policy candidates take roughly 18
minutes, so valid work could be reassigned while its GPU was still training.
The claim API also returned `409` for a collision, but the caller proceeded to
evaluate the rejected candidate. This created duplicate evaluations during a
simultaneous four-worker start.

`doin-node` now retains the historical `600/4` defaults but exposes global
`shared_claim_timeout` and `shared_claim_result_patience` settings. The long
SOL/BTC/ADA jobs use `3600/20`. Simultaneous claims are arbitrated by the
lexicographically smallest persistent `peer_id`; a losing worker confirms the
loss through peer APIs and controlled flooding before GPU evaluation, releases
its local lease, and pulls another free candidate. Candidate status includes
the owner so every replica and the dashboard can audit the assignment.

Polling a peer is replication, not a lease heartbeat. A live incident on
2026-07-16 showed that assigning `time.time()` when importing a peer-observed
claim allowed four expired leases to circulate indefinitely: each replica
renewed the lease immediately after another replica expired it. `doin-node`
now preserves the peer's original `claimed_at` and
`results_since_claim` evidence. Only an explicit claim from the owner renews a
lease, so stale work converges to free on every replica instead of being
resurrected by polling.

The clean deployed start was accepted only after all four APIs returned the
same pool-state hash and exactly four leases:

| Worker | Generation 0 candidate |
| --- | ---: |
| Omega RTX 4070 | 1 |
| Dragon RTX 4090 | 2 |
| Gamma RTX 5070 Ti | 3 |
| Gamma RTX 5090 | 4 |

There were zero ownerless or duplicate leases. The deterministic arbitration
commit is `f060f81`; the lease-replication fix is `6de2bc4`.

## 4. Crash and Restart Semantics

- A filesystem lock permits one supervisor process per host/profile.
- State updates use atomic JSON replacement.
- Worker PIDs include Linux process-start ticks, preventing PID-reuse mistakes.
- A restarted supervisor adopts a matching live `doin_node.cli` process.
- DOIN workers use independent process groups and survive a supervisor crash.
- A missing pre-convergence worker restarts the same job, bounded by a circuit
  breaker; it never advances to another ordinal.
- A worker joining a different bootstrap lineage is stopped and retried without
  touching healthy workers.
- A healthy worker that remains claimless while free candidates exist is
  restarted only after a conservative grace period.
- Runtime monitoring compares bootstrap lineage, component versions, finalized
  anchors and shared-population generation across all workers.
- A process that has become a zombie is reaped and cannot block a stop barrier.
- Runtime history uses SQLite WAL and remains readable while events are added.

The systemd user service uses `Restart=always` and `KillMode=process` so a
supervisor restart reconciles existing GPU work instead of killing and
duplicating it. The service is installed by:

```text
examples/scripts/install_doin_campaign_supervisor.py
```

## 5. Monitoring and History

Every supervisor serves:

| Endpoint | Purpose |
| --- | --- |
| `/dashboard` | Consolidated participants, alerts, current campaign and history |
| `/api/status` | Local immutable-plan and worker lifecycle evidence |
| `/api/network` | Best-effort consolidated view assembled on this node |
| `/api/history` | Completed campaign/champion registry |
| `/health` | Supervisor liveness |

The standard port is `8795`. The dashboard is not a central source of truth;
the same view can be opened from Omega, Dragon, or Gamma. Peer timeouts cannot
block the page and are rendered as high-visibility swarm alerts.

The SQLite `campaigns` table preserves final fitness, finding peer, chain tip,
artifact hash/path, public parameters, metric vector and full evidence. The
`worker_events` table records starts, adoption, convergence barrier, archive,
verified stop and campaign completion. The dashboard also exposes the startup
contract hash, bootstrap worker, join readiness, candidate ownership, genesis
and population block hashes, finalized anchor and free shared candidates.

## 6. Current Queue and Scientific Meaning

Ordinal 0 adopts, archives and closes the completed
`SOLUSDT@4h` shared-v2 single-fit campaign. Its accepted champion is retained,
but the subsequent protected 48-week walk-forward result did not pass:

- compounded 48-week return: `+0.2365605%`;
- full-path maximum drawdown: `3.2999784%`;
- 48-week RAP: `-3.0634179%`;
- coverage: `48/48` weeks.

Fleet rollout archived the exact accepted policy independently on every host:

- champion fitness: `0.06704331595778694`;
- artifact format: `stable_baselines3_zip`;
- bytes: `3,741,513`;
- SHA-256: `cf17e4bc74abec11a16cf8a7235e66fe84cf652516adcc206024642289ff19b1`.

The observed chain had an equal-height terminal fork after finalized block 45.
All replicas proved finalized hash
`72ebc57fb6d0316ce3a283e630aa8473c5059127ce10fba41e4f1860410c37df`
and the exact artifact hash above before the four old workers stopped.

Ordinal 1 is the next Stage A screening campaign:

```text
SOLUSDT@1h / tech_stat_decomp / SAC
rv=0.05 / SL=2 ATR / TP=4 ATR / risk lambda=0.5
```

It starts from the block-41 SOL 4h champion hyperparameters to avoid throwing
away useful search evidence. Its accepted artifact, fully resolved JSON and
chain lineage may enter the first portfolio-research optimization after swarm
closeout. Weekly retraining remains mandatory before release/live deployment,
but is no longer a prerequisite for the portfolio-mechanics experiment.

Ordinals 2 and 3 continue the same four-worker swarm without waiting for a
human status request:

| Ordinal | Cell | Input | Seed geometry | Train / validation |
|---:|---|---|---|---|
| 2 | `BTCUSDT@1h` | `kitchen_sink_guarded` | `rv=.05`, `SL=2`, `TP=3`, lambda `1.0` | 2019-2021 / 2022 |
| 3 | `ADAUSDT@1h` | `kitchen_sink_guarded` | `rv=.05`, `SL=2`, `TP=3`, lambda `1.0` | 2021 / 2022 |

These are the other two evidence-backed short-horizon seeds in the Project 3
selection table. Their immutable manifests record source path, full CSV SHA,
row/column counts, timestamp coverage and untouched 2023 test split.

Ordinals 4 and 5 complete the first balanced six-cell portfolio input library:

| Horizon | Cells |
|---|---|
| Short | `SOLUSDT@1h`, `BTCUSDT@1h`, `ADAUSDT@1h` |
| Long | `SOLUSDT@4h`, `EURUSD@4h`, `DOGEUSDT@4h` |

### 6.1 Execution-order decision, 2026-07-17

The approved pragmatic order is:

1. run all six static train/validation swarm optimizations without repetition;
2. archive the exact accepted model bytes, resolved parameters, config hashes,
   source block and metric evidence for every cell;
3. optimize and exercise portfolio mechanics from that six-cell library;
4. add weekly retraining/walk-forward evaluation to measure adaptation and
   finalize the deployment policy after the portfolio path works end to end.

The campaign supervisor must move directly from one immutable plan job to the
next through the distributed stop/start barriers. Every status and dashboard
view reports current-candidate ETA per worker, current-job ETA and whole-pool
ETA. ETA uses measured per-worker candidate durations and the remaining planned
candidate budgets; stage patience may shorten the observed completion time.

## 7. Verification Evidence

Focused unit coverage rejects:

- duplicate supervisor ownership;
- missing or duplicated worker mappings;
- semantic config drift hidden behind machine overlays;
- offline required peers;
- mismatched chain tips or component versions;
- unverified process stops;
- inconsistent champion artifacts;
- corrupt or incorrectly sized embedded model bytes;
- lost state across supervisor restart;
- transient stop-barrier acknowledgement races.

The integration test launches three real supervisor processes and four fake
DOIN workers, executes two campaigns, verifies exactly one launch per
`(job, worker)` in the mandatory bootstrap order, archives identical
content-addressed models on all hosts, stops all workers, and completes all
replicated histories.

```text
python -m pytest -q tests/unit tests/integration/test_campaign_supervisor_swarm.py
190 passed
```

Coverage includes process adoption from relative `screen` command paths,
deterministic job materialization, terminal-fork safety and the distributed
two-job lifecycle.

The deployed SOL 1h successor was observed on Omega, Dragon, Gamma 5070 Ti and
Gamma 5090 with one domain and one identical shared-pool view, matching
component commits, active GPU/CPU load and no post-restart CUDA or optimizer
errors. Unfinalized blockchain tips may temporarily differ while concurrent
champion blocks are flooded; the completion barrier still requires the safe
finalized-anchor and exact-artifact conditions described above. The supervisor
dashboard is available on every host at port `8795`; Omega's local view is
`http://127.0.0.1:8795/dashboard`.

The `doin-node` focused lease/arbitration tests pass, including preservation of
remote lease age and expiry after repeated observation. The broader related
suite reports `80` focused tests with only the two unrelated pre-existing VUW
assertions red because an expected zero weight currently observes `0.5`. The
full suite reports `353 passed` and those same two VUW failures plus one
independent pre-existing GossipSub full-mesh assertion.

The 2026-07-16 generation-5 recovery exposed two coordination races that the
previous lease rules did not cover. A worker entering a new generation could
reserve the pool while lagging peers rejected its claims, and two cold-starting
workers could begin the same candidate before their claims converged. The
deployed correction in `doin-node@63b3cac` adds:

- one unresolved lease per owner and distributed release propagation;
- a configurable full-membership barrier before shared optimization;
- a per-generation barrier over domain, generation and immutable population
  fingerprint;
- two stable all-peer ownership confirmation rounds before GPU evaluation;
- logical-peer deduplication, route identity resolution and responder identity
  verification;
- full-mesh Tailscale bootstrap routes for this four-worker campaign.

Live acceptance evidence used chain height `19` and common tip
`d6131c28e79cba349afec1b40ce0e9fe5d1a9f8aea61e489f7341aee2cf8b53d`.
All four workers reported the exact component revisions and the same generation
5 claim map: Gamma 5070 Ti candidate `0`, Gamma 5090 candidate `1`, Dragon
candidate `2`, and Omega candidate `3`. The supervisor APIs reported every
worker `running`, owning exactly one candidate, with no active alerts; live GPU
telemetry confirmed compute on all four devices.

## 8. Remaining Scientific Work

The lifecycle problem and the weekly-retrained fitness problem are separate.
The six static artifacts are now intentionally consumed by the portfolio
research phase before weekly retraining. They still cannot be released as live
weekly-adaptive policies until the chronology below is implemented and passed.

Before a component is promoted, the next metric increment must define a
computationally tractable nested chronology:

1. inner train/validation for L1 early stopping;
2. outer validation weeks for L2 candidate fitness;
3. untouched protected test weeks used once after optimization;
4. staged week subsets for early DEAP generations and complete 48-week
   validation for finalists;
5. one releasable artifact recipe plus the latest weekly trained weights.

That increment must be approved and tested independently of campaign
transition mechanics.
