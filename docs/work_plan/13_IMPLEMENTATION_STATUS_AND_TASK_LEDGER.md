# 13. Implementation Status and Task Ledger

Status timestamp: 2026-07-30 23:46 America/Bogota
Plan version: 1.23.1
Current focus: run the protected-entry v2 optimization path to an exact
champion handoff, collect independent read-only Alpaca/IBKR execution evidence,
commission OANDA MT5 when valid demo credentials are available, and close
independently reproduced audit findings without taking compute from DOIN

Immediate parallel runtime added on 2026-07-30:

- Moltbook public collection is active every 30 minutes with deterministic
  hashing, deduplication, injection screening and social OLAP;
- Hermes performs low-cost triage every two hours and a supervised Telegram
  review every six hours. Publishing remains disabled pending a local
  credential and explicit human approval;
- a synthetic USD 100,000 multi-venue shadow portfolio marks nine
  crypto/equity/duration/metal/FX cells every five minutes with zero order
  routes;
- IBKR observations now include normalized delayed/frozen quotes when TWS has
  upstream connectivity;
- a Capital.com Demo GET-only adapter, OLAP, mutation-disabled broker plugin
  and activation scripts are implemented; activation awaits owner-created
  demo credentials.

Runtime acceptance evidence:

- the collector completed with 120 unique posts in social OLAP;
- a real `deepseek-v4-flash` triage consumed the sanitized packet and retained
  URL/hash provenance without tools or side effects;
- a real `deepseek-v4-pro` review produced a 2,459-character bounded report
  and delivered it successfully to the configured Telegram group;
- the multi-venue shadow marked all nine cells with `available_weight=1.0`,
  zero missing/stale cells and `orders_submitted=0`;
- full suites pass: `379` Agent Multi tests and `211` LTS tests.

Account and observer state verified on 2026-07-30:

- Alpaca Trading API Paper is authenticated in read-only mode and observed
  every five minutes. The initial crypto universe is available; positions,
  orders and submitted orders are zero.
- IBKR Individual Margin Paper is authenticated through TWS Paper port `7497`
  after the API disclaimer was accepted. The read-only observer and
  reconciliation run every five minutes; 222 completed sessions were present
  at the verification point with zero positions and zero orders.
- OANDA Global Markets live application remains under compliance review.
- Windows 11 and MT5 are installed in the Dragon VM. A valid OANDA Global
  Markets demo server/session is still blocked on the account/support path, so
  no authenticated MT5 heartbeat is claimed.

Architecture correction 2026-07-29: OANDA Global Markets is not a REST-v20
division. The existing Practice REST client remains preserved for compatible
accounts, but it cannot be activated against this account. LTS will own one
global portfolio and route to OANDA MT5, Alpaca or IBKR through separate
capability-checked adapters.

Decision 2026-07-29, updated 2026-07-30: broker knowledge is measured in
parallel with model optimization. Alpaca and IBKR are active read-only
observers with restart-safe SQLite OLAP, account-identity redaction and
deterministic Telegram monitoring. Watchdog health now requires recent
authenticated observer/reconciliation facts; a reachable IBKR TCP port is
diagnostic evidence only. No protected canary or broker order is enabled.

Decision 2026-07-29: the v1 USDCAD swarm is invalid for champion selection.
One annual validation trade was sufficient to pass its gate, allowing a
near-flat checkpoint to displace an active 98-trade checkpoint by roughly
`2e-6` fitness. All four workers and the old replicated supervisors were
stopped; 60 completed candidates, blockchain state and models were archived.
Four in-progress candidates were intentionally discarded.

The replacement contract is implemented and locally verified:

- train-tail minimum: one completed trade;
- annual validation minimum: 12 completed trades;
- no positive-profit requirement;
- market, limit and stop entries are atomic brackets with mandatory SL and TP;
- protected-plugin failures reject the entry instead of falling back;
- `easy_floor` uses positive commission, spread and slippage;
- the genome includes order family and adaptive-routing thresholds/offsets;
- every split exports mean weekly return, annual return, mean weekly RAP,
  annual RAP, observed weeks and both explicit annualization methods;
- the v2 campaign archives champions after easy and curriculum jobs.

Local preflight evidence: 21 train and 12 validation completed trades, all
submitted as protected limit brackets; zero default orders, plugin failures,
or protection rejections. Focused `agent-multi` tests pass (`84 passed`) and
the complete `gym-fx` suite passes (`73 passed`).

Current protected-v2 operational snapshot at 2026-07-30 22:20 COT:

- plan `phase-1-protected-execution-fleet-v2`, plan hash
  `b43844a7ebd7c85a782c557a8c3459622e1cb353a5d33391816e85f107cb6b21`;
- active job `usdcad-4h-protected-easy-sac-shared-v2`, domain
  `trading-asset-policy-usdcad-4h-protected-easy-v2`, stage 1/4
  `data_observation`, generation 2;
- 56 of 480 planned campaign candidates complete, 16/20 in the current
  generation evaluated and four distinct candidates claimed;
- best L2 fitness `0.00048223070314018903`; exact SB3 artifact SHA-256
  `892fbee0f00ecf7b93ca6ed2bea05ca4db3cd09aa557a8cd16b9698fa7e09842`;
- all four workers share plan, domain, seed, genesis, current generation,
  current population fingerprint, component revisions and finalized anchor;
- measured fleet throughput is approximately 1.73 candidates/hour. The
  remaining full budget is roughly 10-14 days at that rate, subject to L2
  early stopping;
- decision point: at completion of stage 1, compare measured throughput,
  candidate diversity, activity and fitness progression. Continue
  automatically when invariants remain healthy; stop only for an evidence,
  safety or lineage failure, not a profit gate;
- active warning: Dragon currently reports a different unfinalized tip at the
  same height while all workers retain the same finalized anchor and shared
  population. Monitor it as an unfinalized fork warning; do not mutate chain
  state without evidence of a confirmed parallel lineage.

Decision 2026-07-28: retain the useful `data_observation` and `model_training`
search, then stop as the synchronized swarm enters stage 3 before spending
meaningful candidates on zero-cost execution/risk refinement. The agreed champion and five
strong/diverse artifacts become the warm-start source for a new domain with a
visible three-phase cost curriculum and immutable multi-scenario robust
weekly-RAP fitness. Exact sequencing is in
`19_EXECUTION_CURRICULUM_AND_ORDER_ROUTING.md`.

Implementation is deployed on omega, dragon and gamma as a lifecycle-only,
append-safe successor plan:

- `trading-contracts` 0.2.0 adds optional asset-intent urgency;
- `gym-fx` 0.3.0 supports visible cost conditions and native market, limit and
  stop entries with GTD expiry and cancel/market fallback;
- `agent-multi` 0.4.0 adds deterministic three-phase cost selection, robust
  scenario evaluation in weekly-return units, SAC observation expansion, and
  the adaptive router;
- the tracked USDCAD follow-up template remains disabled as source;
- the launch materializer requires decoded parameters, hashes the policy and
  parameter artifacts, and generates one semantic DOIN domain for all workers;
- the router profile is blocked until the robust policy is frozen.

The active worker configs, fitness, seed, domain, genesis and candidate pool
remain unchanged. Supervisor metadata adds only the semantic stage-3
completion boundary, stable artifact handoff, and queued curriculum job.
Migration preserves active worker PIDs and recomputes the replicated lifecycle
contract.

Deployment evidence on 2026-07-28:

- plan hash:
  `f06b0f75c2783d298b23c6f0436d3dae318d862166cc56827e65a1d230103921`;
- replicated lifecycle contract:
  `52875e9f163ecd076fedfef450765de6ba917ab503f4156b93821237434c1109`;
- all three hosts run supervisor code revision `2fef5080`;
- worker PIDs preserved across migration: omega `2466249`, dragon `1664737`,
  gamma-5070ti `1808707`, gamma-5090 `1808951`;
- shared generation, pool fingerprint and champion artifact remained
  identical after migration;
- current pre-boundary budget is at most 280 candidates; the generated
  curriculum schedule is 320 candidates before early stopping.

Current operational snapshot at 2026-07-28 21:45 COT:

- all four workers run `usdcad-4h-full-genome-sac-shared-v1` with matching
  plan, job, domain, seed, dataset, population fingerprint, component versions
  and champion artifact;
- live component revisions are `agent-multi@8935b4b`,
  `doin-core@8573a87`, `doin-node@d7bf671`, `doin-plugins@f5fedf8`,
  `gym-fx@5630734` and `trading-contracts@4675c8f`; the work-plan publication
  commit is newer documentation and is not falsely reported as deployed code;
- active dataset SHA-256 is
  `f2fa13f4ab9df7cb6577e9785d0e5952362c554e24a2e28c79dffdc8b698818b`;
- stage `data_observation`, generation 2: 56/280 pre-boundary candidates
  complete and four distinct candidates claimed;
- measured swarm throughput is 1.688 candidates/hour;
- all GPU temperatures are below the 78 C alert threshold;
- one warning remains for different finalized blockchain anchors. It is not
  accepted as harmless by wording alone: supervisors continue to require
  matching canonical genesis/population evidence, candidate uniqueness and
  champion artifact while monitoring for a true parallel lineage.

Historical evidence-recovery closeout, retained for lineage:

- the transactional E0-E4 pool completed 16,019 jobs and was stopped/disabled
  after its OLAP, configs and load-tested artifacts were archived;
- the earlier BTC recovery and follower-shutdown incidents remain documented
  in sections 2.9, 2.10, 5 and document 15;
- those closed pools are not the current runtime and must not be reported as
  active.

Decision 2026-07-28: order type is not a separate alpha model. After the alpha
handoff, execution uses a deterministic router control, calibrated fill-time,
adverse-selection, path and event-hazard auxiliaries, then one shared learned
encoder with entry and exit heads. Detailed contracts and data-fidelity gates
are in documents 03, 04, 06 and 19.

## 1. Phase Summary

| Phase | Status | Evidence |
| --- | --- | --- |
| Phase 0: contracts and evidence | Implemented and published | Contracts, schemas, metric catalog, shortlist and compatibility gate implemented; `trading-contracts` commit `4675c8f` published |
| Phase 1: evidence recovery | E0-E4 complete and archived | 16,019 jobs, transactional lineage, canonical OLAP facts, validation-only selection and 24 load-tested E4 baseline artifacts |
| Phase 2: heuristic lifecycle extraction | Verified locally | Pure policy, source substitution, packaging and frozen Backtrader requested-action replay pass |
| Phase 3 | Engine selected; vertical slice verified | NautilusTrader 1.230.0 multi-asset replay, costs, margin preflight, rollover, canonical reports and Gym bridge pass; portfolio-native Gym expansion remains |
| Phase 4: full-genome alpha | v1 invalid/archived; protected v2 running on four workers | Mixed data/feature/preprocessing/context/model/training/execution genome, split activity eligibility, protected brackets and exact SB3 artifacts |
| Phase 5: execution curriculum | Protected v2 successor queued fail-closed | Positive-cost curriculum, robust validation, evolvable market/limit/stop/adaptive routing and two-stage artifact handoff; materialization awaits the exact job-0 champion |
| Phase 6 and later | Planned behind frozen cell library | Static portfolio first, then probabilistic rush/event activation and weekly retraining |

### 1.1 Phase 1 closeout sequence

The Phase 1 asset-policy campaign does not authorize live deployment from an
accepted DOIN block by itself. Its closeout sequence is fixed and auditable:

1. reconcile accepted chain evidence into a small validation-only Pareto set,
   deduplicating equivalent re-acceptances caused by the permissive research
   threshold;
2. freeze each candidate's typed parameters, config hash, model artifact hash,
   source block and validation metric vector in a promotion-input manifest;
3. archive every selected static asset/timeframe artifact and expose its
   deterministic action stream to execution and portfolio research;
4. optimize and verify portfolio mechanics without retraining an asset model
   inside every portfolio candidate;
5. run the separate weekly-retrained walk-forward evaluation and retain only
   policies whose complete evidence passes coverage, reproducibility,
   compatibility and release-validation gates before live deployment.

Decision recorded 2026-07-17: weekly retraining follows the first end-to-end
portfolio experiment. It does not block completion of the six static swarm
jobs or use of their exact artifacts in portfolio research.

The initial SOLUSDT 4h result is promising validation evidence, not a promoted
model. Its 2023 protected test remains unopened until step 3.

The first compact reconciliation snapshot is
[`solusdt_4h_promotion_candidates.json`](../../examples/results/phase_1_asset_policy/solusdt_4h_promotion_candidates.json).
At chain height 17 it retains three validation-only, non-dominated recipes:

| Rank | Source block | L2 fitness | Validation return | Validation RAP | Max drawdown | Trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 0.056462 | 27.30% | 22.94% | 4.36% | 575 |
| 2 | 15 | 0.055669 | 23.82% | 20.03% | 3.79% | 300 |
| 3 | 5 | 0.031493 | 13.84% | 10.80% | 3.04% | 116 |

The manifest records validation evidence and artifacts only. It carries the
explicit blocker `WEEKLY_RETRAINED_PROTECTED_TEST_NOT_RUN`.

A bounded local optimization and independent inference verification were run on
Omega before the persistent campaign. On 2026-07-13 Dragon and the two Gamma
GPU islands joined the same live domain after environment, dataset, device,
revision, dashboard and blockchain compatibility checks passed.

## 2. Verified Local Deliverables

### 2.1 `trading-contracts` 0.1.0 foundation

Local repository:

```text
/home/harveybc/Documents/GitHub/trading-contracts
```

Implemented:

- strict Pydantic v2 contract base with unknown-field rejection;
- timezone-aware persisted object semantics;
- canonical asset and cell identity;
- market snapshot and variable-length context token contracts;
- prediction, decision context, asset intent and portfolio intent contracts;
- order intent and execution report contracts;
- component and deployment manifests;
- canonical sorted JSON and SHA-256 content hashes;
- canonical `trading_experiment.v1` configuration;
- typed `candidate_genome_patch.v1` with JSON Pointer keys;
- metric catalog contract and versioned initial catalog;
- JSON Schema exporter and generated schemas;
- representative experiment and candidate-patch fixtures.

Verification:

```text
python -m pytest -q
24 passed
```

The package is installed locally in editable mode for integration testing and
published at `https://github.com/harveybc/trading-contracts`, commit `4675c8f`.

### 2.7 Local-first DOIN trading boundary

Implemented locally, not yet deployed to any machine:

- `agent-multi/app/metrics.py` provides an explicit metric seam while
  preserving the historical agent fitness default;
- `agent-multi/optimizer_plugins/default_optimizer.py` accepts a typed seed
  candidate, receives migrated candidates between generations, and emits
  DOIN-compatible champion/stage/candidate callbacks;
- `doin-plugins.trading.TradingOptimizer` wraps that local optimizer through the
  established external entry-point pattern;
- `doin-plugins.trading.TradingInferencer` performs inference-only scalar
  verification and does not claim to be the rich LTS action-serving API;
- `doin-plugins.trading.TradingScenarioSyntheticData` provides a deterministic
  `fixture_v1` scenario with hash evidence. Learned or heuristic delegated
  scenario backends remain a later promotion gate;
- `gym-fx/metrics_plugins/trading_metrics.py` is the actual simulator metrics
  plugin. It extends `default_metrics` with unit-safe drawdown/RAP fields and
  only annualizes when the elapsed evaluation period is explicitly supplied;
- entry points are `trading_asset` for optimization/evaluation and
  `trading_scenario` for synthetic verification.
- `examples/config/doin/trading_asset_solusdt_4h_sac_v1.json` remains the first
  portable vertical-slice regression seed. The actual incremental campaign is
  now under `examples/config/phase_1_asset_policy/`, with matching `data`,
  `results` and `scripts` directories modeled on predictor's phase layout.
- the local optimizer now executes declared stage-specific bounds, freezes
  inactive parameters, carries its champion between stages, persists atomic
  resume/statistics/parameter files, appends candidate history and preserves
  the exact optimizer checkpoint separately from the final retrain;
- Phase 1 uses independent L1 and L2 semantics: L1 monitors risk-adjusted
  train-tail/validation behavior; L2 maximizes the gap-penalized
  `train_validation_l1_score`;
- candidate optimization sets `evaluate_test_split=false`; a focused firewall
  test proves that `_final_eval` never opens the protected test path.

Contract verification before the vertical smoke:

```text
doin-plugins/tests/test_trading_plugins.py: 8 passed
gym-fx trading metrics + engine tests: 5 passed
agent-multi targeted optimizer/metric tests: 4 passed
doin-core full suite: 278 passed
doin-node metric/config/OLAP suite: 63 passed
agent-multi --load_config ... --agent_plugin random_agent --total_timesteps 0: exit 0
```

The bounded SOLUSDT 4h smoke used real Project 3 input, two candidates, one
generation and 64 training steps. The CLI wrote the resolved config, manifest,
result, final policy and optimizer output. The exact optimizer champion was
194,095 bytes with SHA-256
`47601eee7a6ae7340006f40cc80afaf144ccfe488211355229ed20b5f38c24ee`;
`TradingInferencer` decoded that base64 artifact and reproduced fitness `0.0`.

A second invocation through the external `TradingOptimizer` contract returned
fitness `0.004052591711230477`, 49 evidence fields and a 194,096-byte model;
the decoded SHA-256 exactly matched `model_artifact_sha256`. This run exposed
and fixed a real callback-copy defect: bound DOIN callbacks contain locks and
must be shallow-copied as process-local runtime objects rather than deep-copied
as declarative configuration.

Both values are wiring evidence only. The first policy made no trades and was
diagnosed as `policy_hold_collapse`; neither smoke value is a scientific model
result, annual result, shortlist update or promotion candidate.

On 2026-07-12 the predictor-style Phase 1 smoke executed locally on Omega with
the real hash-verified SOLUSDT 4h source. It evaluated two candidates, preserved
an exact 194,031-byte optimizer champion, wrote all resume/history/statistics
artifacts, and obtained L2 wiring fitness `0.0060425180751335905`. Validation
return was `0.02396380163676559` and validation risk-adjusted return was
`0.006159046907203151`; the protected test summary is explicitly marked
`evaluation_skipped`. Because the smoke caps each environment at 384 rows and
trains for one 64-step epoch, these values are execution evidence only.

The isolated Omega DOIN smoke used two real candidates and completed one
accepted champion block. During candidate evaluation, `/api/candidate` exposed
candidate number, stage, progress and parameters while fitness remained null;
after evaluation it exposed fitness `0.0060425180751335905`. The dashboard was
HTTP 200 at `http://127.0.0.1:8470/dashboard`, the exact 194,031-byte champion
artifact had SHA-256
`f72671fd62d1a6bbc645bb111bd0fb832a2a5648d7bb321a7e813d630c07096d`,
and protected test evidence remained `evaluation_skipped`.

The first attempted node smoke found two real integration defects: callbacks
containing thread/async locks were reaching a deep-copy boundary, and an
all-failed generation could publish the numeric failure sentinel as champion.
Callbacks are now removed from declarative pipeline configuration before model
construction, and a generation whose candidates all carry `evaluation_error`
fails closed without publishing a champion. A first-champion configuration
lookup in `doin-node` was also corrected to read the domain role's
`optimization_config` rather than treating the node dataclass as a dictionary.

This is wiring evidence, not a scientific result. The persistent Omega Phase 1
campaign was the next runtime gate. Dragon, `gamma-5070ti`, and `gamma-5090`
were admitted only after Omega was visibly training. Omega was later restarted
once under a controlled resume check to load the final component-version code;
its chain, resume state and champion were preserved.

The persistent Omega campaign passed that gate on 2026-07-12: the transient
`doin-phase1-omega.service` became healthy on port `8470`, and its dashboard
showed candidate `1/20` in stage `action_behavior` before fitness existed. It
subsequently advanced to candidate `2/20` with one completed evaluation while
the RTX 4070 remained active and host swap remained unused. The service has
`Restart=on-failure`, is not installed for boot-time activation, and remains
running while later node configurations are reviewed.

The first live dashboard review then exposed a publication-granularity defect:
candidate fitness was calculated and persisted, but `on_new_champion` was held
until all 20 candidates in the generation completed. The optimizer now compares
and publishes each successful candidate immediately, including its exact model
artifact and compact generic metric evidence. Omega reproduced the recovered
best seed and accepted block `#1` after candidate 1 with L2 fitness
`0.030345322807962153`; validation return was `0.1337667688568993`, validation
RAP `0.10337026583851741`, max drawdown fraction `0.03039650301838189`, and
validation trade count `117`. The protected test remained disabled.

The DOIN dashboard now recognizes the trading metric contract instead of
labeling it as predictor MAE. Champion and candidate views expose train-tail
RAP, validation RAP, mean score, gap penalty, return, drawdown and trades. Its
monitoring APIs redact embedded model bytes and epoch/split traces, reducing the
live optimization response to about 1.5 KB and the 30-block chain response to
about 2.8 KB while leaving the canonical chain payload unchanged.

During review, a latent island-diversity defect was fixed: the declared
`node_seed_offset` now changes only the local GA seed instead of being silently
ignored.

### 2.8 Historical four-island Phase 1 deployment baseline

The earlier persistent campaign expanded on 2026-07-13 without resetting
Omega's blockchain or optimizer resume state. Its four islands were:

| Island | Compute device | Dashboard |
| --- | --- | --- |
| `omega` | RTX 4070 Laptop | `http://127.0.0.1:8470/dashboard` |
| `dragon` | RTX 4090 Laptop | `http://100.110.215.85:8470/dashboard` |
| `gamma-5070ti` | RTX 5070 Ti Laptop | `http://192.168.0.109:8470/dashboard` |
| `gamma-5090` | RTX 5090 eGPU | `http://100.107.204.49:8471/dashboard` |

All nodes use Python 3.12 from the clean `trading-stack` environment. The
sorted installed-package inventory has SHA-256
`ac6825091b3eca8d72a9181638e3cec662d85ee3cde4a689207d95411d718145`
on all three machines, and `pip check` reports no broken requirements.
`predictor` and `heuristic_strategy` were removed from Omega's new environment
because neither belongs to the canonical lock. Their repositories and old
environments were not deleted.

The exact component revisions in that 2026-07-13 deployment were:

```text
agent-multi       deployment revision recorded by that release
doin-core         8573a87
doin-node         84f371b
doin-plugins      0f23702
gym-fx            20b667b
trading-contracts 4675c8f
```

The dashboard no longer reports a `predictor` repository revision. Peer
handshakes compare all six active component revisions, and all four alert APIs
reported zero compatibility alerts after joining. Headless Chromium confirmed
the same six-revision footer on every rendered dashboard.

The runtime dataset is identical on every machine, with SHA-256
`9a9df280331cec812370e079aae49682a259f8e2da16a045fd2383bff042575c`.
Remote `financial-data` paths are now clean Git checkouts at commit
`538d7d26c94211d8364ca6953f5b99040124223f`, materialized from a complete Git
bundle because the repository is private. The previous 133-179 GB remote trees
remain preserved under
`/home/harveybc/Documents/GitHub/_pre_trading_stack_20260713/financial-data`;
no research data was deleted.

The shared chain had height 7 and tip block 6 when the islands joined. Every
remote node synchronized the current champion from that block before starting
its own candidate. Its reported L2 fitness is computed from the inner L1
evidence as follows:

```text
RAP(split) = total_return(split) - 1.0 * max_drawdown_fraction(split)
L2 fitness = 0.5 * (RAP(train_tail) + RAP(validation))
             - 0.25 * abs(RAP(train_tail) - RAP(validation))
```

For the inherited champion, train-tail RAP is `0.0035364078221380483`,
validation RAP is `0.14666270871929155`, their mean is
`0.0750995582707148`, the generalization-gap penalty is
`0.035781575224288376`, and final fitness is
`0.03931798304642642`. Validation total return is `0.186652148410007`,
validation drawdown is `0.039989439690715445`, and the protected test split is
not evaluated or used by this campaign.

Omega runs as the non-boot-enabled transient user service
`doin-phase1-omega.service`. Dragon and Gamma have `Linger=no`; their transient
user services were correctly terminated when SSH sessions ended, so the three
remote nodes run in detached named `screen` sessions instead. This preserves
the explicit no-boot-autostart requirement while keeping current work alive
across SSH disconnects. Gamma retained about 4.2 GiB available RAM after both
nodes began training and must remain under memory observation.

### 2.9 Controlled-flooding resource gate

The first four-island join exposed a real network defect before acceptance:
forwarding metadata changed the deduplication identity of an otherwise
identical logical message, locally originated messages were not inserted into
the seen cache, and dual LAN/Tailscale routes could broadcast twice to the same
logical peer. Dragon reached its 1,024-file-descriptor process limit after
about nine minutes, with roughly 980 established sockets. All islands were
stopped before changing code; optimizer chains and artifacts were preserved.

`doin-node` commit `abb2971` fixes the failure at three boundaries:

- deduplication hashes exclude transport-only metadata;
- local broadcasts are marked seen and use one endpoint per logical peer;
- the shared `aiohttp` client uses explicit global and per-host connection
  limits with bounded keepalive.

The focused network and unified-node suite passes 42 tests with the two known
stale VUW assertions explicitly deselected. The four-island live soak exceeded
the prior nine-minute failure interval. Maximum observed descriptors were 90
on Omega, 96 on Dragon, 92 on Gamma 5070 Ti and 73 on Gamma 5090; counts later
fell, proving that sockets were being reused and closed rather than accumulated.
No `Too many open files` record, process restart or compatibility alert occurred,
candidate progress remained fresh, and all dashboards rendered identical
six-component revisions.

### 2.10 Decentralized consolidated dashboard

`doin-node` commits `309a64a`, `0164275`, `fd03023` and `c365e56` add a network monitor to every
participant rather than introducing a central supervisor. Node JSONs declare the labels
`omega`, `dragon`, `gamma-5070ti` and `gamma-5090`. Each node exposes a compact
local monitor payload and a cached network overview assembled through the
existing bounded transport client.

The overview first groups alternate LAN/Tailscale routes by discovery identity,
then deduplicates fetched snapshots again by the real peer identity returned by
`/api/monitor`. This prevents provisional route identities from counting one
machine more than once. It tries fallback routes, preserves known unavailable
peers as offline, compares the exact six active revisions, and aggregates recent
alerts with their source node. The dashboard's initial `Network` tab shows participant health, chain
range, active candidate progress, best fitness, alert counts, revision status
and icon links to individual dashboards. A peer timeout cannot block the rest
of the view. Each node advertises every local interface as an `IP:port` alias;
an unresolved offline route is suppressed only when an online peer explicitly
claims the same alias. This avoids false participants when different machines
reach Omega through different LANs while retaining genuine offline nodes.
Focused dashboard/config/transport coverage passes 37 tests; the complete
suite passes 326 tests with only the three documented historical
assertions deselected.

The consolidated chain range exposed one additional routing defect during the
four-island acceptance run: Dragon received block `11` through Omega over
Tailscale but selected the original author's unreachable Gamma LAN route for
the download. `doin-node` commit `bc36999` makes block synchronization prefer
the peer that physically forwarded the announcement. Other directed protocol
responses retain author-first routing. The complete suite then passed 327
tests with the same three historical assertions deselected.

`doin-node` commit `84f371b` extends the same decentralized view with durable
optimization provenance. The `Network` table reports both evaluations completed
by the current process (`run evals`) and `candidate_evaluated` transactions
committed to blockchain (`on-chain`). These counters are intentionally separate:
the first is freshest but resets with the process, while the second survives
restarts but advances only when a block commits pending transactions.

The `Champions` tab and fitness plot now include only true running-best
improvements. Each record distinguishes `transaction.peer_id`, the optimizer
island that found the champion, from `block.generator_id`, the island that
assembled its block. The table also exposes fitness delta, validation RAP,
train-tail RAP, return and drawdown from immutable metric evidence. The complete
suite passes 328 tests with the same three historical assertions deselected.

The first live overview also exposed that Omega knew Gamma 5090 only through
its firewalled LAN port. Omega's canonical bootstrap now includes stable
Tailscale endpoints for Dragon and both Gamma ports, so route fallback is
available immediately after restart instead of depending on incidental PEX.

The dashboard now also makes metric period and runtime capacity explicit. The
Phase 1 summary card says `Best L2 fitness` and visibly identifies it as a
train-tail plus validation composite that is not annualized. Each network row
shows the island's recent median `Cand/h` and planned active-stage ETA; the
summary shows aggregate online throughput. ETA is derived from durable
completion timestamps, excludes clear pause/reboot gaps, exposes a recent
range in its tooltip, and remains explicitly subject to earlier L2 stopping.

### 2.2 Canonical configuration in `agent-multi`

Implemented:

- `--config` alias for `--load_config`;
- optional base profile and DOIN candidate patch;
- precedence: defaults, base, file, CLI, candidate patch;
- canonical nested config validation;
- exact `base_config_hash` enforcement before candidate application;
- JSON Pointer patching only for existing paths;
- deterministic canonical hash and resolution manifest;
- atomic canonical config and manifest output;
- compatibility flattening for current plugins;
- preservation of every key in legacy flat configs;
- embedded-secret rejection;
- explicit namespacing for generic subtrees such as rush, OLAP and portfolio.

Compatibility evidence:

- the real oracle behavior smoke config preserved all 100 source keys with no
  missing or changed runtime values;
- `agent-multi` accepted `--config`, applied a CLI override and wrote canonical
  lineage in an isolated test;
- a canonical nested smoke config resolved to the intended agent, lifecycle,
  optimization, rush and OLAP settings.

Verification:

```text
python -m pytest -q tests/unit
152 passed
```

The suite also exposed and fixed an evidence-only precision defect in the event
token transformer's normalization manifest. Forward computation remains
`float32`; original normalization evidence is retained in `float64`.

### 2.5 Machine-local runtime overlays and code lineage

Implemented:

- separate `trading_runtime_overlay.v1` contract and hash;
- recursive resolution of declared `${NAME_ROOT}` placeholders only;
- fail-closed behavior for missing runtime roots;
- immutable canonical experiment config while runtime paths are materialized;
- independent device, resource, root and environment-reference overlays;
- Git snapshots with repository root, commit, branch, dirty state, status hash,
  bounded status sample and tracked diff hash;
- expected-commit comparison when the canonical config declares one;
- plugin defaults cannot overwrite a loaded config value.

Versioned overlays:

```text
configs/runtime/omega.json
configs/runtime/dragon.json
configs/runtime/gamma_5070ti.json
configs/runtime/gamma_5090.json
```

Gamma's two profiles use separate artifact/model/cache roots. PyTorch 2.13.0
enumerates the RTX 5090 as `cuda:0` and the RTX 5070 Ti Laptop as `cuda:1`, even
though `nvidia-smi` lists the physical adapters in the opposite order. The
profiles and their regression test use the PyTorch order because Stable
Baselines3 executes through PyTorch. Runtime preflight must continue to verify
both framework and physical GPU enumeration before launching work.

### 2.6 Component and deployment compatibility gate

Implemented in `trading-contracts`:

- major contract-family compatibility checks;
- required component-role checks;
- component map key versus manifest-role checks;
- component training cutoff versus deployment cutoff firewall;
- content-hash binding between deployment and component manifests;
- producer-output and consumer-input contract-edge checks;
- execution-contract compatibility;
- fail-closed issues for malformed contract versions;
- deterministic example builder for component, portfolio intent, order intent
  and deployment manifest fixtures.

The gate returns a structured issue report and does not load models, touch
brokers or mutate manifests.

### 2.7 Pure heuristic lifecycle policy increment

Implemented in `heuristic-strategy`:

- dependency-free-from-Backtrader policy module for prediction entry/exit;
- typed and validated policy parameters;
- pure forecast-path entry geometry;
- exact A-G prediction early-close variants;
- legacy reward/risk order-size interpolation and cash cap;
- `DecisionContext -> AssetIntent` policy with no broker units;
- typed optimizable parameter schema;
- Backtrader long/short adapter delegates entry, exit and sizing math to the
  same pure functions;
- frozen JSON decision fixture and adapter delegation regression.
- normalized ideal/mapped, CSV, direct-model, and provider callback adapters;
- source-independent `PredictionBundle` materialization;
- frozen Backtrader requested-action replay derived from the base commit.

Focused verification:

```text
python -m pytest -q \
  tests/unit_tests/test_prediction_entry_exit_policy.py \
  tests/unit_tests/test_prediction_entry_exit_backtrader_replay.py \
  tests/unit_tests/test_prediction_source_substitution.py
17 passed
```

Wheel inspection confirms that the pure policy modules and
`trade_lifecycle_policy.plugins/prediction_entry_exit_v1` entry point are
packaged. Importing `app.policies` does not load Backtrader. Acceptance details:

```text
docs/handoffs/CODEX_HEURISTIC_LIFECYCLE_ACCEPTANCE_2026_07_11.md
```

The historical full suite is not a valid green gate at the starting commit. It
fails during collection on removed functions/modules and on a nested
`timeseries-gan/tests` package collision. The existing prediction-client test
also has five failures because it calls a removed generic `get_prediction()`
API instead of the current entry/exit methods. These baseline failures were
observed before Phase 2 edits and are not hidden or counted as passing.

### 2.3 Metric catalog v1

Source:

```text
trading-contracts/examples/metric_catalog_v1.json
```

The initial catalog includes explicit definitions for:

- coverage weeks;
- mean weekly return;
- compounded annual return;
- mean weekly RAP;
- additive annual RAP;
- mean weekly and full-path drawdown;
- weekly CVaR20;
- logical trade count;
- portfolio turnover;
- L1 gap-penalized score.

Each metric declares unit, period, aggregation, direction, formula,
denominator, coverage rule and failure semantics. Partial evidence cannot carry
an annual label.

### 2.4 Project 3 shortlist import

Generator:

```text
agent-multi/tools/import_project3_shortlist.py
```

Generated manifest:

```text
agent-multi/configs/manifests/project3_doin_shortlist_2026_07_10.json
```

Imported evidence:

- 14 candidates total;
- 5 active short-horizon seeds;
- 6 near-full-year long-horizon seeds;
- 3 partial rush/opportunity seeds;
- exact source file hashes and `financial-data` commit;
- canonical asset/cell identities;
- exact coverage and evidence-scope labels.

The manifest declares these rows as research seeds only. Imported test evidence
cannot select, early-stop, optimize or promote future candidates.

## 3. Delegation Ledger

| Task ID | Owner | Repository | Status | Scope | Acceptance owner |
| --- | --- | --- | --- | --- | --- |
| `CONTRACTS-001` | Codex | `trading-contracts` | verified_local | Contract/schema foundation | Codex |
| `CONFIG-001` | Codex | `agent-multi` | verified_local | Canonical config and legacy translation | Codex |
| `METRICS-001` | Codex | `trading-contracts` | verified_local | Metric catalog v1 | Codex |
| `SHORTLIST-001` | Codex | `agent-multi` | verified_local | Hash-verified Project 3 import | Codex |
| `RUNTIME-001` | Codex | `agent-multi`, `trading-contracts` | verified_local | Machine overlays, path resolution and Git lineage | Codex |
| `COMPAT-001` | Codex | `trading-contracts` | verified_local | Component/deployment contract and hash gate | Codex |
| `HEURISTIC-001` | Codex | `heuristic-strategy` | verified_local | Pure prediction entry/exit lifecycle extraction | Codex |
| `DOIN-CONFIG-001` | Claude + Codex review corrections | `doin-node` | verified_local | Complete unified-node config materialization | Codex |
| `SIM-ENGINE-001` | Codex | `gym-fx`, `agent-multi` | verified_local | Nautilus engine bake-off, cost profiles and canonical execution reports | Codex |
| `SIM-GYM-001` | Codex | `gym-fx`, `agent-multi` | verified_single_cell | JSON-selectable Nautilus Gym compatibility bridge | Codex |
| `DOIN-TRADING-001` | Codex | `agent-multi`, `doin-plugins`, `doin-core`, `doin-node` | verified_four_island | Local optimizer, exact champion artifact, generic metric evidence, external adapter, independent inference and four-island live optimization | Codex |
| `DOIN-MONITOR-001` | Codex | `doin-node`, `doin-plugins` | verified_four_island | Decentralized health, candidate, exact metric basis, throughput, stage ETA, revision and alert dashboard | Codex |
| `ARTIFACT-P2P-001` | Codex | DOIN stack | designed_not_implemented | Content-addressed descriptor, trackerless transfer and multi-peer replication gate | Codex |
| `DOIN-CAMPAIGN-001` | Codex | `agent-multi`, `doin-node` | verified_four_worker_swarm | Replicated six-cell campaign plan, append-only live migration, deterministic bootstrap, ordered joins, seed/data/config/version contract, lineage watchdogs, scoped repair, convergence/stop barriers, crash adoption, champion archive, decentralized history, non-resurrecting leases, strict generation membership barrier, stable claim confirmation and responder identity verification | Codex |
| `GPU-WATCHDOG-001` | Codex | `agent-multi` | verified_three_hosts | Five-minute GPU temperature, GPU-count and NVIDIA telemetry watchdog with Hermes Telegram alert, recovery and hourly repeat semantics | Codex |
| `SWARM-TELEGRAM-001` | Codex | `agent-multi` | verified_three_hosts | Idempotent completion metrics and failover-owned health notifications for frozen machines, unhealthy workers, divergent lineages, parallel swarms and stalled progress | Codex |
| `INPUT-CONTRACT-001` | Codex | `agent-multi`, `gym-fx`, `doin-node` | verified_local_pending_redeploy | Feature-aware observation contract, exact observation-space wiring, neutral causal warm-up, action-collapse rejection, fresh v3 node configs and flat-fitness incident evidence | Codex |
| `EVIDENCE-RECOVERY-001` | Codex | `agent-multi`, `financial-data` | e0_e4_complete | Canonical weekly/annual metric schema, exhaustive parameter registry, explicit external-source bundles, 16,019 completed E0-E4 jobs, transactional pull pool, normalized OLAP facts, bounded-memory execution and 24 load-tested E4 baseline artifacts | Codex |
| `FULL-GENOME-001` | Codex | `agent-multi`, `doin-node`, `doin-plugins` | verified_local_campaign_materialized | Typed mixed feature/preprocessing/context/model/training/execution genome, corrected full-fidelity L1 protocol, exact champion persistence, local smoke, identical four-worker population fingerprint, immutable dataset preflight and first sequential DOIN campaign | Codex |
| `EXEC-CURRICULUM-001` | Codex | `agent-multi`, `gym-fx` | verified_local_disabled_followup | Visible deterministic cost curriculum, immutable robust validation scenarios, weekly-RAP fitness, SAC observation expansion and fail-closed champion materializer | Codex |
| `ORDER-ROUTER-001` | Codex | `agent-multi`, `gym-fx`, `trading-contracts` | verified_local_blocked_on_policy | Account-independent adaptive market/limit/stop router, urgency contract, native GTD execution, expiry and cancel/market fallback | Codex |
| `EXEC-STATE-001` | Codex | `agent-multi`, `financial-data`, `gym-fx` | specified_blocked_on_alpha_and_data_gate | Point-in-time execution-fidelity manifest plus calibrated fill-time, adverse-selection, short-path and event-hazard auxiliaries | Codex |
| `EXEC-POLICY-001` | Codex | `agent-multi`, `gym-fx`, `trading-contracts` | specified_blocked_on_exec_state | Shared causal encoder with entry/exit heads, deterministic risk overrides and market-only/router control comparison | Codex |
| `ACTIVITY-GATE-002` | Codex | `agent-multi` | deployed_four_worker_running | Split-specific activity eligibility at L1 and optimizer boundaries; annual validation minimum 12 without profit gate | Codex |
| `PROTECTED-ORDERS-002` | Codex | `gym-fx`, `agent-multi` | deployed_four_worker_running | Mandatory SL/TP market/limit/stop brackets, adaptive routing genes, fail-closed plugin errors and risk-reducing reversal handling | Codex |
| `WEEKLY-METRICS-002` | Codex | `agent-multi` | deployed_four_worker_running | Equity-trace mean weekly/annual return and RAP for base and curriculum pipelines with explicit units/methods | Codex |
| `OANDA-PRACTICE-001` | Codex | `lts`, `financial-data`, `agent-multi` | verified_local_blocked_for_ogm | REST-v20 Practice capability/quote/transaction observer; retained for compatible account divisions, not OANDA Global Markets | Codex |
| `MULTI-VENUE-PAPER-001` | Codex | `lts`, `agent-multi` | alpaca_ibkr_observers_active_mt5_oanda_blocked | LTS global ledger, authenticated Alpaca/IBKR read-only observers, OANDA MT5 EA bridge, capability snapshots, protected canaries and consolidated OLAP | Codex |
| `SOCIAL-INTEL-001` | Codex | `agent-multi`, future narrow social adapter | specified_not_activated | Hermes-routed low-cost research, social OLAP, Telegram review, bounded Moltbook publishing and DOIN domain discovery | Codex |
| `CONTINUITY-001` | Codex + designated human maintainers | deployment repositories | specified_not_implemented | Reproducible VPS services, encrypted backups, revocation, restore drills and least-privilege human recovery | Codex |
| `CONTINUOUS-AUDIT-001` | Claude audit agent + Codex verification | all active repositories, `agent-multi/docs/audits` | baseline_reviewed_corrections_verified | Read-mostly cross-front audit, stable findings, change-driven cadence, Hermes boundary and versioned Codex role recovery | Codex |
| `AUDIT-SNAPSHOT-COLLECTOR-001` | Codex | `agent-multi` | verified_live_three_hosts | Six-hour deterministic redacted packet with 11-repository provenance, campaign lineage/ETA, three-host/four-GPU telemetry, broker/watchdog evidence, section hashes/deltas and 28-snapshot retention | Codex |

Claude packet:

```text
docs/handoffs/CLAUDE_DOIN_NODE_CONFIG_MATERIALIZATION_TASK_2026_07_10.md
```

Claude's implementation was independently reviewed against `R01` through
`R15`. Codex corrected nested mutable-config leakage and silent unknown
`ResourceLimits` fields, then reproduced focused and full-suite evidence. The
review record is:

```text
docs/handoffs/CODEX_REVIEW_DOIN_NODE_CONFIG_MATERIALIZATION_2026_07_11.md
```

## 4. Immediate Next Tasks

1. Keep Alpaca and IBKR read-only observers and their functional-freshness
   watchdog active until each has at least 24 hours of continuous evidence. Do
   not enable orders before capability review.
2. Continue the active four-worker `USDCAD@4h` easy non-zero-cost full-genome
   job. At the end of stage 1, record the declared duration decision point and
   continue automatically unless evidence, safety or lineage invariants fail.
3. On job-0 convergence, archive and independently load its exact champion
   model, decoded JSON, metric evidence and hashes before crossing the
   replicated stop barrier.
4. Let the replicated campaign materialize and start the separate protected
   easy-to-nominal-to-stress curriculum domain from that exact handoff.
5. Complete the OANDA Global Markets demo/support path, then authenticate MT5,
   compile the tracked EA with zero errors and require a fresh signed heartbeat
   before claiming the bridge active.
6. Freeze the resulting robust alpha policy and generate its deterministic
   action trace.
7. Audit quote/L1/L2 and point-in-time calendar-vintage coverage; materialize
   only the execution claims supported by the available fidelity.
8. Train/calibrate fill-time, adverse-selection, short-path and event-hazard
   auxiliaries, then optimize the deterministic router with Nautilus.
9. Train the shared entry/exit policy and require improvement over market-only
   and deterministic-router controls under identical alpha streams.
10. Materialize the E2-favored market/macro bundles before queuing BTC-perp,
   GBPJPY, NZDUSD and USDJPY; never publish placeholder source genes.
11. Execute one coordinated DOIN campaign per selected asset, using every
   available worker on one chain at a time.
12. Confirm frozen winners across three seeds and build the per-asset champion
   library before portfolio optimization.
13. Expand the verified Nautilus single-cell Gym bridge into the portfolio-native
   multi-asset observation/action contract without creating account state
   outside Nautilus.
14. Optimize the static portfolio, then add calibrated multi-horizon rush/event
   activation and evaluate weekly retraining/fine-tuning.
15. Implement and fault-test the decentralized artifact plane before a
   multi-node trading-domain acceptance run.
16. Complete the consolidated LTS capital ledger and the remaining MT5 adapter
    commissioning, then review the 24-hour read-only observations before
    protected canaries.
17. Implement social-intelligence S0/S1 only after MT5 commissioning: source
    allowlist, deterministic collection, normalized OLAP and daily Telegram
    review with publishing disabled.
18. Benchmark local social models in a declared resource window before
    installing them fleet-wide; never steal unmeasured GPU capacity from DOIN.
19. Provision the continuity VPS from code, restore encrypted state in a clean
    drill and verify credential revocation before treating it as operational.
20. Run the next change-triggered audit from the deterministic snapshot
    contract, independently reproduce material findings, and keep both role
    recovery prompts current as architecture and runtime contracts change.

## 5. Current Risks

- The content-addressed P2P artifact plane is designed but not implemented;
  the first vertical retains the proven base64 champion-in-parameters path.
- The `agent-multi` repository contains a large generated `experiments/`
  directory; only reviewed source/config/documentation files may be committed.
- Current canonical sections intentionally allow plugin-specific dictionaries;
  role-specific typed submodels will be added only after the first vertical
  slice freezes their real fields.
- `doin-node` still has three independently confirmed baseline test failures:
  one GossipSub mesh-capacity failure and two stale VUW zero-weight assertions.
- Controlled flooding is now connection-bounded after the first four-island
  run exposed an FD leak; the live descriptor/socket soak remains a mandatory
  deployment gate for future topology changes.
- Two historical `doin-plugins` network modules target the retired separate
  evaluator service. They remain in the repository but are explicitly skipped
  when that legacy package is absent; active network coverage belongs to the
  unified `doin-node` suite.
- `heuristic-strategy` has pre-existing stale tests and an embedded repository
  test-discovery collision. The Phase 2 policy gate is independently green,
  while repository-wide historical test cleanup remains explicit debt.
- NautilusTrader 1.230.0 does not natively pre-deny insufficient-margin orders
  in its stable Python margin-account risk path. The mandatory adapter preflight
  calls Nautilus's own margin calculator; this guard must remain tested until an
  upstream replacement passes the same fixture.
- The Nautilus replay adapter is multi-asset, but the interactive Gym bridge is
  currently the verified single-cell compatibility slice. It is not yet the
  complete portfolio training environment.

## 6. Evidence Archive Closure (2026-07-27)

E0-E4 services were stopped and disabled after completion. The authoritative
SQLite OLAP remains at:

```text
~/.local/state/agent-multi/evidence-pool-20260725/evidence.sqlite
```

A fresh online backup passed `PRAGMA quick_check`, has SHA-256
`73c46d56c24de8d85e75f7dd621718a5ac6ff4eb0250a3a100eb9acced7887ca`,
and is retained with the two preceding verified snapshots. Reducing 48 hourly
duplicates to three final snapshots reduced that state directory from 32 GB to
4 GB without deleting the source database, normalized OLAP facts, E4 artifacts
or attempt history. Future evidence-service installations default to eight
rolling snapshots.
