# 09. Testing, Security, and Operations

## 1. Verification Layers

### 1.1 Unit tests

- config merge, validation, canonicalization and hashes;
- schemas and version rejection;
- timestamps, calendars, staleness and masks;
- P&L, cost, conversion, margin and protection calculations;
- cell aggregation/netting and attribution;
- sizing, caps and portfolio constraints;
- rush labels/calibration metrics;
- metric formulas and annual aggregation;
- artifact/manifest signatures and compatibility.

### 1.2 Property and metamorphic tests

- zero exposure creates no trading P&L before account fees;
- linear notional scaling produces linear P&L/costs while unconstrained;
- asset/cell order permutation cannot alter results;
- unavailable assets cannot fill;
- future input mutation cannot alter earlier decisions;
- one-cell portfolio matches single-asset behavior;
- same hashes/seed produce identical deterministic replay;
- tighter hard risk limits cannot increase permitted exposure;
- invalid/stale signal cannot create a larger position;
- net instrument target equals sum of virtual cell targets.

### 1.3 Integration tests

- `financial-data` manifest -> `gym-fx` observation;
- `PredictionBundle` -> heuristic policy -> `AssetIntent`;
- `agent-multi` -> artifact bundle;
- DOIN candidate -> reconstructed evaluation;
- DOIN synthetic seed -> same data hash on evaluators;
- provider -> LTS signal/order plan;
- LTS simulation -> OANDA practice parity.

### 1.4 System/acceptance tests

- full validation-year weekly walk-forward;
- multi-node champion migration and chain sync;
- continuous optimization plus weekly release boundary;
- provider channel switch and rollback;
- multi-user LTS portfolios with different risk profiles;
- OANDA practice weekly operation and reconciliation.

## 2. Leakage Controls

- Every feature has observation and availability timestamps.
- Every weekly run persists train cutoff and target week.
- Fit preprocessors/encoders/calibrators on train only.
- Selection and DOIN fitness use validation only.
- Test metrics are access-controlled and excluded from candidate callbacks,
  migration, OLAP ranking views and allocation.
- Rush labels are targets only; causal features stop at decision time.
- Oracle/anti-oracle are diagnostic or training-label sources only and are
  evaluated causally afterward.
- CI includes a future-row mutation test and explicit column provenance audit.

## 3. Determinism and DOIN Verification

- Canonical config/data/code/artifact hashes precede evaluation.
- Seeds are explicit and DOIN verification seed rules remain active.
- Synthetic generators load pre-trained state and generate deterministically.
- Framework nondeterminism is declared; evaluator tolerance is justified from
  measured variation, not convenience.
- Hardware-dependent differences are recorded.
- Evaluators reject incompatible contract/metric versions.

## 4. Failure Matrix

Test:

- missing/gapped/stale data;
- market closure and weekend boundary;
- provider timeout, corrupt artifact and partial asset response;
- optimizer/evaluator crash and restart;
- OOM, disk full and database lock;
- network partition, duplicate/flooded message and chain resync;
- DOIN code/version mismatch;
- insufficient margin and precision rejection;
- OANDA disconnect, rate limit, duplicate submission and partial fill;
- LTS/provider restart during open positions;
- release activation failure and rollback;
- model output NaN/out-of-range or concentration violation.

## 5. Security Boundaries

- Tokens and account IDs are secret-store/environment references.
- No credentials in config, logs, OLAP, chain, checkpoints or screenshots.
- LTS is the only real-order authority.
- Practice/live environments and credentials are separate.
- Artifact and deployment manifests are signed/hash-verified.
- Provider and LTS authenticate requests and authorize model channels.
- Client identity, balance and exact positions are not public/on-chain usage
  telemetry.
- DOIN resource and parameter bounds remain active for untrusted candidates.
- Candidate code execution is constrained to approved plugin/code versions.

## 6. Resource Management

Each worker declares:

- GPU/device ID;
- VRAM/RAM limits;
- maximum runtime, epochs, batch size and artifact size;
- output/data roots;
- concurrency;
- heartbeat and stale threshold.

Gamma's 5070 Ti and 5090 run as explicit independent workers only after a
two-GPU smoke test validates framework placement, host RAM, thermals and output
isolation. A larger GPU does not justify overlapping host-memory workloads that
cause OOM thrash.

## 7. Process Ownership

- DOIN nodes/workers are launched through documented service/config files.
- No hidden Project 3 cron, shell startup or stale supervisor jobs.
- Every service has one owner, PID/state path, logs, resource limit and stop
  command.
- Supervisor failover uses Git, node config overlays, chain/OLAP restore and
  artifact manifests.
- Startup never resumes real trading without reconciliation and explicit live
  enablement.

## 8. Monitoring and Status Contract

Every operational status reports:

- timestamp/timezone;
- machine reachability, GPU(s), worker and fresh heartbeat;
- explicit large alert for offline/stale/idle expected machines;
- domain, stage, generation, candidate and progress;
- pending/running/completed/failed work;
- ETA and estimation basis;
- memory, swap, disk and GPU memory/usage;
- chain height, peers, sync and current champion;
- candidate coverage and provenance;
- mean weekly return, annual return, mean weekly RAP, annual RAP;
- drawdown/downside tail, activity, costs and concentration;
- best full-year candidate per asset/timeframe and best stack;
- exact partial-week count for partial results;
- release/channel/provider/LTS state when deployed.

An unchanged deferred queue can be intentional; status must explain promotion
rules and confirm workers have active eligible work.

## 9. Early Stale and OOM Detection

Do not wait many hours to suspect a worker.

- Mark heartbeat stale after a short configured multiple of expected update
  interval.
- Cross-check actual PID/process, GPU activity and output progress.
- Distinguish long candidate evaluation from deadlock using progress/events.
- Alert on sustained swap growth, memory pressure and OOM-kill journal events.
- Stop/restart only the offending service first.
- Prevent auto-restart loops from repeatedly exhausting memory.
- Preserve candidate state/idempotency before retry.

Omega additionally contains the DOIN service with `MemoryHigh=20G`,
`MemoryMax=24G` and `MemorySwapMax=6G`. A two-minute memory-pressure watchdog
alerts through Hermes Telegram and tracks cgroup `oom_kill` counters. Kernel
oops recovery, AMD hardware-watchdog probing and patch-kernel installation are
documented in `docs/environment/MEMORY_AND_KERNEL_RECOVERY.md`.

## 10. Release Gates

Automatic adaptive-practice release requires:

- complete required validation coverage;
- metric/constraint pass;
- reproducibility and evaluator agreement;
- stack compatibility;
- no leakage/anomaly flags;
- artifact/provider preload health;
- rollback manifest;
- simulation regression pass.

Stable/live release additionally requires human approval and practice evidence.

## 11. Incident Records

Each incident captures:

- detection time and symptom;
- affected services/candidates/orders;
- exact root cause;
- smallest mitigation;
- verification of recovery;
- data/order integrity assessment;
- permanent prevention/test;
- whether metrics/artifacts were invalidated.

## 12. Acceptance Criteria

- CI covers unit, property, integration and contract tests.
- Leakage mutation tests fail when future information is introduced.
- Multi-node deterministic verification succeeds.
- OOM/stale detection alerts before long idle periods.
- Process restart does not duplicate candidates or orders.
- Live safety gates fail closed.
- Status always reports period/unit/coverage for financial metrics.
