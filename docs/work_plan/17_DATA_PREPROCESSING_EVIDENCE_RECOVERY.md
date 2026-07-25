# Data, Preprocessing, and Observation Evidence Recovery

Status: active; DOIN asset campaign frozen; E0/E1 implementation complete

Date: 2026-07-25

## Why This Phase Exists

The transition to per-asset DOIN optimization happened before several material
data and observation choices had executed comparative evidence. The repository
contained a proposed search space, but its own artifacts recorded
`training_launched=false`. In particular:

- all 7,738 inspected archived resolved configs used `window_size=32`;
- 1,296 feature-aware configs used `feature_scaling_window=256`;
- no comparable executed evidence varied observation duration, clipping, or
  representation;
- the 126-feature ADA contract was availability-pruned, not selected for
  out-of-sample target utility;
- presets named `crypto_full`, `fx_full`, and `kitchen_sink_guarded` were built
  only from `trading_asset_features`; they did not merge
  `cross_source_features`;
- CryptoQuant coverage did not span the 2021-2023 evidence period and the
  subscription is now cancelled;
- FXMacroData calendar/actual coverage begins in 2025 and therefore cannot
  support 2021-2023 event selection without another historical source.

These facts make the current ADA DOIN config a valid preserved experiment, but
not a justified final data contract.

## Immediate Runtime Decision

The current shared DOIN campaign is frozen, not deleted. Supervisors and workers
were stopped on omega, dragon, and gamma. Chain state, supervisor state, logs,
configs, and checkpoints were copied to:

```text
~/.local/state/agent-multi/retired-campaign-snapshots/20260725T1810Z/
```

Each machine snapshot contains `SHA256SUMS`. DOIN must not resume until this
evidence-recovery campaign has selected explicit per-asset/timeframe data and
observation contracts.

## Canonical Metric Contract

All return/risk values are stored as decimal fractions. A UI multiplies them by
100 only for display. Every metric row carries `unit`, `horizon`,
`aggregation`, `split`, and `metric_schema`.

Every trading status and comparative table must show:

| Metric | Unit | Horizon | Definition |
| --- | --- | --- | --- |
| `mean_weekly_return` | fraction | week | arithmetic mean of realized weekly equity returns |
| `annualized_return` | fraction | year | compound annualization from start/end equity and elapsed days |
| `mean_weekly_rap` | fraction | week | weekly return minus lambda times within-week drawdown, then averaged |
| `annual_rap` | fraction | year | `52 * mean_weekly_rap` |
| `max_drawdown` | fraction | evaluation period | maximum peak-to-trough equity loss |
| `evaluation_weeks` | count | evaluation period | number of weekly slices represented |

`optimization_score` is always labeled `dimensionless`; it is never formatted
as a weekly or annual percentage.

The implementation is `tools/project3_evidence_metrics.py`, schema
`project3.evidence.metrics.v2`.

## Parameter Registry

The authoritative registry is:

```text
examples/config/evidence_sweep/project3_parameter_registry_v1.json
```

Every material choice has:

- a canonical path;
- ownership layer;
- global, source, asset, timeframe, horizon, or model scope;
- candidate values/range;
- evidence state;
- stage where evidence will be collected.

The registry covers data sources and lags, targets, transform input signals,
feature selection, preprocessing, clipping, observation duration and
representation, learned encoders, policy architecture, RL training, risk,
execution, rewards, and evaluation cadence. An unregistered implicit default
is a defect.

## Hierarchical Experiment Order

No phase requires positive profit to continue. Selection is comparative:
negative controls and weak assets remain useful evidence.

### E0: Data contract and point-in-time coverage

- verify hashes, rows, chronological ordering, duplicates, features, and date
  coverage;
- inventory external sources separately from base feature bundles;
- exclude CryptoQuant;
- measure FXMacroData coverage rather than silently filling unavailable years;
- record publication-lag and point-in-time limitations.

### E1: Base and external source value

- evaluate every available asset/timeframe/base bundle;
- compare rank-IC and mutual-information train-only selectors;
- isolate external context bundles with a common conservative preprocessing
  and context contract;
- report validation and test in the canonical weekly/annual scale;
- use a cheap Ridge trading proxy only for ranking data contracts.

The proxy is not relabeled as SAC performance. Its protocol and score remain
separate dimensions in OLAP.

### E2: Preprocessing and observation context

Run only on robust E1 contracts and compare:

- normalization family and history duration;
- clipping disabled and thresholds 3/5/10/20;
- context durations 24/72/168/336/720/2160 hours;
- last, causal summary, sparse-lag, and later raw/multiscale sequences;
- feature budgets and selectors;
- external publication lag and staleness.

Observation duration is expressed in hours, then converted to bars per
timeframe. A fixed 32-bar window is not assumed to transfer across timeframes.

### E3: Representation and policy family

Compare PCA/autoencoder/TCN/LSTM/Transformer/event-token encoders and compatible
SAC policies on the selected E2 contracts. Transform input signals are explicit
genes; wavelet, Hilbert, multitaper, EMD, and fractional differencing are not
assumed to be best on raw close.

### E4: Weekly retraining confirmation

For one selected contract per asset/timeframe:

- train/fine-tune weekly across the complete validation year;
- report the complete test year using the same cadence;
- save champion weights, resolved config, selected feature contract, data hash,
  and all canonical metrics.

This is the artifact set consumed by portfolio optimization.

### DOIN and portfolio

DOIN Level 2 resumes per asset only after E0-E3 select its contract. Portfolio
optimization begins after E4 produces the required deployable artifact set for
the portfolio cells.

## Pool and OLAP

The new coordinator is independent of the historical 719 MB weekly OLAP and
does not mutate it:

```text
tools/project3_evidence_pool.py
tools/project3_evidence_pool_api.py
tools/project3_evidence_worker.py
tools/project3_evidence_screen.py
tools/project3_evidence_plan.py
```

Properties:

- SQLite WAL and `BEGIN IMMEDIATE` atomic claims;
- one owner per job;
- renewable leases, expired-worker recovery, and bounded retries;
- deterministic config hashes and duplicate-config rejection;
- normalized `parameter_facts`, `metric_facts`, `artifacts`, attempts, events,
  and machine heartbeats;
- OLAP views `evidence_result_olap`,
  `evidence_parameter_effect_olap`, and `evidence_machine_olap`;
- continuous worker polling so a machine claims the next eligible job without
  waiting for a status request.

The initial generated campaign has 1,890 jobs:

| Stage | Jobs |
| --- | ---: |
| `E0_DATA_CONTRACT` | 427 |
| `E0_EXTERNAL_COVERAGE` | 318 |
| `E1_BASE_SOURCE_SCREEN` | 854 |
| `E1_EXTERNAL_SOURCE_SCREEN` | 291 |

Campaign file:

```text
examples/config/evidence_sweep/project3_evidence_recovery_campaign_v1.json
```

E2-E4 are generated from upstream OLAP results, not guessed in advance.

## Selection Discipline

- Feature selection reads training rows only.
- Validation ranks contracts.
- Test is reported but cannot select features or parameters.
- No p-value, positive-profit, DSR, or presentation gate blocks exploration.
- Cross-asset conclusions require a common protocol key: dates, costs, target,
  lag policy, risk lambda, metric schema, and model/proxy protocol.
- A source with inadequate coverage is recorded as such; it is not silently
  treated as an all-zero feature.
- Historical artifacts are preserved and tagged by comparability; they are not
  deleted because the new protocol differs.

## Acceptance Criteria

This recovery phase is complete only when:

1. every registry parameter is either executed, fixed by an explicit contract,
   blocked by a named data limitation, or deferred to DOIN/portfolio with a
   recorded reason;
2. selected per-asset/timeframe contracts have complete configuration and data
   hashes;
3. E4 creates weights and resolved configs usable without reverse-engineering
   source code;
4. all required metrics are present in the same labeled scale;
5. DOIN resumes from a newly materialized contract, never from an implicit
   default.
