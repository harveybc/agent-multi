# 34. Exact ETH Data and Observation Manifest (finding 134 / WP6)

Status: active — v1.1.0, 2026-08-08
Owner: General Satoshi III; verifier: General Musashi
Registry status of every value here: **currently used, NOT
evidence-selected** unless a row says otherwise. Window/scaling/lookback
values remain open genes of the no-default registry until comparative
evidence freezes or excludes them.

## 1. Dataset (immutable)

| Fact | Exact value |
|---|---|
| File | `predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv` |
| SHA-256 | `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f` |
| Total rows | 18,085 |
| Train | 13,699 bars — 2017-09-28 04:00 → 2023-12-31 (≈6.25 years) |
| Validation | 2,196 bars — 2024 (366 days) |
| Disclosed test | 2,190 bars — 2025; **disabled** in N14/EN4_10/E4 |
| Bar interval | 4 hours |
| Manifest sidecar | `…model_ready.manifest.json` (predictor@14a1077) |

The dormant `train_years=4` contradiction is REMOVED from materialized
configs as of agent-multi materializer v2 (this order): explicit dates
govern, and `training.split_contract_note` records that no year-count
shorthand exists in the contract.

### 1.1 Nested decision amendment (document 38)

N14/EN4_10/E4 retain the historical split above. The replacement
decision-bearing curriculum/feature-selection program uses:

| Role | Range | Rows |
|---|---|---:|
| `fit_train` | 2017-09-28 04:00 through 2022-12-31 | 11,509 |
| `train_monitor` | calendar 2022, inside `fit_train` | materialized exactly |
| `inner_validation` | calendar 2023 | 2,190 |
| `outer_validation` | calendar 2024 | 2,196 |
| `sealed_test` | calendar 2025 | 2,190 |

Each score split receives a causal context prefix at least as long as the
largest observation/scaling/feature lookback. Prefix rows initialize state but
cannot trade, mutate account state or enter metrics. Consequently the complete
declared calendar interval is scored; warm-up is no longer deducted from the
evidence year.

## 2. Observation contract (currently used)

| Component | Value | Registry status |
|---|---|---|
| Input features | 83 causal technical/statistical columns | frozen list (hash in dataset manifest); membership optimizable later via feature-group genes |
| Observation window | 32 bars = 128 h = 5 d 8 h | **open gene** — used, not selected |
| Rolling scaling window | 256 bars = 1,024 h = 42 d 16 h | **open gene** — used, not selected |
| Price window | 32 values | tied to window gene |
| Return window | 32 values | tied to window gene |
| Agent state | 4 values | fixed by env contract |
| Flattened observation | **2,724** = 83×32 + 32 + 32 + 4 | derived |
| Scaling | rolling z-score, trailing 256 bars, causal fit | mode = evidence-fixed (observation-contract guard); window open |
| Warm-up | first 256 bars of any stream lack full scaling context; effective evaluable rows = rows − warm-up | derived |

Causal fit boundary: every feature and scaling statistic at time `t` is
computed from bars ≤ `t`; the two-source parity experiment (2026-08-05)
showed the offline `TechStatFeatureEngine` reproduces all 83 columns
from raw OHLCV to CSV serialization precision (~1e-8).

Document 38 now makes the feature-selection sequence executable: FS0 keeps the
83-column control, FS1 evolves an inherited hierarchical sparse mask at L2, and
FS2 learns a sparse feature gate inside a separate SAC L1 plugin. The first
curriculum comparison keeps FS0 fixed for attribution. No current M1 artifact
may be described as having online feature selection.

## 3. Historical fixed-epoch compute contract

| Arm | Steps per seed |
|---|---|
| N14 | 14 × 20,000 = 280,000 |
| EN4_10 | (4 + 10) × 20,000 = 280,000 |
| E4 (diagnostic) | 4 × 20,000 = 80,000 |
| Four seeds total | 2.56 M environment steps |

Repeating epochs re-traverses the same 13,699 training bars; it never
creates new historical observations.

These values describe the preserved N14/EN4_10/E4 calibration only. They are
not the compute contract of document 38's decision run. The replacement uses a
2,000-epoch safety ceiling, a 40-checkpoint minimum floor, patience 60, derived
pass-equivalent timesteps and equal maximum interaction caps; actual stopping
is controlled by paired train-monitor/inner-validation evidence.
