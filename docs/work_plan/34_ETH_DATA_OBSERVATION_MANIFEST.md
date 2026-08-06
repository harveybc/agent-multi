# 34. Exact ETH Data and Observation Manifest (finding 134 / WP6)

Status: active — v1.0.0, 2026-08-06
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

## 3. Compute contract of the decision packet

| Arm | Steps per seed |
|---|---|
| N14 | 14 × 20,000 = 280,000 |
| EN4_10 | (4 + 10) × 20,000 = 280,000 |
| E4 (diagnostic) | 4 × 20,000 = 80,000 |
| Four seeds total | 2.56 M environment steps |

Repeating epochs re-traverses the same 13,699 training bars; it never
creates new historical observations.
