# ETH SAC hyperparameters and DOIN ranges

The easy-to-normal causal experiment freezes the starting point below. DOIN
optimization follows only after the mechanism experiment; ranges are listed so
the next stage is explicit, not because they vary inside the paired test.

## Training and model

| Parameter | Paired-test value | Later DOIN range/choices | Stage |
|---|---:|---|---|
| Architecture | `256x256` | `128x128`, `256x256`, `512x256`, `256x256x128` | model_training |
| Learning rate | fixed `3e-4` | log-uniform `[1e-5, 1e-3]` | model_training |
| Batch size | `64` | `128`, `256`, `512` | model_training |
| Replay buffer | `200,000` | `100,000`, `200,000`, `500,000` | model_training |
| Learning starts | `128` | `2,000`, `5,000`, `10,000` | model_training |
| Gamma | `0.99` | `[0.94, 0.9995]` | model_training |
| Tau | `0.005` | log-uniform `[0.0005, 0.02]` | model_training |
| Train frequency | `1` step | `1`, `2`, `4`, `8` | model_training |
| Gradient steps | `1` | `1`, `2`, `4`, `8` | model_training |
| Entropy coefficient | `auto` | `auto`, `0.01`, `0.05`, `0.1` | model_training |
| Target update interval | `1` | fixed initially | model_training |
| Target entropy | `auto` (`-1` effective) | fixed initially | model_training |
| gSDE | enabled | fixed initially | model_training |

The later DOIN choices still carry historical batch/learning-start values that
differ from the accepted checkpoint (`64`/`128`). Before optimization, expand
those categorical sets to include the starting point; otherwise DOIN cannot
reproduce its own baseline.

## Observation

| Parameter | Starting value | Later DOIN range/choices | Stage |
|---|---:|---|---|
| Feature scaling | rolling z-score | rolling or expanding z-score (`none` repaired away) | data_observation |
| Scaling history | `256` bars | `24`, `42`, `84`, `168`, `336` | data_observation |
| Feature clipping | `10` | `3`, `5`, `10`, `20` | data_observation |
| Context/window | `32` bars = `128h` | `24`, `72`, `168`, `336`, `720h` mapped to bars | data_observation |
| Feature groups | current 83-feature set | oscillators, momentum, statistics, trend, volatility booleans; trend required | data_observation |
| Raw price window | forbidden | not a gene | contract |
| Agent state | 36 values | fixed initially | contract |

The observed 2,692-dimensional input is valid. The historical 2,724-dimensional
input with raw prices is retired.

## Execution and risk

| Parameter | Starting value | Later DOIN range/choices | Stage |
|---|---:|---|---|
| Action threshold | causal screen uses `0`; production baseline `0.1` | `[0.03, 0.4]` | execution_risk |
| Relative volume | `0.05` | `[0.01, 0.25]` | execution_risk |
| Stop loss | `2.0 ATR` | `[0.75, 5.0]` | execution_risk |
| Take profit | `3.0 ATR` | `[1.0, 8.0]` | execution_risk |
| Entry order mode | `adaptive` | adaptive, market, limit, stop | execution_risk |
| Market urgency | `0.75` | `[0.55, 0.9]` | execution_risk |
| Maximum market spread | `4 bps` | `1`, `2`, `4`, `8 bps` | execution_risk |
| Stop breakout threshold | `0.65` | `[0.35, 0.85]` | execution_risk |
| Limit ATR offset | `0.05` | `[0.01, 0.25]` | execution_risk |
| Stop ATR offset | `0.05` | `[0.01, 0.25]` | execution_risk |
| Commission | `0.0002` | fixed by realism profile | environment |
| Slippage | `0` starting profile | scenario-calibrated, not an alpha gene | environment |
| Leverage | `1.0` | risk contract, not optimized initially | environment |

## Optimization schedule after the causal test

| Stage | Generations | Population patience | Parameters |
|---|---:|---:|---|
| Data/observation | 6 | 4 | scaling, context, clipping, feature groups |
| Model/training | 8 | 5 | architecture, LR, batch, buffer, starts, gamma, tau, update ratios, entropy |
| Execution/risk | 4 | 3 | threshold, size, SL/TP and order-placement controls |
| Joint refinement | 6 | 5 | all accepted genes |

These are DOIN population-level generations and patience, distinct from SAC
epoch-level early stopping (`60`, starting at epoch `40`).
