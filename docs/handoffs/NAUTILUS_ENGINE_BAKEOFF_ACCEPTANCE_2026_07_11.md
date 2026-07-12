# NautilusTrader Engine Bake-Off Acceptance

Date: 2026-07-11
Acceptance owner: Codex
Repositories: `gym-fx`, `agent-multi`, `trading-contracts`

## Decision

Select NautilusTrader `1.230.0` for new portfolio simulation work. Retain
Backtrader as the frozen policy/parity oracle and fallback. Do not install or
integrate LEAN unless a later Nautilus acceptance fixture fails.

## Installed Runtime

```text
Python: 3.12.7
Environment: /home/harveybc/.venvs/gymfx-nautilus
NautilusTrader: 1.230.0, pinned by gym-fx[nautilus]
Bootstrap: gym-fx/scripts/bootstrap_nautilus_env.sh
Combined ML bootstrap: agent-multi/scripts/bootstrap_agent_nautilus_env.sh
```

No C#, .NET or paid service is required.

The combined overlay environment was also verified with TensorFlow 2.19.0,
PyTorch 2.11.0, Stable-Baselines3 2.8.0 and NautilusTrader 1.230.0 imported by
one interpreter. It inherits the existing ML installation while pinning NumPy
2.1.3, Pandas 2.3.3 and PyArrow 25 locally.

## Implemented Surface

`gym-fx` now provides:

- `simulation_engines/contracts.py`: engine-neutral replay and cost contracts;
- `simulation_engines/nautilus_adapter.py`: multi-asset authoritative replay;
- `simulation_engines/bakeoff.py`: deterministic fixtures and reconciliation;
- `simulation_engines/nautilus_gym.py`: Gym-compatible single-cell bridge;
- `project3_legacy_v1` and `project3_pessimistic_v1` JSON cost profiles;
- canonical `execution_report.v1` export through `trading-contracts`;
- CLI bake-off, multiprocess smoke and benchmark tools;
- JSON-selectable `simulation_engine`, defaulting to `backtrader`.

`agent-multi/env_plugins/gym_fx_env.py` forwards the engine and profile fields
and builds the selected environment without changing the caller API.

## Accounting Evidence

The deterministic fixture contains EUR/USD at 1 minute and USD/JPY at 5 minutes
in one USD margin account. It opens, partially closes, reverses and flattens
EUR/USD while independently opening and closing USD/JPY.

```text
fills: 6
native final balance: 100001.28 USD
independent fixture expectation: 100001.2798625429553264604811 USD
all positions flat: true
same-process deterministic: true
two-process/four-task deterministic: true
```

The pessimistic profile applies per-side commission `0.0002`, full synthetic
spread `0.0004`, adverse slippage `2 bps` per fill, standard margin, explicit
margin preflight, worst-case intrabar collision and FX rollover.

## Verification

```text
gym-fx default environment:
34 passed, 2 skipped

Nautilus optional suite:
9 passed

agent-multi unit suite:
152 passed

parallel Nautilus smoke:
2 workers, 4 tasks, 1 unique result hash
```

The Nautilus suite covers deterministic replay, independent account
reconciliation, canonical report validation, adverse SL/TP ordering,
insufficient-margin denial, rollover, future-row mutation and the Gym step API.

## Performance Evidence

Fresh-run microbenchmark, 25 runs:

```text
Backtrader one-instrument subset mean: 0.00629 s
Nautilus two-instrument fixture median: 0.03064 s
Nautilus two-instrument fixture mean: 0.05698 s
```

The workloads differ, so this is startup-overhead evidence only. Tens of
milliseconds are acceptable for candidate evaluations dominated by model
training and full-year walk-forward replay. DOIN should continue to isolate
Nautilus nodes by process.

## Findings Requiring Continued Attention

1. Stable Nautilus 1.230.0 returns early from account-risk checks for margin
   accounts. The adapter's mandatory preflight calls Nautilus's own
   `calculate_margin_init`, free balance and xrate; it does not implement a
   separate margin ledger.
2. The legacy EUR/USD sample contains malformed OHLC relationships. Nautilus
   rejected them; the data bridge now normalizes high/low envelopes before
   constructing strict bars.
3. The replay adapter supports multiple assets now. The interactive Gym bridge
   is deliberately a one-cell compatibility slice and requires portfolio-native
   observations/actions in the next Phase 3 increment.
4. The warning about `Timestamp.utcnow` originates inside Nautilus during
   `engine.run`; it does not change deterministic outputs.
5. Omega's GPU incident is resolved after the OS update and reboot. Kernel
   `7.0.0-27-generic` loads NVIDIA module/driver `580.159.03`; `nvidia-smi`
   detects the RTX 4070 Laptop GPU, PyTorch `2.11.0+cu130` reports CUDA
   available, and TensorFlow `2.19.0` detects one physical GPU.

## Reproduction

```bash
cd /home/harveybc/Documents/GitHub/gym-fx
scripts/bootstrap_nautilus_env.sh

PYTHONPATH=. ~/.venvs/gymfx-nautilus/bin/python tools/nautilus_bakeoff.py \
  --profile examples/config/execution_cost_profiles/project3_pessimistic_v1.json \
  --output /tmp/nautilus_bakeoff.json \
  --repeat 3

PYTHONPATH=. ~/.venvs/gymfx-nautilus/bin/python tools/nautilus_parallel_smoke.py \
  --profile examples/config/execution_cost_profiles/project3_pessimistic_v1.json \
  --workers 2 --tasks 4

PYTHONPATH=. ~/.venvs/gymfx-nautilus/bin/python -m app.main \
  --load_config examples/config/nautilus_gym_smoke.json
```
