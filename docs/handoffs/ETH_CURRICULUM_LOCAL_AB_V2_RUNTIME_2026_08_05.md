# ETH Curriculum Local A/B v2 Runtime

Date: 2026-08-05 17:23 America/Bogota
Purpose: use all four GPUs for corrected paired-seed evidence while the rejected
DOIN campaign findings 108-116 remain gated
Promotion authority: none; these are local curriculum comparisons

## Accepted Preflight

Omega seed 2711 passed the corrected easy mechanism smoke before fleet launch:

- 883 closed easy-training trades;
- 895 submitted protected entries;
- zero protected-entry rejections;
- 122 trades after returning to normal validation conditions;
- validation total return 2.9031%, mean weekly return 0.0551%, annualized
  return 2.8984% and maximum drawdown 2.8380%;
- `gym-fx`: 84 tests passed;
- `agent-multi`: 526 tests passed.

Canonical compact evidence:
`docs/audits/evidence/ETH_EASY_ACTIVITY_SMOKE_2026_08_05.json`.

## Runtime Contract

- `gym-fx@9a084ac`
- `agent-multi@b265d65d`
- arm order: `easy,easy_normal,normal`
- per-arm budget: 10 epochs x 20,000 timesteps
- easy minimum: 12 trades plus non-hold, entry-action and submitted protected
  entry evidence; protected-entry rejections must equal zero
- normal validation minimum: 12 trades
- output root:
  `~/.local/state/agent-multi/eth-curriculum-ab-v2-20260805`

| Host | GPU | Seed | User unit | Memory ceiling |
| --- | --- | ---: | --- | ---: |
| omega | RTX 4070 Laptop | 2712 | `eth-curriculum-ab-v2-seed-2712.service` | 12 GiB |
| dragon | RTX 4090 Laptop | 2713 | `eth-curriculum-ab-v2-seed-2713.service` | 24 GiB |
| gamma | RTX 5070 Ti Laptop | 2714 | `eth-curriculum-ab-v2-seed-2714.service` | 6 GiB |
| gamma | RTX 5090 eGPU | 2715 | `eth-curriculum-ab-v2-seed-2715.service` | 6 GiB |

All four services entered active state at 17:23 COT. Initial telemetry showed
35-44% GPU utilization and temperatures of 35-45 C. GPU utilization naturally
oscillates because each epoch alternates CUDA learning with full deterministic
Backtrader rollouts on CPU.

## Boundaries

The old campaign supervisors remain disabled. These jobs must not write DOIN
blocks, share candidate pools, seed a champion archive or authorize Paper/Demo
succession. Their useful output is paired raw train/train-tail/validation
evidence and immutable local model hashes. Any zero-trade easy arm is an
explicit failed run, not neutral evidence.
