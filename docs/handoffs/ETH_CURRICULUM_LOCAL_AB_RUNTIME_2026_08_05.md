# ETH Curriculum Local A/B Runtime

Date: 2026-08-05 15:40 America/Bogota
Owner priority: keep useful GPU work running while findings 108-116 are corrected
Runtime mutation: standalone local training only; no DOIN chain or champion archive

## Purpose

Run four independent paired-seed comparisons of the fixed ETH SAC candidate:

1. normal-only training;
2. easy-only training evaluated under normal conditions;
3. easy-to-normal warm continuation.

Every arm reports raw train, train-tail and validation metrics. The disclosed
2025 test split is not evaluated. These runs measure curriculum behavior and
seed variance; they do not satisfy the corrected DOIN smoke gate and cannot be
promoted as champions.

## Immutable Inputs

- dataset SHA-256: `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`
- fixed config SHA-256: `5df24c0d78a89613041406213692af5f180f4af3c220ac69792a60db8ca70e18`
- code revision: `agent-multi@f5f2c4cb`
- per-arm budget: 10 epochs x 20,000 timesteps
- arms per seed: `normal,easy,easy_normal`

## Allocation

| Host | GPU | Seed | systemd user unit | Output |
| --- | --- | ---: | --- | --- |
| omega | RTX 4070 Laptop | 2703 | `eth-curriculum-ab-seed-2703.service` | `~/.local/state/agent-multi/eth-curriculum-ab-20260805/seed-2703` |
| dragon | RTX 4090 Laptop | 2704 | `eth-curriculum-ab-seed-2704.service` | `~/.local/state/agent-multi/eth-curriculum-ab-20260805/seed-2704` |
| gamma | RTX 5070 Ti Laptop | 2705 | `eth-curriculum-ab-seed-2705.service` | `~/.local/state/agent-multi/eth-curriculum-ab-20260805/seed-2705` |
| gamma | RTX 5090 eGPU | 2706 | `eth-curriculum-ab-seed-2706.service` | `~/.local/state/agent-multi/eth-curriculum-ab-20260805/seed-2706` |

Each output directory must contain `fixture_report.json`,
`fixture_manifest.json` and the model archives produced by its completed arms.
Consolidation must verify hashes before comparing paired validation metrics.

## Operational Boundaries

- Campaign supervisors remain disabled and the rejected chain remains archived.
- The legacy swarm and DOIN-memory watchdog cron blocks are suspended while a
  deliberate audit pause is active. GPU-temperature monitoring, live-trading
  monitoring, Hermes and the incident router remain enabled.
- Reinstall the swarm watchdog with the corrected smoke profile before the
  four-worker smoke begins.
- A failed local unit may be restarted only with the same host, seed, input
  hashes and output namespace; never merge partial outputs across seeds.

## Completion Packet

Produce one consolidated table with, for every seed and arm, raw validation
mean weekly return, total return, maximum drawdown fraction, trades and final
equity. Report paired deltas for `easy-normal - normal` and `easy - normal`,
completion/failure counts, wall time and artifact hashes. Satoshi may use this
evidence while correcting the campaign, but the full swarm remains gated on
Musashi's independent smoke audit.
