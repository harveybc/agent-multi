# Musashi Multi-Front Status Evidence

Observed at: 2026-08-01 14:29 America/Bogota
Collector: Codex technical lead
Method: direct supervisor APIs, local/SSH hardware probes, LTS watchdog state,
systemd timers, social OLAP status and repository tests

## Front 1: DOIN Optimization

- plan: `phase-1-protected-execution-fleet-v2`
- plan hash: `b43844a7ebd7c85a782c557a8c3459622e1cb353a5d33391816e85f107cb6b21`
- current job: `usdcad-4h-protected-easy-sac-shared-v2`
- domain: `trading-asset-policy-usdcad-4h-protected-easy-v2`
- stage/generation: `data_observation`, generation 5 of stage 1
- generation population: 20 total, 15 evaluated, 4 claimed, 1 free
- campaign budget: 115/480 evaluated, 365 remaining, 23.96%
- measured aggregate throughput: 1.7813 candidates/hour
- full-budget ETA: 204.9 hours; early stopping may reduce it
- queued job: `usdcad-4h-protected-curriculum-sac-shared-v2`; its budget and
  ETA remain unknown until materialization, so a displayed zero is not an ETA
- champion fitness: `0.0006247008569073586`
- champion validation total return: `0.2744431976%`
- champion validation risk-adjusted full-period return: `0.2254576330%`
- champion max drawdown: `0.0489855645%`
- champion completed trades: 276
- artifact SHA-256:
  `99fce2e40e3fb8b64103dff038281071118175f8b4c838cf5efe72ca46f8f471`
- artifact format/size: Stable-Baselines3 ZIP, 15,348,290 bytes

All four workers were online, running, candidate-owning and `join_ready`. They
reported the same seed 2703, generation, population fingerprint, champion,
dataset hash, plan hash, domain semantic hash and component versions.

One warning remains open: Omega/Dragon report chain height 13 tip
`1f4478b7...`; both Gamma workers report height 13 tip `ce0c179b...`. All four
share finalized height 6 and hash `cab57051...`. This is an equal-height tip
split above one common finalized anchor, not evidence of separate jobs or
candidate pools. No reorg or active-job mutation was attempted.

## Hardware

| Worker | GPU | Temperature | Utilization | GPU memory | Power |
| --- | --- | ---: | ---: | ---: | ---: |
| omega | RTX 4070 Laptop | 57 C | 33% | 2260/8188 MiB | 34.38 W |
| dragon | RTX 4090 Laptop | 50 C | 39% | 541/16376 MiB | 47.80 W |
| gamma-5070ti | RTX 5070 Ti Laptop | 53 C | 46% | 490/12227 MiB | 81.98 W |
| gamma-5090 | RTX 5090 eGPU | 56 C | 35% | 878/32607 MiB | 108.22 W |

All temperatures were below the 78 C alert threshold. Gamma root storage was
88% used with 48 GiB free and remains the nearest capacity constraint.

## Front 2: Multi-Venue Paper Observation

The consolidated watchdog reported zero active events.

- Alpaca: 738 complete sessions, fresh authenticated session, six crypto
  quotes, zero positions/orders, protected execution not yet eligible.
- IBKR: TWS paper port 7497 reachable, 355 complete sessions, latest
  reconciliation complete, zero positions/orders.
- OANDA Global MT5 demo: bridge reachable through Dragon, heartbeat age 8.6
  seconds, connected, read-only, build 6075, 2,955 heartbeats, 739 snapshots,
  six currently observed symbols and zero positions/orders.
- shadow portfolio: complete, no missing/stale cells, zero orders, NAV
  98,613.61 from 100,000. The -1.386394% value is shadow mark behavior, not a
  model-performance claim.
- Capital.com remains unconfigured and optional.

The tracked MT5 source default was expanded in code to the promising
crypto/selected-FX union. Runtime still has six observed symbols until the EA
is recompiled/reloaded or its Inputs are changed.

## Real-Time Feature and Asset Parity

The new machine-readable contract separates decision feeds from execution
venues and registers 9 input lanes, 12 source families, 8 runtime sources, 3
execution venues, 16 instrument maps and 18 selected asset/timeframe cells.

Current `USDCAD@4h` classification:

- research eligible: yes;
- live inference eligible: no;
- live execution eligible: no;
- blockers: closed-bar runtime feed not integrated, numerical parity not
  tested, USDCAD not yet observed in the active MT5 snapshot and protected
  execution canary not passed.

The active DOIN job remains valid research and is not mutated. Its artifact is
not authorized for live orders.

## Front 3: Social Intelligence

- collector timer active;
- latest run complete: 102 fetched, 18 inserted, 84 duplicates, 0 flagged;
- identity: `Dragon_DOIN`, claimed;
- corpus: 1,524 unreviewed posts;
- model: `deepseek-v4-flash` through `opencode-go`;
- reserved tokens: 144,041/250,000 daily and 144,041/6,000,000 monthly;
- publishing remains disabled/approval-gated; no drafts exist.

## Verification

- `agent-multi` unit suite in `trading-stack`: 404 passed, two expected MLP
  convergence warnings;
- live-parity focused tests: 4 passed;
- LTS MT5 focused tests in `trading-stack`: 5 passed, one dependency
  deprecation warning;
- JSON validation, Python compilation and `git diff --check`: passed.
