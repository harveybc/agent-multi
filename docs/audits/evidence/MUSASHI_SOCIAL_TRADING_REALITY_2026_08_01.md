# Musashi Social-Trading Reality Evidence

Observed at: 2026-08-01 14:49 America/Bogota
Implementation owner: Codex technical lead
LTS commit: `db80d97a8586fb5c110dd6cb8626b53ecbeeac9c`

## Scope

This increment adds a provider-neutral, no-order accounting and allocation
vertical. It does not open an account, read credentials, submit an order,
authorize real capital, publish a signal or mutate the active DOIN campaign.

Tracked implementation:

- `lts/app/social_trading_lab.py`
- `lts/app/social_trading_cli.py`
- `lts/examples/configs/social_trading_platform_registry_v1.json`
- `lts/examples/configs/social_trading_accounting_scenario_v1.json`
- `lts/tests/unit/test_social_trading_lab.py`
- `lts/docs/SOCIAL_TRADING_REALITY_LAB.md`

## Deterministic Scenario

Persisted run ID: `social-2ed043a09e364070`

- scenario SHA-256:
  `ca3255490beca77b165f2b55f89ecc99060b65672cec359a75ea210355b4c8b3`;
- registry SHA-256:
  `cc82c872b966b6694cb2f8dcf7be21c085e04ef3ef8166693df6dac4d17e68f2`.

Inputs:

- manager capital: USD 5,000;
- external investor A: USD 10,000;
- external investor B: USD 25,000;
- strategy path: +8%, then -3%;
- investor A withdrawal: USD 1,000;
- performance fee: 20%;
- management fee example: 2% annualized for 30 days on investor A;
- protected MQL5 and native cTrader Copy allocation comparisons.

Final accounting facts:

- unit NAV: `1.0476`;
- pool equity: `40389.77556976154236428209030948757`;
- external investor equity: `35151.77556976154236428209030948757`;
- manager capital equity: `5238`;
- manager fee balance: `561.0561136478944698122780314561137`;
- orders submitted: `0`.

Allocation facts:

- MQL5 Signals: raw/allocated volume `0.5/0.5`, accepted because subscriber
  SL/TP replication is available under the declared contract;
- native cTrader Copy: raw/allocated volume `0.5/0`, rejected with
  `protected_entry_unavailable` because provider SL/TP levels are not copied.

The persisted local OLAP is
`~/.local/state/lts/social-trading-lab.sqlite`; it is machine state and is not
committed. A separate ephemeral execution reproduced the same final facts.

## Verification

- focused social-trading tests: 8 passed;
- complete LTS suite: 225 passed;
- complete `agent-multi` suite: 405 passed;
- expected warning: one upstream Starlette/httpx deprecation warning;
- `python -m compileall`: passed;
- JSON loading and CLI registry/scenario/report: passed;
- `git diff --check`: passed;
- `pip check`: no broken requirements;
- LTS branch: `main`, pushed to `origin/main` at `db80d97`.

## Current Front 1 Runtime

All four workers directly reported the same active job, domain, seed,
population counts, dataset, plan, finalized anchor and champion:

- job: `usdcad-4h-protected-easy-sac-shared-v2`;
- stage/generation: `data_observation`, generation 5;
- population: 20 total, 15 evaluated, 4 claimed, 1 free;
- campaign: 115/480 evaluated, 365 full-budget candidates remaining;
- measured aggregate rate: 1.7813 candidates/hour;
- full-budget ETA: 204.9 hours without early stopping;
- current candidate ETAs: 0.82-1.51 hours;
- champion fitness: `0.0006247008569073586`;
- champion artifact SHA-256:
  `99fce2e40e3fb8b64103dff038281071118175f8b4c838cf5efe72ca46f8f471`.

The known warning remains: Omega/Dragon and Gamma report two equal-height tips
above common finalized height 6/hash `cab57051...`. They share one population
and candidate pool. No consensus mutation was made during this increment.

Hardware at the same inspection:

| Worker | GPU | Temp | Utilization |
| --- | --- | ---: | ---: |
| omega | RTX 4070 Laptop | 56 C | 32% |
| dragon | RTX 4090 Laptop | 50 C | 38% |
| gamma-5070ti | RTX 5070 Ti Laptop | 54 C | 48% |
| gamma-5090 | RTX 5090 eGPU | 56 C | 36% |

Gamma root remains at 88% with 48 GiB free.

## Current Front 2 Runtime

The consolidated watchdog reported zero active events:

- Alpaca: 742 complete authenticated sessions, six fresh crypto quotes, zero
  positions/orders and no protected execution eligibility;
- IBKR: paper port 7497 reachable, 359 complete sessions, zero
  positions/orders;
- OANDA MT5 demo: fresh heartbeat, terminal build 6075, 3,038 heartbeats, 760
  snapshots, six runtime symbols and zero positions/orders;
- shadow: complete, all cells fresh, zero orders, NAV 98,560.94 from 100,000;
- Capital.com: adapter remains unconfigured and optional.

The expanded 13-symbol MT5 default is tracked but is not runtime evidence until
the EA is recompiled/reloaded or its Inputs are changed.

## Current Front 3 Runtime

- latest collection: complete, 102 fetched, 17 inserted, 85 duplicates, zero
  flagged;
- identity: `Dragon_DOIN`, claimed;
- model: `deepseek-v4-flash` through `opencode-go`;
- reserved tokens: 144,041/250,000 daily and 144,041/6,000,000 monthly;
- corpus: 1,541 unreviewed posts;
- publishing remains disabled/approval-gated.

## Owner Actions and Gates

No immediate spending is required. The next external actions are:

1. create a cTrader ID/demo for Copy investor observation and later official
   Open API preflight;
2. optionally create a free eToro Virtual Portfolio as a manual UX control;
3. review Darwinex Zero country eligibility, terms and recurring price before
   approving membership;
4. do not fund a PAMM manager before ledger parity, legal review and an
   explicit capital limit.
