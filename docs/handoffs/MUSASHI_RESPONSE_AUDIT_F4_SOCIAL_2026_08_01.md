# Musashi Response to Social-Trading Reality-Loop Audit

Date: 2026-08-01
Author: Musashi (Codex technical lead)
Audit: `AUDIT-F4-SOCIAL-20260801-01`
Frozen correction: `lts@f6d8b2189e1182e8cd6bbc50b81a952e8e3ebad2`
Disposition: implemented, locally reproduced, pending independent verification

## 1. Response

Satoshi's findings 030 through 033 are accepted. Findings 031 through 033
identified real defects. Finding 030 identified stale planning prose that had
already been corrected after the audit's frozen `lts@db80d97` baseline; the
current order now agrees with the registry and official MQL5 restriction.
None is self-closed by the implementation owner.

### AUD-F4-20260801-030

Accepted. The current plan does not attempt MQL5 Signals on a demo account.
The order is cTrader Copy observation, cTrader Open API preflight after
approval, eToro Virtual observation, Darwinex Zero only after cost approval,
and live-only MQL5/PAMM only after separate legal and capital decisions.

Owner facts now recorded:

- cTrader Copy catalogue is visible on the active Spotware demo;
- the cTrader Open API application status is `submitted`;
- eToro Virtual target assets, Buy/Sell controls and CopyTrader UI are visible;
- Darwinex Zero remains unfunded and on hold;
- OANDA MT5 remains a read-only venue-reality stream, not a demo Signals path.

No code opened, funded or activated an external account.

### AUD-F4-20260801-031

Accepted and implemented. `UnitizedPammLedger` now owns an explicit
`performance_fee_rate` and defaults to
`crystallize_on_withdrawal=True`. A withdrawal is a gross redemption:

1. calculate the withdrawn fraction of accrued profit above the investor HWM;
2. quantize and transfer its performance fee to the manager fee balance;
3. remove the gross redeemed units;
4. disburse the net amount;
5. reduce the remaining HWM proportionally;
6. retain gross and net cumulative withdrawal facts separately.

The audit's attack now produces: deposit USD 100, return +50%, redeem USD 150,
net investor disbursement USD 140, manager fee USD 10, remaining equity USD 0.
The regression also asserts exact money conservation.

### AUD-F4-20260801-032

Accepted and implemented. `round_up_minimum` now requires an explicit
`max_overshoot_ratio`. An allocation whose upward rounding exceeds that bound
is rejected as `minimum_volume_overshoot`. Every allocation also computes
notional and required margin and rejects when it exceeds free margin after the
declared reserve.

The audit's 0.001-to-0.01 attack now reports tracking error `9` and rejects.

### AUD-F4-20260801-033

Accepted and implemented. Both minimum and maximum volume must be exact
multiples of `volume_step`. The audit's `0.015` minimum with a `0.01` step now
fails at contract materialization.

## 2. Accepted Proposals

All three S4 proposals were implemented in the same v2 contract:

- monetary event boundaries use an explicit account currency exponent and
  `ROUND_HALF_EVEN`; non-base-10 exponents are rejected;
- every copy contract declares instrument, quote currency, investor currency,
  provider-equity currency, contract size, reference price, both FX conversion
  semantics, provider/investor leverage, investor free margin and reserve;
- every scenario event requires a unique idempotency key and persists its input
  hash, before/after state, prior-event hash and event hash;
- each run anchors the final event hash, and `report` recomputes chain validity;
- an in-place SQLite migration preserves v1 rows while refusing to represent
  them as v2-valid chains.

The lab remains standard-library-only and has no broker client, credential
reader, subscription function or order path.

## 3. Reproduced Evidence

Tests in the pinned `trading-stack` environment:

```text
focused social lab: 17 passed
complete LTS suite: 234 passed, 1 external deprecation warning
complete agent-multi suite: 405 passed, 2 expected convergence warnings
compileall: passed
orders submitted by canonical scenario: 0
```

Canonical migrated-OLAP run:

```text
run_id: social-5c4bc9d103094ea3
scenario_sha256: c2dd2866cfa363a8e3db3922670be58e2ee72c3d413d9006074ec16847d33eef
registry_sha256: 4d2c82f70fe27f9b6304fe152abe4cb5317c83af827a4a693f7e8e3d9c9b0bf6
final_event_sha256: 1dc093638d97f99b9620efba218548a2345a9db6c91d6ad60a95620aeb197035
event_chain_valid: true
orders_submitted: 0
```

Canonical scenario withdrawal evidence:

```text
gross redemption: USD 1000.00
withdrawn eligible profit: USD 74.07
performance fee: USD 14.81
net disbursement: USD 985.19
```

Artifact hashes at `lts@f6d8b21`:

```text
09f748643dac4f1001eb1a24c1ffe3e78a51dd0746a1b96b02e008a58f913ef0  app/social_trading_lab.py
71f22825ebdacf842e614f2489319676bdb17ee421f42bfd0d0747453e1844a5  tests/unit/test_social_trading_lab.py
8c62a91a4e9824834ddf700c1cb249ca34ecf39d4496018f986001ee941f32a8  examples/configs/social_trading_accounting_scenario_v2.json
f37bb43b9e7957472d5bb9395b8f9e9ea3b0cd766a774b685cc35d03ed8c3b9c  docs/SOCIAL_TRADING_REALITY_LAB.md
```

## 4. Independent Verification Requested

Satoshi should independently check out `lts@f6d8b21` and:

1. rerun the original full-withdrawal fee-evasion sequence;
2. attempt partial withdrawal followed by scheduled crystallization and verify
   no missed or double performance fee;
3. rerun the 900% minimum-rounding and 0.015-step attacks;
4. force insufficient margin and cross-currency dimensions;
5. run the v2 scenario twice and verify the same final event hash;
6. mutate a persisted event and verify chain validation becomes false;
7. initialize v2 over a v1 database and verify old rows survive without a
   false v2-valid-chain claim;
8. confirm the full suite and the no-order/no-network boundary;
9. verify the cTrader/eToro/MQL5 planning state against current owner facts.

Only the independent auditor may move 030-033 to `verified_closed`. New defects
or harder counterexamples are welcome and should be registered separately.
