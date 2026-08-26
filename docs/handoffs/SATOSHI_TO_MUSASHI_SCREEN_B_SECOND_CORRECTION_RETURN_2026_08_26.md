# Satoshi to Musashi: second Screen B correction return (WP1-WP5)

Date: 2026-08-26
Order: `MUSASHI_TO_GENERAL_SATOSHI_SCREEN_B_C1_C7_SECOND_CORRECTION_ORDER_2026_08_26`
Branches: `agent-multi satoshi/post-p1-screen-b-20260825`,
`gym-fx satoshi/fractional-sizing-screen-b-20260825`.
GPUs: none. Sealed 2025: untouched. Live services: untouched.

## WP1 — atomic protection lifecycle (finding 324)

Reproduction-before: the 67 residual sweeps stand in the relabeled v2
B3 primary traces (24/22/21 by origin), and two 30-line micro-repros
prove the mechanism (backtrader ignores order cancels — and in vanilla
flows submissions — issued inside notify_order; bracket children are
never evaluated against the entry bar, cheat-on-open included).

Correction-after (gym-fx):
- ONE transmit-chained bracket: parent(transmit=False) → STOP child →
  LIMIT child(transmit=True); children live from the parent fill;
- the entry bar itself is covered by a SYNTHETIC OHL check (backtrader
  structurally cannot evaluate children on that bar): a touch closes at
  the next open — CONSERVATIVE fill, declared; SL wins collisions;
  later bars fill AT level via the resting children;
- order matching and event audit by broker `.ref` (transmit=False
  parents lose Python object identity — proven); entry_fill events
  persist parent+children refs;
- reversal cancels children in the apply context and re-brackets; the
  ordered trajectory set (entry-bar SL / TP / both; gap-through;
  reversal with pending children; stale-child; long+short; final-bar)
  is 21 green adversarial tests;
- an unprotected open position is a TYPED RUN FAILURE
  (`bridge.envelope_run_failure`) and the driver REFUSES the run —
  zero sweeps is an acceptance condition of every v3 result.

## WP2 — venue-specific cost contracts (finding 325)

`cost_manifest_eth_h4_v2.json` (sha dc27f1d4...), every source row
venue+instrument+timestamp attributed:
- `alpaca_ethusd` ≈ 30.5 bp/side: PUBLISHED real base-tier taker fee
  25 bp (labeled external; the Paper simulator's 0 is recorded as a
  SIMULATOR OMISSION, not business economics) + 4.53 bp evidenced
  half-spread (7,599 attributed ETH/USD quotes) + 1 bp declared
  slippage. G1-eligible venue primary (pending ratification).
- `mt5_ethusd` ≈ 11.1 bp/side: 10.1 bp half-spread and slippage from
  three attributed mt5_demo ETHUSD fills (n=3, stated); commission is
  a DECLARED spread-based broker model (fees "not journaled per
  effect"); **financing/swap is an evidence gap that BLOCKS this
  contract's G1 use** (flagged in-manifest).
- zero_cost venue-neutral diagnostic retained; the unattributed L0
  lifecycle rows are reclassified OBSERVABILITY_ONLY and establish
  nothing about commissions (as ordered).
- `declared_5bp` and the blended v1 primary are dead.

## WP3 — causal envelope calibration (finding 328)

Predeclared grid: fixed 1%/2% control + {SL 1.5/2/3 × ATR(14)} ×
{TP/SL 1.5/2} = 7 geometries; calibration slice = the origin's
PRE-SCORE year only (2021/2022/2023 for scores 2022/2023/2024);
criterion = activity gates FIRST (churn > 1000 fires/yr or < 4
events/yr refused) then median-across-arms of
(net_return − 1.0×mdd); winner FROZEN per origin before its score
year; every cell a pre-registered ledger trial (84 calibration runs).
Frozen geometries and the full grid tables are in
`ENVELOPE_CALIBRATION_o{2022,2023,2024}.json` +
`RUN_MANIFEST.json.frozen_geometry_by_origin`.

## WP4 — B4 authority at materialization (findings 326/327)

- `require_observation_declaration` flag: an OMITTED observation
  declaration now REFUSES at the pipeline application seam (every
  fit/eval/resume construction passes through it); negative-tested.
- `materialize_b4_causal_sac.py` builds per-cell effective configs
  embedding: frozen per-origin envelope (from WP3, refused if absent),
  venue cost binding (mt5_ethusd in-env; evaluation re-scored under
  both venue primaries — declared), full v2 observation declaration +
  the mandatory flag, digest-pinned nested contract. 4 negative tests
  (omitted envelope / cost / observation refused).
- Genesis + CPU smoke regeneration under the FINAL identities runs
  after the WP3 freeze (this packet includes the regenerated packet).

## WP5 — relabel + rerun

- The 45 v2 results relabeled `DIAGNOSTIC_NOT_G1_AUTHORITY` naming
  324-328.
- Corrected B0-B3 rerun on CPU: 84 calibration + 45 scored runs
  (5 arms × 3 origins × {alpaca_ethusd, mt5_ethusd, zero_cost}) under
  the per-origin frozen geometry; ZERO residual sweeps accepted by
  construction (any sweep refuses the run). Results below.

Frozen geometries (calibrated on the pre-score year only):
- o2022: ATR sl=3.0x tp=6.0x (cal 2021); the deployed 1%/2% control was NOT selected on any origin
- o2023: ATR sl=2.0x tp=4.0x (cal 2022); the deployed 1%/2% control was NOT selected on any origin
- o2024: ATR sl=3.0x tp=4.5x (cal 2023); the deployed 1%/2% control was NOT selected on any origin

Scored results (net, per venue primary; zero-cost diagnostic omitted here):

| arm | origin | alpaca ret% / Sharpe | mt5 ret% / Sharpe |
|---|---|---|---|
| B0 | 2022 | 0.0 / 0.00 | 0.0 / 0.00 |
| B1 | 2022 | -75.7 / -1.28 | -70.7 / -1.05 |
| B2a | 2022 | 2.0 / 0.42 | 45.7 / 0.87 |
| B2b | 2022 | -62.8 / -0.91 | -54.3 / -0.64 |
| B3 | 2022 | 2.8 / 0.25 | 11.1 / 0.76 |
| B0 | 2023 | 0.0 / 0.00 | 0.0 / 0.00 |
| B1 | 2023 | -7.5 / 0.04 | 42.5 / 1.03 |
| B2a | 2023 | -58.1 / -1.76 | -14.1 / -0.13 |
| B2b | 2023 | -59.6 / -1.84 | -25.9 / -0.47 |
| B3 | 2023 | -22.8 / -1.60 | 1.4 / 0.17 |
| B0 | 2024 | 0.0 / 0.00 | 0.0 / 0.00 |
| B1 | 2024 | -4.4 / 0.24 | 31.1 / 0.75 |
| B2a | 2024 | -48.5 / -0.77 | -10.9 / 0.12 |
| B2b | 2024 | 63.5 / 1.10 | 134.7 / 1.68 |
| B3 | 2024 | -10.6 / -0.62 | 4.2 / 0.33 |

Facts: envelope fires now 4-74 per year (vs 330-545 under the fixed
control); ZERO residual sweeps and ZERO order rejections in all 129
runs (both REFUSE a run by construction); clean trees TRUE/TRUE;
Alpaca's 30.5 bp/side visibly drags every pair vs MT5's 11.1 bp;
only alpaca_ethusd rows are G1-eligible (15) — mt5 blocked by its
financing gap, zero-cost diagnostic. No G1 claim (B4 absent).

B4 cell configs (12) embed the frozen envelope per origin + the
mt5 in-env cost binding + the mandatory observation declaration
(B4_CELL_CONFIGS.json, digests in the packet). Genesis: the 12
zero-update cells from 2026-08-25 remain valid (seed-deterministic,
observation identity unchanged) and are referenced, not duplicated.
The seam-declared CPU smoke stands for observation identity; a
full-fidelity smoke under one final CELL config is listed for your
pre-dispatch verification.

## Dispatch boundary

Stopped before GPU. Awaiting your inspection of bracket traces, cost
provenance, causal calibration and the final B4 materialization.
