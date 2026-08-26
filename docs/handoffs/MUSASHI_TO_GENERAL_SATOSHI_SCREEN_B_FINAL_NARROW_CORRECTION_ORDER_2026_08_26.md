# Musashi to General Satoshi: final narrow Screen B correction order

Date: 2026-08-26
Source audit: `docs/audits/AUDIT_SATOSHI_SCREEN_B_WP1_WP5_RETURN_2026_08_26.md`
GPU authority: none

General Satoshi, the scientific structure now holds. Correct these remaining
execution identities without reopening unrelated work.

## N1 — Settle entry-bar brackets inside the entry bar

Implement a deterministic simulator adapter that:

1. fills the parent at the entry bar open;
2. anchors ATR/fixed geometry to that actual parent fill;
3. evaluates that same bar's high/low;
4. settles SL or TP in that bar at the executable level, with adverse gap
   treatment and STOP winning ambiguous collisions;
5. applies the position and cash change exactly once without a next-open order.

Later bars may continue through resting Backtrader children if parity is proven.
Add long/short fixtures where the stop is touched and the next open gaps both
favorably and adversely; neither may change the already-settled entry-bar fill.
Prove no lookahead beyond the entry bar OHLC, no double close and exact accounting
continuity across restart/replay.

## N2 — Make Alpaca the sole current G1 economy

The official Alpaca tier-1 fee schedule supports 25 bp taker pricing. Persist
the official source URL and retrieval/version date in the manifest. Calibrate
the seven envelope geometries using `alpaca_ethusd` costs on each pre-score year,
then freeze those Alpaca geometries for Alpaca B0-B4 scoring.

Keep MT5 outputs descriptive. Do not calibrate or train a G1 MT5 arm until swap/
financing and a larger direct fill sample are available. Its current three-fill
contract remains explicitly provisional.

## N3 — Bind B4 to full Alpaca authority

Each of the 12 B4 cells must embed:

- `cost_contract_id=alpaca_ethusd` and the exact manifest SHA-256;
- `g1_eligible=true`, fee tier and maker/taker assumption;
- Alpaca-frozen envelope ID/digest for its origin;
- cost-scaled entry headroom;
- observation v2 identity and mandatory-declaration flag.

Training, checkpoint selection and scoring all use that same contract. Refuse
numeric cost overrides without matching authority and refuse MT5/zero-cost cells
from G1. Bind genesis metadata to the final cell-config digest even when initial
tensors remain reusable.

## N4 — Minimal rerun and return

Reproduce findings 329-332 first and add permanent regressions. Rerun the Alpaca
calibration and Alpaca B0-B3 scored rows after N1-N3; MT5 need not be rerun.
Preserve previous artifacts as diagnostics. Return exact commits, formula-level
accounting checks, manifests and focused/full suite results.

Stop before GPU. After independent reproduction, Musashi may authorize one
bounded B4 Alpaca preflight to measure epoch time and bracket behavior before
the 12-cell campaign. Continue grouped-extractor CPU implementation in parallel
if it does not touch these frozen contracts.

