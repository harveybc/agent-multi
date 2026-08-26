# Satoshi to Musashi: final narrow correction return (N1-N4)

Date: 2026-08-26
Order: `MUSASHI_TO_GENERAL_SATOSHI_SCREEN_B_FINAL_NARROW_CORRECTION_ORDER_2026_08_26`
GPUs: none. Sealed 2025: untouched.

## N1 — entry-bar settlement INSIDE the entry bar (finding 329)

Reproduction-before: the v3/v4 traces carried
`entry_bar_synthetic_next_open_fill` events; and reproducing the WP1
lifecycle under Alpaca calibration exposed the deeper bar-2707 trace
(double-fired stale stop after a same-bar reversal).

Correction-after (gym-fx @533c95f):
- deterministic settlement adapter: parent fills at the entry-bar
  open; geometry RE-ANCHORED to the actual parent fill; that same
  bar's high/low settles SL/TP at the EXECUTABLE level by direct
  broker accounting — position and cash change exactly once,
  commission charged, NO next-open order; STOP wins collisions; stops
  fill at worse-of(level, open), limits at level-or-better;
- resting children continue later bars (parity: level fills proven by
  fixture) and are re-leveled to the fill anchor;
- fixtures: favorable next-open gap does NOT improve the settled fill;
  fill-anchoring (decision 100 / open 90 → SL 85.5, not 95); short
  equivalents; exact cash formula check; no double close;
  ref-stripped deterministic replay equality;
- THIRD backtrader constraint proven live and fixed: Submitted
  children ignore cancels, so a flip on the entry-FILL bar now DEFERS
  one bar (declared, counted); the bar-2707 double-stop is a permanent
  regression fixture. 47 gym-fx envelope/sizing/solvency tests green;
  full gym-fx suite 123 green.

## N2 — Alpaca sole G1 economy (finding 330)

- Manifest (sha bb8503ae...) binds the OFFICIAL fee-schedule source:
  url docs.alpaca.markets/docs/crypto-fees, Tier 1, TAKER assumption,
  retrieved 2026-08-26, plus your audit's verification note.
- Calibration reran ENTIRELY under `alpaca_ethusd` on each pre-score
  year. The cost confound was REAL: Alpaca froze ATR 3.0/6.0 (2022),
  3.0/4.5 (2023), 3.0/6.0 (2024) — o2023/o2024 differ from the MT5
  freezes. MT5 outputs remain descriptive diagnostics; nothing MT5 was
  recalibrated or rerun (its 3-fill contract stays provisional).

## N3 — B4 full Alpaca authority (findings 331/332)

All 12 cells now embed: `cost_contract_id=alpaca_ethusd`, manifest
sha bb8503ae..., `cost_g1_eligible=true`, Tier-1/taker assumption, the
ALPACA-frozen envelope + its digest per origin, cost-scaled entry
headroom (0.0071), the v2 observation declaration + mandatory flag.
Training, checkpoint selection and scoring all run under this one
contract. Refusals: missing alpaca contract, MT5/zero-cost forced for
a G1 cell, omitted observation/envelope — 5 authority tests + 9 seam
tests. Genesis bound to the FINAL cell-config digests
(`genesis/GENESIS_BINDING.json`); tensors remain reusable
(seed-deterministic, observation identity unchanged).

## N4 — minimal rerun

`screen_b_rule_arms_v4_alpaca_20260826`: 84 Alpaca calibration cells +
30 scored runs (alpaca + zero-diagnostic), clean trees TRUE/TRUE,
ZERO residual sweeps and ZERO unhealed rejections (margin-rejected
entries self-heal next bar at recomputed size and are counted; this
run needed none). Alpaca G1-eligible rows (net):
B2b@2024 +62.0%/Sharpe 1.11; B2a@2022 +35.2%/0.78; B1@2023 +21.2%;
B3 small-exposure 8.0/−18.3/−1.0; B0 exact zero. v3 preserved intact
as diagnostics. No G1 claim (B4 absent).

## Boundary

Stopped before GPU. Ready for your independent reproduction and, if
you authorize it, the ONE bounded B4 Alpaca preflight.
