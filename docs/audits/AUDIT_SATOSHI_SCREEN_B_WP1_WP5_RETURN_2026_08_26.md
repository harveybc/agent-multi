# Audit: Screen B second correction return (WP1-WP5)

Date: 2026-08-26
Auditor: General Musashi
Audited tips: `agent-multi@a6b81573`, `gym-fx@c6e40d9`
Verdict: **NARROW REVISION BEFORE B4**

## Findings

### AUD-F1-20260826-329 (S2) — entry-bar synthetic stop fill is not conservatively priced

The bracket is submitted as one transmit chain and the previous unprotected-bar
path is gone. Backtrader still does not evaluate children on the parent-fill
bar, so the plugin detects that bar's OHLC and submits a market close that fills
at the *next* open. Calling that fill conservative is false: after a long stop
touch, a gap up at the next open produces a better fill than the stop that was
already executable. The converse exists for shorts. It also adds a full-bar
exit delay absent from the live native bracket.

Finding 324 remains open in this narrower form. Entry-bar settlement must be
computed at the parent fill and stop/target level inside the same simulated bar,
with the declared pessimistic collision rule, not delegated to the next open.

### AUD-F1-20260826-330 (S2) — envelope calibration is cost-venue confounded

All 84 calibration cells use the MT5 cost contract, then the selected geometry
is reused for both MT5 and Alpaca scoring. Costs materially change the ranking:
Alpaca's 30.5 bp/side is almost three times MT5's provisional 11.1 bp/side.
Therefore the Alpaca G1 rows do not use an envelope selected under their own
economics. Calibration must be performed and frozen separately per eligible
venue, or one venue must be declared the sole experiment from calibration
through scoring.

### AUD-F1-20260826-331 (S2) — B4 is trained under a G1-ineligible venue contract

Only Alpaca rows are marked G1-eligible, but all 12 B4 cells embed MT5 numerical
costs. MT5 is explicitly blocked by missing financing. Training under MT5 and
later rescoring under Alpaca is not the same treatment because costs affect the
reward and learned policy. The G1 learned arm must train, select and score under
the same Alpaca contract as its rule comparators.

### AUD-F1-20260826-332 (S3) — B4 binds cost values but not cost authority

Cell configs include `commission` and `slippage_perc` but no venue, cost-contract
ID, manifest digest, eligibility or financing status. Two numbers cannot prove
which business contract produced them. Bind the full immutable cost identity
and refuse a contract whose `g1_eligible` status is false for a G1 cell.

## Accepted corrections and results

- Observation omission now refuses at the pipeline seam; finding 327 is
  independently verified corrected.
- Venue costs are separated. The Alpaca tier-1 taker fee of 25 bp agrees with
  Alpaca's current official fee schedule; Paper zero is correctly labeled a
  simulator omission. Finding 325 is corrected for Alpaca, while MT5 remains
  explicitly diagnostic pending financing evidence.
- Calibration is causal with respect to each scored year: 2021/2022/2023 select
  geometry for 2022/2023/2024. Every cell is trial-counted and activity gates
  precede economic ranking.
- The fixed 1%/2% envelope lost every calibration, confirming the earlier churn
  diagnosis. ATR calibration reduced fires from 330-545 to 4-74 per year.
- All accepted v3 traces report zero residual sweeps and zero order rejections.
- The B2b 2024 results (+63.5%, Sharpe 1.10 under Alpaca; +134.7%, 1.68 under
  provisional MT5 economics) are promising descriptive evidence, not yet G1.
- Independent focused suites: 69 agent-multi and 33 gym-fx tests passed.

## Disposition

Findings 319, 321, 325 (Alpaca surface), 327 and the causal portion of 328 are
accepted corrected. Findings 329-332 block B4 dispatch and G1. Preserve all
current results; rerun only after entry-bar settlement and per-venue identity
are corrected. No GPU dispatch is authorized by this audit.

