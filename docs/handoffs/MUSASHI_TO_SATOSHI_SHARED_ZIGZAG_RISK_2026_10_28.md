# Shared ZigZag references, SL/TP rules and normalized order size

For Satoshi, from Musashi. This packet documents measured common inputs to
both strategy designs. It does not merge pure RL with expert-guided learning,
choose the other lane's labels, or qualify a finished account strategy.

## Shared business contract

[BUSINESS_CONTRACT.json](../work_plan/mechanism_synthesis/BUSINESS_CONTRACT.json)
is authoritative: leverage 50; rel_volume 0.1; one simultaneous position;
no strategy daily entry cap; wait 12h after weekly open; forced flat 12h
before weekly close. Early close can act regardless of the opener's horizon.

```text
margin(t) = q * rel_volume * balance(t),  q in [0,1]
```

The 0.1 is applied once. This is margin allocation, not maximum account loss.
Broker rules and gap losses remain distinct from that allocation limit.

## How the ZigZag references were obtained

1. Authenticated admitted H1 EURUSD bars and closure/calendar records,
   four TRAIN years 2020-2023. The tape has 24,895 rows, 19,879 admissible
   entry origins and 209 contiguous chunks. No interpolation or gap crossing.
2. At each origin the retrospective oracle sees the next K chronological
   ZigZag events, strictly later than entry. Both directions evaluate every
   event as a possible exit, then select their best score. A hypothetical
   trade can cross an intervening peak or valley; its adverse path counts.
3. K=2,3,4,5 were compared. K=3 retained the main gain with better support
   than the later variants. No arbitrary timed exit substitutes a pivot.
4. Score per directional candidate is net price profit divided by
   `max(path drawdown, round-trip fill cost, 1 pip) * duration_hours`.
   Per origin choose the best positive score across sides, otherwise zero;
   aggregate over the fixed 19,879 origins. Incomplete next-K support is
   separately censored, not called a valid NOP observation.
5. Horizon eligibility means median pivot spacing lies within
   `[h/sqrt(2), h*sqrt(2)]` with at least 100 spans. **These are nominal
   event horizons on H1 bars, not separately downsampled model training.**
6. DEAP population20, generations300, seeds7/19/43. Compared with corrected
   20x30 runs, all 360 prefix records matched. A separate exhaustive grid of
   2,004 depth/backstep pairs at deviation5 matched all four best fitnesses.
7. Stored both sides' per-origin target, MAE, drawdown, duration and costs.
   Descriptive profiles now separate all support, positive opportunities,
   winning sides, calendar-compatible exits and year strata.

## Optimized ZigZag parameters

| Nominal horizon | Depth | Recorded deviation (pips) | Backstep | Best oracle fitness |
|---:|---:|---:|---:|---:|
| 3h | 3 | 5 | 1 | 0.271176912 |
| 6h | 4 | 5 | 2 | 0.237675675 |
| 12h | 7 | 25 | 6 | 0.152872027 |
| 24h | 12 | 25 | 11 | 0.098117667 |

Bounds used: depth2..96; backstep1..min(24,depth-1); deviation1..120 pips,
0.1 resolution. The 12h best improves 1.215882% from30 to300 generations;
other best scores remain unchanged. These fitnesses are not account Sharpe
or returns and are not directly comparable across different supports.

**Deviation caveat:** code inspection and scale/adapter probes found it
inactive for this implementation and tested domain. It reaches the engine;
this is not a disconnected gene. Fixing it at5 reproduces these champions.
Retain original records; do not silently implement a different reversal rule
or call the differing recorded deviations effective tuned optima.

## Measured directional reference distances

Positive, calendar-compatible side-specific opportunities; medians in pips.
They overlap and are not executable account-order counts. The full report
contains mean/std/quantiles, year strata and alternative populations.

| h | Buy TP / MAE | Sell TP / MAE | Buy / sell opportunity counts |
|---:|---:|---:|---:|
| 3 | 20.0 / 7.4 | 19.6 / 7.7 | 15,033 / 15,053 |
| 6 | 24.0 / 8.5 | 23.4 / 8.9 | 14,946 / 14,845 |
| 12 | 35.3 / 11.9 | 35.8 / 12.7 | 12,932 / 13,191 |
| 24 | 42.1 / 16.0 | 43.1 / 16.2 | 10,284 / 10,765 |

134,916 selected directional paths reconstructed independently with maximum
error0; full profile replay byte-identical. Original input/result artifacts
unchanged. The published profile digest is
`aa2cb1f8f524db4091f8d3cd3b86b7b0f31af6f2ff71bb5a974f49e9d33d9d96`.

MAE means adverse excursion **from entry**, not floating peak-to-trough
drawdown. Using raw MAE as a stop can trigger at equality. Whole endpoint-bar
OHLC is conservative and cannot establish intrabar extreme ordering. Every
complete origin has a positive hindsight opportunity in this corpus, so it
cannot by itself qualify realistic abstention under uncertainty.

## Parameterized rules now implemented

```text
dSL = a_sl * MAE_reference + b_sl * sigma + buffer_pips * pip
dTP = a_tp * TP_reference - b_tp * sigma
SL = entry_quote - direction * dSL
TP = entry_quote + direction * dTP
R = rho * current_balance
L1 = native_units(q=1) * (rounded_SL_distance + cost_per_unit + funding_reserve)
q = c * min(1, R/L1)
```

Round SL away from entry, TP toward entry on the price grid. Reject invalid
or insufficient distances relative to both entry and liquidation-side
quotes; do not inflate a contracted TP. Quotes are already known at this
fill-time construction step. A consumer must not use next-open quotes
prematurely at decision time. Positions require separate calendar/gap
revalidation and an early-close policy.

The adapter is explicitly EURUSD with a USD account. It uses actual native
Sizing integer-unit conversion, leverage50 and margin basis. Loss costs
include market/stop slippage and both commissions. Funding reserve is an
explicit nonnegative input; synthetic base=0. Spread is already in the
entry/liquidation quotes, not added again as a cash fee. This does not revise
the earlier oracle's declared cost convention or assert account replay parity.

| Coefficient | Engineering baseline | Proposed search range |
|---|---:|---:|
| a_sl | 1 | 0.5..2 |
| a_tp | 1 | 0.5..2 |
| b_sl | 0 | 0..1 |
| b_tp | 0 | 0..1 |
| buffer_pips | 1 | 0..5 |
| rho | 0.005 | 0.001..0.01 |
| c | 1 | fixed control until uncertainty calibration |

These are **simulation controls/ranges, not optimized or approved live
risk settings**. References currently offer median and p75. The sigma
estimator and window are not selected by this work; the engineering grid
supplies synthetic values. No change to feature selection or normalization.

The selected rules may differ by horizon, but q retains the same meaning.
Do not hardcode that long horizons deserve larger positions. With a common
loss budget, their larger stop distances can require smaller q. The original
per-unit profit/drawdown metric cannot optimize size: scaling both cancels q.

## Engineering evidence, not strategy performance

3,456 synthetic order configurations passed sizing/geometry, with17,280
native protection witnesses: TP, SL, both touched (SL first), adverse gap
and no touch in both directions.18 additional controls cover NOP, minimum
size, funding and gaps.70 focused tests pass. JSON roundtrip reproduces
config, complete case ledger and report exactly after canonicalizing axes.

Maximum margin was999.988 on a10,000 balance; modeled stop loss never
exceeded the supplied risk budget. The explicit adverse-gap fixture turns
a nominal loss near10 into about90.64: no claim of guaranteed loss bounds.

**Compatibility finding:** your pinned BarEnv._levels clips to multiplier
bounds and potentially `effectiveness_c * sigma_rule`; check_order enforces
that cap too. New absolute levels must not be silently routed through it.
This delivery does not edit your branch or remove those restrictions. We
need a tested absolute-level interface before whole-account optimization.

## What to share and what to keep separate

Share the contract, reference provenance, geometry/sizing formulas, cost
units, synthetic fixtures and same JSON parameters when doing paired work.
Keep separate the learned output semantics, ZigZag demonstration targets,
uncertainty calibration, player rewards and action arbitration. These have
different meanings for pure RL and expert-guided learning.

Please use this packet directly rather than repeating the ZigZag searches.
The useful next cross-review is the absolute-level consumer interface and
position-dependent early-close semantics, especially different positions
under identical market signals. Closing optimum stays the best executable
exit between now and the SL event, not the entry oracle's selected pivot.

## Reproducibility index

- [300-generation results](../work_plan/123_ZIGZAG_K3_300_RESULTS.md).
- [Parameter-effect audit](../work_plan/124_ZIGZAG_PARAMETER_EFFECT_AUDIT.md).
- [Risk profile plan](../work_plan/125_ZIGZAG_RISK_PROFILE_PLAN.md) and
  [measured profiles](../work_plan/126_ZIGZAG_RISK_PROFILE_RESULTS.md).
- [Parameterization plan](../work_plan/127_SHARED_RISK_RULES_PLAN.md) and
  [executed engineering report with commands](../work_plan/128_SHARED_RISK_RULES_RESULTS.md).
- [Full TRAIN aggregates](../audits/evidence/zigzag_risk_profiles_20261028/PROFILES.json).
- [Consumable JSON](../audits/evidence/shared_risk_rules_20261028/CONFIG.json),
  [verification report](../audits/evidence/shared_risk_rules_20261028/REPORT.json),
  [complete synthetic ledger](../audits/evidence/shared_risk_rules_20261028/CASES.json.gz).

Next in the master plan: full absolute-level/early-close integration, paired
account objective tests, coefficient optimization (population20/gen300),
then qualified model-specific floating selection. No RL job was launched.
