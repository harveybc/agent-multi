# TRAIN SL/TP reference profiles and bounded volume

## Completed scope

Plan [125](125_ZIGZAG_RISK_PROFILE_PLAN.md) executed on the four K=3,
population 20 / 300-generation champions. All four years of TRAIN were used;
no validation, TEST, neural training, new optimizer or indicator change.
The source arrays and receipts are unchanged.

Independent reconstruction of **134,916 directional paths** matched net
profit, MAE, floating drawdown, duration and score with maximum absolute
error **0.0**. A second full execution produced the same report SHA-256:
`aa2cb1f8f524db4091f8d3cd3b86b7b0f31af6f2ff71bb5a974f49e9d33d9d96`.
Source run identity:
`d54d51768a5ac257b7428a8e7afc8fb61bd467009dea77e8224a64c9ee824a9e`.

Machine-readable aggregates, including all quantiles, year strata, support
and code/source hashes:
[PROFILES.json](../audits/evidence/zigzag_risk_profiles_20261028/PROFILES.json).

## Directional references

The table uses positive side-specific opportunities whose selected exit
releases no later than the existing conservative forced-flat boundary.
Both directions are measured independently; these overlapping hypothetical
opportunities are **not executed account trades**. They are not restricted
to the side winning the comparison. The JSON also reports winning-side and
all-complete populations separately, including losing side opportunities.

One pip = 0.0001. Standard deviation is population std (ddof=0).
MAE is the raw adverse distance from entry, not an executable stop.

| Nominal h | Side | Opportunities | TP mean +/- std (pips) | TP median | MAE mean +/- std (pips) | MAE median | Duration median / p90 (h) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | Buy | 15,033 | 28.17 +/- 26.03 | 20.0 | 11.24 +/- 12.16 | 7.4 | 5 / 12 |
| 3 | Sell | 15,053 | 28.06 +/- 27.73 | 19.6 | 12.03 +/- 14.01 | 7.7 | 5 / 12 |
| 6 | Buy | 14,946 | 33.47 +/- 30.45 | 24.0 | 13.05 +/- 14.20 | 8.5 | 7 / 16 |
| 6 | Sell | 14,845 | 32.75 +/- 30.85 | 23.4 | 13.53 +/- 14.80 | 8.9 | 7 / 16 |
| 12 | Buy | 12,932 | 46.52 +/- 37.71 | 35.3 | 17.64 +/- 17.94 | 11.9 | 12 / 27 |
| 12 | Sell | 13,191 | 47.86 +/- 40.95 | 35.8 | 19.22 +/- 20.48 | 12.7 | 12 / 29 |
| 24 | Buy | 10,284 | 53.90 +/- 41.67 | 42.1 | 22.06 +/- 20.89 | 16.0 | 17 / 41 |
| 24 | Sell | 10,765 | 54.52 +/- 44.37 | 43.1 | 23.51 +/- 24.04 | 16.2 | 17 / 42 |

No new time cap was added. Nominal horizon constrains the median spacing of
ZigZag events, not order duration. The oracle still selects among the next
three chronological events; the calendar subset filters its chosen endpoint
without replacing it by a different pivot or a timed exit.

| h | Complete origins / 19,879 | Censored | Positive buys excluded by calendar | Positive sells excluded by calendar | Zero MAE buys / sells in table |
|---:|---:|---:|---:|---:|---:|
| 3 | 19,620 | 259 | 696 | 824 | 256 / 253 |
| 6 | 19,108 | 771 | 796 | 1,041 | 221 / 221 |
| 12 | 16,191 | 3,688 | 581 | 768 | 144 / 144 |
| 24 | 12,539 | 7,340 | 364 | 363 | 95 / 100 |

## Findings affecting the next design step

- Every complete origin has some positive hindsight opportunity. This does
  not imply that the live system should always open: oracle foresight removed
  estimation uncertainty. NOP needs explicit costs/risk/uncertainty scenarios;
  this oracle population alone cannot qualify learned abstention.
- Zero raw MAE exists in both directions at every horizon. A zero-distance
  stop is not admissible. Costs, broker minimum distance, tick rounding and a
  deliberate cushion must enter the executable rule; none is invented here.
- Stop at exactly MAE can be touched. Whole H1 endpoint-bar extremes include
  possibly post-pivot adverse movement; their ordering is unknown. Raw MAE
  is a conservative reference, not proof of stop survival in the native engine.
- Dispersion is large. Setting TP and SL to two independent means/medians
  would not preserve the actual joint path or its exits. TP/MAE correlations
  in these subsets range from -0.0205 to +0.0695; year summaries are retained
  and observations overlap, so no independent-sample confidence is claimed.
- The source tie rule selects buy at exact equality; zero positive exact
  ties occurred here. A regression exposes the rule instead of implying
  perfect directional symmetry in tied decision cases.

## Volume semantics implemented and tested

For every horizon:

```text
margin(q) = q * rel_volume * balance(t)
rel_volume = 0.1;  0 <= q <= 1
q = quality * min(1, monetary_risk_budget / loss_at_q1)
0 <= quality <= 1
```

The caller supplies `loss_at_q1` in account currency using the applicable
SL, units, leverage, currency conversion and execution costs. This is the
estimated loss for margin `rel_volume * balance`, not a second application
of `rel_volume`. The helper assumes loss scales linearly with size; it is
not a broker lot adapter and must not be used unchanged for fixed ticket
fees or nonlinear fills. Broker-minimum q is explicit; a smaller permissible
q yields zero instead of rounding upward. Gaps can exceed a modeled SL loss.

No risk fraction has been selected, and `quality` is an explicit input, not
an asserted calibrated confidence. The helper is not wired into live or
historical execution. There is no automatic volume premium for 24h over 3h.
The per-unit optimization score cannot identify volume because scaling
profit and drawdown together cancels q.

Verification examples, not trading recommendations: balance 10,000 and
q=1 yield margin 1,000; q=0.3 yields margin 300. Leverage is not applied a
second time to that margin: it belongs in the separate units/notional and
loss-at-SL calculation.

## Verification and reproduction

47 focused tests passed (18 new risk-profile cases plus 29 existing search
and extension cases). Targeted Ruff check passed. RED first: missing new
module, followed by implementation and GREEN. Native trading tests were not
run because this step changes no execution path.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/zigzag_risk_profiles.py --root <completed-k3-run> \
  --output <new-report-directory>/PROFILES.json

CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/unit/test_zigzag_risk_profiles.py \
  tests/unit/test_zigzag_opportunities.py \
  tests/unit/test_zigzag_extension.py
```

An initial profiling pass repeatedly decompressed NPZ members inside the
scalar loop. It was stopped before writing a report and replaced by one
materialization per horizon. Both completed runs use that correction; source
artifacts were never changed.

## Next step in the master plan

Construct configurable joint SL/TP proposals from these TRAIN references,
with explicit cushion/rounding and risk-budget candidates. Test the resulting
orders and position-dependent early-close rule in the native simulator before
optimizing the account strategy. Early-close optimum remains the best
executable exit between the current decision and the SL event, not the entry
pivot optimum. No strategy or floating-selection admission follows from this
descriptive report alone.
