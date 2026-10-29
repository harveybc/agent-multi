# TRAIN ZigZag DEAP: measured results

Date: 2026-10-28. Subplan: [114](114_TRAIN_ZIGZAG_OPPORTUNITY_OPTIMIZATION.md).
Status: 12/12 searches complete; 30 individuals, 40 generations, seeds 7/19/43
for each of four scales. Runtime: 222.91 seconds on CPU.
No GPU, neural fit, VALIDATION or TEST evaluation. This is the requested
retrospective opportunity study, not a qualified trading strategy.

Evidence: [machine-readable results and audit](../audits/evidence/zigzag_opportunity_20261028/RESULTS.json).
Identity: `80246f27faf7ed91f76a5575857d422a3fc9801ff33a4b9694323965424151e0`.

## Parameters and opportunity performance

Fitness F is the mean per-origin max(NOP=0, buy score, sell score), where score
is net unit profit / (floored intratrade drawdown upper bound * hours).
Unit: inverse hours. It is neither Sharpe nor a percentage return.

| Nominal h | Depth (H1 bars) | Deviation (pips) | Backstep (H1 bars) | F | Median pivot spacing h | Mean selected holding h | Mean selected net pips |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 3 | 5 | 1 | 0.27239645 | 3 | 5.0302 | 28.0487 |
| 6 | 4 | 10.7 | 2 | 0.24521714 | 5 | 6.5938 | 33.8297 |
| 12 | 9 | 21.2 | 3 | 0.18087886 | 9 | 11.4553 | 45.3881 |
| 24 | 12 | 21.2 | 11 | 0.14269387 | 17 | 16.2727 | 55.1667 |

All four use the same 19,879 eligible hourly origins from 24,895 admitted bars
in TRAIN 2020-2023. 5,016 origins excluded by calendar/path endpoints, not by
candidate fitness. 209 contiguous session chunks. No missing-hour interpolation.
Costs are the inherited assumed spread/slippage/commission/financing schedule;
they have not been newly validated against a broker.

## Diagnostics that matter

1. Deviation is empirically neutral at all four finalists: values 1, 5, 60
   and 120 pips produce identical fitness and pivot counts. The reported
   deviation is a retained genotype, not an identified economically optimal
   threshold. This needs interpretation before adopting these as profiles.
2. The fitness rewards speed. Median spacing moves to the lower edge of the
   declared [h/sqrt(2), h*sqrt(2)] scale band: 3/5/9/17 hours, not exact
   3/6/12/24-hour holding durations. Do not silently relabel 17h as 24h.
3. These are best-found parameters, not certified global optima. Different
   seeds disagree especially at 24h. See the seed table below.
4. The 24h winner has backstep=depth-1, an active constraint. The 3h winner
   has the minimum backstep. Boundary sensitivity is not resolved by this run.
5. The adverse-excursion denominator gives much larger fitness than drawdown
   because gains can retrace without crossing entry. It was reported, not
   used to change the selected winner after seeing results.

| Nominal h | F seed 7 | F seed 19 | F seed 43 | Unique genomes | F using adverse-excursion denominator |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.27239645 | 0.27239645 | 0.27239645 | 841 | 1.88291866 |
| 6 | 0.23182211 | 0.24521714 | 0.24521714 | 930 | 1.75006194 |
| 12 | 0.17951862 | 0.18087886 | 0.18087886 | 973 | 1.34727352 |
| 24 | 0.13808761 | 0.14269387 | 0.13301194 | 1013 | 1.09386728 |

## Side coverage and yearly fitness

Buy/sell/NOP percentages below are origin-wise hypothetical choices, not
simultaneously executable trades. Negative candidates are retained in NPZ.

| h | Buy % | Sell % | NOP % | F 2020 | F 2021 | F 2022 | F 2023 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 50.209 | 49.469 | 0.322 | 0.273899 | 0.269644 | 0.279124 | 0.266891 |
| 6 | 50.299 | 49.454 | 0.246 | 0.251740 | 0.240486 | 0.253456 | 0.235152 |
| 12 | 49.399 | 50.163 | 0.438 | 0.184473 | 0.173078 | 0.188942 | 0.177019 |
| 24 | 49.338 | 49.857 | 0.805 | 0.142405 | 0.139692 | 0.149032 | 0.139626 |

## Non-overlapping retrospective reference

Dynamic programming maximizes additive net profit over the finalist's fixed
unit intervals, permitting waiting and only one open interval. This is a
different diagnostic objective from DEAP's per-origin score. No balance or
margin series, variable sizing, SL/TP strategy or learned actions is simulated.

| h | Selected intervals | Net pips, sum | Buy % | Winning % |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 4084 | 122037.20 | 50.318 | 100.0 |
| 6 | 3121 | 111779.45 | 50.368 | 100.0 |
| 12 | 1669 | 87842.40 | 50.270 | 100.0 |
| 24 | 1088 | 73554.85 | 49.908 | 100.0 |

100% winning is a consequence of a future-informed positive-profit interval
selector with NOP and no minimum-trade quota. It is not evidence that a model
can achieve that win rate. Pivots fill ideally at H1 extrema, not demonstrated
tick-level executable fills. Drawdown is an adverse intrabar-order upper bound.
No account Sharpe is fabricated from these unit opportunity scores.

## Verification and replay

- 47 focused tests passed (35 exact-game tests plus 12 new opportunity tests).
- Independent scalar audit reconstructed 2,048 real-data side/path records:
  net, costs, financing, duration, MAE, drawdown and score; max error 0.
- All 480 generation records verified; fixed-unit schedules do not overlap.
- Input, engine, code, records and receipt digests retained. No sealed source
  campaign was modified.

Operator commands (private paths supplied locally):

```bash
python tools/zigzag_opportunity_deap.py prepare --out <new-output> \
  --source-spec <existing-private-spec> --engine-root <pinned-engine>
python tools/zigzag_opportunity_deap.py run --out <new-output>
python tools/audit_zigzag_opportunity_results.py --root <new-output>
```

The runner chains all 12 units. Completed units require identity and content
hash equality; an interrupted unit restarts from its deterministic seed.

## Next discussion

Interpret the neutral deviation parameter and the lower-band horizon choices
before deriving SL/TP/volume profiles. Keep the whole opportunity map, not just
winning entries, for that analysis. Strategy qualification and phase 2 floating
selection remain subsequent steps; this result does not authorize their launch.

