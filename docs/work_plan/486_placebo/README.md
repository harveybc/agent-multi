# 486: Does rejected-BUY timing add value beyond waiting in cash?

Scope: seasonal/no_vol_side, the 485 arm for which the hypothesis arose.
Matched archive, not independent confirmation. The BUY cash rule compounded
to +12.14% versus -17.96% staying available. There is no robust evidence
that its within-week choice of pause times accounts for the gain.

## Design

Each placebo uses both positive-net filters, q=1, one slot, original weekly
calendar and execution costs. For each observed week, preserve the number,
order and elapsed durations of the actual pauses. Each substitute start is
flat on its **own** path, in the same UTC six-hour time-of-day band, with
the same count of blocked eligible origins as the original interval. Never
pass the known flattening deadline or interrupt a position. No booked return
is used to choose a schedule. Dynamic programming counts feasible suffixes
and samples entire admissible schedules uniformly (with replacement).

Cross two controls: same weekday required vs unrestricted within the week;
and all flat origins vs flat origins where neither an order nor a fallback
would execute (selected side's score nonpositive; no BUY requirement).
The latter prevents taking away an otherwise immediate entry just to start
the placebo. Run 8,192 draws per year per design (131,072 annual paths).
Also report the subset within 5% of the actual annual order count **and**
hours between trade entries and exits. Quartiles are not used for selection.
The four-year summaries combine independently drawn annual paths (65,536
compounds per design/subset). Percentages equal or above actual include ties.
These are conditional reference distributions, not formal out-of-sample
significance tests; exposures are matched retrospectively. Alternative
schedules depend on the observed weekly budgets and feasibility; they are
not deployable random policies. They do not test the value of selecting
which weeks or how many times to pause.

| Exposure-matched calendar | Pause eligibility | Median compounded % | 5th / 95th % | % at least as good as actual |
|---|---|---:|---:|---:|
| Week + UTC band | Any flat origin | 6.80 | -5.78 / 21.98 | 27.62 |
| Week + UTC band | No trade otherwise | 14.56 | 8.15 / 21.90 | 71.67 |
| Same weekday + UTC band | Any flat origin | 10.70 | 0.48 / 17.46 | 38.33 |
| Same weekday + UTC band | No trade otherwise | 12.53 | 7.88 / 17.55 | 55.15 |

2023 alone looks favorable against broad timing (1.77% of matched schedules
equal or exceed actual). With both restrictions it is 49.63%, and only two
of the 17 pauses move on the median path (16 feasible full-year schedules).
The stricter designs have less power to separate triggers. In 2022, the
broad matched percentile comparison is 24.89% at least as good as actual.
Do not infer that any random cash pause earns a positive forward return.
Next independent test: freeze a rule that learns a weekly pause budget from
prior data and compare prospective schedules without the evaluated year's
observed pause counts, alongside no pause, on new chronological data.

[Annual results](annual.csv) include realized DD, actual/median exposure,
eligible matched draws and log10 feasible schedule counts; [compound results](compound.csv)
include both all-draw and exposure-matched distributions. Replaying actual
starts reproduces 485 entries, exits, balances, annual return, realized DD,
order counts and hours. Uniform weighting is checked on enumerable fixtures.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.role486_panel "$RUN477" "$HISTORY397" "$MARKET" "$BUNDLE" docs/work_plan/479_gross_cost docs/work_plan/485_cash_wait "$PANELS"
# The two commands below are independent; merge their disjoint outputs when distributed.
OPENBLAS_NUM_THREADS=1 python tools/role486_placebo.py "$PANELS" "$DRAWS" --strict 0 --workers 6
OPENBLAS_NUM_THREADS=1 python tools/role486_placebo.py "$PANELS" "$DRAWS" --strict 1 --workers 8
python -m tools.role486_report "$DRAWS" docs/work_plan/485_cash_wait "$OUT"
```
