# TRAIN risk profiles after K=3 refinement

## Placement and scope

This substep follows plans 122-124 and precedes strategy qualification and
phase 2 model-specific floating selection. It consumes the completed 20x300
K=3 champions, without rerunning DEAP, modifying the indicator, or reading
VALIDATION/TEST. Historical receipts and opportunity arrays remain immutable.

## Requirements and acceptance design

1. Authenticate the source spec, input and code pins, result envelopes and
   each opportunity array. Refuse changed identities or malformed support.
2. Describe each horizon and direction separately. Include all complete
   origins, positive side-specific opportunities, and positive winning-side
   opportunities. These are overlapping hypothetical trades, not account
   orders or estimates of live profitability.
3. Report both unrestricted oracle support and the subset whose selected
   endpoint releases before or at the existing conservative forced-flat
   boundary. Do not choose a replacement endpoint to improve this subset.
4. TP reference is signed target-minus-entry distance. SL reference is maximum
   adverse excursion FROM ENTRY (MAE), not peak-to-trough floating drawdown.
   Reconstruct MAE and net profit independently on every retained side/path.
5. Summarize n, mean, population std, min, p10, p25, median, p75, p90, p95,
   max; add year strata, joint TP/MAE correlation and zero-MAE counts.
   Empty populations yield null summaries, never fabricated zeros.
6. H1 whole endpoint-bar OHLC is conservative and does not reveal the order
   of intrabar extremes. MAE is a reference, NOT an executable SL: a stop
   exactly at MAE can be hit. No buffer, broker rounding, or stop level is
   silently selected by this descriptive report. Early-close design stays
   separate from the ideal pivot exit.
7. Margin = q * rel_volume * current balance, q in [0,1], rel_volume=0.1
   exactly once for every horizon. A pure sizing helper can consume an
   explicit monetary risk budget and loss-at-SL estimate for q=1, both in
   account currency, plus an explicitly supplied quality factor in [0,1].
   It does not estimate those inputs or claim uncertainty calibration.
   q = quality * min(1, budget / loss_at_q1). Below broker minimum -> NOP,
   never round upward. The linear-loss assumption excludes fixed ticket
   fees and nonlinear fills; native broker validation remains separate.
8. No automatic larger size for longer horizons. The per-unit DEAP ratio
   cannot identify position size: scaling profit and drawdown cancels q.

## Test design before implementation

- Known long and short paths distinguish MAE from floating drawdown.
- Mirrored prices swap directional summaries; equal positive scores are
  counted explicitly (the source selector breaks ties toward buy).
- A release after flat is excluded, release at flat accepted; no substitution.
- Censored rows stay censored; empty subsets and constant vectors are typed.
- Altered MAE/net or source hashes are rejected.
- q=1 and q=0.3 with balance 10000 give margin 1000 and 300, not 100 and 30.
- Quality reduces, never increases the risk-limited q; invalid inputs fail;
  below-minimum size becomes zero; linear estimated loss stays within budget.
- Real source hashes are unchanged after the report, deterministic rerun
  matches, and results contain aggregate TRAIN evidence only.

## Architecture and execution

`app.zigzag_risk_profiles` owns pure summaries and bounded sizing;
`tools.zigzag_risk_profiles` owns authentication, TRAIN loading, scalar audit,
and atomic output to a new report path. Existing opportunity generation,
optimization and execution code are not edited. CPU sequential processing
of 3/6/12/24h, no model fitting. No method-state validator exists in this
checkout; method state and requirement/evidence links are checked directly.

## Next use

After these summaries: propose joint per-horizon SL/TP rules, explicit
buffer/rounding and monetary risk-budget candidates, then qualify them in
the native execution and early-close scenarios. Descriptive oracle profiles
alone do not admit a trading strategy or floating selection.
