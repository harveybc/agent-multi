# Crossed panel and unattended account optimization

Date: 2026-10-29. Owner authorized execution after repairs in139.
Precedes floating feature selection and learned-policy experiments.

## Frozen design before implementation

- Calibration seed bank 50100..50109; confirmation 60100..60109. Smoke tests use
  separate banks70100/80100 and cannot consume scientific-bank paths. Three price
  families: trend, reversal and overlapping frequencies. Thirty independent
  innovations per bank; three volatility envelopes (constant/increasing/decreasing),
  two spreads (0.5/3 pips), and exact mirrored price paths are paired transforms.
  360 cells per bank, **not 360 independent observations**. Costs do not change
  bars; report quality does not change bars or costs. Multiple random shock times,
  strengths and trend reversals augment the existing three-family vocabulary.
- Five report rungs: ideal, correlated error, delayed shape, omitted reversal,
  intermittent availability. Ideal is a positive diagnostic, never DEAP fitness.
  Correlated errors use phi=0.6 as a declared stress parameter, not a fitted fact;
  independent samples and nonzero shared error prevent antithetic cancellation.
  Delayed shape has three bars of lag; omitted reversal extrapolates the observed
  prefix and understates its dispersion. Missing reports cause no entry/HOLD,
  while native SL/TP/forced-flat remain active. Stale timestamps still fail.
- DEAP calibrates on the **correlated-error** rung, all360 cells. Other rungs
  measure sensitivity, not extra fitness terms. Final confirmation runs all five
  rungs only after finalists are frozen. No use of the old confirmation bank.
- Same fixed risk geometry and11 genes as the predecessor. Entry/close uncertainty
  penalty remains explicitly shared. No new feedback features, live-data scaling,
  financial labels or TEST. No change to the account business contract.
- Objectives remain mean account return and negative worst MTM drawdown, activity
  >=3 entries/week on average. No positivity requirement for progressing from
  diagnostic controls to optimization; correctness must not mean cherry-picking.
- Population20, generations300, seeds7/19/43 sequential. Choose representatives
  across distinct objective vectors, with lexicographically smallest genes for
  identical vectors. Report all duplicates; do not call them independent strategies.

## Tests and gates

| ID | Expected property before results | Evidence |
| --- | --- | --- |
| CP01 | Full Cartesian cost/envelope/mirror grid; disjoint banks; ten draws/family | Specification tests |
| CP02 | Cost and quality changes leave prices identical; mirrors invert OHLC exactly | Generator metamorphic tests |
| CP03 | Separate worlds have different report errors; noise mean is not truth | Report identity/sample tests |
| CP04 | OHLC valid, external clock/deadline exact, terminal spread explicit | All rung contract tests |
| CP05 | Missing report never causes an entry; protections keep executing | Native full-account tests |
| CP06 | Known forced-position closes obey gap priority, SL/TP order and terminal costs | The repaired native witness suite, both directions |
| CP07 | Repeated execution/cache gives identical accounts; return and cash agree | Native control replays |
| CP08 | Restart authenticates receipts, never reuses another code identity | Driver smoke and identity-tamper test |

The preflight runs the focused suite on the pinned code and records its XML outside
the checkout. Controls (NOP, guarded, greedy, four single horizons) then run on the
full calibration bank at every quality, sequentially with durable result files.
The ideal witness tests, not high control returns, establish implementation validity.
Failures stop with a typed state and retained evidence. No automatic retuning of
witnesses, costs, activity threshold or scenario parameters.

## Automation and reporting

Single resumable process, exclusive lock, source/config/version manifest, fsynced
candidate receipts, bounded caches, four persistent single-thread CPU shards and a
service memory ceiling. Only scenarios within one candidate are concurrent;
candidates and experiments remain sequential. Native results must match serial
evaluation exactly, including ordered aggregation and full ledgers.
Completed control cells and seed results are authenticated before reuse. An
interrupted seed repeats deterministic evolution using the receipt cache, without
re-evaluating completed candidates. Heartbeat includes completed controls/candidates
and elapsed time. ETA uses measured full-panel evaluation cost; it is not a promise.

Frozen-control quality tables report return, worst DD, total orders, buy/sell,
directional win counts, early closes and activity. Final tables add paired finalist
comparisons and action-sequence diversity at fixed size. Weekly short-sample native
Sharpe is diagnostic only; no significance claim treats mirrors as independent.

No global optimum or learned report sufficiency follows from this finite panel.
This version does not yet solve multi-week global opportunity scheduling; it tests
the audited account mechanism on a less favorable, independently crossed panel.

## Pre-search correction

The first end-to-end smoke consumed two paths of the originally proposed 20100
confirmation bank. Before any full DEAP generation, the initial service was
stopped during controls (7/35 completed). Its identity9dfeae88 remains separate
and is not scientific evidence. Both full banks were replaced with the unused
50100/60100 banks above; smoke now has its own namespace and disjointness test.
No original control or candidate result is reused in the replacement run.
