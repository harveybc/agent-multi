# Multiscale strategy execution: completed acceptance run

Owner: Musashi. Local date: 2026-10-26, America/Bogota.
Producer `d7ede10e`; execution engine `cd6489f`; independent verifier and
expanded mutation suite `74fbded5`. This supersedes the missing-dimensional
coverage statement in report 96 for the finite grid specified below.

## Completed results

| Measurement | Result |
|---|---:|
| Complete Cartesian market cases | 47,628 / 47,628 |
| Failed cases | 0 |
| Prescribed command/size/protection checks | 656,736 |
| Flat-state alternative comparisons | 95,256 |
| Long-state alternative comparisons | 92,676 |
| Short-state alternative comparisons | 92,808 |
| Natural positioned states not reached after entry | 5,028 |
| Focused regression/integration tests | 103 passed, zero skipped |
| Grid execution time (excluding final file hashing) | 471.29 s |
| Peak service memory | 4,372,807,680 bytes |
| Service restarts / exit status | 0 / 0 |
| Financial fits / TEST access | 0 / none |

The run used six single-thread CPU workers under an 8 GiB limit; no GPU.
Serial and six-worker smoke execution matched exactly on all 120 cells before
the numeric repair; corrected parallel results are independently reconstructed
from the full closed run. Closure and independent verification receipts are
bound by the [public evidence manifest](../audits/evidence/multiscale_strategy_20261026/MANIFEST.json),
not by a verbal PASS declaration. Independent reconstruction verified all
47,634 closed artifacts and balanced marginal counts for every axis.

## Systematically crossed dimensions

Every combination, not a sample, of the following declared finite axes ran:

| Axis | Values |
|---|---|
| Drift before AND after | -0.00045, -0.00015, -0.00005, 0, +0.00005, +0.00015, +0.00045 price units/bar |
| Innovation amplitude before AND after | 0.00002, 0.00015, 0.00060 |
| Transition duration | 0 (abrupt), 4, 12 observed bars |
| Oscillation period | 8, 24, 72 observed bars |
| Post-decision holding/comparison horizon | 1, 4, 12, 24, 48, 72 observed bars |
| Costs | configured and zero-cost diagnostic |

Total: 7 x 7 x 3 x 3 x 3 x 3 x 6 x 2 = 47,628.
Transition starts at bar 32; phase is zero. Each path has 600 observed warmup
bars and 240 target bars, with weekends absent from the observed-bar clock.
The prior phase/location grid is separate evidence, not silently multiplied
into this grid. Positive OHLC and deterministic/prefix-local construction are
tested. Both innovation amplitude and realized increment standard deviation
are recorded; they are not treated as interchangeable.

Two decision anchors bracket each transition: two bars before its start and
two bars after its end. Horizons are independent of oscillation period and
transition duration. Calendar/protective closes override a longer horizon.
The 5,028 states that did not remain open are recorded individually: 3,732 SL
and 1,296 TP closes, with 2,580 long and 2,448 short states. Every case actually
entered; none was an entry refusal. They are not counted as successful
open-position tests. The broker-event fixtures separately exercise
both directions' HOLD, early close, SL, TP, both-hit, gaps and forced-flat paths.

## Outputs and execution semantics

Command is categorical: NOP/HOLD, OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT.
There is no continuous scalar bucketed into five commands. Only legal commands
are unmasked; the same business engine owns position and restrictions. No
agent-state feature vector is supplied to the encoder by this validation.
Both open scores may be positive; the larger wins. Exact score ties use the
declared first-index rule. NOP while positioned means HOLD, not CLOSE.

For each direction the JSON contains `tp_fraction`, `sl_fraction`,
`order_fraction`, each in [0,1]. This run uses (.5,.5,1) on both sides, the
explicit midpoint CONTROL. Let d=+1 for long and -1 for short, S be volatility
available at decision close t, R the causal range scale and C the current close:

1. Intersect the configured TP/SL multiple ranges with the broker minimum
   distance / S and effective maximum distance / S (with the pinned numerical
   boundary guards). An empty range means no executable entry.
2. Map each directional fraction linearly within its own feasible interval.
   TP = C + d * tp_multiple * S; SL = C - d * sl_multiple * S.
3. At next open, recheck that SL and TP straddle the fill price and satisfy the
   minimum distance. A gap cannot silently create an invalid order.
4. Capacity = floor(current_balance * rel_volume * leverage / fill_price) for
   margin sizing; units = floor(order_fraction * capacity). q=1 uses the full
   configured rel_volume (0.1); q=0 submits no order. Below-minimum size is
   refused rather than increased. Required margin must remain affordable.
5. Commands execute at the next open. Existing gap protection has priority;
   then agent close/entry, intrabar protection (SL first if both hit), margin
   events and mandatory calendar flat. Daily cap and reopening delay remain
   hard rules, not optimizer genes.

Each reachable state receives literal prescribed commands separately from the
counterfactual ranking experiment. TP/SL, units, direction and fill bar are
reconstructed from config and raw data. Spread, commission, slippage and
rollover charges are independently recomputed, without the engine cost helper.
Native tests cover q endpoints, invalid outputs, missing windows and real policy
integration; no training is used to make a witness pass.

## Two numeric errors found and corrected

The first expanded run (`8da62c24`) recorded six failing cells and was stopped.
No failing receipt was rewritten. Their corrected replays all passed.

- Diagnostic monetary scores extremely close to one another collapsed after
  float32 conversion. Injecting exact ordinal ranks now preserves the SAME
  expected winner; this does not change prices, profits or the comparison rule.
- The production distribution's deterministic path used argmax of softmax
  probabilities, which could erase a representable difference such as logits
  0 versus 1e-8. It now uses argmax of masked original logits. Stochastic
  sampling/log-probability is unchanged. A PRE regression failed, then passed.

The mutation suite detects SELL-to-BUY routing, ignored early close, ignored
SL, calendar bypass, removed execution costs, missing cells and forged cell
status despite a zero-failure summary. The ignored-close witness now reports a
typed contract failure instead of an incidental empty-list IndexError.

## Interpretation and audit handoff

The alternative-ranking portion selected NOP/HOLD 102,194 times, BUY 41,052,
SELL 45,679, CLOSE_LONG 48,493 and CLOSE_SHORT 43,322. These are diagnostic
comparisons, NOT a financial trading performance table or learned policy.

Independent prescribed-output and ledger checks establish the tested execution
properties. Counterfactual P&L still comes from the same simulator and therefore
is not an independent economic optimality oracle. The finite grid is complete
within its published bounds, not a theorem about every continuous market path.

No new DEAP winner was selected by this run. Historical DEAP decision-use
remains rejected; the JSON interface and midpoint control are not relabelled as
an optimized strategy. The earlier C4 learned-policy failure is also unchanged.
No financial feature selection or model release is authorized by this report.

The [new audit request](100_SATOSHI_MULTISCALE_EXECUTION_AUDIT.md) asks Satoshi
to reproduce the completed evidence, attack the numeric fixes and check all
four scopes separately: execution, coverage, optimization and learning.
