# Causal history experiment and unattended execution

Date: 2026-10-29. Parent: plans 144--147. Priority: before feature selection.

## Requirements and tests defined before implementation

The completed diagnostic has 27/27 receipts, 184 passing tests and no failed
units. It does not qualify a learned reporter. The next experiment tests a
controller-side history penalty, not a replacement for market features.

| ID | Requirement | Test / evidence |
| --- | --- | --- |
| CH01 | Reports and quality use only completed market prefixes | Mutate every later OHLC; reports, quality and decisions unchanged |
| CH02 | Entry and close reliability are independent controls | Four arms: neither, entry only, close only, both; independent close uncertainty |
| CH03 | Additional close penalty cannot create a new close | Known value matrix, both directions, exact zero-penalty equivalence |
| CH04 | Cached causal values bind position, risk, report and bytes | Tampered value/report/identity rejection; direct parity |
| CH05 | Cold and restarted runs reproduce | Sealed per-stage outputs, source identity, full receipt chain, cold account equality |
| CH06 | Position, costs, calendar and protections remain native | Existing boundary/account tests plus full native ledgers and counterfactuals |
| CH07 | No cost-free or hindsight fitness | Prefix-only 32-path reporter, real synthetic spread/funding; separate new seed banks |
| CH08 | No silent continuation after an engineering failure | Durable FAILED state, bounded process restart, independent jobs continue |
| CH09 | No human approval between executable stages | Tests -> controls -> conditional search -> frozen confirmation -> counterfactuals -> report/audit request |
| CH10 | Reports distinguish experiment from qualification | Negative results and no-activity outcomes are terminal evidence, not model qualification |

## Mechanism under test

Use the existing prefix bootstrap reporter with median or zero drift. At each
completed bar, mature old forecasts, then issue the new forecast. For each
1/3/6/12/24-hour lead retain the last 64 matured errors, with minimum 20.
No future prices determine the ledger or its availability.

For a bid with unit stop risk R and horizon h, subtract
`entry_history_weight * historical_MAE(h) / R` from gain/risk. For an open
position subtract `close_history_weight * historical_MAE(h) / unit_R` from
the existing close-versus-wait advantage. These are empirical reliability
penalties, NOT calibrated joint-path confidence intervals. A required history
that is not ready forbids new entry and voluntary close; native SL/TP and forced
flat still execute. At late-week horizons use the largest tracked lead no
longer than remaining support, recorded in the evidence. No new input clipping,
rolling z-score or agent-state feature is introduced.

This can suppress useful closes as well as harmful ones. That possibility is
why fixed-position SL/TP cohorts and whole-account no-close controls are
mandatory. A positive result is not assumed. The closer's uncertainty coefficient
is independent from the entry coefficient. Signal comparison fixes size at 1;
the existing rel_volume=0.1 is applied once by the native order builder.

## Execution and selection

Separate median and zero-drift lanes are preregistered; they are not chosen
against previous calibration coverage. Each has independent output identity.
Calibration uses new seeds starting 150100; confirmation starts 160100; smoke
uses 250100/260100. Both lanes share draws for paired comparisons. Calibration
uses 36 transformed accounts per lane; confirmation uses 108. This is a bounded
engineering study, not an exhaustive market or learned-adapter qualification.

Run the four history arms, no-close and NOP controls first. A new exploratory
DEAP study (population 20, generations 300, seeds 7/19/43) is conditional on
tests passing, at least one active history arm improving mean calibration return
over its corresponding no-history control without increasing worst DD, and
mean activity >=3 orders/week. Failing this rule produces the complete negative
report automatically, without retuning gates or manufacturing a winner.
This conditional study is NOT the final R6 optimization: full PC1--PC6 and a
learned adapter remain separate acceptance requirements from plan 144.

If eligible, optimize fixed-size controller genes, freeze up to five Pareto
representatives on calibration, then evaluate those plus all controls on
confirmation. Always report cost, directions, activity, win rates, Sharpe,
return, DD, no-close differences and native close counterfactuals. Confirmation
never selects a new winner. The final report and audit request are generated
even when there is no eligible candidate or improvement.

Two independent, resumable lanes can run on separate CPU hosts, with memory
ceilings and no GPU claim. Transient process restarts replay deterministic
search order against existing receipts. Integrity or test failures stop the
affected lane with explicit evidence. Scripts cannot safely invent a code fix
or change a scientific criterion; those failures require engineering work,
not another user approval.

Automatic financial RL training, feature floating selection, final strategy
promotion and external audit approval are excluded from this diagnostic queue.
