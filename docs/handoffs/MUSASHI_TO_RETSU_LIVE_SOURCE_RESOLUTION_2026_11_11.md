# Selected69: Resolve Forward Sources Independently Of Historical STL

Date:2026-11-11. Supplements the selected69 and consumer-activity handoffs.
This is an actionable integration disposition, not a receipt of a running feed.
No broker runner, account, position or acquisition service was changed here.

## Decision

Preserve the69-input actor and its frozen producer/scaler recipe. Build current
real observations through the existing forward adapter. Do not make the demo
wait for the lateral STL experiment's2019 prehistory or a2022 missing quote.
Those historical failures are not proof that the same features cannot be
produced now. Conversely, historical producer parity does not prove a live feed.
STL is not part of the deployed candidate's recipe; do not add it implicitly.

Verified artifact hashes, rechecked locally on this date:

- Actor: `8ab11f607dc18c53ce45b34dc628e21272767702d60e0bfbe272620761de4e40`.
- Consumer activity: `fdc95a0cdabf9a5327c991187f62c8be928476f385551c7e8db3972045167d6b`.

Locations remain those in the earlier handoffs. The70-input actor is a separate
reference; do not pass69 inputs to it. Historical heartbeat files inspected
here are not a current fleet-status check.

## Required Production Paths

| Dependency | Production path | Acceptance requirement |
| --- | --- | --- |
| EURUSD and six FX crosses | Existing authorized terminal/feed, with real bid observations for EURUSD,GBPUSD,USDCHF,USDJPY,EURGBP,EURJPY,AUDUSD | Verify actual symbol coverage, timestamps, bid side, complete closed bars and available history; an installed terminal alone is insufficient |
| `fac.fx_f1.ret_4`, `fac.fx_f1.ret_24` | Original frozen factor over the six crosses | Exact producer/input order and full-precision fitted state; unchanged return and aggregation rules |
| `wv.usd.A4` | Original USD-strength producer and causal wavelet | All four actual legs, correct window and endpoint; no missing-leg zero |
| `px.hours_since_session_open` | Original calendar producer with current session/closure contract | Explicit2026 calendar, weekly/DST and closure semantics; do not interpret a feed outage as a scheduled close |
| `act.p_hit` | Recovered full consumer estimator, index0, on its14 ordered lagged inputs | No historical OOF fold model or averaging in forward use; no refit or substitute global estimator |
| Daily releases and COT | Original series/report definitions with actual release and retrieval evidence | Last genuinely released value under the frozen as-of rule; not a newly fabricated hourly publication |
| Kalman/GMM and deterministic features | Original recovered fitted parameters and causal producers | Restore producer state or replay enough real history; verify restart/prefix parity |

The admitted historical calendar covers2020-2025. Its2019 extension belongs to
STL recovery; its2026 extension is a distinct forward-feed task and is required
now. Neither can be assumed from the existence of the other.

### Activity's Transitive Inputs

Preserve this exact order from the authenticated consumer receipt:

```text
cal.utc_dow_cos@0
cal.utc_dow_cos@1
cal.utc_dow_cos@120
cal.utc_dow_sin@24
cal.utc_dow_sin@48
dly.dax.ret_5d@48
dly.spx.ret_1d@48
dow_cos@0
dow_cos@1
dow_cos@120
dow_sin@48
ev.tokyo_fix.hours_since@8
ev.tokyo_fix.hours_to@8
px.hours_since_session_open@2
```

The suffixes above are observed-row lags, not new LOG/DIFF encodings. Reproduce
the original lag builder. DAX and S&P500 are acquisition dependencies even when
absent from the actor's direct feature list. An index CFD, ETF or futures price
is not automatically the original index series. Do not bypass missing inputs
because the classifier implementation happens to accept NaN.

For forward rows after the frozen fit scope, use the full estimator. Historical
replay retains its recorded fold selectors. The recovery has reproduced31116
historical consumer scores exactly; current input acquisition remains separate.

## Bounded Execution Order

1. Inventory the actual authorized source and terminal: seven FX symbols,
   quote side, timestamps, available history, latest closed bars and release
   dependencies. Report missing access explicitly without exposing credentials.
2. Collect real current observations and real warmup history into a NEW forward
   snapshot. Persist publication/retrieval times and producer state. Do not
   overwrite the sealed historical store or pretend its2024 endpoint is live.
3. Extend the actual calendar through the current forward period and run the
   frozen producers, including activity's transitive inputs. Keep the original
   fixed TRAIN scaler, transforms, column order and13-observation stride2 grid.
4. Check the source overlap, batch/incremental output, restart recovery and final
   tensor/action parity. Source migration is a versioned acquisition change,
   not a requirement that two different brokers have identical prices. Quantify
   disagreements and resulting policy sensitivity before accepting migration.
5. Publish one actual current-hour complete observation and real actor decision
   in the isolated shadow account. Its receipt must include source and model
   identities, newest bar close, each producer's age and any refusal reason.
6. Continue hourly acquisition/inference. Paper/demo routing follows the existing
   route authorization and native protection checks. No real-capital routing.
   Candidate succession still requires the agreed common forward week and flat
   position transfer; historical replay does not count toward that week.

Bound retries by the scheduled decision deadline and checkpoint acquisition
progress. A failed source must not cause an endless retry loop. Reject that
model's new-entry decision when its required observation is unavailable; do not
report the refusal as learned NOP. Existing positions retain native protective
SL/TP, forced-flat and outage handling independent of model inference.

Freshness is measured against the latest completed eligible market bar: no more
than one hour old at decision time. Daily/COT inputs keep their true release
ages under the recipe; they are not required to publish every hour. Restarts
must not silently reset recursive filters or shrink warmup to25 raw rows.

## Alternatives If A Source Cannot Be Sustained

Preferred: one qualified continuous FX source, ideally the execution broker,
plus the original release series. MT5's official APIs expose historical bars
and bid/ask ticks; actual broker availability must be measured, not inferred
from API existence. [Bar API](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py),
[tick API](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksrange_py).

Second: qualify another real feed with overlap and forward shadow evidence.
Failover needs its own approved mapping and producer-state handling. Do not
switch providers inside a feature window without a tested source contract.

Third: create a separately identified deployable-input candidate, remove every
dependent feature/branch, and repeat the matched selection/refit and forward
comparison. Loss of USDCHF affects the factor returns and USD wavelet input;
loss of calendar or activity parents also affects `act.p_hit`. Compute the full
dependency closure before deciding a removal list. Removing one source is not
necessarily removing one tensor column. No automatic promotion or deletion of
the existing champion is authorized by this fallback design.

Synthetic prices, stale quotes relabeled current, interpolation and missing-value
zero fills are not alternatives. Neither is substituting SPY/ETH models for this
EURUSD policy. Keep acquisition/integration with Retsu; Musashi's original phase2
actor qualification remains independent of the historical STL recovery.
