# External audit repair plan

Date: 2026-10-29. Scope: plans 127-137, before model-specific floating selection.

## Evidence and decisions

Satoshi's dynamic-account, close/hold and shared-risk verdicts, and Retsu's
dynamic-account review independently corroborate the archived numerical runs.
They do not qualify the report generator as a learned or causal forecaster.
Keep the original archive and source identity immutable. No financial training,
feature selection or new parameter optimization is authorized by this repair.

## Requirements and tests (declared before implementation)

| ID | Requirement | Failing witness / acceptance |
| --- | --- | --- |
| AR01 | Reference protection equals native arithmetic | Both directions at exact TP plus through boundary; SL wins ambiguous bars |
| AR02 | Deadline liquidation uses deadline spread | Both directions with a spread jump at the forced-flat timestamp; explicit terminal spread |
| AR03 | External calendar owns the report deadline | A self-consistent shortened report must be rejected against the independent deadline |
| AR04 | External clock owns availability | Stale and future reports rejected by cached and direct consumers, including warmed bid caches |
| AR05 | Cache never substitutes another position/report | Warm cache, then change entry/geometry, report content or costs; reject or recompute |
| AR06 | Repairs preserve unaffected historical executions | Replay every archived account with new code; compare full accounts, not only aggregate metrics |

Architecture: the native engine remains unchanged. The report declares terminal
spread; the controller requires external decision time and deadline. The panel
obtains both from the engine, validates before cache access, and keys values by
position, report and funding identity. No new learned inputs or state feedback.

## Scientific corrections, not silent changes to old experiments

- Twelve panel cells represent only three price draws plus paired transformations.
  Volatility envelope and spread were confounded. Confirmation was not untouched.
- The empirical frontier contains exposure variants, not 632 independent behaviors.
  Shared entry/close uncertainty penalty is an explicit coupling, not two genes.
- Zero realized SL exits does not demonstrate absence of SL risk: the future-aware
  closer preempts many SL paths. Early-close benefit dominates rotation benefit.
- P1's mean edge over the 3h control is small and concentrated. Neither P1 nor
  any other representative is promoted as a robust winner.
- A profitable unilateral deviation is not necessarily a Pareto improvement.
  SL sizing bounds modeled stop losses, not gap losses. Six daily returns do not
  support a reliable annualized Sharpe inference.

## Next experimental version

After repair replay, freeze a new panel identity with spread crossed independently
against volatility, independent price draws (at least ten per family), mirrored
pairs, and separate report-quality levels. Include delayed/missing/biased reports,
correlated path-shape errors, missed SL modes, gap-before-CLOSE and terminal cost
changes. Freeze expected behavior before observing policy results. Compare fixed
controls on identical paths and exposure before any new DEAP search. Do not reuse
the old confirmation as an unseen evaluation. Keep population >=20 and generations
>=300 for a subsequently authorized full search, without changing old receipts.

Shared contract: leverage 50, rel_volume 0.1 applied once, no daily entry cap,
12h reopening wait and forced flat 12h before weekly close. State and evidence
will record the exact completed tests and the next executable action.
