# Causal history v2 launch receipts

Date: 2026-10-30. Parent:[150](150_CAUSAL_HISTORY_SEARCH_GATE_CORRECTION.md).
Runtime is pinned at `3e6450fc`; native engine sources remain the e0a1c2b export.

| Lane | Identity | Workers | Runtime tests |
| --- | --- | ---: | ---: |
| Zero drift | `a7ecac6dac2bd812992c074ef42723010e3740ed1e7b249d458799fc92f3be2e` | 2 CPU | 209 pass,0 skip |
| Median drift | `452931b1fe97840255ac60abe87911a97724053ffc7d6c1249318c5c5026d36e` | 2 CPU | 209 pass,0 skip |

Both independent hosts launched at approximately00:11 Bogota. At the initial
receipt, both had passed runtime tests and entered calibration controls. These
controls no longer veto DEAP based on performance. Each lane automatically runs
20 individuals,300 generations, seeds7/19/43, then frozen confirmation,
per-close native counterfactuals, RESULTS.md and AUDIT_REQUEST.md. The persistent
collector authenticates both completed lanes and assembles the combined report.

Per-lane resource limits:4GiB RAM,3GiB soft ceiling, no swap,200% CPU quota and
two workers. GPU is not used by this native CPU simulator. Services are enabled
at boot with lingering and bounded abnormal-process restart. Source, config,
unit artifacts and candidate receipt chains are checked on resume; no prior run
root was overwritten or reused for the new identity.

Preflight:209 focused tests passed; complete57-candidate/two-generation smoke,
12 confirmation jobs and authenticated completed resume. Observed variation
covered all13 free genes; only position size remained fixed by design.

Scientific scope remains synthetic mechanism/controller search, not financial
model training, floating feature selection or a certified optimal strategy.
Absence of early closes is explicit telemetry, not a successful closer result.
Activity is a candidate constraint (>2 mean orders/week), not a launch gate.

ETA must come from the measured candidate timings of the full lanes; the smoke
does not provide a valid full-run estimate. Live status and receipts are stored
in the operator-managed output roots, outside this public repository.
