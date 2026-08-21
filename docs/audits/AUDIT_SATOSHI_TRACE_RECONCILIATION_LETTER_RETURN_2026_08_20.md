# Audit: Trace Reconciliation Letter-Complete Return

Date: 2026-08-20 America/Bogota
Auditor: General Musashi
Commits: `fe4de817`, `f88c7e86`
Disposition: **NEAR ACCEPTANCE; TWO EXECUTING BYPASSES BLOCK GPU**

## Reproduced

- 68/68 focused tests pass.
- Settlement is now an appended explicit row; the final market row is intact.
- Monotonic/integral checks and physical bound exist in the reconciliation
  primitive.
- CPU smoke passes the evidence gate and selects a checkpoint.
- Report exposes cumulative counts and a non-null boundary.
- Duplicate model files are absent from the current tree.

## Blocking findings

### TR-L1: call site bypasses strict count typing

`rl_pipeline_with_validation.py` calls:

```python
reconcile_trace_trades(
    trace_rows,
    int(summary.get("trades_total") or 0),
    ...)
```

The `int(...)` coercion occurs before `_integral_count()`. Numeric strings,
fractional floats and booleans can therefore be normalized/truncated and evade
the strict validator proven by the direct unit tests. Pass the raw value into
the primitive. Add executing-call-path tests for `"3"`, `3.7`, `True`, NaN,
inf and negative values.

### TR-L2: sealed-test proof uses validation fallback

`wp4_cpu_smoke.py` chooses:

```python
evaluation or validation_epoch or train_epoch
```

The report therefore labels `validation_epoch` ending 2018-03-07 as the
`diagnostic_internal_test_split`. This proves the validation date, not the test
boundary or test firewall. The committed evidence contains
`test_return_trace.csv`, but `facts_from()` does not load it.

Load the actual test trace explicitly and refuse if absent. Report its split
label, first/last timestamp, hash and `contains_heldout_rows`; never fall back to
train/validation. Separately prove from the executing selection call path that
this split cannot influence checkpoint or stopping state.

## Immediate correction

1. Remove pre-validation count coercion and add executing-path regressions.
2. Add `test` to trace fact discovery and require the actual diagnostic test
   trace for the boundary proof, without fallbacks.
3. Rename the report field from ambiguous `trades` to
   `closed_trades_cumulative`.
4. Run focused/full suites, CPU smoke and CUDA preflight.
5. Return one commit. No owner phrase is required; courier remains secondary.

After independent reproduction of these two corrections, launch the bounded
single-GPU smoke immediately.
