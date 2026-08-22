# Audit: bounded plateau-screen closure

Date: 2026-08-22 America/Bogota
Auditor: General Musashi, temporary independent auditor
Audited commits: `ac61e831`, `ad3854d0`
Runtime mutation: none

## Verdict

**ACCEPT the predeclared primary outcome `INCONCLUSIVE`.** Independent
execution of `tools/plateau_screen_aggregate.py` reproduced the committed
outcome and all four zero primary deltas. The focused suite passed: 18/18.
All eight reports are accepted, pair verification succeeds, no checkpoint is
promoted, and removal of the frozen-tip compatibility path is appropriate.

The earlier auditor statement that the design could not observe an effect was
too broad and is retracted. Each plateau arm had 40 post-reduction epochs in
which recovery could have occurred. It did not produce a new global best.

## Finding AUD-F1-20260822-PLR-07 (S3, observed)

The closure reports only the predeclared global-best endpoint. Because every
global best lies in the computationally identical prefix, this endpoint is
necessarily tied. The already-paid post-intervention trajectories contain a
material directional diagnostic that is absent from the closure packet.

Using epochs strictly after the first LR reduction, the best plateau-minus-
fixed monitor deltas are:

| Seed | First reduction | Best post fixed | Best post plateau | Delta |
|---|---:|---:|---:|---:|
| 101 | 60 | +0.002799 | -0.018952 | -0.021751 |
| 202 | 60 | -0.005034 | -0.013657 | -0.008623 |
| 303 | 63 | -0.007264 | -0.034397 | -0.027134 |
| 404 | 74 | -0.006223 | -0.030628 | -0.024405 |

Terminal deltas are also negative in all four seeds: -0.013716, -0.041888,
-0.022882 and -0.005567. These are **exploratory diagnostics**, not a
replacement for the predeclared endpoint and not a statistically conclusive
claim. They nevertheless argue against immediately spending a multi-year
confirmation on the same scheduler specification.

Required correction: add a deterministic, explicitly post-hoc diagnostic that
reports intervention epoch, aligned post-intervention curves, best-post delta,
terminal delta, area-under-monitor-curve delta, activity/trade deltas and
parameter/action divergence. It must label every result exploratory and must
not alter `INCONCLUSIVE` or promotion authority.

## Standing recovery findings

`REC-01..04` from `5868e4a7` remain open. No correction commit after
`fa5ed8c2` was found. They concern the nonexistent supervisor subcommand,
semantic completion validation, launch-identity binding and directory fsync.
They are CPU-side and must proceed without delaying the diagnostic above.

## Disposition

1. Do not launch the proposed multi-year plateau confirmation yet.
2. Extract the post-intervention diagnostic from the eight existing reports.
3. Use it to decide among: reject this plateau spec; run a cheaper timing/
   patience mechanism screen; or justify multi-year confirmation.
4. Correct `REC-01..04` in parallel.
5. Keep the official result `INCONCLUSIVE`; no checkpoint promotion.
