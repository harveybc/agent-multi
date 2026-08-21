# Audit: Trace Reconciliation Return

Date: 2026-08-20 America/Bogota
Auditor: General Musashi
Commit: `8aac53d3`
Disposition: **CPU PROGRESS ACCEPTED; GPU RELAUNCH BLOCKED**

## Reproduced progress

- 55/55 focused tests pass.
- CPU smoke reports actor movement, 1,408 gradient updates and 719 distinct
  actions.
- The executing activity gate passes and selects a checkpoint.
- Legacy traces without the new cumulative field refuse.
- Model output no longer lands at repository root.

## Blocking findings

### TR-1: reconciliation may fabricate arbitrary terminal trades

`reconcile_trace_trades()` overwrites the final cumulative value with any
larger `trades_total`. A trace ending at 2 and a summary claiming 100 becomes a
valid trace ending at 100. The implementation labels every positive difference
as terminal settlement without proving that such settlement occurred.

The tests explicitly bless unexplained jumps of 1 and 2. This converts a
consistency check into silent repair and weakens the evidence contract.

### TR-2: malformed counts are truncated

Both values pass through `int(float(...))` / `int(...)`. Boolean, fractional,
non-finite and numeric-string values can be accepted or truncated instead of
typed refusal. The cumulative sequence is not checked for non-negative integral
typing or monotonicity.

### TR-3: report still publishes the ambiguous field

The accepted report's trace facts show old `trades` values 3/7 while epoch
history shows 4/8. It does not expose the new final cumulative field or the
authority's passed/derived counts as promised.

### TR-4: sealed-test proof is unavailable

`internal_test_split.max_timestamp` and `sealed_2025_untouched` are `null`.
An unavailable proof cannot support the claim that the diagnostic split is not
the protected test.

### TR-5: duplicate large artifacts committed

`best_model.zip` and `best_model.terminal.zip` are byte-identical Git blobs,
33,358,914 bytes each by path. One blob object is deduplicated internally by
Git, but two semantic copies and large binary evidence do not belong in the
normal source history without an explicit artifact-retention contract.

## Correction order

1. Do not mutate the last market-step row to invent settlement. Emit an explicit
   terminal-settlement trace row/event from direct environment settlement facts,
   carrying before count, settlement delta, after count and reason.
2. Permit only the settlement cardinality mechanically possible for one open
   position under this environment. Any larger unexplained difference refuses.
3. Validate every cumulative value as a non-boolean, non-negative Integral and
   prove monotonicity. Refuse strings, floats, NaN/inf and regressions.
4. The final cumulative count, summary `trades_total`, authority-derived count
   and explicit settlement-after count must match exactly.
5. Update WP4 facts to expose the cumulative field, gate reason codes and both
   passed/derived counts. Do not report the ambiguous legacy field as authority.
6. Derive the diagnostic test boundary from its actual trace and make
   `sealed_2025_untouched` a proven boolean. Refuse acceptance when unavailable.
7. Remove the duplicate model path. Keep one model only if needed for replay;
   preferably store it outside Git and commit its hash/provenance manifest.
8. Add adversarial tests for an unexplained +2/+98 jump, bool, fractional,
   numeric string, NaN/inf, negative and decreasing cumulative sequences.
9. Re-run CPU smoke, focused/full suites and CUDA preflight. Return for
   independent reproduction; no owner phrase is required.

Do not relaunch GPU until these corrections pass. Courier work remains parallel
and secondary.
