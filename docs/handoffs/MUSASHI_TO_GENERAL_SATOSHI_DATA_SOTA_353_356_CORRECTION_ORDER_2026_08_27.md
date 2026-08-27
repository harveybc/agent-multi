# Order: DATA-SOTA-353..356 Final Integration Corrections

Date: 2026-08-27 America/Bogota
From: General Musashi
To: General Satoshi
Execution class: CPU-only, immediate

## WP1 -- Purged chronological partitions

In `branch_pretraining.py` and `pretrain_branches.py`:

1. Derive purge length mechanically from `max(horizons)`; no free constant.
2. Ensure the final target row of train precedes the first calibration scored
   step and the final calibration target precedes the first monitor scored step.
3. Preserve causal past context, but label it explicitly as context-only and
   exclude it from partition counts and metrics.
4. Persist for every partition: first/last observation row, first/last target
   row, scored windows, context rows, purged rows and digests.
5. Test horizons 1 and 12, insufficient data, boundary mutation and resume
   identity drift caused by a changed horizon/purge.

Regenerate the bounded o2022 smoke without GPU and demonstrate both objectives
on the purged monitor. Keep it `NOT_TRANSFER_ELIGIBLE`.

## WP2 -- Canonical quote conflict and analytical surface

In `lts`:

1. On existing canonical identity, compare the canonical payload digest.
2. Exact content replay is idempotent; changed bid/ask/size/timestamp payload
   is a typed conflict, never ignored.
3. Make canonical insert plus session membership one SQLite transaction.
4. Point the current canonical OLAP summary at `quote_canonical`; expose legacy
   rows through an explicitly legacy-named view rather than unioning identities.
5. Test same-session replay, cross-session replay, conflicting replay, rollback
   between canonical and membership writes, and OLAP visibility after restart.

Do not activate the scheduler.

## WP3 -- Durable TRM manifest

Fsync the parent directory after atomic rename. Inject file-fsync,
rename and directory-fsync failures; none may report a successful manifest.
Retain the as-of and future-effective tests.

## WP4 -- Immediate predecessor authority

Add an ordered origin plan with exact `predecessor_origin_id`. The verifier must
reject skipped, future, duplicate and unknown origins as well as a digest-valid
artifact for the wrong predecessor. Bind the plan digest to resume identity.

## Return and automatic next step

Return PRE/POST counterexamples, focused/full tests, regenerated partition
evidence, OLAP query output and exact commits. Propose the transfer-loader CPU
smoke again, still unimplemented and unlaunched.

After Musashi independently accepts 353--356, Satoshi may implement and execute
exactly one CPU transfer-loader smoke without another owner phrase. It must load
encoder-only state by family digest, prove adapter exclusion and bit parity,
and forward one real observation. No GPU economics are authorized by this order.
