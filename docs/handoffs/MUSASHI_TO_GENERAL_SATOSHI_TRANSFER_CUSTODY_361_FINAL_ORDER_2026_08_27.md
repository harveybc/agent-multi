# Final Order: DATA-SOTA-361 Completion Uncertainty

Date: 2026-08-27 America/Bogota
From: General Musashi
To: General Satoshi
Execution class: model-free CPU tests only

## Implement

Add a no-clobber completion-intent marker keyed by the dispatch ledger key.
Its lifecycle must be durable and independently observable:

1. Before completion, create marker with expected evidence path, digest, schema,
   run id and dispatch id; fsync file and ledger directory.
2. `verified_render` checks the marker first and refuses with
   `completion_uncertain` if present.
3. Commit completed ledger evidence and fsync it plus parent.
4. Remove marker only after successful completion commit; fsync parent after
   unlink. Only then may `complete()` acknowledge success.
5. If any operation fails, marker remains. Completed-looking canonical state is
   subordinate to the marker and cannot render or rerun.

Do not automatically recover or delete uncertain markers. Provide a separate
read-only diagnostic that reports expected/actual digests and state; resolution
is outside this order.

## Regressions

Inject failure at each boundary:

- intent file fsync;
- intent directory fsync;
- evidence file/directory fsync;
- completed-record file/directory fsync;
- intent unlink;
- post-unlink directory fsync.

For every failed completion, assert both:

- a second reservation refuses; and
- renderer refuses, including after constructing a fresh `DispatchLedger`
  instance to simulate process restart.

Also prove the successful path has no marker, renders repeatedly and remains
single-use. Preserve the independent counterexample output in PRE evidence.

## Return

Return PRE/POST output, tests and proposed replacement command. Do not construct
a model or execute a forward. Acceptance of 361 will itself authorize exactly
one replacement CPU smoke; no further owner phrase will be needed.
