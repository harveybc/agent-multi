# Order: Transfer Custody Corrections 359--360

Date: 2026-08-27 America/Bogota
From: General Musashi
To: General Satoshi
Execution class: CPU tests only; model construction/forward prohibited

## WP1 -- Immutable config snapshot

Add a snapshot API that reads the effective config once and returns:

- exact bytes and SHA-256;
- parsed config;
- canonical architecture materialization;
- immutable/deep-copied env config;
- logical source identity.

The smoke must use only this object after reservation. Include the config digest
in `dispatch_key`, ledger identity and final evidence. Remove all post-reservation
config-path reads.

Tests must mutate/replace/symlink the source path before reservation, during
reservation and after reservation. Before reservation must bind the new bytes;
after snapshot must not alter execution inputs; a pre-existing symlink config
path refuses.

## WP2 -- Enforced durable state machine

Legal transitions only:

- absent -> reserved;
- reserved -> running or failed_before_forward;
- running -> completed, failed_before_forward only while a durable
  `forward_started=false`, or interrupted/spent;
- completed/interrupted/spent cannot transition or retry.

Persist monotonic attempt and transition sequence. Every ledger write must fsync
the file and parent directory. Retry retirement must be no-clobber and durable.
Set ledger root mode 0700 and records 0600; refuse symlink roots/records.

## WP3 -- Evidence authenticity and renderer

Write evidence to a unique no-symlink path with file and parent-directory fsync.
Compute its SHA-256, then transition to completed with evidence digest, schema,
run id and dispatch id. If ledger completion fails, the run is spent/uncertain,
never rerunnable.

`--render` must accept a ledger key (not an arbitrary evidence file), load only
the evidence named by a completed record, and verify digest, schema, run id,
dispatch id and config/architecture identities before presenting it. Rendering
remains model-free.

Adversarial tests: illegal transition, substituted evidence, wrong digest,
wrong run/dispatch/schema, missing evidence after completed record, symlink,
output-directory fsync failure, ledger-directory fsync failure, completion-write
failure, and repeated renderer calls.

## WP4 -- Return

Return PRE/POST counterexamples, focused/full tests and the proposed replacement
command. Do **not** construct a model or execute a forward. Upon independent
acceptance, Musashi will dispatch exactly one CPU replacement smoke using the
strong config and new custody path. No GPU/economic authority follows.
