# Audit: weekly-flat durable custody D1-D3 return

Date: 2026-08-29

Audited commits: `gym-fx@9a2939c`, `agent-multi@a8e84720`

Verdict: **D1 accepted; D2/D3 revise.** The watchdog is now read-only and the
claim survives restart, but terminal transitions and evidence authority are
not yet safe enough for deployment.

## Accepted

- The in-memory ledger was removed.
- Claim election uses a persistent exclusive final path.
- Repeated watchdog reads do not mutate custody.
- Identity fields and record digest are persisted.
- Owner-only file/root modes and root-symlink refusal are tested.

## Critical: terminal transitions are not atomic

`finish()` performs `read(active)`, constructs a terminal record, then calls
`os.replace` without a lock or compare-and-swap. Two processes can both read
`active`; one may write `completed` and the other `failed`, with the last rename
winning. Terminal immutability is therefore not guaranteed across processes.

The advertised concurrent-claim test is also sequential: its list
comprehension calls `subprocess.run` twice, waiting for the first process before
starting the second. It does not test simultaneous contention.

## Critical: direct evidence can be fabricated

`native_protection_digest` accepts any non-empty string and carries no evidence
timestamp, source, position binding or freshness check. Completion accepts any
dictionary with truthy `flat_confirmed` and `fresh` values. Reproduced:

- `native_protection_digest="not-a-digest-or-fresh-evidence"` authorizes;
- `{"flat_confirmed":"yes","fresh":"yes"}` completes custody.

Thus neither native protection nor zero/zero reconciliation is established by
a typed direct-evidence envelope.

## High: protocol durability gaps

- The record advertises `prepared`, but no prepared record/state exists.
- Update temporaries use `O_TRUNC`, not exclusive creation.
- A final empty placeholder can remain if the process dies between exclusive
  final-path creation and replacement; this fails closed but has no diagnosed
  recovery/disposition state.
- Final record symlinks are followed by `read()` after creation and should be
  explicitly refused.

## Reproduced tests

Focused custody + session suite: `49/49`. The counterexamples above remain
possible because they are absent from the suite.

## Disposition

D1 remains accepted. Correct D2/D3 under the attached order. C5 may be coded
in a separate branch, but the combined package cannot be accepted and no WP3,
WP4 or live deployment may proceed until custody passes independent audit.

