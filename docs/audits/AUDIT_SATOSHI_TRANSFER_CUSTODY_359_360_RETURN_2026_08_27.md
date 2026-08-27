# Audit: Transfer Custody 359--360 Return

Date: 2026-08-27 America/Bogota
Auditor: General Musashi
Reviewed tip: `agent-multi@d390d2e1`

## Verdict

**REVISE BEFORE REPLACEMENT SMOKE.** DATA-SOTA-359 is accepted as corrected.
DATA-SOTA-360 is materially improved but its failed-completion guarantee is
false under the exact directory-fsync failure it claims to handle.

Independent Tier-A suite: **82 passed**. No model was constructed or forwarded.

## Accepted Facts

- Effective config bytes are snapshotted once for execution; digest,
  architecture and env config derive from that snapshot.
- Post-snapshot source mutation does not change execution inputs.
- State transitions are checked and terminal records reject ordinary retries.
- Ledger root/record modes and symlink refusals are implemented.
- Evidence carries schema/run/dispatch and identity bindings.
- Renderer authenticates ordinary completed evidence against the ledger.

## Finding

### DATA-SOTA-361 (S2): failed completion can still render as completed

Independent counterexample:

1. Reserve and transition to `running`.
2. Write valid evidence.
3. Inject failure only on parent-directory `fsync` during `complete()`.
4. `complete()` raises `OSError`, but the visible ledger state is `completed`.
5. After restoring normal I/O, `verified_render()` returns the packet.

Observed output:

```text
complete_error OSError
visible_state completed
render_after_failed_completion s
```

The committed regression explicitly tolerates this state and checks only that
re-reservation refuses. That prevents duplicate execution but does not satisfy
the stronger requirement: failed completion must never be acknowledged or
rendered.

## Required Semantics

Use a durable completion-intent/uncertain sidecar:

- create and directory-fsync it before attempting completion;
- renderer refuses while it exists, regardless of visible canonical state;
- write and fsync evidence and completed record;
- remove the sidecar only after all completion durability operations succeed,
  then fsync the directory again;
- any failure leaves the marker and the dispatch permanently
  `completion_uncertain`/spent until an independent recovery tool verifies every
  digest and explicitly resolves it; the ordinary execution path cannot retry
  or render it.

Do not weaken the guarantee to "rerun refused". The required property is both
**rerun refused and render refused** after any failed completion acknowledgement.

## Disposition

No replacement smoke is authorized. Correction 361 is model-free CPU work.
After independent reproduction, the single strong-config replacement smoke may
be dispatched immediately.
