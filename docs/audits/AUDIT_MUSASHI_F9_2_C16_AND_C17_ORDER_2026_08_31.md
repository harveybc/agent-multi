# Audit F9.2 / C16 and correction order C17

Date: 2026-08-31
Auditor: General Musashi
Reviewed identities:

- `agent-multi@43797e3d` (F9.2)
- `lts@872359e` (C16)
- return package `agent-multi@76365ae2`

## Disposition

### F9.2: ACCEPTED

The executing callback now caps the next SB3 training block at the exact
remaining update budget in `_on_rollout_end`, stops at `>=`, preserves the
configured `gradient_steps` identity, and restores it after training. The real
SAC regressions cover multi-update blocks, zero, one, one-remaining, and resume
at or above the limit. This closes the update-budget overshoot finding.

Acceptance authorizes reuse of this guard in future reviewed runs. It does not
authorize a training dispatch or resurrect evidence whose terminal artifact
was lost.

### C16: REVISE (one remaining TOCTOU defect)

The path-containment, descriptor-first hashing, mode/owner checks, reviewer
pinning, sealed acta, and least-privilege decision for `trade_allowed` are
accepted in principle. One authority-bearing path is still reopened:

1. `_sha256_descriptor_first(root, acta_rel)` hashes the reviewed acta through
   a verified descriptor.
2. `(Path(root) / acta_rel).read_text()` opens the path again to parse it.

A replacement between those operations can make the judge verify one byte
stream and consume another. The claim that the acta is consumed
descriptor-first is therefore false at the executing seam.

## C17 correction order

Implement only the following correction before collector activation:

1. Replace the digest-only helper for authority-bearing structured artifacts
   with a descriptor-bound reader returning the exact bytes and digest from one
   verified descriptor.
2. Parse the acta from those returned bytes. Do not reopen its path after
   verification.
3. Keep diff, rollback, and backup hashing descriptor-bound. Audit all callers
   and assert structurally that no verified structured artifact is subsequently
   reopened by path.
4. Freeze the exact adversary: substitute the acta path after the verified
   descriptor is opened/hashed but before parsing. The consumed acta must remain
   the verified bytes or the operation must refuse; it must never consume the
   replacement.
5. Cover symlink replacement, rename replacement, malformed verified bytes,
   digest mismatch, and the clean GO path.

No network, service, EA, position, order, collector activation, GPU, training,
or economic grid is authorized by this order. The activation verdict remains
`COORDINATED_WINDOW_REQUIRED` while the MT5 position exists or any evidence
precondition is absent.

## Next gate

Acceptance of C17 may close the effect-free collector preflight implementation.
Actual collector activation remains a separate coordinated-window decision and
may authorize only the read-only session-evidence collector.
