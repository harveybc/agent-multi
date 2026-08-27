# DATA-SOTA-362 — Registered Required Hardening (work plan)

Registered per the acceptance audit of DATA-SOTA-361 and the
replacement-smoke dispatch (2026-08-27). S4, NON-BLOCKING for the
bounded mechanics smoke; REQUIRED before any authority beyond it.

**Finding:** after a post-unlink directory-fsync failure, the
completion-intent marker restoration is best-effort but not itself
file/directory-fsynced; a real power loss at that exact point leaves
the restored marker's persistence ambiguous (canonical completed state
still prevents duplicate execution).

**Required hardening (before economic authority, promotion or
reusable one-shot execution):** replace marker DELETION with an
append-only durable completion-ack protocol — retain the intent marker
permanently and require a separately fsynced ACK record for
renderability.

Status: OPEN — scheduled ahead of any post-smoke authority request;
disposition remains with General Musashi.
