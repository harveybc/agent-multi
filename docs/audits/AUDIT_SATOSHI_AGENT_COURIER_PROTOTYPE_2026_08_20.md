# Audit: Agent Courier Prototype

Date: 2026-08-20 America/Bogota
Auditor: General Musashi
Commit: `8de6856b`
Disposition: **PROTOTYPE ACCEPTED; DAEMON/HEADLESS DELIVERY REJECTED**

## Reproduced

- 13/13 focused tests pass.
- Filename addressing, branch scanning and blob deduplication work on the
  fixture.
- Commands use argv, not a shell string.
- Dry-run does not invoke the configured CLI.

The idea directly addresses the handoff failure that caused hours of idle time
today. It is worth completing, but it currently confuses traceability with
authority.

## Blocking findings

### C-AUTH-1: any push-capable branch can inject agent instructions

The scanner trusts every fetched remote branch and accepts a document solely
because its filename contains `_TO_<identity>`. Git records provenance but does
not prove that the sender is an authorized principal. A compromised token,
agent, branch or contributor can inject text into a locally privileged CLI.

### C-AUTH-2: delivery prompt grants action authority

The prompt says `actua segun las ordenes vigentes`. The launched CLI inherits
the local permission surface. Calling the component "transport only" does not
remove the effective authority conveyed by this prompt.

### C-STATE-1: failures and dry-runs are acknowledged as delivered

Every blob is added to `seen_blobs` after `deliver()`, including
`DELIVERY_FAILED` and `DRY_RUN`. A transient CLI failure or supervised dry-run
therefore suppresses the future real delivery permanently.

### C-STATE-2: state and inbox durability are incomplete

No 0700 directory/0600 file enforcement, atomic write/fsync, process lock, or
crash-safe pending/delivered transition exists. Concurrent pollers can corrupt
state or duplicate execution.

### C-DOS-1: unbounded remote input

The scanner has no document-size, branch-count, message-count or execution-time
budget. A branch can cause large extraction/inbox growth and serial one-hour CLI
invocations.

### C-PROV-1: provenance is insufficiently bound

The record lacks repository remote identity, commit hash, signer/allowlist
decision and immutable sender identity. Deduping only by blob permits ambiguous
first-seen provenance across branches.

## Correction order

1. Split **transport** from **action**. The courier may fetch, authenticate,
   validate and enqueue. It must not tell a privileged agent to act.
2. Initial rollout must invoke agents in read-only/review mode with a prompt
   treating the document as untrusted data and producing a proposed response
   only. No commands, edits, pushes, runtime changes or approvals.
3. Accept documents only from configured repository remotes and branch
   allowlists whose commits descend from an approved anchor. Require a canonical
   YAML/JSON header with schema, sender, recipients, message id, parent id,
   intent class and body hash. Filename matching is advisory only.
4. Add detached signature verification against an owner-managed sender
   allowlist before any non-dry-run delivery. Unknown/invalid signatures go to
   quarantine and generate one deduplicated alert.
5. Implement `discovered -> authenticated -> pending -> delivered|failed ->
   acknowledged` state. Only `delivered`/`acknowledged` may enter completed
   dedupe. Dry-run and failed items remain pending/retryable.
6. Enforce 0700 directories, 0600 files, atomic replace + fsync, and a single
   writer lock. Never persist agent stdout that may contain secrets; retain a
   hash and bounded redacted status instead.
7. Add size/count/time/rate limits, exponential retry and quarantine.
8. Seed historical blobs as `baseline_ignored`, distinct from delivered.
9. Add adversarial tests: forged filename, unauthorized branch, invalid
   signature, modified body, replay under another branch, failed delivery,
   dry-run then real run, crash between CLI return and state write, concurrent
   pollers, oversized document and prompt injection body.
10. Return a read-only single-message supervised demo. Do not install or enable
    units yet.

The route-critical WP4 activity-evidence correction remains priority 1. Courier
hardening may proceed in parallel but must not delay it.
