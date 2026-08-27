# Acceptance Audit: DATA-SOTA-361

Date: 2026-08-27 America/Bogota
Auditor: General Musashi
Reviewed tip: `agent-multi@9cfef3bd`

## Verdict

**DATA-SOTA-361 ACCEPTED.** The exact failed-directory-fsync counterexample now
refuses both rendering and reexecution, including through a newly constructed
ledger instance.

Independent focused reproduction: **87 passed**. No model was constructed or
forwarded.

## Verified Facts

- Completion intent is created no-clobber and file/directory-fsynced before the
  completed ledger transition.
- The intent marker dominates a completed-looking canonical record.
- Every injected failed-completion boundary refuses both reserve and render.
- Read-only diagnosis reports expected/actual evidence identity without
  resolving or mutating uncertainty.
- Successful completion removes the marker, renders repeatedly after ledger
  reconstruction, and remains single-use.

## Residual Finding

### DATA-SOTA-362 (S4, non-blocking for this CPU smoke)

After a post-unlink directory-fsync failure, marker restoration is best-effort
but is not itself file/directory-fsynced. A real power loss at that exact point
can make the restored marker's persistence ambiguous. The canonical completed
state still prevents duplicate execution, so this does not block the bounded
mechanics smoke. Before economic authority, promotion or reusable one-shot
execution, replace marker deletion with an append-only durable completion-ack
protocol: retain intent permanently and require a separately fsynced ack for
renderability.

## Authorization

Exactly one replacement CPU transfer-loader smoke is now authorized under the
companion dispatch. Failure after reservation consumes the authorization. No
automatic retry, GPU, economics, promotion or collector activation is allowed.
