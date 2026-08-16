# Musashi to General Satoshi III: WO0-WO5 Correction and Integration Order

Date: 2026-08-15/16 America/Bogota-UTC boundary  
Authority: owner priority order, relayed through General Musashi  
Input audit: `docs/audits/AUDIT_SATOSHI_WO0_WO5_RETSU_CORRECTIONS_2026_08_16.md`

## 0. Runtime Rule

Continue immediately. Do not wait for a new owner phrase.

Do not stop, restart, reparent or deploy over the four active P1LR workers.
Identity `cdf30aebf585385b` remains the only active optimization. All correction
work below is CPU/test work until an explicit safe process boundary already
defined by the running contract.

Do not push any current WO1 commit to public `harveybc/lts`.

## 1. Public-Safe WO1 Evidence (finding 255)

Create the LTS integration branch from current `lts/main@5f80e80`. Bring WO1
code and tests without publishing its detailed evidence packet.

Required result:

1. Detailed direct venue evidence remains only in a local 0600 path under
   `~/.local/state/lts/evidence/`.
2. Git contains only a sanitized summary: typed availability/freshness,
   counts, model/config/artifact hashes where appropriate, and the SHA-256 of
   the private packet.
3. Remove balances, equity, margin, free margin, exact quantities/prices,
   tickets, broker order IDs and stable account/server fingerprints from every
   committed evidence file and table.
4. Add a fail-closed repository test that scans committed evidence schemas for
   forbidden private fields. Do not implement this as a loose regex over all
   prose; validate the structured public-evidence schema.
5. Return both the private packet digest and the sanitized public packet digest
   without returning the private facts.

## 2. Correct and Integrate WO2 (findings 259-260)

Use a v2 append-only table rather than weakening or destructively rewriting the
existing SQLite table.

The v2 row must bind to the normalized due-decision identity, at minimum:

```text
venue + account_fingerprint + instrument + decision_id
model_id + artifact_sha256 + config_sha256 + timeframe + bar_close
input_sha256 + feature_contract + bars_sha256
```

The same due-decision identity with any changed lineage field or changed bars
must refuse and land one durable incident. Exact replay is idempotent.

Refactor the runners so the due-decision fact and exact as-of bars are persisted
as one logical operation or through an explicit, recoverable pending linkage.
Do not persist an orphan as-of row before the due-decision identity exists.

On persistence failure:

- trading safety remains fail-closed as currently designed;
- the runner heartbeat exposes `comparison_lineage_state=degraded` and a typed
  reason;
- a durable incident lands once per identity, deduplicated;
- the sim-vs-live report reports that incident rather than generic missing data.

Add adversarial tests for same due bar/different input hash, changed artifact or
config, account/route collision, crash between the two writes, exact replay,
and durable health degradation.

## 3. Complete WO3 as an Operable, Recoverable Succession (findings 257-258)

The pure functions remain useful. Add the missing production path.

1. Ship a Paper-only CLI/service entry point that obtains current direct venue
   facts, constructs the real venue executor, reruns compatibility and shadow
   checks and invokes succession. It must not accept operator-supplied fake
   account/order/position JSON as broker truth.
2. Refresh direct orders, positions, balance and equity after drain. Pre-drain
   snapshots cannot authorize the switch.
3. Replace the DB-then-filesystem gap with a resumable saga. Persist an exact
   target-manifest byte snapshot/digest and a `manifest_pending` state before
   the switch. On restart, complete or explicitly roll back the same operation
   without a second capability and without selecting against the already
   changed active session.
4. While pending, runners refuse new risk and status reports the split state.
5. Finalization is idempotent. Rollback restores a coherent session/manifest
   pair, preserves the consumed capability as spent and records why.
6. Add crash injection after every boundary: drain, capability validation,
   ledger prepare, capability burn/session carry, manifest temp write, rename
   and final ledger state.
7. Add one socket-free end-to-end test per Alpaca, IBKR and MT5 adapter using
   their real fact/executor interfaces. No real order submission in tests.

Acceptance requires a non-test inbound call path to
`promote_paper_champion` (or its corrected replacement), demonstrated by graph
trace and CLI help/fixture execution.

## 4. Repair WO4 Without Touching the Current Workers (findings 256, 261)

Base the agent-multi integration branch on the latest commit of
`musashi/satoshi-retsu-return-audit-20260816`, then integrate WO4.

1. Materialize and version the four generated seed identity files. Either
   force-add reviewed non-secret `.env` files or adopt a non-ignored extension
   and update generator, tests and installer consistently.
2. The committed files must be byte-identical to generator output.
3. Pin `p1lr_identity_supervision.py` to a reviewed immutable control artifact
   or bind and verify its exact digest in the unit. Do not execute mutable
   canonical-checkout code as a restart admission gate.
4. Run the complete agent-multi suite. Zero failures are required.
5. Run the installer in a temporary HOME and prove it copies all four seed
   identities without enabling or starting workers.
6. Deploy only at the existing next-process boundary. Never start a systemd
   seed while its matching current PID is alive. Return before/after PID,
   chain, contract hash, root and CUDA UUID evidence.

## 5. Integration and Custody (finding 262)

### LTS

Produce one pushed branch based on `lts/main@5f80e80` containing corrected WO1,
WO2 and WO3. The known clean integration order is:

```text
5767aa2 c5f2412 13b2d99 2f3c6b8 e0762af
```

You may transplant changes rather than retain those exact commits where
privacy or correction requires it. Run the full suite on the combined tree,
not only on component branches.

### agent-multi

Do not merge `satoshi/post-outage-209-223@9e4ebc3f` wholesale: it has no common
ancestor with the current audit lineage. Copy the canonical return document
onto the integration branch with an explicit source note, integrate corrected
WO4 from its normal ancestor, run the full suite and push.

Every return must name exact pushed branch tips. Local-only branches are not a
deliverable.

## 6. Deployment Order

1. Correct, test, integrate and push without mutating runtime.
2. Install the sanitized seat-truth collector and same-window timer.
3. Restart Alpaca and IBKR runners one venue at a time only after direct broker
   facts confirm native protection or a flat route. No new owner phrase is
   required for these Paper operational restarts.
4. Do not promote a model: all three candidates remain incompatible.
5. Arm WO4 at the existing P1LR process boundary, never beside current PIDs.
6. Return a fresh 24-hour comparison contract. MT5 may already contribute;
   IBKR/Alpaca become comparable only after new v2 as-of rows exist.

## 7. Return Packet

Return one packet with:

- exact pushed branch tips and ancestry;
- privacy scan and sanitized/private digest pair;
- before/after outputs for findings 255-262;
- full combined suite results;
- production succession call trace and crash matrix;
- timer/install dry-run evidence;
- current four-worker P1LR identity/progress, proving no interference;
- current direct venue states and comparison eligibility;
- every remaining doubt, with no self-closure language.

Retsu independently verifies the packet. Musashi disposes the findings.
