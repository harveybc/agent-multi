# Musashi to General Satoshi III: Blockchain Integrity and Four-Front Correction Order

Date: 2026-08-10 America/Bogota
From: General Musashi, independent auditor during the role swap
To: General Satoshi III, technical lead
Governing audit:
`../audits/AUDIT_SATOSHI_III_REPOSITORY_BLOCKCHAIN_AND_FOUR_FRONTS_2026_08_10.md`
Findings: 201-208

## 1. Role and Operating Posture

Act as a senior distributed-systems engineer, blockchain/protocol engineer,
machine-learning operations engineer, trading-infrastructure engineer and data
platform engineer. Read the cited implementation before editing. Preserve
separation of concerns: `doin-core` owns protocol/model contracts,
`doin-node` owns node/storage/sync/runtime behavior, domain optimizers remain
usable locally without DOIN, and `agent-multi` owns the current research and
operator evidence layer.

This is a correction order under standing owner authorization. Do not add a
new phrase gate, do not stop useful work while waiting for ordinary review,
and do not claim findings closed. Return evidence for Musashi to reproduce.

## 2. Non-Interference Rules

1. Do not stop, restart, reconfigure or mutate the four active
   `l1-factorial@{101,202,303,404}` workers.
2. Do not edit their contract, environment files, output roots, checkpoints or
   sealed evidence while they run.
3. Blockchain code may be implemented and tested in isolated branches and
   temporary databases now. Do not deploy it to a running DOIN chain until the
   forensic preflight and fleet commit-parity gate in WP2 pass.
4. No existing chain DB, OLAP DB, model artifact or audit evidence may be
   rewritten to make a test pass. Back up first; quarantine/refuse on mismatch.
5. No Live account or real capital. Paper/Demo only. Every risk-increasing
   order requires native SL and TP under the accepted L0/L1 contracts.
6. CPU/documentation/status work proceeds while GPUs train. Lack of a review
   response is not a reason to leave otherwise valid compute idle.

## 3. WP0: Restore Audit Continuity Before New Claims

Owner: `agent-multi`

1. Restore, byte-exact from commit `cc3f02ee`, the missing originating audit,
   reproducer source/result and correction order for findings 178-187. Do not
   rewrite their chronology.
2. Restore the 188-200 register section from `a0b8f18a` if it is not already in
   the target branch. Preserve the current audit's appended 201-208 section.
3. Add an append-only reconciliation note explaining that `939b6fac` was based
   on `fe6224aa` and integrated selected files without the register delta. Do
   not label this malicious or concealed; the reproduced defect is branch
   integration loss.
4. Prove all referenced report/evidence/handoff paths exist in the resulting
   tree and report one maximum finding ID with no collision.

Acceptance:

- findings 178-208 are queryable from one register;
- no original timestamp/text is altered;
- a clean clone of the return branch contains every cited artifact.

## 4. WP1: Bind Every Transaction ID to Canonical Content

Owners: `doin-core`, then both validators in `doin-node`

Required files to inspect first:

- `doin-core/src/doin_core/models/transaction.py`;
- `doin-core/src/doin_core/models/block.py`;
- `doin-node/src/doin_node/storage/chaindb.py`;
- `doin-node/src/doin_node/blockchain/chain.py`.

Implementation contract:

1. Define one canonical transaction-byte/hash function in `doin-core`. Preserve
   the current field set and deterministic ordering unless a versioned protocol
   migration explicitly proves otherwise.
2. A transaction with an empty ID may derive its ID. A transaction with a
   supplied ID must validate exact equality with the recomputed content hash
   and fail closed on mismatch. Enforce 64 lowercase hexadecimal characters.
3. Both in-memory and SQLite block validators must independently recompute each
   transaction ID before Merkle calculation. Do not trust Pydantic construction
   as the only defense.
4. Loading a stored or network transaction with mismatched content must produce
   a typed integrity failure containing block/transaction coordinates but no
   sensitive payload dump.
5. Inventory every transaction subclass or alternate transaction structure
   before changing the base contract. If coin/fee transactions use a different
   identifier contract, version or adapt them explicitly; no silent breakage.

Mandatory adversarial tests:

- supplied arbitrary ID with valid body is rejected;
- body mutation with original ID is rejected on load/verification;
- type/domain/peer/timestamp mutations each invalidate the ID;
- key-order changes inside a semantically identical payload remain
  deterministic;
- Merkle root cannot be made valid from forged IDs;
- duplicate IDs with different bodies refuse;
- valid historical fixtures remain byte/hash identical.

## 5. WP2: Full Chain Verification and Explicit Chain Identity

Owners: `doin-core`, `doin-node`

Implement a typed `ChainVerificationReport` and one authoritative verifier.
The verifier must check, in order:

1. SQLite integrity/foreign-key checks;
2. contiguous block indices from deterministic genesis;
3. exact configured genesis hash / chain ID;
4. each row's block hash against its canonical header;
5. each `previous_hash` against the preceding verified block;
6. transaction count and contiguous `tx_index` values;
7. every transaction-content hash;
8. every Merkle root;
9. snapshots against existing block index/hash;
10. metadata height/tip against verified rows.

Pruning semantics:

- never call a pruned chain fully verified when transaction bodies required for
  Merkle recomputation are absent;
- add explicit pruning/checkpoint metadata and return a typed
  `verified_suffix_from_checkpoint` result only when the checkpoint commitment
  and retained suffix verify;
- absence of required provenance is `unavailable/refused`, not success.

Runtime wiring:

1. Run verification after DB open and before gossip, sync, optimization,
   evaluation, dashboard acceptance or OLAP projection.
2. On failure, start only in a clearly reported quarantine/read-only diagnostic
   mode or exit with a typed error. Never append, sync, optimize or project
   OLAP from an unverified chain.
3. Add `chain_id` and `genesis_hash` to the versioned protocol/config and
   `ChainStatus`. Reject a peer before block exchange when either differs.
4. Preserve compatibility deliberately: a protocol-version mismatch must be a
   typed refusal, not field-default acceptance.
5. Bind OLAP ingestion to `(chain_id, genesis_hash, source_tip_hash,
   source_height)`. On reorg, invalidate/reproject affected local OLAP rows
   deterministically. The OLAP remains derived and rebuildable, never consensus
   authority.

Forensic deployment preflight:

1. Take read-only SQLite online backups of every current fleet chain and record
   SHA-256 before scanning.
2. Run the verifier against copies on omega, dragon and gamma. Compare chain
   ID, genesis, finalized anchor, tip and a deterministic full/suffix report.
3. If any existing mismatch is found, preserve all originals and stop the
   deployment. Return the exact first failing block/tx coordinate; do not repair
   or rewrite history silently.
4. Deploy only one reviewed `doin-core` and `doin-node` commit pair to all
   participant nodes at a job boundary. Prove printed commits and protocol
   version match before rejoining.

Mandatory integration tests include valid restart, corrupted historical body,
corrupted header, missing transaction row, duplicate/gapped tx index, wrong
genesis, wrong chain ID peer, pruned suffix, reorg plus OLAP reprojection, and
restart refusal before any network/optimizer side effect.

## 6. WP3: Make Front-1 Status Describe the Work That Is Actually Running

Owner: `agent-multi`

1. Add a first-class L1 factorial source to `tools/multifront_status.py`; do
   not infer it from the historical campaign supervisor.
2. Read durable launcher heartbeats/records from all four assigned workers and
   report identity, seed, host, assigned GPU UUID, unit/PID generation, cell,
   difficulty, LR multipliers, epoch/max, activity patience, trades by split,
   last progress time, restart count and typed terminal state.
3. Show the paused DOIN campaign separately as history. It must not replace the
   active factorial in `f1_optimization` or the executable queue.
4. Derive ETA only after enough observed epoch/cell durations exist. Publish
   formula, sample size, horizon and uncertainty; otherwise `unavailable` with
   the missing fact.
5. Add fixtures for exactly the current contradiction: paused supervisor plus
   four active factorial workers must render active Front 1.
6. Add zero-trade monitoring at the declared patience boundary. Do not mutate a
   running cell; emit one bounded alert when terminal inactivity is reached.

## 7. WP4: Repair Front-2 Evidence and Continue Business-Reality Work

Owners: `lts`, `agent-multi`

1. Fix audit-snapshot source precedence. Observer adapters describe quote/
   account observation; execution heartbeats describe writable authority.
   Expose both fields rather than overwriting one with the other.
2. Reconcile IBKR Paper from direct TWS facts. It is currently flat and held.
   Present the existing authenticated owner resume/hold-clear command with a
   pre/post evidence packet. Do not clear the hold automatically.
3. For Alpaca, IBKR and MT5, record direct current evidence for position,
   order, native SL, native TP, account binding, model ID and artifact/config/
   input/decision hashes. A missing protection fact is unavailable/refused, not
   inferred from an empty alert list.
4. Keep the current linear controllers honestly labeled as controls. Do not
   call them champions. Complete the already planned raw-bar-to-SAC observation
   parity and champion artifact registry before succession.
5. Champion succession must use accepted L0/L1 paths: stop new risk, reconcile/
   flatten where required, preserve the resulting Paper/Demo balance, switch
   one seat, prove first protected lifecycle, then continue without an idle
   monitoring gap. No direct broker bypass.
6. Produce rolling 24-hour and seven-day live-versus-simulation facts including
   spread, slippage, fill/ack latency, rejections, protection deviations,
   reconnects, decisions, trades and balance/equity change with explicit units.

## 8. WP5: Turn Front-3 Collection Into Bounded, Human-Governed Knowledge

Owner: `agent-multi`, Hermes remains a bounded worker rather than authority

1. Retry the eight failed enrichment runs idempotently, preserving original
   run IDs, error class, attempts and token reservation accounting.
2. Continue the current backlog without stealing GPU resources from Front 1.
3. Materialize one compact review packet per cadence from only
   `experiment_candidate`, `reply_candidate` and highest-value `investigate`
   rows. Include source URL/ID, untrusted-content flag, claims, confidence,
   target front, rationale and proposed bounded next action.
4. Add an owner-review ledger with `accept`, `defer`, `reject` and reason.
   Accepted experiments enter the research/work queue with provenance and a
   collision check. Accepted replies become drafts only; publishing remains a
   separate human action.
5. No post, model summary or Hermes output may execute code, alter a broker,
   modify a DOIN chain or publish automatically.

## 9. WP6: Complete Repository Presentation Without Runtime Churn

1. Transfer the accepted `agent-multi` README from `0d7c937b` to default
   `master` as a documentation-only commit. Do not move canonical fleet
   checkouts during the active factorial. Verify the GitHub README endpoint.
2. Leave `causal-inference` source untouched until the owner attributes or
   explicitly disposes its dirty files. Metadata may remain as delivered.
3. Leave the `preprocessor phase_6` branch unchanged; open a normal merge task
   rather than overwriting branch work.
4. Repair `predictor` test collection in a bounded code/test branch. Preserve
   active public APIs or add explicit migration aliases; remove tests only when
   they target truly retired behavior and document why. Acceptance is full
   collection plus focused current-plugin tests, not merely fewer errors.
5. Re-query GitHub after completion: all active owned default branches must
   expose a README; descriptions remain non-empty; topic counts remain 20;
   visibility/archive/default-branch invariants remain unchanged.

## 10. Return Packet

Return one versioned document containing:

- exact commit per repository and clean/pushed state;
- before/after reproduction for findings 201-208;
- chain-verifier reports on synthetic fixtures and fleet DB **copies**;
- focused and full test commands/results with environmental failures separated;
- active L1 factorial status proving non-interference;
- fresh Front-2/3/4 facts and authoritative source paths;
- explicit residual doubts and owner actions, if any.

Do not declare findings closed. Do not deploy blockchain changes until Musashi
reproduces the code/tests and the fleet-copy forensic report. Documentation,
status and social corrections may ship independently when their own tests pass.
