# Independent Audit: Blockchain and Four-Front Return

Date: 2026-08-10 America/Bogota
Auditor: General Musashi (Codex), independent auditor during the role swap
Subject: `SATOSHI_III_BLOCKCHAIN_FOUR_FRONT_RETURN_2026_08_10.md`
Subject revisions: `agent-multi@6a762a82`, `doin-core@00397f5`,
`doin-node@d61cdba`, `predictor@92eac34`
Runtime mutation by this audit: none

Reproducer and captured output:

- `evidence/MUSASHI_BLOCKCHAIN_RETURN_REPRO_2026_08_10.py`;
- `evidence/MUSASHI_BLOCKCHAIN_RETURN_REPRO_RESULT_2026_08_10.json`.

## 1. Verdict

**RETURN NOT YET ACCEPTED.** The principal blockchain repair is real:
transaction IDs are now content-derived and independently recomputed by both
append validators, startup has a ten-check read-only verifier, peer status
carries protocol/chain/genesis identity, and OLAP projection records chain
provenance. Independent suites reproduce:

- `doin-core`: 316 passed;
- `doin-node`: 460 passed;
- `agent-multi`: 924 passed;
- `predictor`: 21 passed, 1 skipped, 1 strict xfail.

The audit nevertheless reproduces seven remaining defects. Two affect the
chain-integrity claim, three make the new owner status materially misleading,
one leaves sensitive evidence reachable in a public Git history, and one is a
real production bug deliberately left as xfail. Findings 201, 204 and 208 are
therefore only partially corrected; 202-203 remain open through the new
counterexamples. Findings 205-207 are code/document corrected but await a
clean integration/deployment proof after this round.

The active L1 run was not mutated. At the audit sample it had 11/16 landed
records and four running workers under identity `2de49ea9225e2baf`.

## 2. Reproduced Improvements

1. A forged transaction ID is refused by the model and both block validators.
2. Historical transaction-body mutation is detected by `verify_chain_db()`.
3. Wrong protocol/chain/genesis peers are refused before block exchange when
   explicit identities are supplied.
4. Startup verification precedes gossip, sync, optimization, evaluation,
   dashboard acceptance and OLAP projection.
5. The first-class L1 status source correctly detects the active factorial and
   keeps the paused DOIN job as history.
6. The audit collector now separates observer read-only capability from
   execution write authority.
7. Findings 178-208 are present in one append-only register, the accepted
   README is on `agent-multi` master, and predictor collection is restored.

## 3. Findings

### AUD-F1-20260810-209 (S2): metadata integrity can pass incomplete claims or crash untyped

`doin-node/src/doin_node/blockchain/verify.py:644-671` treats `(height,
tip_hash)` as optional independently. Deleting either member from a valid DB
still returns `fully_verified`; replacing height with `not-an-int` raises raw
`ValueError` from `int()` because `_Verifier.run()` does not translate check
exceptions into a typed report.

Independent result:

```text
missing_tip_hash -> fully_verified / check 10 PASS
missing_height   -> fully_verified / check 10 PASS
bad_height       -> ValueError
```

Both metadata values must be absent for a declared legacy DB or present and
valid as a pair. Every malformed value must yield a typed `FAIL` or
`UNAVAILABLE`, never crash the verifier.

### AUD-F1-20260810-210 (S2): verified startup does not bind later appends to the verified history

`ChainDB.append_block()` validates the incoming block and immediate tip
(`chaindb.py:282-342,397-450`) but carries no verified-history cursor or
incremental integrity token. This audit fully verified a temporary chain,
mutated a historical transaction row through a second SQLite connection, and
then appended the next valid block successfully. A subsequent full verifier
correctly failed, but the append had already committed:

```text
append_after_history_tamper=ACCEPTED height=3
subsequent_full_verify=failed
```

Do not rescan O(chain) before every block. Bind append to a startup-verified
tip/height plus SQLite `data_version` or an equivalent tamper-evident cursor,
verify the current metadata/tip in the same write transaction, and run a
periodic/background full verifier with quarantine on failure. Archive and OLAP
projection must require a fresh successful report.

### AUD-F1-20260810-211 (S3): chain identity exists in code but no working config pins it

All 97 shared-population JSON examples omit `chain_id` and `genesis_hash`.
`UnifiedNode` therefore defaults every deployment to the same deterministic
`doin-<genesis-prefix>` identity (`unified.py:315-321,410-419`). Two unrelated
networks with the default genesis pass the new early identity check and are
distinguished only later as competing histories.

Shared-population production configs must require one explicit, common
network/chain identity across all participating machines. The materializer
must emit and validate it; omission must fail closed for production/shared
population mode. Existing legacy chains remain read-only evidence.

### AUD-F1-20260810-212 (S3): L1 status reads stale, cell-unbound logs as current facts

`tools/multifront_status.py:667-703` reads the global
`<output_root>/logs/seed*.log`, accepts it regardless of age or active cell,
and combines it with a fresh launcher heartbeat. In the live sample those logs
were about 12.7 hours old. The packet consequently reported epochs 34/61/72/70
while direct service journals reported 15/54/62/5 for the current cells. It
also displayed the previous cell's `attempt` path as if it belonged to the
current heartbeat.

Epoch, trade and patience facts must be bound to `(identity, seed, cell,
attempt)` and carry source freshness. A stale or differently bound source is
`unavailable`, not current telemetry. The launcher should publish its active
attempt and progress directly or write one per-attempt timestamped log.

### AUD-F1-20260810-213 (S3): aggregate ETA serializes parallel work

`_l1_cell_eta()` (`multifront_status.py:501-524`) computes mean cell duration
times all remaining cells. With four concurrent workers it reported about
16.2 hours for five cells although the critical path is Omega's current cell
plus one queued cell, approximately eight hours from observed per-seed
durations. The existing unit test explicitly enforces the serial formula.

Compute remaining duration per worker from its active/queued cells and report
the maximum worker path. Expose current-cell and full-experiment ETAs
separately, each with sample count and uncertainty.

### AUD-F2-20260810-214 (S3): executable queue still calls completed IBKR L1 work dependency-blocked

`multifront_status.py:1327-1337` unconditionally appends the historical
`ibkr-paper-l1-canary` as dependency-blocked. Fresh durable evidence shows a
write-enabled IBKR runner with four lifecycles and a currently enforced hold.
The correct state is operational-but-held, not missing-adapter/preflight.

Derive this queue item from execution heartbeat/journal facts. A broker hold
is an operational state with its exact reason and owner action, not an old
development dependency.

### AUD-SEC-20260810-215 (S2): removed evidence remains public in Git history

`harveybc/agent-multi` is public. The social and Front-2 packets removed at
`42578f70` and `047892b5` remain retrievable from earlier public commits, and
the topology sample remains at public HEAD. The return reports no credentials,
but it does include third-party content and operational/account metadata.

No further sensitive evidence may be pushed. Make the repository private as
the immediate containment, then perform an owner-approved history rewrite and
force-push from a verified mirror, or scrub the history before restoring
public visibility. Pointer-only public evidence is the standing contract.

### AUD-GEN-20260810-216 (S4): predictor production bug is hidden behind a strict xfail

`predictor/app/data_handler.py:39` calls `.strip()` on integer labels whenever
`load_csv(headers=False)` uses pandas `header=None`. The default public API is
therefore broken. `test_load_csv_without_headers` marks this as xfail rather
than correcting it. Collection is restored, but the active-core repair is not
complete while a default path knowingly crashes.

Normalize the label to `str(c)` for detection (or rename before detection),
remove the xfail, and prove header/no-header/date/max-row cases.

## 4. Four-Front Status

### Front 1: optimization and research

- Active identity: `2de49ea9225e2baf`, ETHUSD L1 matched factorial.
- Landed: 11/16 cells; four workers running; no service restarts.
- All 11 landed cells stopped after 80 phase-2 epochs with no
  activity-eligible checkpoint. This is evidence, not yet the typed factorial
  result; five cells remain.
- All GPUs are thermally healthy. Do not mutate or stop this run.

### Front 2: Paper/Demo business reality

- Alpaca Paper: write-enabled SPY short; direct broker snapshot contains the
  filled parent and both native bracket children.
- IBKR Paper: TWS reachable, flat, write-enabled runner fresh, `halt=hold`;
  decisions are being rejected for that exact reason.
- OANDA MT5 Demo: connected, execution-enabled, one authorized ETHUSD
  position, server-side protection previously evidenced.
- No active watchdog event. Controllers remain linear controls, not the
  promoted SAC champion.

### Front 3: social intelligence

- 8,513 posts collected; 500 enriched; 394 eligible backlog.
- 63 complete and 9 failed enrichment runs; 24 experiment candidates and 15
  reply candidates; zero drafts/publications.
- Execute retry only through the idempotent token-budgeted path, then present
  the private owner-review packet. Posts never gain execution authority.

### Front 4: audit and continuity

- Audit/register continuity and full test packets are present.
- The status/audit fixes are branch-delivered, not yet one deployed canonical
  revision. Findings 212-215 block declaring the operator surface complete.
- The public-history incident requires owner disposition now.

## 5. Deployment and Migration Disposition

Do not rewrite 61 legacy chains into v2: changing transaction IDs changes
Merkle roots, block hashes and every descendant. Preserve each as immutable
legacy evidence with hashes and its OLAP projection. At the next DOIN job
boundary, after findings 209-211 are independently verified, start one new v2
chain with an explicit shared chain ID/config hash on every participant. No
mid-chain rollout and no deployment during the active L1 factorial.
