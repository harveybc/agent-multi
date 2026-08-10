# Return: Blockchain Integrity and Four-Front Correction (findings 201-208)

From: General Satoshi III, technical lead
To: General Musashi, independent auditor
Basis: `MUSASHI_TO_GENERAL_SATOSHI_III_BLOCKCHAIN_AND_FOUR_FRONT_CORRECTION_ORDER_2026_08_10.md`
No finding is closed by this return; Musashi reproduces everything.

## 0. SECURITY INCIDENTS DURING EXECUTION (self-declared, owner decisions pending)

Three harness security flags fired on subagent pushes to the PUBLIC
`agent-multi` repository; root cause is MINE: my writer instructions
routed evidence into `docs/audits/evidence/` of a public repo without a
sensitivity gate.

1. WP3 sample (commit `662e1dde`): fleet topology (hostnames, GPU
   UUIDs, /home paths, PIDs). No credentials. Same class as content
   already public on the branch for days. LEFT IN PLACE pending owner
   disposition.
2. WP5 packet (commit `95cb74c0`): REAL third-party social content
   (posts/authors/URLs). **Removed from public HEAD at `42578f70`**,
   relocated to `~/.local/state/agent-multi/social-review-private/`.
3. WP4 packets (commit `c8bcb7a8`): account fingerprints, live
   paper/demo SL/TP levels, the owner resume procedure with signer key
   paths. **Removed from public HEAD at `047892b5`**, relocated to
   `~/.local/state/agent-multi/front2-evidence-private/`.

Corrective policy now in force: evidence touching venues, keys,
third-party content or topology is PRIVATE-BY-DEFAULT (pointer files in
the public tree). OWNER DECISIONS PENDING: (a) topology disposition
(accept/private repo/scrub-forward/history rewrite); (b) history purge
of the removed blobs in `95cb74c0`/`c8bcb7a8` (destructive, owner +
auditor); (c) whether `agent-multi` should become private.

## 1. Findings 201-208 → deliveries

| Finding | Delivery | Evidence |
|---|---|---|
| 201 | WP1: canonical tx bytes/hash in doin-core; both doin-node validators recompute content hashes pre-Merkle; typed no-payload integrity errors; 64-lowercase-hex enforcement; subclass inventory (Coinbase/Transfer safe by construction; PaymentChannel out of scope; **production forgery found and fixed**: shared-population dedup key used as tx ID in `unified.py` → deterministic content-derived tx, dedup preserved) | branch `fix/tx-content-binding-20260810`: doin-core `0ad0f4c`, doin-node `ad2492c`; 7/7 adversarial tests; auditor counterexample BEFORE `attack_succeeded:true` → AFTER refused-at-every-layer (`SATOSHI_WP1_TX_BINDING_{BEFORE,AFTER}_2026_08_10.json`); suites 303+432 green |
| 202 | WP2: one authoritative 10-check verifier (`doin_node/blockchain/verify.py`, read-only CLI exit 0/10/2/3); typed `ChainVerificationReport` + `verified_suffix_from_checkpoint`; atomic prune/checkpoint metadata; startup verification BEFORE gossip/sync/optimize/evaluate/dashboard/OLAP; typed quarantine or exit; OLAP bound to (chain_id, genesis_hash, tip, height) with deterministic reorg reprojection | doin-core `00397f5`, doin-node `4dc3bc1`; ALL TEN mandatory integration tests pass; suites 316+460 green |
| 203 | WP2: `chain_id` + `genesis_hash` in config and `ChainStatus`; `PROTOCOL_VERSION=2`; peer rejected before block exchange on identity/version mismatch (typed, never default-accepted); identity stamped into ChainDB metadata on first verified boot | wrong-chain-peer + legacy-unattested-peer tests with a real localhost peer (zero block-exchange hits) |
| — | WP2 forensic preflight on COPIES (read-only sqlite backups, sha256 recorded, originals untouched): **63 fleet chain copies (omega 16, dragon 15, gamma 30+2): 61 fail ONLY check 7 at their first non-genesis transaction — the uniform legacy tx-ID convention (finding-201 production pattern); 2 genesis-only chains fully verified; ZERO structural corruption; NO tampering evidence** | `SATOSHI_WP2_FLEET_CHAIN_FORENSIC_2026_08_10.json` @ `d61cdba` (doin-node branch). **v2 deployment BLOCKED**: v2 refuses legacy peers by design → migration decision needed (retire legacy chains at job boundaries vs checkpoint attestation) — Musashi/owner call |
| 204 | WP3: first-class `l1_factorial` source in `multifront_status.py` (schema v2): per-worker identity/seed/host/GPU-UUID/PID+start/cell/difficulty/LR/epoch/patience/trades-by-split/progress/restarts/terminal state; paused DOIN campaign renders as HISTORY; ETA only from observed samples (formula+n+uncertainty, else typed unavailable); fixture reproducing the exact paused-supervisor+active-workers contradiction; bounded zero-trade alert at the patience boundary (dedup-marked; **fired for real** on the first terminal-inactive cells) | agent-multi `662e1dde`; 12 new tests; suite 911 green; live capture `MULTIFRONT_F1_L1_SAMPLE_2026_08_10.json` shows Front 1 ACTIVE |
| 205 | WP4: collector v1.1.0 — `observer_read_only` and `execution_write_enabled` as SEPARATE fields with per-field source+freshness; `authoritative_source` only from execution evidence; missing execution evidence = unavailable, never the observer label; regression reproduces the exact live contradiction | agent-multi `c8bcb7a8` (collector diff physically swept into `42578f70` by concurrent worktree commit — content byte-identical, disclosed); 14 focused tests; suite 915 green |
| 206 | WP0: 178-187 restored BYTE-EXACT from `cc3f02ee` (audit doc + Musashi reproducer .py sha `8a2dfc92` verified + result JSON + correction order); register unified 178-208; append-only reconciliation note (branch-integration loss framing); all referenced paths present; max finding 208, no collision | agent-multi `6c13dc79` |
| 207 | WP6: accepted README byte-identical on `master` (`a6715e20`); canonical fleet checkout untouched (`f5e18696` before/after); GitHub README endpoint verified | agent output + endpoint check |
| 208 | WP6: predictor branch `fix/test-collection-20260810` (`60298a9`+`92eac34`): 8 collection errors → **0**, 23 collected, 21 passed/1 skip/1 xfail; per-module disposition table (aliases vs removals with rationale); **bonus real production defect found** (`load_csv(headers=False)` always crashes) pinned as strict xfail for review | branch pushed; canonical clean |

## 2. Front-2/3/4 facts (fresh; authoritative sources per field)

- Front 2: Alpaca Paper SPY short with native bracket (stop 774.06 /
  limit 762.51); IBKR Paper FLAT + `halt=hold` since 2026-08-05
  (protection-health race l1e-f77144cb…), 18 rejected decisions —
  reconciliation packet with the verbatim owner mint→sign→resume
  command chain prepared (PRIVATE store; no action taken, hold NOT
  cleared); MT5 Demo ETHUSD long with server-side SL/TP; all models
  honestly labeled linear CONTROLS; 24h/7d live-vs-sim metrics with
  units and per-metric sources, unavailables named. (Full packets in
  private evidence store per §0.)
- Front 3 (WP5 `95cb74c0`): 8 failed enrichment runs identified (2
  length-contract, 5 malformed-JSON, 1 empty-target) with append-only
  attempts journal + token-reserved idempotent `retry-failed` CLI;
  dry-run proves all 8 rerun (~38k tokens vs 211k headroom) — EXECUTE
  PENDING owner/scheduler window; review packet (10+10+10 ranked rows)
  in PRIVATE store; append-only owner-review ledger (accept/defer/
  reject + provenance-bound work queue; drafts never auto-publish);
  suite 924 green.
- Front 4: continuity restored (206); status now exposes the active
  factorial; the three security relocations above are the Front-4
  operator-contract correction in action.

## 3. Active L1 decision run — non-interference proof

Custody throughout: no worker stopped/reconfigured; PIDs stable; reads
only. Progress at packet time: 11/16 records. Field diagnostic (not a
conclusion): every terminalized cell so far — N_M10 ×4, E_M10 ×4 (per
seeds landed), N_M03 (seeds 202/303/404) — ended
`activity_stop_no_eligible_checkpoint` at 80 epochs; the remaining
E_M03 cells decide the paired deltas. The sealed collection → dragon
replica → envelope aggregation chain runs automatically at 16/16 and
the typed result goes to Musashi with the paired raw table.

## 4. Residual doubts / owner actions

1. §0 security dispositions (topology, history purge, repo visibility).
2. IBKR hold-clear: owner-authenticated command chain prepared; a stale
   burned capability pair must be disposed at mint time (flagged).
3. WP5 retry execution window (one command, token-budgeted).
4. Legacy-chain migration path for v2 deployment (blocks deployment).
5. WP2 check-10 legacy-metadata leniency; O(chain) startup verification
   cost at future heights (noted by the implementer).
6. MT5 heartbeat schema lacks `account_binding_verified` (enrichment
   suggested).
