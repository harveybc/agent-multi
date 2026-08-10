# Independent Audit: Repository Presentation, Blockchain Integrity and Four Fronts

Date: 2026-08-10 America/Bogota
Auditor: General Musashi (Codex), independent auditor during the role swap
Subject branch: `satoshi/m0-aggregation-hardening`
Subject revisions: `agent-multi@3d9eaec6`, plus the per-repository commits in
`SATOSHI_III_REPOSITORY_PRESENTATION_DELIVERY_2026_08_10.md`
Runtime mutation by this audit: none

## 1. Verdict

**REPOSITORY PRESENTATION: ACCEPT WITH TWO EXPLICIT RESIDUAL ITEMS.**

Satoshi's delivery is substantially correct. All 20 claimed README commits
exist on GitHub. GitHub independently reports non-empty descriptions and
exactly 20 topics for all 21 owned repositories in the inventory. Nineteen of
the 20 rewritten READMEs are visible on their default branches; `agent-multi`
is the one exception because its README exists only on the campaign branch.
The deferred `causal-inference` README was correctly left untouched because
its worktree contains unattributed changes.

The reported cheap test evidence also reproduces: `gym-fx` collects 84 tests,
`agent-multi` collects 898, and `predictor` collects three tests but fails on
eight stale import errors. The last result is honestly documented, but it is
still active-repository test debt.

**BLOCKCHAIN INTEGRITY: THE BLOCK HEADER CHAIN IS REAL, BUT FULL CONTENT
INTEGRITY IS NOT CURRENTLY ENFORCED.**

DOIN computes each block hash from its header, and the header includes
`previous_hash` and the transaction Merkle root. Incoming append validation
checks index, immediate predecessor, Merkle root and block hash. However,
Merkle leaves are the supplied transaction IDs, and `Transaction` verifies a
transaction ID only when it creates one locally. A caller or stored row may
provide an arbitrary ID unrelated to the transaction body. Startup also does
not walk and validate the persisted history before the node starts operating.

The result is not theoretical: this audit constructed a transaction whose ID
did not match its body, appended it successfully, changed the persisted JSON
payload, and then appended the next block successfully. The current focused
chain suite remains green because it has no adversarial case for this defect.

**FOUR-FRONT RUNTIME: COMPUTE IS ACTIVE AND THERMALLY HEALTHY, BUT THE
CONSOLIDATED STATUS IS NOT DESCRIBING THE ACTIVE FRONT-1 WORKLOAD.**

All four assigned GPUs are running decision identity `2de49ea9225e2baf`.
Front 2 has two writable Paper/Demo venues carrying exposure, IBKR is safely
held and flat, Front 3 is collecting and enriching, and Front 4 is producing
snapshots. Findings below cover the mismatches that prevent those facts from
being one reliable operator view.

## 2. Findings

### AUD-F1-20260810-201 (S2): transaction IDs are not bound to transaction content

`doin-core/src/doin_core/models/transaction.py:73-75` computes an ID only when
the field is empty. A supplied ID is accepted without comparing it to
`compute_id()`. Both block validators compute the Merkle root from `tx.id`, not
from independently recomputed transaction-content hashes:

- `doin-node/src/doin_node/storage/chaindb.py:273-297`;
- `doin-node/src/doin_node/blockchain/chain.py:163-199`.

Independent temporary-database reproduction:

```json
{
  "forged_id_constructed": true,
  "tamper_not_detected_on_load": true,
  "next_block_accepted_after_tamper": true,
  "height": 3
}
```

Impact: the Merkle root commits to transaction ID strings, but those strings
do not necessarily commit to the stored payload/type/domain/peer/timestamp.
The claim that changing historical transaction content necessarily breaks the
chain is false in the current implementation.

### AUD-F1-20260810-202 (S2): append and startup validate only the tip transition, not persisted history

`UnifiedNode` opens `ChainDB`, creates genesis only for an empty DB, and then
continues startup (`doin-node/src/doin_node/unified.py:761-773`). No
`verify_chain`/`validate_chain` routine exists in the active source tree.
`append_block()` validates the new block against `height` and `tip_hash` but
does not verify earlier rows. The separate OLAP projector accepts a caller's
block list and inserts facts without a verified source-chain anchor
(`doin-node/src/doin_node/stats/olap_db.py:293-344`).

Impact: local corruption or a malformed imported history can survive restart,
feed OLAP and accept subsequent blocks. Any pruned-body mode also lacks an
explicit verified-checkpoint contract distinguishing full verification from
suffix-only verification.

### AUD-F1-20260810-203 (S3): peer chain identity is implicit rather than an explicit protocol field

Genesis itself is deterministic and therefore a valid identity anchor.
However, `ChainStatus` currently carries only height, tip hash/index and
finalized height (`doin-core/src/doin_core/protocol/messages.py:182-188`). It
does not carry a protocol/network chain ID or genesis hash for rejection before
sync/fork work begins.

Impact: same-chain membership is discovered indirectly and late. A configured,
explicit identity would make accidental parallel-chain participation and wrong
seed configuration diagnosable before blocks or populations are exchanged.

### AUD-F1-20260810-204 (S3): the consolidated status omits the active L1 factorial

At 2026-08-10 00:30 COT, `tools/multifront_status.py` reported Front 1 as the
paused historical job `eth-4h-anchored-full-sac-shared-v2`, generation 0,
1/20. The code and its tests contain no L1 factorial source. At the same time,
direct systemd/process/GPU evidence showed these active workers:

| Seed | Host/GPU | Epoch sampled | GPU util | Temperature |
| --- | --- | ---: | ---: | ---: |
| 101 | omega / RTX 4070 Laptop | 17/1996 | 35% | 54 C |
| 202 | dragon / RTX 4090 Laptop | 32/1996 | 38% | 51 C |
| 303 | gamma / RTX 5070 Ti Laptop | 33/1996 | 40% | 49 C |
| 404 | gamma / RTX 5090 | 26/1996 | 44% | 56 C |

All were in `L1_N_M10`, identity `2de49ea9225e2baf`. All four had zero trades
on train/train-tail/validation. That is not yet a terminal result: activity
patience starts at epoch 40. It is nevertheless a mandatory live diagnostic.

Impact: the owner-facing status can say Front 1 is paused while every GPU is
working, and cannot report current cell, progress, activity patience or ETA.

### AUD-F4-20260810-205 (S3): the audit snapshot contradicts execution truth for writable venues

The six-hour audit snapshot labels Alpaca and IBKR `read_only=true` using
observer adapter facts. The fresh execution heartbeats used by
`multifront_status.py` correctly label both `write_enabled`/`read_only=false`.
At the same sample Alpaca had one order and one position; IBKR was flat with
`halt=hold`; MT5 had one authorized Demo position.

Impact: the evidence product can describe a writable execution seat as
read-only. Source precedence must distinguish observer capability from
execution-runner authority instead of allowing the observer label to overwrite
the runtime fact.

### AUD-GEN-20260810-206 (S3): audit findings 178-200 were fragmented during branch integration

The subject branch's findings register ends at 177. Commit `a0b8f18a` contains
the accepted 188-200 register section, but `939b6fac` was created from the
earlier `fe6224aa` tree and re-added selected reports without restoring that
section. Findings 178-187 and their originating report remain only on the
`cc3f02ee` audit branch.

Impact: the active technical-lead branch contains the corrections and final
acceptance reports but loses part of the append-only audit chronology and the
current disposition of 23 findings. This audit restores the 188-200 summary to
its own register and orders exact-object restoration for 178-187.

### AUD-GEN-20260810-207 (S4): `agent-multi` still has no README on its default branch

Commit `0d7c937b` exists and its README is accepted, but the GitHub default
branch is `master` and the GitHub README endpoint returns 404. Waiting for the
entire campaign branch to merge is unnecessary: a documentation-only commit
can be transferred without moving the fleet checkout or changing runtime
identity.

### AUD-GEN-20260810-208 (S4): active `predictor` test collection is broken

The README states this honestly, and the audit reproduces it: three tests are
collected and eight modules fail import due to removed/moved symbols and stale
module paths. This is not a defect in the README delivery. It is active-core
maintenance debt that weakens safe future changes to the external DOIN domain
plugin.

## 3. Four-Front Audit

### Front 1: research and optimization

- **Runtime:** four durable `l1-factorial@*.service` workers active; exact GPU
  assignment holds; temperatures healthy; no OOM evidence sampled.
- **Current scientific state:** first normal/baseline-LR cell running on every
  seed; zero activity so far; no typed factorial result exists yet.
- **Do not mutate:** preserve the active cells. Let the declared activity rule
  terminalize them. Do not change thresholds, learning rates or checkpoints in
  place.
- **Required correction:** status integration (finding 204) and blockchain
  integrity work on isolated branches/copies (findings 201-203).

### Front 2: business reality and live Paper/Demo

- **Alpaca Paper:** writable runner fresh; one SPY position and one open order;
  current controller `spy-daily-linear-live-v1`.
- **IBKR Paper:** TWS reachable, runner fresh and account-bound, zero positions
  and orders; `halt=hold`, so decisions are rejected safely.
- **OANDA MT5 Demo:** connected, execution enabled, one authorized position,
  zero open orders; bridge v2 fresh.
- **Truthful limitation:** all three seats still name linear live controllers,
  not a promoted loadable SAC champion. This is already an open work-plan gap,
  not a duplicate finding. The next champion switch must preserve balance,
  flatten/reconcile at the accepted boundary, and bind artifact/config/input/
  decision hashes.
- **Required correction:** reconcile the IBKR hold through the existing owner
  path, prove current SL/TP/protection directly per venue, and fix audit-source
  precedence (finding 205). No Live capital is authorized.

### Front 3: social intelligence

- 8,234 posts collected; 436 enriched; 432 eligible backlog; 55 complete and
  eight failed enrichment runs; zero drafts and publishing remains human-gated.
- The OLAP view contains 170 `investigate`, 19 `experiment_candidate` and 13
  `reply_candidate` rows. The content includes potentially relevant multi-agent
  failure, experiment-design, live/simulation and security observations.
- **Required correction/improvement:** idempotently retry failed batches,
  produce a small owner review packet, and record accepted items in a
  provenance-bound experiment/research queue. Posts must never directly mutate
  code, runtime, models or publishing state.

### Front 4: audit, continuity and operator evidence

- Audit snapshots, watchdogs and test packets are active; no current fleet
  alert was observed.
- The snapshot sees healthy GPUs and no OOM delta, but Gamma has only about
  0.50 GiB memory available and remains the host requiring trend monitoring.
- Findings 205-206 show that evidence exists but is not yet one internally
  consistent operator contract.
- **Required correction:** restore the append-only finding history, expose open
  finding/closure counts in status, and make each status field cite its
  authoritative source and freshness.

## 4. Verified Non-Findings

1. The block hash does include the previous block hash through the header.
2. Genesis uses a fixed timestamp and is deterministic for a fixed generator.
3. All four current GPU workers are active on their assigned physical devices.
4. The README/About/topics work did not mutate the running factorial.
5. GitHub visibility, archive state and default branches were preserved.
6. Current broker holds/refusals are fail-closed; no evidence of a Live account
   or real-capital submission was found.

## 5. Disposition

- Satoshi executes
  `MUSASHI_TO_GENERAL_SATOSHI_III_BLOCKCHAIN_AND_FOUR_FRONT_CORRECTION_ORDER_2026_08_10.md`.
- No new owner phrase is required for corrective code, documentation, tests or
  status work already inside the approved mission.
- Clearing the IBKR Paper hold remains an authenticated owner action after
  direct flat/protection reconciliation; Satoshi must present the exact action
  and evidence rather than inventing an approval ceremony.
- Findings 201-208 remain open. Satoshi closes none; this auditor reproduces
  the return packet before any blockchain deployment or finding closure.
