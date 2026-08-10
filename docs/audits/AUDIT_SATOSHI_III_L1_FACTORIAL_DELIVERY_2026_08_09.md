# Independent Audit: Satoshi III L1 Matched Factorial Delivery

Date: 2026-08-09 America/Bogota
Auditor: General Musashi (Codex), independent auditor during the role swap
Subject revision: `agent-multi@8deccdb30da1982b504d36610191acf75016c05e`
Subject branch: `satoshi/m0-aggregation-hardening`
Runtime mutation by this audit: none

## 1. Verdict

**REJECTED AS A FINAL DELIVERY. NO L1 OUTCOME EXISTS YET.**

The only committed delivery document is explicitly a draft and says that all
16 decision cells and the typed outcome are pending. The corrected experiment
identity `16acf854c83b5051` had `0/16` cell records at the independent runtime
sample. Only Omega seed 101 and Dragon seed 202 were executing that identity;
Gamma's two GPUs were busy with doomed first-dispatch processes, not queued
corrected seeds. Therefore none of `EASY_CONTRIBUTES`, `LR_ONLY`,
`INTERACTION`, `EASY_HARMFUL` or `INCONCLUSIVE` is currently supported by
runtime evidence.

The nested chronology, paired comparator and ordered outcome function are
substantial and their focused tests pass. The decision envelope is not safe:
the aggregator can promote with missing raw metrics or falsified code identity,
the runner does not bind the exact terminal artifact/system/budget to its
record, the dispatcher has no exclusive claim or idempotent recovery, and no
distributed collection path brings the four hosts into one verifiable packet.

Do not stop the currently useful Omega/Dragon diagnostic work. Do not publish
or consume a promotion outcome from this experiment identity.

## 2. Findings

### AUD-F1-20260809-178 (S2): completion and four-worker claims contradict durable runtime facts

The subject document is `DRAFT - NOT an audit request` and retains all runtime
facts as `PENDING`. At 2026-08-09 19:07 COT the corrected identity had zero
records. Omega seed 101 and Dragon seed 202 were valid corrected launches.
Gamma seed 303 and 404 were processes started at 18:20 COT from the lost
dispatch; the corrected launch was only described as queued. No durable queue,
service or watcher exists to perform that transition.

Impact: the package cannot support any outcome and the corrected run is not a
four-worker execution. A prose queue can silently become two idle GPUs.

### AUD-F1-20260809-179 (S2): dispatch and record publication are duplicate-writer unsafe

`dispatch_l1_factorial_fleet.sh` performs unconditional `nohup` launches with
fixed seed log paths. It has no host/seed assignment assertion, `flock`, durable
claim, existing-process check or completed-record reuse. `run_cell()` writes
directly to a fixed `l1_cell_record.json` path without an atomic temporary-file
replace. The first dispatch already produced the predicted duplicate seed-404
writer. Relaunching also overwrote the first attempt's seed logs.

Impact: two writers may train and publish into one physical identity; crash or
manual relaunch can destroy incident evidence and create an ambiguous result.

### AUD-F1-20260809-180 (S2): absent mandatory raw metrics can still promote

The repair specification requires missing metrics to force refusal or
`INCONCLUSIVE`. `raw_metrics()` merely adds an `absent` label. It never adds a
refusal and `decide_outcome()` never consumes raw-metric completeness.

Independent counterexample: remove `results.json` from one otherwise accepted
cell. The aggregator reports the missing file but still returns
`EASY_CONTRIBUTES` with zero refusals. The existing test only asserts that the
word `absent` is emitted; it does not assert the required outcome.

### AUD-F1-20260809-181 (S2): execution identity omits and does not validate decision-bearing facts

The required identity is contract + system manifest + asset + seed + cell
factors + anchor artifact/tensor + data/config/observation hashes + budgets +
metric schema + actual code identity. The current experiment ID contains only
contract SHA, nested-split-contract SHA, two Git revisions and profile. Records
without the following fields are accepted with no binding refusal:

- system-manifest SHA;
- resolved-config SHA;
- observation-manifest SHA;
- terminal artifact SHA and terminal tensor SHA;
- requested and realized phase budgets.

The runner also calls the ETH-specific `_base_config()` instead of the ordered
typed system materializer. `code_identity_expected` remains a stale short hash
(`5322d42a`) and is not consumed by any executable.

Independent counterexample: replace a record's `code_revisions` with arbitrary
values. Aggregation still returns `EASY_CONTRIBUTES` with zero refusals.

### AUD-F1-20260809-182 (S2): terminal-model continuity is computed, not bound to the record

The record stores a mutable path but no terminal artifact digest and no terminal
tensor digest. `probe_terminal()` loads whatever bytes occupy that path later.
It rehashes the phase-1 artifact against the boundary, but does not compare the
terminal artifact/tensor against an immutable value recorded by the producing
cell. The output itself declares this deviation.

Impact: a replaced but loadable terminal can become the policy evaluated by the
aggregator without proving it is the model produced by the cell.

### AUD-F1-20260809-183 (S2): no distributed collection and replica gate exists

Each worker writes its seed subtree to its own local filesystem. The aggregator
scans one local `output_root`; neither the dispatch script nor the delivery
contains a collector, collision-safe import, source-host manifest or independent
replica. Sixteen successful remote records therefore do not automatically form
one auditable aggregation root.

Impact: a final packet cannot be produced from the delivered workflow without
an unreviewed manual copy step.

### AUD-F1-20260809-184 (S3): the code-motion guard observes a different checkout from executed code

`d1._git_rev()` is hard-coded to `/home/harveybc/Documents/GitHub/<repo>`.
Omega executes `l1_factorial_screen.py` from a temporary Satoshi worktree while
the identity and before/after guard inspect the canonical checkout. At the
runtime sample, the executing worktree was at `8deccdb3` while the canonical
checkout used for identity was `9b6f0745`.

Impact: the guard can claim stable code while the actual source tree differs or
moves. The subject-code revision published later by the aggregator is also the
aggregation-time canonical revision, not necessarily the producing revision.

### AUD-F1-20260809-185 (S3): clean-checkout bootstrap does not pin DOIN

`bootstrap_test_fixtures.py` clones the current default branch of `doin-node`
and reports the resulting revision, but has no required revision or tree hash.
The clean suite is runnable today but is not reproducible from the same subject
commit after the remote default branch moves.

### AUD-F1-20260809-186 (S3): CLI and raw-return semantics are unsafe for automation

The aggregator returns exit 0 for `INCONCLUSIVE` because that value belongs to
`OUTCOMES`. A scheduler checking only the process exit code can advance after a
refusal. `total_return` also divides by a hard-coded 10,000 rather than a
hash-bound initial-cash fact from the materialized cell.

### AUD-GEN-20260809-187 (S3): the full suite mutates tracked campaign files

`test_full_v2_recovery_plan_has_one_fresh_shared_domain()` calls the production
materializer without redirecting its outputs to `tmp_path`. An independent full
suite run modified the tracked full-v2 optimization config, campaign plan and
three machine profiles in the audit checkout. This was reproduced twice and the
audit branch was restored afterward.

Impact: a green clean-checkout run does not leave a clean checkout, can alter
decision-bearing fixtures, and makes subsequent source-identity checks depend on
test order.

## 3. Reproduction Evidence

Executable evidence:

- `docs/audits/evidence/repro_runs/MUSASHI_L1_FACTORIAL_DELIVERY_REPRO_2026_08_09.py`
- `docs/audits/evidence/repro_runs/MUSASHI_L1_FACTORIAL_DELIVERY_REPRO_2026_08_09.json`

Observed counterexamples, socket-free:

| Mutation | Required | Actual |
| --- | --- | --- |
| delete one `results.json` | `INCONCLUSIVE` | `EASY_CONTRIBUTES`, 0 refusals |
| falsify one record's code revisions | `INCONCLUSIVE` | `EASY_CONTRIBUTES`, 0 refusals |
| omit seven identity/artifact/budget fields | binding refusal | 0 binding refusals |
| invoke fixed dispatcher twice | second launch refused/reused | no lock, claim or reuse path |

Tests independently run from a detached clean checkout of `8deccdb3`:

- focused L1/nested/curriculum set: **107 passed**;
- clean full suite after declared fixture bootstrap: **845 passed**, 2 warnings,
  but it left five tracked campaign files modified (finding 187);
- adversarial reproducer: all four defects reproduced, `network_used=false`.

Green tests therefore establish implementation consistency, not contract
completeness.

## 4. Runtime Facts

Independent sample: 2026-08-09 19:07 COT.

| Worker | Process | Correct identity | Records | GPU sample |
| --- | --- | --- | --- | --- |
| Omega / seed 101 | active since 18:45, ~100% CPU | yes | 0 | RTX 4070, 54 C, 35% |
| Dragon / seed 202 | active since 18:45, ~96% CPU | yes | 0 | RTX 4090, 52 C, 38% |
| Gamma / seed 303 | active since 18:20 | no, lost dispatch | 0 | RTX 5070 Ti, 49 C, 37% |
| Gamma / seed 404 | active since 18:20 | no, lost dispatch | 0 | RTX 5090, 56 C, 44% |

Corrected contract SHA:
`a4cb963fac8c1e2b7c84ed51de8a2291529e7af16383193b7fe8937d4b13fead`.
Corrected decision experiment ID: `16acf854c83b5051`.

All GPUs were physically occupied, but only two workers were producing evidence
for the corrected identity. Utilization was sampled, not averaged.

## 5. Verified Non-Findings

The following areas did not fail this audit and should not be rewritten:

1. The nested split implementation preserves chronological roles, excludes the
   sealed 2025 split from this decision run, and reproduces the declared row
   counts.
2. The pure ordered outcome rules implement the declared five labels in the
   specified order.
3. Smoke records are schema-ineligible for decision aggregation.
4. Missing activity facts are invalid rather than silently inactive.
5. The fixed L1 stopping parameters use the owner-approved 2,000 safety cap,
   patience 60 and start floor 40; this audit does not revive the rejected
   14-epoch design.
6. The first-dispatch loss and new experiment identity are disclosed in prose;
   finding 179 concerns durable raw evidence and writer safety, not concealment.

## 6. Disposition

- No owner decision is required now.
- Preserve the current outputs as diagnostics; they are not promotion evidence.
- General Satoshi III implements the accompanying correction order.
- General Musashi independently reruns the counterexamples, clean suite,
  distributed collection and a bounded corrected smoke before accepting a new
  full decision identity.
- A final audit request is permitted only after 16/16 corrected records, one
  verified collection root, one independent artifact replica and one typed
  aggregation exist.
