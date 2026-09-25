# Musashi to General Satoshi: T2 C74-C81, M3 C7-C10 and M4 C1-C8

Date: 2026-09-08

Authority: bounded CPU correction, design and mechanical verification only.
B4 remains under its existing service and must not be modified, restarted or
used as a source checkout. This order grants no T2 scoring, M4 scientific
intervention, venue, live, key, sealed-2025, checkpoint-promotion or GPU
authority.

Read first:

- `docs/audits/MUSASHI_AUDIT_T2_C66_C73_M3_C1_C6_AND_M4_0_2026_09_08.md`;
- T2 candidate `a8fe2be62d6ff33720cfd249295b89a76364e12c`;
- M3/M4 candidate `16e12a2573ae270fb98eb93e2b9d59fd1b0bcaa8`; and
- the immutable T2 v6 and M3 v1-v3 design chains.

Priorities are P0 T2 custody, P0 M4 intervention semantics, P1 M3 exact global
population, then the corrected mechanical preflight. Keep all worktrees and
evidence roots separate.

## 1. P0: T2 C74-C81

### C74: close the post-revalidation replacement window

Freeze the audit's exact adversary: call the real root revalidation, replace
the root immediately before that call returns, and then invoke the productive
heartbeat. At the reviewed tip the call succeeds and the heartbeat lands on
the old invisible inode.

Every descriptor-relative read, write, rename and fsync that can influence
campaign state must prove the declared root identity both immediately before
and immediately after the operation. This includes heartbeat, claims, arrays,
records, terminals, stop reports, locks, release witnesses and the wall ledger.

A failed post-check is a typed custody loss. It must:

- prevent the effect from being accepted as successful;
- leave any in-flight unit `UNCERTAIN`;
- prevent lock release and process success; and
- require external disposition before continuation.

Do not claim that a concurrent same-uid rename can be prevented merely by
holding a directory descriptor. The required guarantee is fail-closed
authorization and adjudication under before/after identity, not impossible
write prevention.

### C75: descriptor-bound reads and wall chain

Apply the same before/after root identity proof to `read_private()`, directory
inventory and wall-ledger append/replay/close. A path replacement during a read
must not allow bytes from an obsolete root to become evidence. A path
replacement after wall append but before post-check must preserve the original
charges as uncertain evidence and refuse any successful exit or new budget.

Re-run the full chain from fresh descriptors after the final identity check and
before release. Release itself needs a final post-write check.

### C76: transparent resource-only successor

The 14,400-second v6 cap is physically incompatible with the corrected work
census and the recorded approximately 34-hour projection. Preserve v6 byte for
byte and create a successor design changing only:

- schema/version and supersession bindings;
- `resource_contract.max_wall_seconds` to **216,000 seconds (60 hours)**; and
- metadata that truthfully classifies the change as
  `DISCLOSED_PRE_EXECUTION_RESOURCE_LIMIT_AMENDMENT_FROM_MEASURED_IMPLEMENTATION_COST`.

The 60-hour value is a hard campaign ceiling, not an ETA or target. Preserve
one logical worker, 8 GiB RSS, nice 15, stop-file semantics, task population,
models, origins, estimands, thresholds, multiplicity and every scientific
field byte-equivalent. No confirmatory outcome exists yet, so the chronology is
pre-execution and must say so.

Materialize an executable field-by-field v6-to-successor diff. Any scientific
delta refuses. Rebind templates and candidate submissions, but do not create
or install the external Musashi review or execution record.

### C77: budget projection and early feasibility

Persist the projection basis as non-authoritative planning evidence: number of
observations, corrected dev wall, work census, linear-scaling assumption and
headroom. Before execution authority is consumed, the runner must reject a
design whose hard limit is below its committed feasibility estimate. Runtime
still obeys the hard wall and may stop earlier; the projection never grants
extra time.

### C78: exact final inventory under races

Run final adjudication against a fresh descriptor chain after the post-operation
root check. Reject a root that was replaced and restored between operations,
foreign objects, duplicate semantic objects, record-plus-terminal, and any
object not bound to the successor design and execution record.

### C79: adversarial battery

Add isolated regressions for:

- replacement immediately after pre-write revalidation;
- replacement immediately after pre-read revalidation;
- replacement during wall append and during release;
- replace-then-restore with the same pathname but another inode;
- a custody loss followed by attempted resume;
- a successor changing one scientific field;
- use of the old four-hour design; and
- a forged projection granting time beyond the hard cap.

Mutations removing each post-check must bite the productive path. Use real
processes for at least heartbeat-vs-replacement and release-vs-replacement.

### C80: verification

Run the focal battery and the full suite at the final tip. Report passed,
failed, skipped and collection errors separately. Name inherited failures; do
not merge a stale count from an earlier commit.

### C81: stop before authority and scoring

Do not author or install either external record. Do not score any member of the
242-unit population and do not create a scientific ledger. Required
disposition:

`T2_RESOURCE_SUCCESSOR_AND_CUSTODY_READY_FOR_FINAL_MUSASHI_REVIEW`

Musashi will independently rerun the race battery, verify the v6 successor
delta, install the records and launch the bounded CPU service if and only if
the return passes.

## 2. P1: M3 C7-C10

The numerical M3 result remains accepted. Do not regenerate or replace the
immutable v1-v3 evidence.

### M3-C7: exact global cell population

Before regenerating outcomes, assert exact equality between:

- observed `(K, ratio)` keys in task records;
- observed `(K, ratio)` keys in controls; and
- the sealed grid keys.

Reject an unknown cell even when its record is self-integral and the summary
file digest is repaired. Require the exact total record census as a consequence
of the per-cell adaptive populations plus 20 controls per cell.

Freeze the exact attack: append one valid task body under `K=999`, repair its
self-digest, the records-file digest and summary self-digest. It must refuse
before the expensive solver replay.

### M3-C8: strict design and summary schemas

Replace permissive parsing for design and summary with duplicate-key and
non-finite rejection. Validate exact recursive schemas, strict primitive types
and domains. Reject unknown/missing fields, bools in numerics, invalid cell
identities, inconsistent totals and duplicate cells.

The default verifier may report the accepted scientific label only for the
exact reviewed v3 design identity. An explicitly supplied foreign design may
be tested, but its output must be labeled non-authoritative consistency and
must not inherit the accepted v3 disposition.

### M3-C9: tests and cost ordering

Add focused regressions for foreign task/control cells, duplicate summary keys,
non-finite summary values, extra summary cells, foreign design identity and
wrong total counts. Perform cheap exact-population and schema checks before the
9,800-solver regeneration so malformed evidence fails promptly.

Re-run the corrected verifier over immutable v3. The expected result remains:

`COVER_CALIBRATION_REPRODUCED_AFTER_DISCLOSED_PRECISION_AMENDMENT`

### M3-C10: stop

Do not rerun M3 execution or change any result record. Return the corrected
consumer and its independent reproduction evidence with:

`M3_EXACT_GLOBAL_VERIFIER_READY_FOR_MUSASHI_REVIEW`

## 3. P0/P1: M4 C1-C8

### M4-C1: cumulative residual-capacity endpoint

Supersede the design before any scientific intervention. The primary endpoint
must count random associations that remain jointly acquired, not batches once
learned and later forgotten.

After adding batch `b`, evaluate the union of batches `0..b`. A batch advances
the count only when every cumulative association satisfies the frozen
acquisition criterion and the original-task retention criterion remains met.
Once an earlier association is forgotten, the endpoint cannot continue to
increase.

### M4-C2: freeze rehearsal and sampling semantics

State exactly how original-task examples and cumulative random associations
enter every update. Use a fixed, pre-result primary rule: balanced minibatches
with 50% original-task examples and 50% uniformly sampled cumulative
associations. Freeze batch size, replacement policy, update count, learning
rate and evaluation cadence.

Preserve a no-rehearsal arm as a diagnostic only if its multiplicity and role
are frozen now. Do not choose the better policy after seeing intervention
outcomes.

### M4-C3: implement the retention rule exactly

The current design says two consecutive retention failures; the code stops on
one. Implement the sealed consecutive rule with the evaluation cadence fixed.
Reset the streak only after a passing evaluation. Publish both raw losses and
the derived streak.

### M4-C4: real restart continuation

The mechanics preflight must save the complete training state, reload it in a
fresh process, apply the next sealed association batch, and compare the result
with an uninterrupted branch under the same state and batch. Equality must
cover parameters, optimizer/RNG state, cumulative-association inventory,
retention streak and counters.

### M4-C5: executable limits and controls

Implement and exercise the design's wall, RSS, stop-file and heartbeat
contracts. Execute the matched-compute control and derive its update difference;
the current string declaration is not evidence. Freeze typed outcomes for wall,
RSS, stop request, numerical failure and optimization-limited acquisition.

### M4-C6: write-once reconstructible artifacts

Refuse a nonempty output root before writing anything. Use private directories
and exclusive, durable files. The report must carry full SHA-256 identities for
every checkpoint, metadata object, batch ledger and heartbeat/stop evidence it
consumes.

Add `verify_preflight()` using a fresh reader. It must validate exact schemas,
inventory and full file digests and reconstruct the stated facts from the
artifacts. A replaced checkpoint or metadata file must invalidate the report.
A second invocation must make zero changes before refusing.

### M4-C7: adversarial battery

Freeze regressions for:

- forgetting batch 0 while fitting batch 1;
- training only on the newest batch;
- one retention failure treated as two;
- reload without an actual continuation update;
- absent optimizer/RNG state;
- heartbeat, stop file or matched-compute control omitted;
- checkpoint changed after report creation; and
- second invocation overwriting an artifact before refusing.

Each bypass mutation must bite the productive verifier or preflight.

### M4-C8: bounded stop

Run only the corrected two-unit CPU `MECHANICS_ONLY` preflight. No M4
confirmation generator, intervention outcome, DOIN gene or production gate is
authorized. Stop with:

`M4_CUMULATIVE_CAPACITY_DESIGN_AND_MECHANICS_READY_FOR_MUSASHI_REVIEW`

## 4. Runtime reporting

Observe B4 through read-only telemetry only. Report completed/current cells,
service restarts, wall and thermal facts without touching the process. T2 is
blocked by this correction and external review, not by the owner. No current
owner action is required for T2, M3, M4 or B4.
