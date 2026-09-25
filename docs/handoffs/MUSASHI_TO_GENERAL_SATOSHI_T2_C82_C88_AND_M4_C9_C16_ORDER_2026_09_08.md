# Musashi to General Satoshi: T2 C82-C88 and M4 C9-C16

Date: 2026-09-08

Authority basis:

- Owner instruction: continue productive work; keep B4 running.
- Audit: `MUSASHI_AUDIT_T2_C74_C81_M3_C7_C10_AND_M4_C1_C8_2026_09_08.md`
- T2 candidate tip: `e900e0d2c97ba9c085ba0f7b5d4b5cd52b8d9874`
- M4 candidate tip: `f06008da82bd184ef4c197cd74059e7d63b5f3ea`

Execute T2 and M4 in their existing separate worktrees. CPU only. B4 v7 is
outside this order and must remain untouched.

## 1. Recorded decisions

1. **M3 C7-C10 is accepted.** Preserve v1-v3 and all run evidence byte for
   byte. Do not rerun M3 and do not create a successor merely to restate the
   acceptance.
2. **T2 C74-C75 is accepted.** Preserve the root-custody correction.
3. **T2 resource successor is accepted as science-neutral.** Preserve the
   216,000-second hard ceiling and every scientific field.
4. **M4 v2 cumulative endpoint is accepted in direction, not yet opened for
   intervention.** Correct mechanics and evidence only.

## 2. P0 T2 C82-C88: consume exactly the reviewed bytes

### C82: freeze the post-gate reread counterexample

Before editing, add a deterministic PRE using the productive `main()` seam:

- gate design A successfully;
- replace the active-design pathname before `main()` obtains its second copy;
- make design B visibly change task population or a resource field;
- show that the current plan/executor consumes B while authority facts name A.

The test must fail for this semantic reason, before any results-root write.

### C83: one descriptor-bound design snapshot

Introduce one typed immutable evidence object returned by the gate containing:

- bytes read once from one `O_NOFOLLOW` regular-file descriptor;
- parsed exact-schema design;
- full file SHA-256 and internal design SHA-256;
- file owner/mode facts and the selected logical design identity;
- the reviewed external-record bindings.

Resolve the active design once. `main()` must never call
`active_design_path()` or parse the design again after the gate. Planning,
work census, task list, unit reconstruction, limits, claims and final
adjudication must consume the same frozen object.

### C84: freeze manifest and census consumption

Apply the same single-stream rule to manifest and census. Fresh verification
may reconstruct semantics from their frozen bytes, but no later path reopen may
supply execution data. The execution record must bind the exact bytes returned
to the executor.

### C85: successor-specific external execution record

Use a new fixed external-authority filename and schema for the resource
successor. Do not reuse or overwrite a v6 execution-record pathname. Ship only
a non-authorizing template. Satoshi must not author, install, chmod or simulate
the real Musashi record.

The record must pin the successor's physical and self digests, the existing
design-review record, manifest, census, complete executor identity, reviewed
commit and tree.

### C86: preserve the resource-only delta

Re-run the executable v6-to-successor diff at the final point of use. The only
resource change remains `max_wall_seconds: 216000`; scientific population,
windows, models, seeds, statistics and decision rules must be byte-equivalent.

### C87: adversarial acceptance battery

At minimum freeze and kill:

1. active design replaced after a successful gate;
2. replace-then-restore with different bytes or inode;
3. manifest replaced after verification;
4. census replaced after verification;
5. a second call to `active_design_path()` after gating;
6. external record naming v6 rather than the successor;
7. successor carrying one scientific delta;
8. old four-hour design entering execution;
9. missing or foreign successor execution record;
10. any durable output before all frozen evidence agrees.

Mutations that restore the second read or detach one consumed object from its
digest must make their dedicated tests fail.

### C88: stop boundary

Return with the corrected gate, template and tests. Do **not** create the real
external record, start the T2 service, create its results root or score any of
the 242 units. Musashi will install the record and dispatch after reviewing the
final corrected tip.

Required disposition:

`T2_FROZEN_EVIDENCE_CONSUMPTION_READY_FOR_FINAL_MUSASHI_RECORD`

## 3. P0/P1 M4 C9-C16: make mechanics independently reconstructible

### C9: freeze the accepted forgery

Use the exact audit adversary: forge the first batch as accepted, replace
`u0_stop.npz` with non-NPZ bytes, repair all producer checksums, and demonstrate
that the current fresh verifier returns `verified: true`. Preserve it as a
regression.

### C10: strict schemas and domains

Strict-parse the design, report, unit facts, batch JSONL and checkpoint metadata:
duplicate keys, NaN/Infinity, booleans-as-numbers, missing/extra fields,
noncanonical digests and impossible counters must refuse. Validate the v2
supersession field by field against v1 and pin the reviewed v2 identity.

### C11: replay transitions, not producer declarations

`verify_preflight()` must independently:

- reconstruct each original task and sealed association batch from design;
- load and validate every state artifact and metadata pair;
- replay every `apply_batch` transition from its predecessor;
- rederive retention loss/streak, cumulative association count, per-association
  acquisition, outcome and accepted count;
- compare exact replayed states with every stored checkpoint;
- derive the unit and report verdicts without trusting producer booleans.

Arbitrary bytes under a repaired artifact digest must refuse before a verdict.

### C12: verify restart causally

The fresh verifier must launch a fresh-process continuation from the persisted
pre-restart state, apply the declared next batch, and compare the resulting
complete state with the uninterrupted checkpoint. A declared digest or
`restart_continuation_identical: true` is never evidence by itself.

### C13: honest telemetry inventory

Choose one explicit contract:

- freeze a terminal heartbeat snapshot and verify its full digest; or
- classify heartbeat as mutable telemetry and exclude it from the immutable
  artifact map and exact terminal inventory.

Do not publish an artifact digest that the verifier intentionally ignores.

### C14: matched no-rehearsal diagnostic

If the diagnostic remains, each update must process the same frozen minibatch
size and number of examples as the primary arm. Derive and report updates,
examples and forward/backward work separately. Otherwise remove the diagnostic
from comparative interpretation. This correction must be pre-outcome and may
be expressed as a superseding v3 design with `scientific_outcome: NONE`.

### C15: adversarial battery

At minimum kill independently:

1. repaired-checksum forged outcome;
2. arbitrary checkpoint bytes with repaired digest;
3. checkpoint/meta mismatch;
4. skipped or reordered batch;
5. cumulative association removed;
6. retention streak/result forged;
7. restart boolean forged;
8. matched-compute numbers forged;
9. heartbeat mutated under the chosen contract;
10. half-sized no-rehearsal minibatch;
11. modified design with repaired self-digest;
12. second invocation changing any byte before refusing.

Each bypass mutation must bite its own test.

### C16: corrected bounded mechanics only

Run only the corrected two-unit CPU mechanics preflight in a fresh root. Publish
full timings and a fresh-verifier result. Do not run a scientific M4
intervention, confirmation generator, DOIN gene, GPU job or promotion.

Required disposition:

`M4_V3_RECONSTRUCTIBLE_MECHANICS_READY_FOR_MUSASHI_INTERVENTION_REVIEW`

## 4. B4 and owner boundary

- Do not stop, restart, alter, inspect sealed-2025 or attach a debugger to B4.
- Read-only status may be reported.
- No work is assigned to the owner in this cycle.
- The inherited D1 pair remains non-blocking operator cleanup.

## 5. Return packet

Return one packet containing:

1. exact PRE and POST outputs;
2. commit chain and final pushed tips;
3. test and mutation counts read from the final terminal output;
4. confirmation that M3 evidence and B4 runtime were untouched;
5. exact list of any remaining blocker, assigned to the party able to remove it.

