# Musashi audit: T2 C74-C81, M3 C7-C10 and M4 C1-C8

Date: 2026-09-08

Reviewed tips:

- T2: `e900e0d2c97ba9c085ba0f7b5d4b5cd52b8d9874`
- M3/M4: `f06008da82bd184ef4c197cd74059e7d63b5f3ea`
- Return: `GENERAL_SATOSHI_TO_MUSASHI_T2_C74_C81_M3_C7_C10_AND_M4_C1_C8_RETURN_2026_09_08.md`

No T2 confirmatory unit, M4 intervention unit or B4 campaign object was
written during this audit. B4 was observed read-only.

## 1. Disposition

| Lane | Disposition |
|---|---|
| T2 C74-C75 custody | `ACCEPT` |
| T2 resource-only successor | `ACCEPT_SCIENTIFIC_DELTA_NONE` |
| T2 execution opening | `REVISE_BEFORE_EXTERNAL_RECORD` |
| M3 corrected verifier and result | `ACCEPT` |
| M4 cumulative scientific semantics | `ACCEPT_DIRECTION` |
| M4 mechanics verifier and intervention opening | `REVISE` |
| B4 v7 | `CONTINUE_UNTOUCHED` |

The return therefore does not authorize a T2 service or an M4 intervention.
It does close M3 and supplies bounded corrective work for Satoshi immediately.

## 2. Accepted findings

### T2 custody and resource amendment

The exact stale-root adversary from the preceding audit now refuses with
`RESULTS_ROOT_CUSTODY_LOST_AFTER_HEARTBEAT`. `ResultsRoot` performs a declared
path identity check before and after state operations, and `WallAuthority`
does the same around append and close. The committed POST completed its
three-unit mechanical rehearsal with three `COMPLETED_VERIFIED` units and no
sealed-bank score.

The resource successor changes the hard wall from 14,400 to 216,000 seconds
and restricts all other differences to schema, self identity, supersession
bindings and disclosed amendment metadata. The field-by-field verifier rejects
a changed scientific field. The 60-hour value is a ceiling, not an ETA, and
the projection cannot grant runtime.

### M3

The corrected verifier rejects the exact `K=999` foreign-cell record before
the expensive solver replay. Task cells, control cells, summary cells and the
sealed grid must now be equal as global populations. Design, summary and JSONL
records receive duplicate-key, non-finite, exact-schema and domain checks.

The immutable v3 evidence still reproduces:

`COVER_CALIBRATION_REPRODUCED_AFTER_DISCLOSED_PRECISION_AMENDMENT`

This is accepted only for the reviewed v3 identity. M3 requires no rerun and
no additional work in this cycle.

### M4 scientific correction

The v2 design correctly replaces sequential acquisition throughput with a
cumulative endpoint: every association acquired so far is evaluated after
each batch. The 50:50 rehearsal rule and the two-consecutive-evaluation
retention rule are appropriate, and the state digest now includes parameters,
cumulative associations, counters and retention state.

## 3. Findings requiring correction

### P0 T2: the executing design is reopened after the gate

`t2_confirmatory_executor.main()` calls `verify_confirmatory_gates()` with one
result of `active_design_path()`, then calls `active_design_path()` and
`strict_json_load()` again. The second document supplies the task population,
unit map, role geometry, seeds and resource limits used by the executor, but it
is not the byte stream whose digest and external execution record were checked
by the first call.

Consequently, the authority facts can describe design A while the campaign
consumes design B after a path replacement between the two reads. The new
results-root checks begin only after this split. The fix is not another path
check around each call: the gate must return one descriptor-bound design
snapshot, and that exact parsed object and byte digest must be the sole design
consumed by planning, claiming, fitting and adjudication.

The manifest and census should follow the same one-read rule so the external
record pins the bytes actually consumed.

### P0 M4: a self-consistent producer forgery passes `verify_preflight()`

The verifier hashes artifacts but does not reconstruct their contents. It
does not load the saved states, replay deterministic batches or compare the
replayed states and outcomes. Batch JSONL is parsed with ordinary
`json.loads()`, and report facts such as restart identity and matched compute
are trusted as producer booleans.

The following adversary was executed against the reviewed tip:

1. Change the first batch outcome from `ACQUISITION_ENDPOINT` to `ACCEPTED`.
2. Repair its record checksum and the report's accepted count.
3. Replace `u0_stop.npz` with arbitrary non-NPZ bytes.
4. Repair the two artifact digests and the report checksum.

Observed result:

```json
{
  "forged_outcome": "ACCEPTED",
  "invalid_checkpoint_bytes": "not-an-npz-but-self-consistent-producer-metadata",
  "verifier": {"verified": true, "units": 2, "artifacts": 15}
}
```

This defeats the stated reconstructibility guarantee. Checksums establish
internal consistency only; the fresh verifier must derive the transition facts
from the deterministic generator, the previous state and the sealed design.

### P1 M4: the no-rehearsal diagnostic is not compute matched

In `apply_batch(..., rehearsal=False)`, the association minibatch has only
`MINIBATCH - half` examples. With the frozen values this is eight examples,
while the primary arm processes sixteen examples per update. Equal update
counts therefore do not imply matched training work. The diagnostic must use
the full frozen minibatch size or be removed from comparative claims.

### P2 M4: terminal telemetry is listed but not verified

`M4_HEARTBEAT.json` appears in `artifacts_sha256`, but
`verify_preflight()` explicitly skips its digest. Mutable telemetry should be
outside the immutable terminal inventory, or a terminal snapshot should be
frozen and verified like every other artifact. It cannot be both digest-bound
and deliberately unchecked.

## 4. Verification performed

```text
T2 committed POST: PASS; 3/3 mechanical units COMPLETED_VERIFIED
T2 focal battery: 90 passed, 1 skipped in 554.81 s
M4 focused battery: 6 passed in 1.54 s
M4 repaired-checksum forgery: ACCEPTED by the reviewed verifier
M3 source and exact-population path: reviewed; corrected logic accepted
B4 read-only observation: active/running, NRestarts=0
```

## 5. Runtime observation

At the final read-only observation, `b4-v7-campaign-20260907.service` was
`active/running`, `MainPID=307630`, `NRestarts=0`. Two terminal cell records
were present and the third cell was training. The GPU was at 100 percent,
81 C and 2,013 MiB of 8,188 MiB. Nothing was changed.

## 6. Owner action

None. T2 and M4 are blocked by code/evidence corrections assigned to Satoshi,
not by the owner. M3 is closed. B4 is already running.
