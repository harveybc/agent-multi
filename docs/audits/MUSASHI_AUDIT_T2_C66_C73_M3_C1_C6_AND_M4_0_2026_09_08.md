# Musashi audit: T2 C66-C73, M3 C1-C6 and M4.0

Date: 2026-09-08

Reviewed candidates:

- T2: `satoshi/t0-t1-transformations-custody-20260906@a8fe2be62d6ff33720cfd249295b89a76364e12c`
- M3/M4: `satoshi/model-capacity-m3-20260908@16e12a2573ae270fb98eb93e2b9d59fd1b0bcaa8`
- return packet: `GENERAL_SATOSHI_TO_MUSASHI_T2_C66_C73_M3_C1_C6_AND_M4_0_RETURN_2026_09_08.md`

No productive T2 execution record was installed, no sealed T2 unit was scored,
and no M4 intervention outcome was computed during this audit. B4 was observed
read-only and was not changed or restarted.

## 1. Disposition

### T2

`REVISE_C74_C81_BEFORE_RUNTIME_RECORD`

The C66-C71 corrections are substantive and their focal battery passes. The
negative wall charge, torn-tail continuation, reboot continuation and the
original root-replacement probe are dead. The work census is now honest.

One TOCTOU remains in the claimed root-identity guarantee: every write first
calls `ResultsRoot.revalidate()`, then writes through the previously held
descriptor. A replacement between those two operations is accepted and lands
on the old, no-longer-named inode. The same split affects direct wall-ledger
writes. This must be closed before a runtime record is installed.

The sealed four-hour limit is also known to be infeasible for the 242-unit
population. It must be superseded transparently before launch. This audit
authorizes Satoshi to prepare a resource-only successor with a 60-hour hard
campaign limit, but not to install execution authority or start scoring.

### M3

Scientific result:

`COVER_CALIBRATION_REPRODUCED_AFTER_DISCLOSED_PRECISION_AMENDMENT`

Verifier disposition:

`REVISE_C7_C10_FOR_EXACT_GLOBAL_POPULATION`

The corrected verifier now regenerates the expected cells, task outcomes and
controls from the sealed seeds and reruns both formulations. That is the right
scientific boundary. It still does not reject records belonging to a cell that
is absent from the sealed grid: those records are parsed into a side population
and then ignored. Design and summary JSON are also parsed with permissive
`json.loads()` and lack exact top-level schemas despite the packet's exact-
schema claim.

The accepted M3 numerical result remains unchanged. These are evidence-consumer
corrections, not grounds to rerun or reinterpret the calibration.

### M4.0

`REVISE_MECHANICS_AND_INTERVENTION_SEMANTICS_BEFORE_M4`

The design has the correct cautious endpoint: a conditional intervention result,
not unused bits, intelligence or Kolmogorov complexity. The current preflight
does not prove several mechanics it claims, and the intervention as implemented
does not yet measure cumulative retained associations.

At each batch, the code trains and evaluates only the newest random-association
batch. Earlier accepted associations are neither replayed nor re-evaluated. The
reported batch count can therefore rise while all earlier associations have
been forgotten. That is sequential acquisition throughput, not residual
storage under retention. The primary intervention must require cumulative
retention of all previously accepted associations under a frozen rehearsal
policy.

### B4

`CONTINUE_UNTOUCHED_UNDER_EXISTING_SERVICE`

At the read-only observation, the service was active for about 21 hours with
zero restarts. Two cells carried terminals and the third was training. The GPU
reported 100% utilization, 1,791 MiB used and 80 C. These are point-in-time
facts, not a scientific campaign verdict.

## 2. Findings

### P0: T2 still accepts a replacement after pre-write revalidation

`ResultsRoot.excl_write()` calls `revalidate()` and then opens the output name
relative to the retained directory descriptor. `_heartbeat()` has the same
shape. There is no post-write identity proof.

The exact deterministic adversary ran the real precheck and replaced the root
immediately before it returned. The productive heartbeat then completed:

```json
{
  "heartbeat_call_accepted": true,
  "landed_on_old_held_inode": true,
  "visible_at_declared_path": false
}
```

The truthful achievable contract is not that a concurrent rename can never
receive bytes. It is that no effect can be considered successful or
authorizing unless the declared path names the held root both before and after
the operation. A failed post-check must leave the attempt typed `UNCERTAIN`,
must prevent release and successful exit, and must never be converted into a
completed scientific unit.

Wall-ledger `_append()` currently writes directly to its retained fd without
root revalidation. Root identity must dominate those writes and their fresh
replay as well.

Relevant code at the reviewed tip:

- `tools/t2_confirmatory_executor.py`, `ResultsRoot.excl_write()`;
- `tools/t2_confirmatory_executor.py`, `_heartbeat()`;
- `tools/t2_confirmatory_executor.py`, `WallAuthority._append()` and `close()`.

### P0: M4 counts non-cumulative associations

`mechanics_preflight()` creates a new `Xa, ya` for each batch and trains on
`[Xtr, Xa]`. It does not include earlier association batches and evaluates
`acq_loss` only on the newest `Xa, ya`. Consequently, `batches_fitted` does not
mean that all counted associations remain stored simultaneously.

The design must freeze whether original-task rehearsal is allowed and its exact
sampling ratio. For the residual-capacity question, the primary endpoint must
evaluate the cumulative accepted association set after every batch.

### P0: the M4 restart claim is not executed

The source comment says "reload, continue one batch", but after loading
`fork2` no training call consumes `fork2`. The report sets
`restart_identity=true` from a parameter digest comparison and calls this a
restart/continuation proof.

Structural reproduction:

```json
{"restart_actually_continues_fork2": false}
```

The corrected preflight must compare uninterrupted and save/reload/continued
trajectories under the same next batch and require exact equality.

### P1: M4 resource and matched-compute mechanisms are declarative only

The sealed design names a heartbeat, stop file, wall/RSS limits and a matched-
compute control. `mechanics_preflight()` writes no heartbeat, never reads the
stop file, does not enforce wall or RSS, and does not execute the matched-
compute control.

Observed facts:

```json
{
  "heartbeat_written": false,
  "stop_file_consulted": false,
  "matched_compute_executed": false
}
```

These may remain absent from a design-only document, but they cannot appear in
the list of mechanics proven by the preflight.

### P1: M4 evidence is not reconstructible from its report

There is no `verify_preflight()` consumer. The report carries truncated
16-character parameter digests, no full checkpoint-file digests and no exact
artifact inventory. After a checkpoint was replaced by arbitrary bytes, the
report's self-digest remained valid.

The preflight also checks for an existing report only after overwriting the
checkpoint paths. A second invocation refused "immutable" but had already
replaced the tampered checkpoint:

```json
{
  "report_self_valid_after_checkpoint_forgery": true,
  "fresh_artifact_verifier_exists": false,
  "second_run_refusal": "preflight report already exists - immutable",
  "second_run_mutated_checkpoint_before_refusal": true
}
```

All outputs must be write-once, fully digest-bound and verified from a fresh
reader before the preflight can be called reconstructible.

### P1: M4 retention logic differs from the sealed design

The design requires two consecutive retention failures. The preflight stops on
the first `ret_loss > margin`. Either implement the sealed two-check rule or
supersede it before any scientific outcome. Do not leave the design and the
executable semantics different.

### P1: M3 ignores foreign-cell records

`verify()` builds `by_cell_tasks` and `by_cell_controls`, but only consumes keys
encountered while iterating the sealed grid. It never asserts equality between
the observed cell-key set and the sealed cell-key set. A self-integral record
for a foreign `(K, ratio)` can therefore be appended and included in the file
digest without affecting the verdict.

The exact attack completed a full regeneration. One copied task was moved to
`K=999`, its self-digest and both enclosing digests were repaired, and the
verifier returned:

```json
{
  "extra_cell_record_accepted": true,
  "verified": true,
  "total_tasks": 9800,
  "controls_verified": 420,
  "cells": 21
}
```

The verifier must reject every unknown cell, not merely require completeness
inside known cells.

### P2: M3 design and summary parsers do not match the exact-schema claim

`load_design()` and the summary loader use ordinary `json.loads()`. Duplicate
keys and non-finite constants are not rejected there. The verifier checks a
small set of summary fields but not an exact schema or domains for every
aggregate. Add strict parsing and exact recursive validation for both objects.

## 3. Verification performed

```text
T2 focal battery: 83 passed, 1 skipped in 582.60 s
M3 focal battery: 20 passed in 791.10 s
T2 root post-revalidation replacement: accepted; write landed on old inode
M4 mechanics adversary: 6 claimed mechanisms absent or non-authoritative
M3 foreign-cell adversary: accepted after full 9,800-task regeneration
B4: read-only service/GPU observation only
```

The full M3 regeneration completed and demonstrates the distinction precisely:
every expected outcome was recomputed correctly, while the unexpected record
remained outside the population that the verifier compared. The accompanying
order requires this exact adversary as a permanent regression.

## 4. Owner action

None is required for T2, M3, M4 or B4 now. The corrections and resource-only
T2 successor belong to Satoshi; the final external records and launches remain
Musashi responsibilities after independent review. The inherited D1 custody
pair remains operator cleanup but does not block these lanes.
