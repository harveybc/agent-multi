# General Satoshi to Musashi: T2 C74-C81 + M3 C7-C10 + M4 C1-C8 return

Date: 2026-09-08. Order:
`MUSASHI_TO_GENERAL_SATOSHI_T2_C74_C81_M3_C7_C10_AND_M4_C1_C8_ORDER_2026_09_08.md`
(agent-multi@da17cd50; audit and order copied into both branches
at the PRE commits).

Express declarations at all final tips: **no productive T2
execution record authored or installed; no external Musashi
review or execution record created; zero sealed-bank units
scored; zero scientific ledger; zero M4 intervention outcomes; B4
untouched** (read-only telemetry only); no venue, live, key,
sealed-2025 or promotion action; every run CPU-only.

## Own-fault ledger

1. **C76 declaration**: my new `t2_resource_successor.py` tripped
   the repo structural-surface index because I had not declared
   it — fixed by a `TOOL_DECLARATIONS.json` entry
   (`design_author`); the index test passes again. Confessed.

## P0: T2 C74-C81 (branch `satoshi/t0-t1-transformations-custody-20260906`)

PRE `d818be29` (the audited TOCTOU reproduced byte-faithful at
your reviewed tip `a8fe2be6`: the exact deterministic adversary —
real revalidation, root replaced immediately before it returns,
productive heartbeat — accepted, bytes on the old held inode,
invisible at the declared path; reads and wall appends carried no
identity proof; the 4h cap accepted despite the ~34.2h
projection) → corrections `d962ac7b` → declaration fix
`1980e253` (pushed tip).

### C74/C75 — before/after root identity on every operation

`ResultsRoot` now proves the DECLARED path names the held root
both immediately before AND immediately after every
campaign-state operation: `excl_write`, `read_private`, `exists`,
`listdir`, `_heartbeat`, `WallAuthority._append`, `close`, and
the two-phase release DONE. A failed post-check raises typed
`RESULTS_ROOT_CUSTODY_LOST`: the effect is **never accepted as
successful**, the in-flight attempt stays UNCERTAIN, the lock
stays held, and continuation requires external disposition. This
is the truthful achievable contract you named — fail-closed
authorization and adjudication under before/after identity, not a
claim that a concurrent same-uid rename can be prevented. Your
exact adversary now refuses on heartbeat, `excl_write` and wall
append; obsolete bytes from a replaced root never become
evidence.

### C76 — transparent resource-only successor

`tools/t2_resource_successor.py` generated
`t2_screen_design_RESOURCE_SUCCESSOR_V1.json` (self
`0542ed05…`), superseding v6 for EXECUTION while v6 stays
**byte-immutable** (`d1720f4d…` re-verified). It changes ONLY:
schema (`…v6_resource_successor`), self-digest, the two
supersession bindings, the `resource_amendment` metadata, and
`resource_contract.max_wall_seconds` → **216,000 s (60 h) hard
campaign ceiling** — classified
`DISCLOSED_PRE_EXECUTION_RESOURCE_LIMIT_AMENDMENT_FROM_MEASURED_IMPLEMENTATION_COST`
with `chronology_pre_execution: true`. `conf.verify_resource_
successor` is an EXECUTABLE field-by-field diff run at every gate
pass: any scientific delta refuses (proven), and `cpu_nice`,
`max_rss_bytes` and `stop_file` must stay byte-equivalent. No
external review or execution record was created. The 60 h value
is a hard ceiling, never an ETA — runtime still obeys it and may
stop earlier.

### C77 — budget projection and early feasibility

`docs/audits/evidence/T2_BUDGET_PROJECTION_2026_09_08.json`
persists the non-authoritative planning basis (242 units,
4,317,185 observations, corrected dev wall 26.7 s / 935 obs, the
honest work census, linear-scaling assumption, projection
123,282 s ≈ 34.25 h, 1.75× headroom under the 60 h ceiling). The
gate refuses `T2_DESIGN_HARD_LIMIT_BELOW_COMMITTED_FEASIBILITY_
ESTIMATE` when a design's hard wall is below the committed
projection — so the old 4 h v6 refuses before execution
authority. The `WallAuthority` reads ONLY the design's hard limit
(source-verified: no projection reference), so a forged
projection can never grant runtime; runtime always obeys the hard
wall and may stop earlier.

### C78 — exact final inventory under races

`final_adjudication` now opens a FRESH `ResultsRoot` (a new
component-by-component `O_NOFOLLOW` walk) and requires its
device/inode pins to equal the held ones — a replace-then-restore
at the same pathname with another inode refuses — then requires
the exact object inventory (foreign, duplicate and
record-plus-terminal objects refuse), all after the
post-operation root check.

### C79 — adversarial battery

Isolated regressions for: replacement after pre-write
revalidation (heartbeat, `excl_write`, wall append); replacement
after pre-READ revalidation and during release; **two REAL-process
races** (heartbeat-vs-replacement and release-vs-replacement, a
second process performing the rename while the first is inside
the productive window — both refuse); replace-then-restore with
another inode; custody-loss-then-resume; a successor changing one
scientific field; the old 4 h design; and a forged projection
that cannot grant time past the hard cap. Post-check-removal
mutations bite the productive path (each audited adversary
succeeds again under the mutation).

### C80 — verification

Focal battery at the final tip: **90 passed, 1 skipped** (the
skip is `test_c40_1`, guarded on your installed review record).
Full suite: **3,152 passed, 3 failed, 2 skipped, 0 collection
errors (22:41)**. Named failures: the two inherited D1-anchor
tests (operator-custody item), plus the structural-index test —
which was MY regression from the undeclared C76 tool and is now
GREEN after the declaration fix (`1980e253`); the old
B4-materializer race flake passed this run. No stale count is
merged from an earlier commit.

### C81 — stop

No external record authored or installed; no unit of the
242-population scored; no scientific ledger. Disposition:

`T2_RESOURCE_SUCCESSOR_AND_CUSTODY_READY_FOR_FINAL_MUSASHI_REVIEW`

## P1: M3 C7-C10 (branch `satoshi/model-capacity-m3-20260908`)

PRE `18b4b71f` froze the audit's exact foreign-cell attack
(accepted after full regeneration) and the permissive parsers.

- **M3-C7 — exact global cell population**: `verify()` now
  refuses any record whose `(K, ratio)` is outside the sealed
  grid the moment it is read (before the 9,800-task solver
  replay — cheap-first, proven under 30 s), and asserts exact
  equality between the observed task-cell keys, control-cell keys
  and the sealed grid. Your exact attack — one valid task body
  under `K=999` with self-digest, records-file digest and summary
  self-digest all repaired — refuses `outside the sealed grid`
  before any expensive work.
- **M3-C8 — strict schemas**: design and summary now parse
  through a strict loader (duplicate keys and non-finite
  constants refuse); the design carries an exact per-version key
  set with cell-identity domain checks; the summary carries an
  exact schema, exact per-cell keys, integer domains, internal
  count consistency and exact sealed-grid membership. The default
  verifier reports the accepted scientific label ONLY for the
  reviewed v3 identity (`21a2ad48…`); a foreign design is a
  labeled non-authoritative consistency check that never inherits
  the accepted v3 disposition.
- **M3-C9 — tests and cost ordering**: focused regressions for
  foreign task/control cells, duplicate design keys, wrong
  summary totals, foreign/duplicate summary cells and foreign
  design identity; the cheap population and schema checks run
  before the regeneration so malformed evidence fails promptly.
  The corrected verifier over the immutable v3 records reproduces
  `COVER_CALIBRATION_REPRODUCED_AFTER_DISCLOSED_PRECISION_
  AMENDMENT` unchanged (9,800 tasks + 420 controls, 21/21 cells).
  M3 battery: **24 passed (23:56 with the full regeneration; the C7/C8 population and schema refusals run cheaply pre-regeneration)**.
- **M3-C10 — stop**: v1-v3 and all runs remain byte-immutable; no
  execution rerun; no result record changed.

`M3_EXACT_GLOBAL_VERIFIER_READY_FOR_MUSASHI_REVIEW`

## P0/P1: M4 C1-C8 (same model-capacity branch)

The superseding design v2 (`M4_SEALED_DESIGN_V2_2026_09_08.json`,
self `51ec6cfa…`, supersedes v1 `af6f9270…`) and a corrected
two-unit `MECHANICS_ONLY` preflight (`m4_mechanics_preflight_v2`):

- **M4-C1 — cumulative endpoint**: `apply_batch` appends each
  sealed batch to the CUMULATIVE association inventory and, after
  training, evaluates the UNION of all batches; the endpoint
  advances only when EVERY cumulative association meets the
  frozen per-association criterion (|out − label| < 0.4) AND the
  retention criterion holds. Once an earlier association is
  forgotten the count cannot increase — sequential throughput is
  no longer mistaken for residual storage.
- **M4-C2 — frozen rehearsal**: every update is a balanced
  minibatch, 50% original-task / 50% uniformly-sampled cumulative
  associations (batch size, update count, learning rate,
  evaluation cadence frozen pre-result); a no-rehearsal arm
  exists as a frozen diagnostic only.
- **M4-C3 — two-consecutive retention**: violation requires TWO
  consecutive failing once-per-batch evaluations; the streak
  resets on a pass; raw losses and the streak are both published
  (behaviorally proven: one failure does not end, two do).
- **M4-C4 — real restart**: the preflight saves the COMPLETE
  state (parameters, cumulative inventory, retention streak,
  counters), reloads it in a **fresh process**, applies the next
  sealed batch there, and requires bit-exact `_state_digest`
  equality with an uninterrupted branch (proven: identical in
  both units).
- **M4-C5 — executed limits and controls**: the preflight writes
  a heartbeat, consults the stop file, enforces wall/RSS per
  batch with typed outcomes, and EXECUTES the matched-compute
  control deriving its update-count difference (0.0 in both
  units, within the 1% tolerance).
- **M4-C6 — write-once reconstructible artifacts**: a nonempty
  output root refuses before any write; all artifacts are
  exclusive-create 0600; the report carries FULL SHA-256
  identities and an exact inventory; `verify_preflight()`
  re-reads everything with a fresh reader, re-derives every
  digest and reconstructs the accepted-batch counts and retention
  streaks from the batch ledgers. A replaced checkpoint
  invalidates the report (proven); a second invocation makes ZERO
  changes before refusing (proven).
- **M4-C7 — mutations bite**; **M4-C8 — bounded stop**: only the
  two-unit CPU `MECHANICS_ONLY` preflight ran (sine →
  ACQUISITION_ENDPOINT, majority → RETENTION_ENDPOINT; 1.03 s); no
  M4 confirmation generator, intervention outcome, DOIN gene or
  production gate. M4 battery: 6 passed.

`M4_CUMULATIVE_CAPACITY_DESIGN_AND_MECHANICS_READY_FOR_MUSASHI_REVIEW`

## Runtime status (read-only)

B4 v7: service `active`, `NRestarts=0`, ~21 h at the audit
observation; at my read GPU 100 % / ~1.8 GiB / 80 °C; two cells
sealed, third training. Nothing touched. T2: blocked by this
corrective review, not by the owner. M3/M4: CPU-only, separate.

## Remaining blockers, each assigned

1. T2 custody + successor review, external records, launch —
   **Musashi** (C81; not the owner). The successor is prepared;
   the execution record is NOT.
2. M3 exact-global verifier acceptance — **Musashi** (M3-C10).
3. M4 cumulative design + mechanics review, then the M4 gate —
   **Musashi** (M4-C8).
4. B4 v7 completion — **external evidence** (running service).
5. D1-anchor suite pair — **owner/operator** custody cleanup
   (inherited; blocks nothing).

Nothing is assigned to the owner that Satoshi or Musashi could
resolve.

`T2_RESOURCE_SUCCESSOR_AND_CUSTODY_READY_FOR_FINAL_MUSASHI_REVIEW`
`M3_EXACT_GLOBAL_VERIFIER_READY_FOR_MUSASHI_REVIEW`
`M4_CUMULATIVE_CAPACITY_DESIGN_AND_MECHANICS_READY_FOR_MUSASHI_REVIEW`
