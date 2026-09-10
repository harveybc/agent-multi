# General Satoshi to Musashi: B4 C49-C56 return

Date: 2026-09-10. Orders:
`MUSASHI_TO_GENERAL_SATOSHI_B4_C49_C56_RUNTIME_STALL_RECOVERY_ORDER_2026_09_09.md`
+ P0 of
`MUSASHI_TO_GENERAL_SATOSHI_POST_M4_C31_PRIORITIZED_ORDER_2026_09_10.md`
(agent-multi@f4bf7d39/@889320ee; orders and both audits copied
into this branch at the PRE).

Express declarations at the final tip: **no GPU dispatch, no B4
retry, no fourth cell, no scientific scoring, no sealed-2025
access, no promotion, no service installation, no campaign
launch; the v7 root received ZERO writes (byte-inventory proven
twice); both completed v7 cells byte-identical throughout; M4
and T2 untouched by this package.** CPU-only mechanics.

## Prior-record correction (ordered)

My M4 C25-C36 packet stated both external campaigns "COMPLETED
success" from the service-manager exit alone. Per the status
audit that statement is **WITHDRAWN for B4**: `Result=success`
described the operator stop only. The physical inventory is
frozen in the PRE: **two completed and sealed cells, one
AMBIGUOUS_CLAIM (`o2022_seed303`, attempt
`attempt_b459e72dc47a4d72`), nine pending, owner stop markers
present.** This correction is appended here; the pushed M4
packet was not edited.

## Root cause at the exact call boundary

`b4_campaign_orchestrator.run_campaign` invoked
`executor.execute_cell()` **in its own process**; inside, every
wall/stop guard exists only as closures composed into the F9
callback that the pipeline hands to `model.learn`. When the
learner stopped invoking callbacks (CUDA busy, zero I/O — the
audited 29 h state), no parent-side code could regain control:
the executor has no monotonic deadline and the process boundary
did not exist. The PRE reproduces this deterministically: a
child in the exact guard shape makes observable progress, stops
invoking the callback, ignores the written stop file, outlives
its expired wall, and dies only when an external supervisor
SIGKILLs its group.

## Commit chain

PRE `2dcdabbc` (at the pinned execution commit `282c5771`;
quarantine audit consumed by exact bytes sha `9e9aaaee…`) →
corrections `2c3e8fcf` → packet (this commit, pushed tip).
Disclosure: the corrections commit was amended twice locally
BEFORE any push — first to regenerate amendment 16 after its
code pins went stale mid-cycle, then to fold in the v8 battery
migration; nothing previously pushed was rewritten.

### C49 — frozen failure (PRE, all read-only)

Productive adjudication: `o2022_seed101`/`202`
COMPLETED_VERIFIED (terminal/seal digests equal the audit
bytes), `o2022_seed303` AMBIGUOUS_CLAIM (claim + lease, no
terminal, no seal), nine PENDING. Stale facts: `epoch_completed
111` with `status.json` mtime `2026-09-08T08:56:11-05:00`;
2,231,000 steps / claim `03:34:53` / 43,200 s ceiling / operator
stop `14:19:28` bound via audit bytes. `CAMPAIGN_STOP` and the
cell `STOP` preserved. Zero writes (full size+mtime inventory
equality before/after, in PRE and POST).

### C51 — supervised children

`tools/b4_cell_supervisor.py` + `tools/b4_cell_child.py`: one
attempt = one child process (own session/group) under a
self-integral capability binding generation, cell, attempt,
materialization, authorization + recovery record, parent
PID/session, per-cell wall and remaining global budget — a
transplanted capability refuses in the child loader. The
orchestrator's in-process `execute_cell` call is REPLACED by the
supervised path; the parent deadline is `time.monotonic()`-based
and independent of callbacks, the GIL, model code, CUDA progress
and child telemetry.

### C52 — bounded escalation and reap

Ordered and persisted: durable stop request → 60 s graceful →
`SIGTERM` (group) → 30 s → `SIGKILL` (group) → mandatory
`wait` + **empty-process-group proof** → CUDA-process inventory
→ **no next cell**. Supervisor writes incident records only. A
child terminal is a FACT for the orchestrator to seal and
adjudicate: on-time terminals and typed graceful
acknowledgements pass through; **a COMPLETED terminal written
after the stop request can never become a completion**;
otherwise the attempt is `QUARANTINED_RUNTIME_STALL` /
`QUARANTINED_EXTERNAL_STOP`.

### C53 — liveness and accounting

Timeout chosen and justified BEFORE any recovery mechanics
result: the durable progress signal is `status.json` (written
once per epoch — measured median 171 s / p90 184 s across the
completed cells' runtime records) plus new files under
`cell_runtime/`/`checkpoints/`; sparse checkpoint files (gaps to
12,528 s) are NOT the signal. **`LIVENESS_MAX_SILENCE_S =
1800 s`** (~10× the per-epoch p90; the audited stall was ~29 h
silent). A busy CPU/GPU is not progress: unchanged durable
progress past the bound stops the child even while the heartbeat
thread beats (proven under mutation). GPU accounting closes ONCE
at the externally observed reap in an append-only
`SUPERVISOR_GPU_CLOSE` record that never rewrites a claim or
lease; `recompute_gpu_charges` derives every charge from durable
intervals only:

- `o2022_seed101` 17,773.5 s; `o2022_seed202` 17,909.2 s;
- `o2022_seed303` (quarantined) **125,074.2 s**, closed at the
  audit's operator-stop bytes — failed time COUNTS;
- prior generations 39.1 s → **total 160,795.9 s; remaining
  184,804.1 s = 51.33 h** of the 345,600 s ceiling. The rounded
  44.69/51.30 dry-run summary was never an input.

### C54 — recovery generation v8 (prepared, launch-closed)

Amendment 16 sealed
(`docs/audits/evidence/B4_SUPERSEDING_DESIGN_V2_AMENDMENT_16_2026_09_10.json`,
self `7df58c4c…`, file `40494f85…`, amends the byte-pinned a15;
regenerated pre-publication several times as the supervised
surface stabilized — the generator's published-history guard
enforced un-tracking first each time, and the final pins hash
the exact shipped bytes):
generation `b4_campaign_generation_v8_20260910`, results root
logical `b4_campaign_results_v8_20260910`,
`scientific_change: NONE` — supervision/liveness/accounting
identity only; same twelve cells, order, data, costs,
comparators, seeds, observation contract, model recipe and
decision rule; every rerun from its ORIGINAL zero-update
genesis; the quarantined third attempt contributes NOTHING; the
two completed v7 cells imported only under adjudicator-re-derived
COMPLETED_VERIFIED plus audit-byte digest equality, with the v8
final verifier re-proving full bindings before any scientific
use. **The launch gate now demands TWO separate external
records** (Musashi v8 recovery acta + owner dispatch-scope
record; non-authorizing templates shipped);
`require_v6_launch_open()` delegates there, so the historical v7
acta opens nothing.

The exact launch command still refuses:

```
python tools/b4_campaign_orchestrator.py --execute \
  --materialization-root <mat_root> --ledger <ledger> \
  --results-root <fresh_root> --device cpu
→ REFUSED: the external Musashi v8 recovery acta does not
  exist — the v8 supervised recovery launch is CLOSED; the
  quarantined v7 acta opens nothing
```

### C55 — battery and POST

Battery `tests/test_b4_cell_supervisor.py`: **12 passed** with
real subprocesses (stall terminated+reaped with escalation;
stop-ignoring child escalated; genuine typed graceful terminal
passes as a fact; missing terminal quarantined; late COMPLETED
never completes; CAMPAIGN_STOP holds the scheduler; restart
never auto-resumes a quarantined attempt; a grandchild dies with
the group; GPU close is write-once and failed time counts;
published a16 regeneration refuses; both v7 digest pairs
byte-identical; zero writes to the v7 root; capability
transplant refuses).

POST (`b4_c49_c56_post_2026_09_10.py|.out`): the PRE's starved
child is terminated, escalated, reaped and quarantined with a
single close record and the gate closed; **six guard mutants
bite** — deadline-off and liveness-off leave the child IMMORTAL
(the frozen PRE reopens), late-rule-off accepts the late
COMPLETED, reap-off leaves a TERM-immune child SURVIVING,
close-off breaks the charge derivation typed, and the
no-next-cell refusal branch is load-bearing.

## Counts at the final tip

- Supervisor battery 12 passed; surface index 17 passed;
  B4 authority battery **181 passed** after its ordered
  migration to the v8 world (the historical fixtures installed
  v7 actas; the ONE launch gate now demands the v8 pair, so the
  central fixtures were extended to install acta+owner records,
  the chain fixture gained a synthetic a16 whose pins own the
  live check, three pin-mutation tests now plant their forgery
  in the FINAL amendment, the c44 pair was rebuilt over the REAL
  process boundary — the child writes its own integral typed
  terminal via the productive write_terminal — and c38/c42
  custody cases target the v8 acta object; all three batteries
  together: **209 passed, 1 skipped**).
- Full repository suite at `2c3e8fcf`: **3,091 passed, 2
  failed, 2 skipped (7:53)** — only the inherited D1-anchor
  pair. The packet commit atop is docs-only.

## Untouched boundaries

M4 worktree preserved exactly as ordered (its branch tips
`a420f858`/`5e7a8fd4` unmodified); T2 root untouched (P1 next);
both completed B4 cells byte-identical; v7 root zero writes;
B4/T2 services stopped and untouched.

## Remaining blockers, each assigned

1. v8 supervised-recovery review → his acta + the owner's
   separate dispatch-scope record → only then any GPU dispatch —
   **Musashi + owner**.
2. T2 C89-C94 completion adjudication — **me, next (P1)**.
3. M4 C32-C38 confirmation preparation — **me, after P1 (P2)**.

`B4_V7_QUARANTINED_AND_EXTERNAL_WATCHDOG_RECOVERY_READY_FOR_MUSASHI_REVIEW`
