# M4 CONFIRMATION — the missing execution body (2026-09-26)

Author: **Satoshi, successor technical lead.** Under the owner's grant of
2026-09-26. Frozen corpus: `agent-multi@0e99ad1a` (branch
`satoshi/model-capacity-m3-20260908`, the M4 C32-C38 return tip). Work branch:
`satoshi/m4-confirmation-exec-body-20260926`, in its own worktree; the owner's
checkout was not touched.

**Lead answer: yes — the screen can now run, once and only once its two
records exist.** The 3024-unit CONFIRMATION census is driven end to end by
`run_confirmation()`, resumably, through the sealed v5 limit machinery, at a
measured **0.36 s and 103 MiB per unit → ≈18 min of one CPU** inside the
sealed 172,800 s wall. **And the gate still refuses without the records**:
`run_confirmation()` with either record absent exits
`REFUSED: … record is ABSENT …`, creates **nothing** (the output root does not
exist afterwards), and — proven by stubbing the body to raise — **never enters
the body at all**. I authored neither record, installed neither, and ran no
CONFIRMATION unit.

---

## 1. What was missing

`tools/m4_confirmation_runner.py::execute_confirmation()` ran the whole gate
chain (C32 bind → C33 successor → `require_both_records` → clean-checkout
identity → 3024-unit census → O_EXCL pre-result ledger) and then **returned**.
Its closing note said

> "unit execution proceeds only beyond this point, through the sealed v5 limit
> machinery, with per-array disjointness verification"

— beyond a point that did not exist in the module. The only call to
`rn._run_intervention_unit_v5` in the file was inside
`development_mechanics_probe`. Confirmed at the frozen commit: with both
approvals installed, `execute` would have produced a PENDING ledger for 3024
units and **zero fitted units**. Three further gaps the body had to close:

1. **Nothing resumable.** The v5 intervention loop treats "the summary file
   exists" as "the unit is done", and `_excl_json` creates the final name
   *before* writing, so a process killed mid-write leaves a truncated record
   that reads as complete.
2. **A unit interrupted mid-arm was unresumable at all.** A completed arm
   deletes its durable state, so re-entering that unit meets the sealed
   `partial intervention log without its durable predecessor state —
   UNCERTAIN` refusal forever. Verified live (§4.2).
3. **The declared per-array disjointness check was domain-mismatched.**
   `prior_role_digest_census()` digests whole persisted `*.npz` **files**; a
   generator **array** digest can never equal a zip-archive digest, so that
   census alone could only ever catch a byte-identical file.

## 2. What I wrote

Two files changed (+461/−11: runner +448/−9, v5 runner +13/−2), one test
battery (324 lines, 16 tests) and one POST (336 lines) added. No sealed
test, no sealed evidence file, no design, no successor and no record was
edited.

### `tools/m4_v5_runner.py` (+13/−2)

`_run_intervention_unit_v5(design, u, out, acct, allow_confirmation=False)` —
one **pass-through** to the generator bank's C33 kill-17 construction guard,
defaulting to `False`. Every pre-existing caller (`execute_v5`,
`verify_run_v5`, the sealed DEVELOPMENT probe) keeps refusing CONFIRMATION
bytes byte-for-byte as before; only a caller that has already passed the
two-record gate may set it. It grants no authority and opens no gate. Proven
still closed by `test_confirmation_unit_still_refuses_by_default` and by POST
Phase 1 ("kill 17 (default caller, CONFIRMATION unit): REFUSED").

### `tools/m4_confirmation_runner.py` (+448/−9)

- **`execute_confirmation_units(...)` — the body.** The unit loop, shaped on
  the existing v5 intervention loop (`m4_v5_runner.py` ~line 648): per-unit
  record path, skip-if-recorded, `rn._limits()` before each unit, sealed
  `rn._run_intervention_unit_v5()` for the four checkpoint arms, and the same
  accounting dict (`optimization_updates`, `evaluations`,
  `descriptor_seconds`, `descriptor_evals`) that `_limits` reads. It drives
  the sealed machinery; it replaces none of it. It contains **no authority
  decision** — it never consults a record, and it must already have been
  allowed to run.
- **Per-array role disjointness, in the array domain.**
  `generator_array_digests()` takes the six byte-disjoint array digests the
  bank hands the consumer (after `gb.consumer_verify()` re-derives them from
  the bytes in use); `role_array_digests()` re-derives the
  DEVELOPMENT+CALIBRATION array census for exactly the cells about to run.
  The body checks each generator's arrays against the **union** of the sealed
  file census and this array census, through the sealed
  `verify_role_disjointness()`. This only ever *enlarges* the forbidden set;
  the sealed comparison and its refusal text are untouched. An empty census
  refuses rather than passing vacuously.
- **Atomic per-unit records.** `write_unit_record()` writes and fsyncs under a
  temporary name and only then `os.replace`s into place (plus a directory
  fsync), so a record carrying its **final** name is never a partial.
  `read_complete_unit_record()` accepts a unit as complete only if the record
  is strict JSON, its `record_sha256` re-derives, and it binds that unit id —
  otherwise a typed refusal (UNCERTAIN), never a silent re-run and never
  counted.
- **Unit-atomic resumption.** A pre-pass classifies all 3024 units as COMPLETE
  or PENDING. `abort_partial_unit()` **preserves** an interrupted unit's arm
  logs and durable states by moving them to
  `ABORTED_PARTIALS/<unit>/attempt_NNN/` — nothing is deleted — and the unit
  runs again from scratch. At 0.36 s per unit, re-running the single in-flight
  unit is cheaper and stricter than any partial resume, and it is what makes
  the sealed UNCERTAIN refusal survivable instead of terminal.
- **Typed resumable stops.** `WALL_STOP` / `RSS_STOP` / `STOP_REQUESTED` from
  the sealed `_limits`, plus an operator `--batch-units` bound
  (`BATCH_FILLED_RESUMABLE`), each end the session with
  `census_complete: false` and an exact pending count instead of a crash.
- **One append-only `SESSION_NNN_REPORT.json` per session** (deliberately not
  CONFIRMATION-named, so the sealed zero-artifact assertion is unchanged; the
  role is a field), self-digested, carrying census digest, ledger path, both
  gate record digests, accounting and wall seconds.
- **`run_confirmation()` — the complete execution path**: gate chain first,
  then the body, then the session report. The `execute` subcommand now runs
  this; `gate-only` keeps the gate stage reachable alone.
- **`execute_confirmation()`**: resumption tolerance (an existing root must
  carry a pre-result ledger whose self-identity, census digest and *scientific*
  gate identities — successor, both records, executing head — match, or the
  session refuses; `prior_role_digests` is excluded from that comparison
  because it is a live count of the state root), `chmod 0700` on the root, and
  the closing note now names the function that actually continues.
- **`development_execution_probe()`** and the CLI subcommands
  `development-execution-probe` and `gate-only`; the execution subcommands set
  `CUDA_VISIBLE_DEVICES=""` and `nice 15` per the sealed `resources` block.

### One design judgement, declared not slipped

**The body lives one call below `execute_confirmation`, not inside it.** The
sealed POST battery
(`docs/audits/evidence/repro_runs/m4_c32_c38_post_2026_09_10.py`, mutant
`A_gate_off`) proves the two-record gate is load-bearing by **removing it** and
calling `execute_confirmation()` directly with no records installed. Had I put
the 3024-unit loop inside that function, re-running that sealed proof would
itself manufacture an unauthorized CONFIRMATION census. Gate stage and body
are therefore separate functions and the execution path (`execute`) runs both;
the sealed proof still reaches exactly census + ledger and nothing more
(§4.3). This is the only place where I chose a shape rather than copying one,
and it is the reviewer's to overrule.

## 3. What I did not do

- **No gate weakened, bypassed, softened or reordered.**
  `cp.require_both_records(successor)` is the same unchanged call in the same
  position — the first statement in `execute_confirmation` that can refuse
  after the C32/C33 binding, and still ahead of every mkdir, array and ledger.
  The hardcoded role tokens, key sets, decision strings, pinned identities and
  refusal texts are unchanged. Every line I added *runs* strictly after the
  chain has passed.
- **Neither record authored, installed, mocked, stubbed or committed** — not
  the Musashi design-review record, not the owner execution record, not any
  new fixture resembling one. Both paths are still absent on this host
  (asserted in POST Phases 1 and 4). No test in my new battery installs a
  record; the only record doubles that exist are the sealed battery's own.
- **The CONFIRMATION screen was not run.** No CONFIRMATION array, score or
  ledger was created by me, anywhere, at any point. The body was exercised
  over DEVELOPMENT units only.

## 4. The three proofs

All runs CPU-only, `CUDA_VISIBLE_DEVICES=''`, every one through
`crispdm-run -m <MEM> -t <WALL> -n m4body --` with the cap read from live free
RAM (13.2 GiB available, 3–5 GiB requested). Single artifact:
`docs/audits/evidence/repro_runs/m4_confirmation_exec_body_post_2026_09_26.{py,out}`,
**exit 0**.

### 4.1 The gate still refuses, and creates nothing

| check | result |
|---|---|
| `run_confirmation()` with the review record absent | `REFUSED: Musashi design-review record is ABSENT …` |
| output root after that refusal | **does not exist** |
| body entered during that refusal (stubbed to raise) | **0 times** |
| owner record absent alone | refuses (chain) |
| `plan` | 3024 units, 21 slots, `execution_open: false`, both records `False` |
| kill 17: CONFIRMATION unit through a default caller | `REFUSED: CONFIRMATION generators are RESERVED …`, no artifact |
| sealed `development_mechanics_probe` | 2 records, 0 CONFIRMATION artifacts (unchanged) |

Battery: `tests/test_m4_confirmation_execution_body.py::`
`test_execution_path_refuses_without_records`,
`test_gate_refusal_never_reaches_the_body`,
`test_owner_record_absent_alone_still_refuses`,
`test_no_confirmation_artifact_is_created_by_this_battery` — all passing.

### 4.2 The body end to end over DEVELOPMENT units, real process boundary

Through the CLI, in separate processes:

| run | numbers |
|---|---|
| fresh census, 4 DEVELOPMENT units | 4/4 complete, 4 new, `CENSUS_COMPLETE`, 4 generators disjointness-verified against 384 prior-role array digests, 0 CONFIRMATION artifacts |
| accounting | `optimization_updates` 16,350 · `evaluations` 19 · `descriptor_evals` 16 · wall 1.43 s |
| cost basis | **0.36 s/unit**, RSS flat at **102.8 MiB** (no accumulation across units) → 3024 units ≈ **18.1 min** of one CPU; records ≈ 1.6 KiB/unit ≈ 5 MB total |
| idempotent re-run | **0** new units, 4 verified complete |
| batched 2 + 2 | session 1: 2 new, 2 pending, `BATCH_FILLED_RESUMABLE`; session 2: complete; the first two records **byte-identical** after the resume |
| SIGKILL **inside** a unit | rc −9 with 1 record and 5 arm logs on disk, 0 session reports |
| restart after the kill | 4/4 complete, 3 new, **1 partial attempt set aside and preserved**, `CENSUS_COMPLETE`, 0 CONFIRMATION artifacts |
| mutant `set_aside_off` on that same interrupted state | `REFUSED: … partial intervention log without its durable predecessor state — UNCERTAIN` — so the set-aside step is load-bearing, not decoration |

Every probe re-asserts the sealed assertion that **no CONFIRMATION-named
artifact appears**; all record modes are `0600`. The produced records also
satisfy the independent verifier's input contract — four checkpoint arms,
integer `restricted_endpoint`/`updates_done`, and
`paired_primary_difference` re-deriving from the arm records within 1e-9
(`test_body_records_satisfy_the_verifier_contract`).

### 4.3 The sealed evidence, re-run unchanged

| battery | before | after |
|---|---|---|
| `tests/test_m4_confirmation_protocol.py` (frozen C37) | 28 passed | **28 passed** |
| `tests/test_m4_v5_protocol.py` | 20 | **20** |
| `tests/test_m4_intervention.py` | 13 | **13** |
| `tests/test_m4_numeric_incident.py` | 12 | **12** |
| `tests/test_m4_residual_capacity.py` | 19 | **19** |
| sealed M4 total | **92 passed** | **92 passed** |
| new body battery | — | **16 passed** |
| combined | — | **108 passed, 0 failed** (38.5 s) |
| frozen POST `m4_c32_c38_post_2026_09_10.py` | exit 0 | **exit 0**, mutant `A_gate_off` still `EXECUTED` with `ledger_exists: true, units: 3024` and **no unit run**, mutant `B_diverge_off` still `VERIFIED` at threshold 11, successor bytes restored |

No sealed test was edited. Nothing failed, so nothing had to be reported as a
stop.

Two pre-existing conditions, neither mine: `docs/audits/evidence/repro_runs/`
`m4_c32_c38_pre_2026_09_10.py` is pinned to the PRE tip `5e7a8fd4` and is not
re-runnable at `0e99ad1a` (it asserts the protocol is absent);
`tests/test_engineering_surface_index.py` cannot be collected in this
environment (`ModuleNotFoundError: trading_contracts`) — both fail identically
before my change. I added no `tools/` file, so `TOOL_DECLARATIONS.json` and
`ENGINEERING_SURFACE_INDEX.json` are untouched (the index snapshot does not
list the M4 confirmation tools at all).

## 5. Exact remaining preconditions for a real run

The screen is executable; it is not authorized. In order:

1. **Musashi's review of the C32-C38 packet is still outstanding.** The
   disposition at the frozen tip is
   `M4_CONFIRMATION_PROTOCOL_READY_FOR_EXTERNAL_MUSASHI_REVIEW`: the
   successor's 12/16 rule, floor 39, M2 exclusion and the declared
   Holm-supersedes-Bonferroni override are not yet ruled on. **This body is
   itself unreviewed**, including the §2 design judgement.
2. **The Musashi design-review record**, installed by Musashi at
   `<state_root>/m4_confirmation_authority/`
   `MUSASHI_M4_CONFIRMATION_DESIGN_REVIEW_RECORD.json`
   (`<state_root>` = `~/.local/share/agent-multi`): exactly the nine keys
   `_REVIEW_KEYS`, schema `musashi_m4_confirmation_design_review.v1`, role
   `EXTERNAL_AUDITOR`, decision
   `M4_CONFIRMATION_DESIGN_APPROVED_FOR_EXECUTION`,
   `reviewed_successor_sha256` =
   `6a50d97ddfb3a8e8dd1b5fbc83ebd95e60e1c087b3fc5e01697c2d783a50608c`,
   `reviewed_tip` = `5e7a8fd430c8231a049baf03f00e720ba24ec994`, a
   re-deriving `record_sha256`, owned by the operator uid, no `TEMPLATE` text
   and no `<placeholder>` values.
3. **The owner execution record**, installed by the owner at the same
   directory as `OWNER_M4_CONFIRMATION_EXECUTION_RECORD.json`: exactly the ten
   keys `_EXEC_KEYS`, schema `owner_m4_confirmation_execution.v1`, role
   `OWNER`, decision `M4_CONFIRMATION_EXECUTION_AUTHORIZED_CPU_ONLY`, the same
   `authorized_successor_sha256`, and `authorized_review_record_sha256` equal
   to the review record's `record_sha256` — the gate is a chain, not a pair of
   islands. **Neither record is mine to write.**
4. **A clean executing checkout** at the commit that will be pinned:
   `git status --porcelain` empty, or the run refuses. The head is written into
   the pre-result ledger and every resumed session must present the same one.
5. **A fresh output root** for the first session (the pre-result ledger is
   written once, before the first observation, and never rewritten); later
   sessions reuse the same root.
6. **Resources**: one CPU, `CUDA_VISIBLE_DEVICES=""`, `nice 15`, ≈18 min and
   ≈103 MiB RSS against the sealed 172,800 s wall and 8 GiB ceiling, ≈5 MB of
   records; on this host through
   `crispdm-run -m <MEM> -t <WALL> -n <name> -- …`.
7. **Then, and only then**: `python3 tools/m4_confirmation_runner.py execute
   --out <root>` (optionally `--batch-units N`), followed by the independent
   `verify --run-root <root>`, which re-derives effects, attrition against
   floor 39, costs and all 16 contrasts from the raw records.

## 6. Confessions

1. My first mid-unit interrupt harness killed the child *between* units, so it
   proved resumption but not the abort path. I tightened the trigger to fire
   only when an in-flight unit has started writing arms, and that is what
   exposed the terminal-UNCERTAIN hole of §1.2 — which I then had to close
   rather than describe.
2. My first `set_aside_off` mutant assertion matched on the word `UNCERTAIN`
   inside a 120-character truncation that did not reach it; the assertion
   passed a wrong reason once before I widened the slice and matched the
   sealed sentence instead. Reported rather than quietly fixed.
3. I considered putting the loop inside `execute_confirmation()` and rejected
   it only after reading what the sealed POST's `A_gate_off` mutant actually
   does. Had I not read it, re-running the sealed proof of my own change would
   have produced an unauthorized 3024-unit CONFIRMATION census. §2 records the
   choice for the reviewer to overrule.

---

Satoshi, successor technical lead — 2026-09-26, under the owner's grant of
2026-09-26. No CONFIRMATION screen was run; neither authorizing record was
authored or installed; no service was started, stopped or restarted; no
hostname, address, token or account identifier appears in this package.
