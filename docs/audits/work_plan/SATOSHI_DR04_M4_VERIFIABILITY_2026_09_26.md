# DR04 — M4: verifiability repaired before confirmation

**Satoshi, successor technical lead. 2026-09-26.**
Order: `docs/handoffs/SATOSHI_DAY_REVIEW_CONTINUATION_2026_09_26.md` §DR04.
Dictamen answered: `docs/audits/work_plan/MUSASHI_DAY_REVIEW_2026_09_26.md` F2 / P1,
with its evidence `docs/audits/evidence/DAY_REVIEW_2026_09_26/{M4_PROBE.txt,M4_FINDINGS.md}`
(findings F1–F6).
Repository: `agent-multi`, branch `satoshi/dr04-m4-verifiability-20260926`,
off the reviewed candidate `0de54534`.

**This document is not an approval.** The auditor will review the repaired
candidate; my POST is my own evidence, not his verdict. **No external auditor
signature was created, installed, mocked or simulated anywhere in this work**:
the role token `EXTERNAL_AUDITOR`, the decision strings, `REVIEWED_TIP` and the
two record paths are exactly as they were, and both record paths are still
absent on this host. **CONFIRMATION was not run**, and no CONFIRMATION array
was fitted or scored. Everything below is CPU-only, `CUDA_VISIBLE_DEVICES=''`,
through `crispdm-run -m 2G|3G -t <WALL> -n m4verify --`.

## 1. The answer first

**No.** After the repair, a fabricated summary, a repeated seed and an
incomplete population cannot reach `VERIFIED`, and none of them can produce a
statistical rejection. Each is refused **by name**, on the same public path the
auditor probed, and each refusal is a permanent test:

| the auditor's counterexample | before (PRE, frozen at `0de54534`) | after (POST) |
|---|---|---|
| 234 fabricated summaries, no raw logs, no record hashes, `gates={}` | `records_verified=234`, 39 generators, `p_holm=5.4399453636652144e-33`, **reject** | refused when the ledger is **written** (`gate map is not the exact schema`) and, over a real root, refused at the record (`omits the identity field 'kind'`) |
| a consistently altered pair of endpoints (declared difference agrees) | not detectable — the verifier compared a summary with itself | refused: `declared restricted endpoint 8 does not re-derive from the raw log (0)` |
| three differently named copies of `s0`, generators `g9000..g9038` out of census | 39 "complete" generators, **reject** | refused: `g9000… is NOT in the exact census`; and the copy refusal **names the repetition**: `REPEATS model seed s0 of generator DEVELOPMENT-chirp-clean-g0` |
| 21 eligible slots below the floor, the 15th contrast still evaluated | `p_holm=4.1244935474544496e-25`, **reject** | the 15th contrast is `NOT_EVALUABLE`, `reject_at_alpha=false`, `checkpoint_population={}`, `population_complete=false` |
| 60 distinct identities collapsed into 30 by the bare `gN` index | `n_generators=30`, **reject** | 8 distinct identities give 8 observations; indices are never pooled across families, noise regimes or widths |
| a record of `{unit_id, hash}` counted complete, disjointness skipped | `units_complete=1`, `census_complete=true`, `generators_disjointness_verified=0` | refused: `omits the identity field 'kind'`; and a resumed unit's disjointness proof is **revalidated** (`generators_disjointness_verified=4` on a 24-unit resume) |
| the same (absent) authority accepting two different code heads | both recorded, both verified | the review record must pin `reviewed_implementation_sha256`, the gate compares it with the live bytes, and a ledger written by other code never yields a verdict |

The exact census is unchanged and still re-derives: **3024 units,
`12cfd9ad785b41e788ffce575ec575ab2a78ab772151b5e3b94c8f0c71169ea0`**.

## 2. The PRE battery: his probes, frozen before the repair

`docs/audits/evidence/repro_runs/m4_dr04_verifiability_pre_2026_09_26.{py,out}`
— committed at **`bba6932d`**, *before* one line of the repair existed
(the repair is the next commit, `0cddcb6b`).

The auditor extracted function ASTs from Git blobs and drove them with
in-memory fixtures. The PRE battery reproduces the same six findings **through
the module's public surface only** — `cp.bind_calibration_evidence`,
`cp.verify_confirmation_successor`, `cr.materialize_census`,
`cr.write_pre_result_ledger`, `cr.verify_confirmation_run`,
`cr.execute_confirmation_units`, `cr.execute_confirmation` — with no AST
surgery, no monkeypatched verifier and no mocked authority reader. It
reproduces his two Holm p-values digit for digit and his census hash.

One deliberate difference: his F5 used a mocked record reader returning
sentinel digests. I replaced that with a **record-free** structural probe (the
public review schema has no key naming the implementation; the public ledger
writer accepts two different fabricated `executing_head` values and the
verifier reads neither), because simulating an auditor signature is not mine
to do.

Nothing confirmatory was constructed in the PRE: the "confirmation" documents
are fabricated summary JSON with no arrays, no logs and no lineage. That they
were counted verified **is** the finding.

## 3. The six repairs, each with a positive and a negative case

Permanent tests: `tests/test_m4_dr04_verifiability.py` (**42 tests**).
POST evidence: `docs/audits/evidence/repro_runs/m4_dr04_verifiability_post_2026_09_26.{py,out}`,
exit 0.

Every positive case runs over a **DEVELOPMENT fixture population** — 2 cells ×
2 widths × 2 generators × 3 nested seeds = **24 real units** with real
generators, real fits and real raw per-batch logs — through **the same code**
the CONFIRMATION path runs. No new confirmatory data was constructed. The
fixture can never yield a confirmatory verdict: the verifier requires the
order-pinned successor whenever the expected role is CONFIRMATION, refuses the
pinned successor for a fixture role, the census and the ledger both carry the
role, and the verdict carries
`NON_CONFIRMATORY_DEVELOPMENT_FIXTURE_NO_SCIENTIFIC_VERDICT`.

### R1 — authority bound to the reviewed implementation, not a recorded HEAD

`cp.IMPLEMENTATION_FILES` names the six modules that generate, execute,
analyse and verify the screen; `cp.implementation_digest()` is the sha256 over
the canonical `(path, file digest)` table of their working-tree bytes.

- `_REVIEW_KEYS` gains the **required** key `reviewed_implementation_sha256`;
  `read_musashi_review_record` refuses when it is absent, not 64 hex, or not
  the implementation that would run — *"a review of other code authorizes
  nothing"*. The check lives inside `cp.require_both_records`, so the sealed
  mutant that removes the two-record gate still behaves exactly as sealed.
- `execute_confirmation` refuses when any implementation file differs from the
  commit it claims, writes `executing_implementation_sha256` into the ledger
  gates, and adds it to the resumption identity set.
- `verify_confirmation_run` refuses a ledger written by a different
  implementation, and for CONFIRMATION re-reads both records and requires them
  to be the records the ledger was written under.
- New `authority-digest` subcommand prints what an approval must pin. It
  installs nothing.

**Positive:** the live digest is stable, covers the six files, and a
byte-identical copy of the tree gives the same digest; a review double pinning
the live digest satisfies the check and chains to the owner record
(`tests/test_m4_confirmation_protocol.py`, existing `_mk_records` machinery).
**Negative:** one appended byte changes the digest; an absent module refuses;
a review double pinning `ab…` refuses with *a review of other code authorizes
nothing*; a placeholder refuses as a template; a ledger carrying `a*64`
refuses with *never read out of a run produced by other code*.

`REVIEWED_TIP` (`5e7a8fd4…`) is **unchanged** — it pins the calibration
evidence, and pointing it at my own tip would be self-authorization. The
implementation binding is the separate, code-level requirement. See §8.

### R2 — exact census, distinct seeds, complete identity

- `census_units(successor, role)` is the one enumeration; `confirmation_units`
  is now a call to it, and the CONFIRMATION census bytes are **byte-identical**
  (its sha256 is quoted in accepted evidence; a non-CONFIRMATION census carries
  its role explicitly so it can never be mistaken for the confirmatory one).
- Every record file must carry the **canonical** name of the unit it declares;
  the unit must be **in** the census; no identity may appear twice. The
  refusal **names the repeated model seed and its generator**.
- Every identity field (`kind, role, family, noise_coord, width,
  generator_index, generator_id, model_seed, unit_role`) must equal the census
  unit, and `generator_id` must re-derive from role/family/noise/index.
- The analysis stage consumes a **seed-keyed** map
  (`{cell: {width: {gN: {model_seed: effect}}}}`), so a row count can never
  stand in for the sealed seed set; a list of rows is refused by name.

**Positive:** the exact seed set `{0,1,2}` makes a generator complete; the
24-unit fixture verifies with `records_verified=24`.
**Negative:** `g9000` refused as out of census; three named copies of `s0`
refused, naming the repetition; `model_seed` and `generator_id` tampering
refused by field name; `{"g0": [1.0,1.0,1.0]}` refused as *not keyed by model
seed*.

### R3 — raw, authenticated evidence and numerical re-derivation before VERIFIED

`reconstruct_unit()` rebuilds, for **every** unit: the generator (and its
manifest), the association tape (digest and id), the four checkpoints and
their lineage, the stop trajectory and the selected stop update, then replays
**all four arms** and compares every fact — restricted endpoint, cap,
stopping cause, updates, retention margin, final parameter digest,
descriptors, stop-split checkpoint loss — and compares **every raw per-batch
line** with the replay. `authenticate_arm_log()` additionally re-derives the
endpoint, the update count, the cap, the stopping cause, the failing batch and
the retention streak from the raw log alone, and refuses a surviving durable
resume state beside a "complete" record. The verifier reports
`reconstruction: FULL_REPLAY_FROM_RAW_RECORDS`.

The pre-result ledger's gate map is now **evidence that is validated**
(`validate_ledger_gates`), at write time and at verification time; the empty
gate map of the counterexample is refused when written.

**Positive:** the 24-unit fixture returns `VERIFIED` with
`records_numerically_rederived=24`, 96 raw per-batch logs present, and a
replay accounting of 99 500 optimization updates.
**Negative:** a fabricated summary; a missing arm log; a single altered raw
line; **a consistently altered pair of endpoints whose declared difference
still agrees**; an empty gate map; a gate map with a non-digest; a
CONFIRMATION ledger carrying the fixture token — each refused by name.

### R4 — attrition and a complete population for every contrast, the 15th included

- **A missing record refuses**: `POPULATION_INCOMPLETE: n of N census units
  have no record … an incomplete population refuses, it never narrows the
  denominator`. It does not narrow, and it does not report a verdict.
- The attrition floor is applied per eligible slot **before** any contrast.
- The **fifteenth** contrast is built from verified complete generator
  **identities** (`fam::nz::wW::gN`) that survive attrition, with the
  aggregation rule stated in the output
  (`checkpoint_population_rule`, `checkpoint_population` per slot):
  *one observation per verified complete generator identity, nested seeds
  averaged first, below-floor slots excluded; generator indices are NEVER
  pooled across families, noise regimes or widths.*
- Nothing reads `VERIFIED` while any identity is partial or any slot is below
  its floor: the verdict becomes
  `POPULATION_INCOMPLETE_NO_PRIMARY_VERDICT` and every contrast carries
  `population_complete: false`.

**Positive:** the complete fixture gives
`checkpoint_population={sine::clean::w16:2, w64:2, chirp::clean::w16:2, w64:2}`,
`n_generators=8`, `population_complete=true`.
**Negative:** deleting one of 24 records refuses; every slot below floor makes
the 15th contrast `NOT_EVALUABLE` with `reject_at_alpha=false` and an empty
population; two families sharing index `g0` remain two observations.

### R5 — resumption carrying the complete schema

"Complete" now means the **complete terminal schema**: identity, a typed
terminal state (the four arms, or `NUMERICALLY_INVALID_TASK_TRAINING` with its
update), `checkpoint_lineage` over the four checkpoints, `genesis_digest`
consistent with it, tape id/digest/tolerance, `manifest_sha256`,
`stop_trajectory_digest`, `selected_stop_update`, a paired difference equal to
its own arm endpoints — **and** the authenticated raw per-batch logs with no
surviving durable state. In addition, the pre-pass **revalidates the
role-disjointness proof** for the generators of resumed units and requires each
resumed record to bind the manifest its generator re-derives.

**Positive:** resuming a complete 24-unit root runs nothing new
(`units_new_this_session=0`), rewrites no record (byte-identical digests) and
proves disjointness for all 4 generators.
**Negative:** `{unit_id, hash}` refused; a record missing
`checkpoint_lineage` refused; a leftover `.state.npz` refused as UNCERTAIN; a
poisoned prior-role census refuses **on resume alone**; a record binding a
foreign manifest refused. A CONFIRMATION record in a DEVELOPMENT census and a
DEVELOPMENT record in a CONFIRMATION census are both refused —
*one census, one role*.

### R6 — cumulative accounting, no double execution under interruption or concurrency

- **Run lock** `M4_RUN_SESSION.lock`, `O_EXCL`, carrying pid + boot id: a live
  holder **refuses**; a lock left by an interrupted session is **preserved**
  under `ABORTED_LOCKS/` (never deleted) and the session continues.
- **Per-unit claim** `…__CLAIM.json`, `O_EXCL`: a unit is executed exactly
  once even if the run lock were bypassed. A live claim refuses; a stale claim
  is preserved under `ABORTED_PARTIALS/` and replaced, exactly as partial
  bytes already were — so the sealed *"partial intervention log without its
  durable predecessor state is UNCERTAIN"* refusal still bites unchanged
  (verified: the sealed exec-body POST's `set_aside_off` mutant still reports
  that same refusal).
- **Cumulative accounting**: session reports are self-verified and carry
  `cumulative{sessions, wall_seconds, optimization_updates, evaluations,
  descriptor_evals, units_new}`; an unverifiable prior report makes the
  account UNCERTAIN and refuses.
- **The sealed wall now counts the whole run.** `new_accounting` moves `t0`
  back by the wall already spent in the root, so the *sealed, unmodified*
  `rn._limits` enforces the frozen 172 800 s over the cumulative cost instead
  of restarting at every interruption.

**Positive:** a stale lock is set aside and preserved, the run continues, the
lock is released at exit; cumulative accounting over the fixture run reports
1 session, 24 new units, 99 500 updates, 9.07 s.
**Negative:** a live lock refuses; a live unit claim refuses; with the wall
tightened below the wall already spent, the next session returns `WALL_STOP`
and runs **0** new units; a tampered prior report refuses as UNCERTAIN.

### The complete entrypoint

New subcommand `development-verification-probe` runs **census → pre-result
ledger → execution body → independent verifier** in one process, over the
DEVELOPMENT fixture population, and prints the verdict. Positive: exit 0,
`VERIFIED`, `NON_CONFIRMATORY…`. Negative: with `--batch-units 3` the census is
incomplete, the probe reports `NOT_ATTEMPTED_CENSUS_INCOMPLETE`, and the
verifier **refuses** that root rather than narrowing it. `verify --run-root`
on a fixture root refuses (role mismatch), and `execute` still refuses with
both records absent and creates nothing.

## 4. What I changed in the sealed batteries, and why

Four tests in `tests/test_m4_confirmation_protocol.py` drove their scientific
properties through **fabricated CONFIRMATION summary documents fed to the
verifier** — precisely the route the auditor showed can manufacture a
rejection, and the route the repair closes. Their properties are preserved and
re-expressed on the stage where they actually live, the pure and exact
`cr.analyse_complete_population`:

| sealed test | before | now |
|---|---|---|
| `test_missing_nested_seed_never_completes` | fabricated run world | seed-keyed effects; the incomplete identity is **named** and `population_complete=false` |
| `test_attrition_beyond_allowance_never_favorable` | fabricated run world | same, **plus** the new assertion that the 15th contrast no longer escapes attrition |
| `test_numerical_failure_stays_in_denominator` | fabricated run world | the generator that lost a seed is named, never silently removed |
| `test_forged_producer_aggregate_refuses` | fabricated run world | `cr.validate_unit_record` refuses a declared difference that does not equal its own arm endpoints |

The record layer that *feeds* that stage is covered end to end, over real
records, in `tests/test_m4_dr04_verifiability.py`. The guard-removal mutant
`rederive_off` was retargeted to the paired-difference check it now lives in,
`floor_off` is unchanged, and both still bite (mutant admits, live code
refuses). `_mk_records` pins the live implementation digest; three new tests
cover the implementation binding.

Test counts: **108 → 153** across the seven M4 batteries (`tests/unit/*` has
20 pre-existing collection errors unrelated to M4; I did not touch them).

## 5. No gate weakened, reordered or bypassed

- `records = cp.require_both_records(successor)` is the **same line, byte for
  byte, in the same position**; the sealed C32–C38 POST anchors on that text
  and still passes, with mutant A reaching census + ledger and mutant B
  verifying a changed threshold exactly as sealed.
- The kill-17 construction guard, `verify_role_disjointness` and its refusal
  text, `refuse_foreign_role_record`, the pinned identities, the decision
  strings, the role tokens, `rn._limits`, `rn._run_intervention_unit_v5` and
  `rn._excl_json` are untouched. `refuse_role_mismatch` is **added**, and it
  only ever refuses more.
- Every check I added either refuses where nothing refused before, or runs
  strictly after a chain that had already passed. There is no flag anywhere
  that downgrades re-derivation, completeness or authority.
- Sealed evidence re-run today on the repaired candidate:
  `m4_c32_c38_post_2026_09_10.py` **passes**, and
  `m4_confirmation_exec_body_post_2026_09_26.py` **passes**. I did **not**
  overwrite either frozen `.out`: rewriting the artifact of a past run would
  falsify the record. (The exec-body POST's final printed sentence carries the
  wording corrected in §7; that wording is superseded, the script's behaviour
  is not.)

## 6. Cost and containment

| run | command | wall |
|---|---|---|
| PRE battery | `crispdm-run -m 2G -t 900 -n m4verify` | ~4 s, exit 0 |
| POST battery | `crispdm-run -m 3G -t 1800 -n m4verify` | ~75 s, exit 0 |
| 7 M4 suites | `crispdm-run -m 3G -t 3000 -n m4verify` | 96.9 s, 153 passed |
| sealed C32–C38 POST | `crispdm-run -m 3G -t 1800 -n m4verify` | ~40 s, exit 0 |
| sealed exec-body POST | `crispdm-run -m 2G -t 2400 -n m4verify` | ~90 s, exit 0 |

All CPU-only with `CUDA_VISIBLE_DEVICES=''`; no GPU was used, no heavy compute
was dispatched to the coordinator, and one `crispdm-run` request at 3 G was
refused by the admission guard (`SLICE_AGGREGATE_BUDGET`) and I re-requested
2 G rather than bypassing it.

Full re-derivation is now the verifier's cost. Measured basis on the fixture:
0.37 s per unit to run and ~0.37 s per unit to replay, so the 3024-unit
CONFIRMATION census projects to roughly **19 minutes** of one CPU to execute
and about as much to verify, inside the sealed 172 800 s wall and the sealed
8 GiB RSS ceiling. That is the price of a verdict that rests on raw records.

## 7. The sentence I published, corrected

The previous return said: *"No CONFIRMATION array, score or ledger was created
by me, anywhere, at any point."* As written that is **false**, and it blurred
four facts that are not interchangeable. Corrected in place as an erratum in
`docs/audits/work_plan/SATOSHI_M4_CONFIRMATION_EXECUTION_BODY_2026_09_26.md`
§3, asserted mechanically in `test_f6_four_facts_about_confirmation_arrays`,
and stated here:

| fact | holds? | how it is established |
|---|---|---|
| **1. Disjointness proof — in-memory CONSTRUCTION** of CONFIRMATION arrays | **HAPPENED** | `gb.generate("CONFIRMATION", …, allow_confirmation=True)` is called by the explicit byte-level role-disjointness proof: two call sites in `tests/test_m4_confirmation_execution_body.py`, one in the exec-body POST. `tools/m4_generator_bank.py:198` permits exactly this exception. A filesystem name scan cannot see it. |
| **2. MATERIALIZATION** — any CONFIRMATION array, run root, ledger or record on storage | **did NOT happen** | nothing persisted: both authority record paths absent, zero `*m4*confirmation*` in the state root, zero CONFIRMATION-named artifacts in any probe root. |
| **3. FITTING** — a model trained on CONFIRMATION bytes | **did NOT happen** | the kill-17 guard refuses for every default caller; the two-record gate refuses before the execution body; no tape, checkpoint or arm was ever built on those bytes. |
| **4. SCORING** — an endpoint, paired difference, contrast or Holm p-value from CONFIRMATION bytes | **did NOT happen** | no CONFIRMATION record, session report or verification document has ever existed. |

**The protocol exception, resolved explicitly.** C35/C36's *"no CONFIRMATION
arrays before the gate"* is read as **no materialization, no fitting, no
scoring**. The bank's named disjointness-proof exception is the one
construction it allows; it is in-memory only, and it is what makes the
byte-level disjointness claim *checkable* instead of asserted. Anything
broader than that reading would make the disjointness proof itself
unperformable. If the auditor wants the stricter reading — zero construction,
therefore no byte-level disjointness proof — that is his call to make, and it
is on the list below.

## 8. What must still correspond before CONFIRMATION may run

Nothing here may be supplied by me. The screen stays closed until **all** of
these correspond to one another:

1. **Musashi's design-review record** at
   `~/.local/share/agent-multi/m4_confirmation_authority/MUSASHI_M4_CONFIRMATION_DESIGN_REVIEW_RECORD.json`,
   authored and installed by him, decision
   `M4_CONFIRMATION_DESIGN_APPROVED_FOR_EXECUTION`, pinning
   `reviewed_successor_sha256`, `reviewed_tip`, and now
   **`reviewed_implementation_sha256` — the digest of the implementation he
   actually reviewed**, printed by
   `python3 tools/m4_confirmation_runner.py authority-digest`. The template at
   `docs/audits/evidence/MUSASHI_M4_CONFIRMATION_DESIGN_REVIEW_TEMPLATE_2026_09_10.json`
   carries the new placeholder. On the current tip that digest is
   `68e4c0ed22bb910cad1ba5722f5d5647821064dd1db3ff0f1b1474e4923fa83c`; **it
   changes with any further change to the six M4 modules**, so it must be read
   from the reviewed checkout, not copied from this document.
2. **The owner's execution record**, chained to the review record's
   `record_sha256`, decision
   `M4_CONFIRMATION_EXECUTION_AUTHORIZED_CPU_ONLY`.
3. **`REVIEWED_TIP` vs the executing tip.** The protocol still pins
   `5e7a8fd430c8231a049baf03f00e720ba24ec994` — the calibration-evidence tip —
   and I did **not** repoint it at my own branch, because that would be
   self-authorization. Either the review record is authored against that tip
   while its `reviewed_implementation_sha256` names the code that will run
   (the chain then binds both), or the auditor orders `REVIEWED_TIP` moved to
   the tip he reviews. **This is an open correspondence, and it is his to
   settle, not mine.**
4. **The aggregation rule of the fifteenth contrast.** I froze it explicitly
   (one observation per complete generator identity, seeds averaged first,
   below-floor slots excluded, indices never pooled) because the sealed
   documents did not state it and the old code pooled by bare index. Under the
   sealed design this changes the contrast's population from ≤ 48 pooled
   values to up to 21 × 48 identities. **It needs his explicit acceptance or
   his substitute rule** before it decides anything.
5. **The reading of the array-construction exception** in §7 — accepted as
   written, or replaced.
6. **Whether an incomplete population should refuse or report.** I chose
   refuse: the verifier raises `POPULATION_INCOMPLETE` and produces no verdict.
   Progress is read from the session reports instead. If he wants a typed
   non-verdict document rather than a refusal, that is a change of his design,
   not a defect of this one.
7. **Re-verification cost.** Full replay doubles the run. If the frozen budget
   is to cover execution *and* verification, someone must say so; I did not
   change any budget.
8. **What this repair does NOT certify.** The auditor's own caveats stand and I
   add nothing to them: no CONFIRMATION unit has been fitted or scored, so no
   scientific outcome of M4 exists; `NO_NEW_MEASUREMENT` for M4 is unchanged by
   this work; hard resource containment under true multi-host concurrency was
   not exercised (my concurrency proofs are lock/claim-level, single host); and
   the 20 pre-existing `tests/unit/*` collection errors are untouched and
   unrelated.

**NO_NEW_MEASUREMENT.** Nothing in this package is a scientific measurement,
so no closure table is due: the only numbers reported are refusal behaviour,
record counts, digests and wall times. The DEVELOPMENT fixture's 24 units are
mechanics, explicitly labelled
`NON_CONFIRMATORY_DEVELOPMENT_FIXTURE_NO_SCIENTIFIC_VERDICT`, and they are not
a result about anything.

## 9. Artifacts

| what | where |
|---|---|
| PRE battery (committed before the repair, `bba6932d`) | `docs/audits/evidence/repro_runs/m4_dr04_verifiability_pre_2026_09_26.{py,out}` |
| POST battery | `docs/audits/evidence/repro_runs/m4_dr04_verifiability_post_2026_09_26.{py,out}` |
| permanent tests (42) | `tests/test_m4_dr04_verifiability.py` |
| repaired implementation | `tools/m4_confirmation_runner.py`, `tools/m4_confirmation_protocol.py` |
| sealed batteries updated, with reasons in §4 | `tests/test_m4_confirmation_protocol.py`, `tests/test_m4_confirmation_execution_body.py` |
| erratum on the published sentence | `docs/audits/work_plan/SATOSHI_M4_CONFIRMATION_EXECUTION_BODY_2026_09_26.md` §3 |
| review-record template with the new pinned key | `docs/audits/evidence/MUSASHI_M4_CONFIRMATION_DESIGN_REVIEW_TEMPLATE_2026_09_10.json` |

Signed: **Satoshi, successor technical lead, 2026-09-26.**
Nothing in this document is written under Musashi's name, and nothing in it is
his approval.
