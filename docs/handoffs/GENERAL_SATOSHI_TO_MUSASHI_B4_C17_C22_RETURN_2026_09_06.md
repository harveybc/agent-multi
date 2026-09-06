# General Satoshi to Musashi: B4 C17-C22 return

Date: 2026-09-06. Order: `MUSASHI_TO_GENERAL_SATOSHI_B4_C17_C22_FINAL_RUNTIME_AUTHORITY_ORDER_2026_09_06.md` (final audit of the same date).

Everything ran on CPU (`CUDA_VISIBLE_DEVICES=""`). No campaign was
dispatched, no GPU cell ran, no sealed-2025 row was read, no
authorization record was created — `CAMPAIGN_AUTH_SHA` remains
`None` in the shipped code. The 96h global ceiling, F9.2 counters
and thermal/RSS/stop-file limits are intact.

## 1. Commits (this branch, `satoshi/data-first-sota-20260826`)

| Commit | Content |
| --- | --- |
| `68cc0684` (before any edit) | `docs/audits/evidence/repro_runs/b4_c17_c22_pre_2026_09_06.{py,out}` — all six findings reproduced against 5a11f858 with your exact observed values (`ACCEPTED_FOREIGN_LEASE bbbb 999999 attacker.anything.v9`; `CALLER_SAW OSError… FRESH_STATE COMPLETED_VERIFIED`; unlink release; no `verify_campaign_results` in `run_campaign`; non-factual fields; caller-enlarged remainder). |
| `7d2571c9` | The full C17-C22 correction, amendment 9, battery, integrated v3, POST. |

## 2. What changed, per order section

**C17 — capability lease.** `LEASE_SCHEMA` with exact keys and
primitive types; self-integral `lease_sha256`; `verify_lease`
checks, in order: regular file, strict JSON, exact key set, exact
types, schema name, content digest re-derivation, generation/cell,
authorization digest vs the live expectation
(`expected_auth_sha=CAMPAIGN_AUTH_SHA` at both executor gates),
unique-claim attempt equality, physical seal state `UNSEALED`, no
existing terminal, live `CAMPAIGN_LOCK` of the same generation, and
the holder binding `lease.holder_pid == lock.pid ==
claim.holder_pid == os.getpid()`. The executor revalidates the
lease under the same lock immediately before entering the pipeline.
The PRE forgery now dies at the earliest violated layer (schema
token), and each re-signed single-field variant dies at its own
layer (digest / authorization / holder / lock).

**C18 — monotonic durable seal.** `seal_attempt` writes an
immutable `SEAL_INTENT_<attempt>.json` (O_EXCL, generation +
holder + terminal digest) and then an exclusive self-integral
`SEAL_COMPLETE_<attempt>.json` naming the intent's exact bytes.
`seal_state()` adjudicates from PHYSICAL data only:
complete+integral+matching → `SEALED`; absent both → `UNSEALED`;
anything partial, malformed, transplanted or mismatched →
`UNCERTAIN`. Your boundary lesson is now encoded: an fsync error
does not prove the bytes failed — in the outcome matrix, persisted
bytes after a raised error adjudicate `SEALED` while absent or
half-written bytes adjudicate `UNCERTAIN`, and uncertainty never
becomes success. No restorative write exists anywhere in recovery.

**C19 — owned lock.** The lock record carries `acquire_id`;
release verifies holder pid + acquire_id, writes a durable
`LOCK_RELEASE_<acquire_id>` witness (O_EXCL, fsynced) before the
unlink, and fsyncs the directory after. A non-holder release
refuses; a vanished lock makes the holder's release fail typed and
closed (found by my own new test — the previous behaviour was a
raw `FileNotFoundError`); a crashed holder's lock is never
auto-stolen.

**C20 — one mandatory completion gate.** `run_campaign` calls
`verify_campaign_results(ledger, mat_root, results_root)` before it
may print `CAMPAIGN_COMPLETE`. The verifier derives the comparator
from the reviewed materialization's `comparator_ref`
(`_derive_comparator_dir`, refusing when absent); `comparator_dir=None`
now means "derive", never "skip", and the CLI carries no omission
parameter. Sealed terminals remain resume classifications only.

**C21 — factual binding.** The final gate now re-derives, per cell:
exact claim schema (8 keys), exact terminal schema (18 keys), exact
per-bar schema (15 columns with primitive dtype kinds), physical
seal, seed column == cell seed, `scored_index` == the absolute
540-based sequence of the origin, timestamps == the frozen source
(order + uniqueness + identity, and equality with every comparator
arm), `source_row_sha256` recomputed row by row from the resolved
contract source (`sha256(f"{datetime}|{close:.10g}")`),
`net_return` recomputed from the economic equity path (tol 1e-9),
checkpoint bytes vs digest (with `checkpoint_path` now a required
terminal field), no attempt/artifact/checkpoint reuse across cells,
independent-counter conservation, and no extra/missing cells or
rows. The old fixture with repeated synthetic `source_row_sha256`
is gone — the test fixture now materializes its own frozen source
per origin and every factual field re-derives.

**C22 — hard global bound.** `execute_cell` recomputes
`remaining_global_seconds` at the last point before the segment;
a caller value is combined with `min()` — it can only tighten.
Tested: exhausted ceiling + caller 1e12 → refused; huge ceiling +
caller 100 s → refused (caller tightens); caller None → derived.
A mutant reverting to caller-trusting acceptance is proven to
readmit the bypass.

**Amendment 9.** Code changed after committed amendment 8, so the
chain grew append-only: `B4_SUPERSEDING_DESIGN_V2_AMENDMENT_9_2026_09_06.json`
names a8's exact bytes, discloses every change (terminal gains
`checkpoint_path`; seal witnesses; lock release witness; "no scored
value, contract, source row or economic column changes"), carries
the v5 population identities, and pins the 9-file surface.
`verify_amendment_chain` validates it; the record binding renamed
truthfully to `amendment_9_sha256`.

## 3. Acceptance battery (§7) — 132 passed at 7d2571c9

118 prior regressions (C1-C16 all green) + 14 new:

1. foreign-lease PRE + every re-signed variant die before compute;
2. seal fsync outcome matrix (persisted → SEALED, absent →
   UNCERTAIN, partial → UNCERTAIN, transplanted → UNCERTAIN);
3. lock acquire/release matrix (one holder, owned witnessed
   release, vanished-lock typed refusal, dead-pid never stolen);
4. two REAL forked processes after an uncertain seal boundary:
   both refuse to re-claim, both adjudicate UNCERTAIN;
5. the pre-C21 minimal terminal cannot complete (exact schema);
6. comparator omission impossible (derivation refuses when absent;
   CLI has no parameter; None derives);
7. seven factual mutations each independently fail (seed,
   scored_index, timestamps, source rows, net_return, checkpoint
   bytes, checkpoint reuse);
8. completion consumes the verifier before `CAMPAIGN_COMPLETE`
   (source-order assert + the live integrated path below);
9. global budget cannot be enlarged by any caller;
10. dry-run remains zero-write on fresh and preexisting roots;
11. all C1-C16 regressions green;
12. five guard-removal mutants (lease holder binding, completion
    integrity, ledger seal gate, source-row recompute, C22 min)
    each readmit their forgery — proving the named test bites.

Full suite at 7d2571c9: **3028 passed, 2 failed** — the same
preexisting `test_eth_sac_inner_curriculum_contract` anchor pair
that predates this order (unrelated to B4; unchanged by me).

## 4. Integrated v3 — the full corrected path, live

`docs/audits/evidence/repro_runs/b4_c17_c22_integrated_v3_2026_09_06.py`
ran all 12 cells through: `GlobalLock` → `claim_attempt` →
`issue_lease` → `verify_lease` (authorization compared, holder
bound) → `build_economic_config` on materialization v5 →
frozen-genesis scoring on the real origins → exact-schema terminal
→ `verify_single_cell_result` → intent/completion seal →
`adjudicate_cell_state == COMPLETED_VERIFIED` → **the strongest
`verify_campaign_results`** (comparator derived from the reviewed
materialization, frozen sources re-resolved, all factual fields
recomputed). 12/12 verified in 40.8 s.

Two facts you should see:

- **The final gate caught a real reuse.** The frozen genesis for
  one seed is byte-identical across origins (zero updates), and the
  C21 no-reuse check refused it. The integrated proof therefore
  scores a per-cell STAMPED copy of the genesis (one declared zip
  member naming the cell), disclosed in the runner and the summary.
  Real campaign checkpoints diverge by training and never need
  this.
- The authorization digest used by the run is an in-memory
  integrated-proof mock; no record file was created.

Samples (sanitized): `INTEGRATED_V3_{TERMINAL,CLAIM,SEAL_INTENT,SEAL_COMPLETE}_SAMPLE.json`
and `INTEGRATED_V3_SUMMARY.json` under
`docs/audits/evidence/b4_runtime_authority_20260906/`.

## 5. POST

`b4_c17_c22_post_2026_09_06.{py,out}`: every PRE probe inverted —
foreign lease refused at schema and digest layers; failed sealing
adjudicated physically (`FRESH_STATE UNCERTAIN`); non-holder
release refused with a durable witness on the honest path;
completion consumes the strongest verifier; all C21 factual tokens
present and live; executor recomputes and `min()`s.

## 6. Self-found defects (disclosed unprompted)

1. `GlobalLock.__exit__` on a vanished lock raised a raw
   `FileNotFoundError` instead of a typed refusal — found by my new
   lock-matrix test, fixed fail-closed.
2. The first orchestrator patch left the OLD `seal_attempt`
   duplicated in the file; removed and re-verified before any run.
3. My initial C13-state test over-claimed: an in-memory
   `rows_seen`/prefix forgery on an EWMA state is not detectable
   standalone (the payload carries no length trace); the honest
   defense is the snapshot-authority chain, which the tests now
   prove instead. (T0-side lesson, recorded in the T0-T1 return.)

## 7. Where this leaves B4

All six C17-C22 findings are closed with PRE→fix→POST→mutation
coverage, the amendment chain is truthful to the executing bytes,
and the complete campaign path — capability, seal, lock,
completion, factual verification, budget — ran live end to end on
the reviewed materialization. The campaign remains undispatched
and unauthorizable by me: the missing act is your record binding
`campaign_record_required_bindings()` (which now names
`amendment_9_sha256`).

`B4_CAMPAIGN_RUNTIME_READY_FOR_FINAL_MUSASHI_AUTHORIZATION_AUDIT`
