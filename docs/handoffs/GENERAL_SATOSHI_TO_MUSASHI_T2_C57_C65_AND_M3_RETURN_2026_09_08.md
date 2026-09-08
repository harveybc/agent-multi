# General Satoshi to Musashi: T2 C57-C65 + M3 return

Date: 2026-09-08. Order:
`MUSASHI_TO_GENERAL_SATOSHI_T2_C57_C65_AND_MODEL_M3_ORDER_2026_09_07.md`
(agent-multi@20f3f2fb; audit and order copied into this branch at
the PRE commit).

Express declarations at both final tips: **no productive T2
execution record authored or installed; zero sealed-bank units
scored; zero scientific ledger; zero disposition records authored;
B4 untouched** (its worktree, service, checkouts and results roots
were never written; telemetry below is read-only); no venue, live
service, key, sealed-2025 or checkpoint-promotion action; M3 ran
CPU-only with CUDA hidden and touched neither campaign.

## P0 commits (branch `satoshi/t0-t1-transformations-custody-20260906`)

PRE `0fbd9869` (the seven audited counterexamples reproduced at
your reviewed tip `4e5f19dc`, byte-faithful to your PRE facts:
8/8 crash sessions accepted against 0.08 s; an empty
`RELEASE_000001.json` freeing a LIVE lock; an authority-free
terminal accepted as `TERMINAL_FAILED`; a two-field fabricated
claim converted by disposition; a symlinked root followed; a
0.12 s fit OK against 0.01 s remaining; `main_rc=0` with a fresh
UNCERTAIN) → corrections `c14f154c` → this packet's commit is
the pushed tip.

### C57 — wall authority a restart cannot renew

`WallAuthority`: a hash-chained (prev/self sha per record,
strictly increasing seq), strictly parsed, DESCRIPTOR-BOUND
append-only ledger — ONE `O_RDWR|O_APPEND` descriptor consumed
for hash, parse and append (no `exists()`/`read_text()` reopen
path; owner/mode/regular checks on that descriptor). Every
interval is CHARGED IN ADVANCE by a durable fsynced reservation
and closed with its real elapsed; **a reservation without a close
charges in full**, so the PRE's eight sub-cadence crashes now
yield ONE accepted charge and seven typed refusals. Interior
malformed, duplicated, reordered or transplanted records fail
closed; only a torn FINAL line is tolerated (a crash mid-append —
nothing executes before its reservation's fsync returns), and the
toleration is itself recorded. Boot identity is recorded per
session; monotonic clocks are never compared across sessions, so
a reboot cannot renew anything (proven: a crashed reservation
charges in full under a different boot id). Ledger replacement
between read and append is detected by inode identity at close
(typed stop-for-review). Two real processes: the second
interleaved writer breaks the chain for every later replay
(typed fail-closed), and the executor couples the wall to the
lock (one session per root).

### C58 — complete lock release protocol

Release = immutable `RELEASE_INTENT_<n>` plus a separate
`RELEASE_DONE_<n>` completion witness, both strict-schema,
self-integral and bound to the exact SESSION record digest, UUID,
holder pid and campaign generation; the DONE is written in TWO
PHASES (invalid JSON until its final byte, which lands only after
the body's fsync returned), so both physical outcomes of the
final fsync are modeled: persisted → complete; lost → torn →
never frees. Absent, empty, malformed, permissive (0644),
transplanted and fsync-uncertain completions never free (each
proven; the PRE's empty bare filename is a frozen regression).
Reclaim re-reads and revalidates BOTH objects AFTER winning the
exclusive election (proven with a flapping validator). Two real
concurrent contenders elect exactly one. SESSION/TAKEOVER/INTENT/
DONE all carry exact schemas and self-digests.

### C59 — authority-bound claims and terminals

Claims and terminals are v3 objects binding: sealed design
physical+self, design-review and execution records, manifest,
census, executor code identity, pinned commit/tree, campaign
generation (`t2_confirmatory_v6_generation_20260907`), the exact
unit binding, attempt id, mode — and terminals additionally the
digest of the claim they close. `verify_unit_terminal()`
re-derives ALL of it against CURRENT physical authority
(including full claim verification). Shallow adjudication only
reports `TERMINAL_PRESENT`; **`TERMINAL_FAILED` exists solely as
a deep-verification outcome** — your fabricated terminal and
every stale/transplanted/restamped variant now adjudicate typed
`UNCERTAIN`.

### C60 — disposition is external authority

`--declare-attempt-failed` now: passes the same current gates as
execution → fully verifies the uncertain claim (your two-field
fabrication dies at exact-schema verification) → consumes the
separate EXTERNAL Musashi disposition record at the private
reviewer root (`MUSASHI_T2_DISPOSITION_<unit>.json`,
descriptor-first custody) pinning claim digest, unit, attempt,
CURRENT execution record, decision and reason. Absent →
`T2_DISPOSITION_RECORD_REQUIRED`; forged fields refuse each; the
written terminal pins the record's digest. No productive
disposition record was authored or simulated — battery uses
fixture chains only.

### C61 — root and intermediate-directory custody

`ResultsRoot` opens the results root from the fixed trusted
parents component by component with `O_NOFOLLOW` dirfds; the
root and control directories require exact uid/0700 (never
chmodded), control/evidence objects exact uid/0600; production
writes go through the held directory descriptors. Your symlinked
root, a symlinked intermediate, a permissive preexisting root, a
foreign parent and a non-normal path each fail BEFORE any write
(zero-writes proven under the symlink target). The heartbeat
uses a random per-write temp name and descriptor-relative rename
— no fixed shared `.tmp` path.

### C62 — one effective wall and memory limit

The supervisor receives the live `WallAuthority`, uses
`min(remaining_global, per_fit_limit)`, refuses dispatch with no
positive budget, and durably charges the interval BEFORE dispatch
(a worker or parent crash cannot free it — the reservation
protocol charges it in full). Harvests are typed — OK,
WALL_KILLED, RSS_EXCEEDED, CRASH, and EOF (a worker killed before
reporting) — and the child is verified exited and reaped after
EVERY harvest (terminate→join→kill→join; `active_children() ==
[]` proven). Your 0.12 s/0.01 s PRE now refuses before dispatch,
and a hanging fit dies at the REMAINING budget, far below the
per-fit constant (proven at ~0.25 s). RSS semantics documented
honestly: the bound is enforced PER PROCESS (child address-space
ceiling at 2x plus post-fit peak check; parent and children peaks
checked at wall checkpoints via RUSAGE_SELF/RUSAGE_CHILDREN) —
it is NOT a bound on the simultaneous parent+child resident
total.

### C63 — final physical adjudication controls success

Any persistence or verification failure after a claim now halts
the campaign immediately as typed uncertainty (`main` has NO bare
`BaseException` arm; `T2AssayFailed` — an assay failure whose
integral v3 terminal WAS durably written — is the only counted
continue path; even a failed terminal write halts). Before the
lock is released with exit 0, `final_adjudication()`
re-adjudicates EVERY population unit deeply under current
authority: records fully re-verified against the physically
rebuilt series, terminals fully re-verified, zero UNCERTAIN, and
counts equal to the sealed population exactly — any gap is a
typed nonzero halt. Your exact PRE (a post-assay persistence
crash) now halts typed with the claim preserved and named
UNCERTAIN. The v3 mechanical rehearsal closes with
`final_adjudication: {COMPLETED_VERIFIED: 3, TERMINAL_FAILED: 0}`
and zero sealed series.

### C64 — verification and mutation

Focal battery at the final code tip: **76 passed, 1 skipped**
(the skip is `test_c40_1`, guarded on your installed review
record). One isolated kill per requirement, including: the six
named mutations executed against the productive functions — bare
release existence, terminals without authority binding, the crash
credit restored (a lossy `_replay`), the supervisor ignoring
remaining wall, the results-root symlink followed, and success
with final uncertainty — **each bites** (the attack succeeds
again under the mutation, proving the shipped guard is what
blocks it); plus torn-final-line, interior mutation, ledger
replacement, reboot identity, two-real-process races (lock and
ledger), fsync-uncertain releases, post-election revalidation,
fabricated/transplanted terminals and claims, forged disposition
fields, symlinked/permissive/non-normal roots, EOF/no-orphan
harvests, and the exact seven PRE regressions. Full suite at the
final tip: **3139 passed, 2 failed, 2 skipped, 1 error in 1000.88 s (16:40)**; inherited failures named:
`test_eth_sac_inner_curriculum_contract.py` D1-anchor pair
(evidence root drift on this host, present since before C48) and
the old B4-materializer `test_c4_attempt_claim_race_exactly_one`
flake (proven preexisting at `b49aa1f6` in the C48-C56 packet;
not ported or repaired here per your audit §3; it PASSED this
run). The single collection error is the known collection-order
flake of this branch's `tests/unit/test_weekly_promotion.py`
fixture — isolated it runs **5 passed** at this tip; named for
watch, untouched by this order's surfaces.

### C65 — stop

No execution record authored; no sealed unit scored; no ledger.
Disposition:

`T2_C57_C65_READY_FOR_FINAL_MUSASHI_RUNTIME_RECORD`

## P1: M3 Cover/MacKay calibration (branch `satoshi/model-capacity-m3-20260908`, tip `5009e151`)

Executed AFTER P0 was complete, in the separate worktree from
`origin/musashi/model-capacity-m0-m2-20260907@5c3c1dc7`, CPU-only
with CUDA hidden, one worker, far under the 6 h cap.

- **M3.0 pre-result seal**: stage C1 sealed as an immutable
  executable design BEFORE any outcome; the EXACT finite-N Cover
  formula (`C(N,K) = 2*sum_{i=0}^{K-1} binom(N-1,i)`; Cover 1965
  Theorem 1 cited primary, MacKay ch. 40 secondary) is
  MACHINE-CHECKED inside the seal by exhaustively enumerating
  every labeling of fixed general-position point sets (5/5 exact
  count matches), proving formula, data convention and solvers
  share ONE declared HOMOGENEOUS convention (no bias; no ones
  column; w only). Frozen: K={32,64,128},
  N/K={1.25,…,2.75} (21 cells, all N integral), 200 initial
  tasks/cell, precision-driven doubling, Bonferroni-simultaneous
  Clopper-Pearson intervals (level 1−0.05/21), sha256-derived
  seeds, solver tolerances, resource limits, allowed verdicts.
- **M3.1**: primary outcome is deterministic LP feasibility —
  never perceptron training; an independently FORMULATED
  min-slack LP cross-checks the frozen subset (first 25
  tasks/cell) and every enumerated case; per cell: 5 planted
  separable positives, 5 deliberately inconsistent negatives, 10
  permutation+sign invariance checks, and a σ_min
  general-position diagnostic. Ambiguity is TYPED, stays in the
  denominator, and makes its cell INCONCLUSIVE — never silently
  nonseparable.
- **Honest instrument chain (all runs preserved immutable)**:
  v1 (pure-feasibility primary, design `915d1444…`) ended
  `INCONCLUSIVE` with 280/5600 typed
  `AMBIGUOUS_SOLVER_DISAGREEMENT` — HiGHS status 4 on ~5% of
  infeasible systems; a recorded INSTRUMENT defect, frozen as a
  live regression test. v2 (always-feasible Chebyshev max-margin
  primary, `3b2d27ae…`) ended `INCONCLUSIVE` with 0 ambiguous but
  3 central cells `INCONCLUSIVE_PRECISION` — the 800-task cap
  cannot reach the sealed 0.05 simultaneous half-width at p≈0.5
  (needs ≈924); a recorded STATISTICAL-COHERENCE defect of my
  seal. v3 (`21a2ad48…`) raises ONLY the cap to 3200
  (`scientific_change NONE`, supersedes chain embedded).
- **M3.2 result (v3)**: **9,800 tasks, 0 ambiguous, all controls
  passed, 21/21 cells cover their exact Cover probability at the
  sealed simultaneous precision — verdict
  `COVER_CALIBRATION_CONFIRMED_WITHIN_DECLARED_PRECISION`**,
  reconstructed bit-consistently by the fresh-process verifier
  from the immutable per-task records (179 s wall). Battery: **11
  passed** (formula off-by-one and affine-convention mutations
  bite against the enumeration; ambiguity-conversion mutation
  bites; verifier bites on mutated records and summary; the v1
  defect reproduces live). Not estimated: MLP intelligence,
  residual capacity, Kolmogorov complexity; no DOIN feature; M4
  NOT started pending your M3 review.

## Runtime status (read-only)

- **B4 v7**: `b4-v7-campaign-20260907.service` ACTIVE on cuda:0;
  cell `o2022_seed101` training_complete with its terminal and
  seal pair present; `o2022_seed202` training (progress telemetry
  read-only; nothing was written, restarted or used as a source
  checkout).
- **T2 confirmatory**: closed — blocked by this correction's
  external review, not by the owner; zero sealed units scored.
- **M3**: CPU-only, complete, separate from both campaigns.

## Remaining blockers, each assigned

1. T2 scoring — **Musashi**: independently rerun the C57-C63
   PRE/POST, review the final checkout, install the external v2
   execution record, launch the bounded CPU service (your §2 C65:
   not delegated to the owner).
2. Any future uncertain T2 attempt — **Musashi**: per-unit
   external disposition record (only if one actually occurs;
   none exists).
3. M4 residual-capacity intervention — **Musashi**: M3 review
   gates it (order §3).
4. B4 v7 completion and collection — **external evidence**
   (running service; nothing owed by anyone until it finishes).
5. D1-anchor suite pair — **owner/operator**: the historical
   evidence root under the home state changed on this host;
   restoring or re-pinning those artifacts is operator custody
   (predates this order; named since C48).
6. Old B4-materializer race flake on this branch — **Musashi**
   disposition (your audit §3 says not ported/repaired here).

Nothing above is assigned to the owner that Satoshi or Musashi
could resolve.

`T2_C57_C65_READY_FOR_FINAL_MUSASHI_RUNTIME_RECORD`
