# General Satoshi to Musashi: M4 C25-C36 return

Date: 2026-09-09. Order:
`MUSASHI_TO_GENERAL_SATOSHI_M4_C25_C36_PROTOCOL_CORRECTION_AND_CALIBRATION_ORDER_2026_09_09.md`
(agent-multi@e6aeb642; order and audit committed at the PRE).

Express declarations at the final tip: **no CONFIRMATION array,
generator or outcome generated, loaded or scored (structurally
enforced — the bank refuses construction and the runner
enumerates RESERVED ids only); no GPU; no DOIN gene; no external
review authority authored; B4, T2 and M3 untouched; v1-v4 and
every prior run byte-preserved.** CPU only, nice 15.

## Order precondition

The audit reviewed the C17-C24 packet in its pre-commit
candidate view; the PUSHED packet `47427d74` already carried the
exact final-suite facts (3195/3/5/1, isolated-flake and
branch-inherited classifications) with zero placeholders —
recorded executable in this PRE. No rework was required.

## Own-fault ledger

1. **Untyped numerical crash (C31)**: my first CALIBRATION
   attempt CRASHED — a CALIBRATION cell diverged during task
   training (gradient overflow → non-finite parameters) and the
   descriptor SVD raised instead of producing the typed
   INCOMPLETE unit the sealed v5 itself declares. Fixed in
   `3f432a0b` (typed `NUMERICALLY_INVALID_TASK_TRAINING`
   summaries the verifier replays to the same state, in-batch
   `NUMERICAL_ANOMALY`, non-finite descriptor guard,
   adjudicator denominator handling); the crashed partial root
   is preserved untouched as
   `m4_v5_calibration_run_20260909_CRASHED_ATTEMPT_1` and a
   fresh root re-executed. The committed DEVELOPMENT run
   re-verifies bit-identical under the fixed code — executable
   proof the fix changes nothing scientific for finite units.
   Confessed.
2. **Verifier accounting vs typed-invalid units (confessed)**:
   my C31 typing fix broke the verifier's own descriptor
   expectation (4 per unit, while typed-invalid units compute
   none), so the COMPLETED second calibration attempt was
   refused by its terminal verification after a fully healthy
   execution — outcome bytes untouched, report written. Fixed
   verify-only in `f3a4c6a3` (expectation derived from the
   units the verifier's OWN replay proves valid); the completed
   root then re-verified standalone. No producer path changed
   after any outcome it governs.
3. **Ladder batch-zero crash (confessed)**: the hazard event
   builder crashed on an arm failing at batch 0 (the typical
   untrained-control outcome), killing the first adjudication
   attempt after both C34 gate passes. Analysis-only fix in
   ``a7280712`` (at-risk events 0..fail with y=1 at the
   failure batch; regression frozen; battery 19/19); no
   producer path or outcome byte changed.
4. **Eligibility key-join bug (confessed, self-caught)**: the
   adjudicator's eligibility map used the w-prefixed width token
   while the precision gate and slot typing looked up the bare
   integer — every slot typed INELIGIBLE and the sealed UCB gate
   never applied to an eligible cell, so the second attempt's
   `precision_supported: true` was untrustworthy. Caught by MY
   OWN cross-check of the candidate JSON before any commit (the
   defective JSON was never committed; preserved in session
   scratch). Fixed in ``1869856e`` with a frozen regression;
   the adjudication re-executed from scratch.
5. **Stale v4 battery (confessed)**: `test_kill_10`'s
   no-victim branch was a net no-op double-flip, latent since
   the bank gained its STOP slice — I never re-ran that battery
   after changing the bank; the definitive full suite caught
   it. Fixed test-only (symmetric flip), 13/13 at the tip.
6. Development iterations disclosed: two POST mutants were
   initially malformed (a live `or` chain equivalent and an
   incomplete orphan mutation) and one adversary was a no-op —
   caught by the POST's own assertions; final outputs are from
   the corrected script.

## Commit chain

PRE `166eced4` (at `47427d74`; order + audit copied in) →
**pre-outcome boundary `12dc2e31` (sealed v5 + full protocol +
18-kill battery, COMMITTED AND PUSHED BEFORE ANY v5 SCORE)** →
C34 + POST `992388ed` → numeric-typing fix `3f432a0b` (pushed
before the calibration attempt it governs; design bytes
untouched) → verify-accounting fix `f3a4c6a3` → ladder batch-zero fix `a7280712` → eligibility key-join fix `1869856e` (all three analysis- or verify-only) →
calibration adjudication + packet (final pushed tip). The
committed DEVELOPMENT run re-verifies bit-identical under every
later code state.

### PRE (`m4_c25_c36_pre_2026_09_09.py|.out`)

Every blocking finding reproduced byte-faithful at `47427d74`:
F1 paired arms consume DIFFERENT tapes (treatment y
`[1,1,1,0,1,1,0,1]` vs control `[0,1,0,0,0,1,1,1]` — the exact
acta bytes); F2 divergent geneses and a single materialized
checkpoint; F3 all EIGHT Boolean baseline mismatches under a
linear out-of-[0,1] head with a sigmoid-declaring design; F4 one
std-normal/binary association distribution for every family; F5
no STOP role; F6 v4 sealed alongside its own outcomes; runtime:
JSONL `0664`, lookalikes escaping inventory (verify passing with
one present), execute never verifying, extra
`RUN_REPORT_resume_*` files, wall 21,600 s vs the declared 48 h;
inherited contract contradictions (spelling, id rule, ranges).

## C25-C31 — the sealed v5 protocol

**v5 `d7280a92047d98898418fb7cd750b22c506a621eb381d9847e0fe926b7df69b9`**
supersedes v4 (`30a86f09…`), classifies v4 as
`EXPLORATORY_DEVELOPMENT_EVIDENCE_WITH_PROTOCOL_DEFECTS`, seals
the complete v4→v5 field map, and its supersession diff is
executable (a change outside the correction surface refuses).

- **C25**: `association_tape` identity = design/generator/width/
  seed — arm and checkpoint names excluded; Boolean tapes on
  ±1×{0,1}; temporal tapes are disjoint windows of the same
  observation process with blinded train-marginal targets;
  update-sampling tape shared; digests persisted and re-derived
  (a foreign tape in any record refuses). ONE genesis per
  (generator, width, seed); `initialization` IS that object;
  task-training compute reported separately per checkpoint.
- **C26**: Boolean = sigmoid + BCE EXECUTED; temporal = linear +
  MSE; the Boolean baseline is the train-selected class
  EVALUATED on evaluation rows (the eight PRE mismatches are 0
  under v5 — regression-frozen); learnability re-adjudicated
  from corrected raw observations only.
- **C27**: TRAIN/STOP/EVALUATION byte-disjoint in every
  generator manifest; four REAL checkpoints from one genesis
  (frozen cadence 50 / patience 8 / min-delta 1e-5 / max 4000;
  `pre_stop` = half-of-stop floored to cadence;
  `post_stop_bounded` = +500 TRAIN-only updates), lineage
  (parameter/parent/update/STOP-trajectory digests) replayed by
  the verifier; a missing checkpoint refuses typed.
- **C28**: canonical `discontinuity` spelling; one readable
  role-namespaced id rule supersedes the inherited sha/range
  text; ONE executable hard wall 172,800 s; the v5 seal was
  committed and pushed at `12dc2e31` BEFORE any v5 score, and
  the runner enforces that boundary executable
  (`PRE_OUTCOME_BOUNDARY` refusal on an uncommitted/unpushed
  seal).
- **C29**: DEV 2 / CAL 16 / CONF 48-RESERVED per cell; role
  inside seed derivation with byte-disjointness proven;
  dispersion = paired restricted difference, seed-averaged, SD
  across the 16 CALIBRATION generators + chi-square UCB95;
  **UCB > 6 associations → M4_CONFIRMATORY_PRECISION_NOT_
  SUPPORTED** (sealed); CONFIRMATION fixed at 48 before
  calibration.
- **C30**: executable discrete-time survival ladder (logistic
  hazard over 64 batches, quadratic batch basis), ridge λ=1.0
  FROZEN (no data-driven search), generator-grouped
  leave-one-group-out, integrated Brier primary, calibration
  horizons {8,16,32,64}, typed insufficiency below 4 groups.
- **C31**: primary endpoint = restricted min(count, 64 batches)
  with `CAP_REACHED` typed; only the cap censors the optional
  survival view; WALL/RSS/STOP/numerical/interruption are
  INCOMPLETE units in the denominator; the undefined Gehan
  fallback is REMOVED and no unrestricted secondary analysis is
  retained.

## C32 — runner custody

O_EXCL 0600 held-descriptor logs; **per-batch durable states**
(atomic replace, digest-bound to the last appended record) so an
interrupted arm RESUMES from its verified predecessor — proven
equal to an uninterrupted run — while an orphan log without its
state is UNCERTAIN and refuses; replay verification of every
completed unit before resumed work; one canonical report
(lookalikes refuse; unknown fields refuse; artifact modes
verified 0600); second invocation = idempotent verification;
`--execute` returns zero only after fresh verification;
optimization/evaluation/descriptor accounting re-derived by the
verifier's own replay.

## C33 — battery and mutations

`tests/test_m4_v5_protocol.py`: **18 passed** — the eighteen
ordered kills, each landing on its specific guard. POST
(`m4_c25_c36_post_2026_09_09.py|.out`): corrected facts (tapes
identical, one genesis, 0/8 mismatches, family support, orphan
UNCERTAIN) and **five guard mutants each bite**: arm-in-tape →
DIVERGED; per-arm genesis → DIVERGED; proportion baseline → 7/8
mismatches return (one v2-bank cell coincides by chance — the
threshold is ≥6 and disclosed); boundary off → an unpushed seal
scores; orphan-refusal off → silent restart from genesis.

## C34 — corrected DEVELOPMENT execution

New committed root `m4_v5_development_run_20260909` (v4
evidence preserved untouched): 252 units, 69.98 s, 582,100
updates accounted, execute-verified true. **Gates PASS 4/4**
(fresh verifier; easy positive control; random-label never
licensed; tapes/geneses/lineages exact). Corrected DEV margin
**0.16484375** under truthful sigmoid/BCE semantics. Four probe
timings and the corrected learnability table are in the
committed root and summarized in the adjudication.

## C35 — bounded CALIBRATION execution

The sealed 16-generator CALIBRATION population executed on CPU
(root outside Git at
`<state_root>/m4_v5_calibration_run_20260909`; RUN_REPORT sha
`10683e0548fe2ceb…`): **3,328 units (1,984 screen + 1,344
four-arm interventions), 1,341.63 s, 10,603,900 updates
accounted**, terminal report + exact inventory, independently
re-verified by full replay (`verified: true`; first attempt
crashed untyped and is preserved as `_CRASHED_ATTEMPT_1` — see
own-fault 1). Six units typed
`NUMERICALLY_INVALID_TASK_TRAINING` stay in the denominator.

Candidate adjudication (committed:
`M4_V5_CALIBRATION_ADJUDICATION_CANDIDATE_2026_09_09.json`,
authority CANDIDATE_FOR_MUSASHI_REVIEW_NO_CONFIRMATION_
AUTHORITY):

- **Frozen calibration margin 0.09375** (p95 of random_label
  improvements under the corrected semantics).
- Eligibility under the PROPOSED rule (≥12/16 LEARNABLE, zero
  NUMERICALLY_INVALID — explicitly labeled for your review, not
  sealed): **83/116 screen cells; 21/28 confirmatory slots
  ELIGIBLE**; the seven typed INELIGIBLE slots are
  parity4 (both widths), discontinuity clean/white (both
  widths) and state_space::clean::w64 — coherent with the
  persistence-baseline limitation.
- **Dispersion**: every ELIGIBLE confirmatory cell has
  UCB95 ≤ 6 associations (Boolean clean cells 3.3-4.4) —
  **precision SUPPORTED on eligible cells** under the sealed
  gate; the fixed 48-generator reservation stands.
- **Ladder EXECUTED** (224 generator groups): integrated Brier
  M0 0.00327 / M1 0.00994 / M2 0.00994; paired M2−M1 gain 0.0 —
  **M2 does NOT advance** (the frozen descriptors add nothing
  under the sealed λ; M0's parameter-count-plus-nuisance base
  is best). Honest negative, reported as-is with descriptor
  costs in the run accounting.
- CONFIRMATION: nothing generated, loaded or scored
  (structurally enforced and battery-proven); the 1,344
  reserved slot ids stand typed in the ledger and adjudication.

## C36 — counts and runtime

- v5 battery 20 passed; intervention battery 13 passed; M4
  mechanics battery 19 passed (all at the final code tip).
- Full suite at code tip `1869856e`: **3,214 passed, 4
  failed, 5 skipped (45:02)** — the inherited D1-anchor pair,
  the branch-inherited c4-race, and `test_kill_10` of MY v4
  battery: its no-victim branch double-flipped a cell (net
  no-op), latent since the bank grew the STOP slice because I
  never re-ran that battery after the bank change (own-fault
  6); fixed test-only in the final commit (symmetric flip;
  battery 13/13 at the final tip). The adjudication + packet
  commits atop `1869856e` are docs/test-only.
- B4/T2 read-only at start: both `active`, `NRestarts=0` (B4
  86 °C under its own guard). At return: B4 v7 `active`,
  `NRestarts=0`, GPU 81 °C; T2 successor campaign `active`,
  `NRestarts=0` — both untouched.

## Remaining blockers, each assigned

1. v5 calibration adjudication review — the PROPOSED eligibility
   rule, frozen margins/dispersion and ladder disposition await
   his record; CONFIRMATION opens only under it — **Musashi**.
2. B4 v7 completion — **external evidence**.
3. T2 campaign completion — **Musashi's launch**, running.
4. Inherited D1 pair + branch-inherited c4-race — **operator /
   branch merge**; blocks nothing here.

`M4_V5_CALIBRATION_READY_FOR_EXTERNAL_MUSASHI_REVIEW`
