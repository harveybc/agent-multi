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

## C31A-C31F — the numeric incident correction (order
@f4bf7d39, executed inside this return)

**Chronology confession (C31A)**: the incident order was
authored while attempt 2 ran; in my timeline attempt 2 had
finished and I had already run AND PUSHED the C35 adjudication
derived from it before receiving the order. That artifact is
RETIRED — renamed
`M4_V5_CALIBRATION_ADJUDICATION_ATTEMPT2_NON_GOVERNING_2026_09_09.json`
(history preserved) — and attempt 3's adjudication below is the
only governing one.

Attempt identities and dispositions (all preserved byte-for-byte
under `<state_root>/`):

1. `m4_v5_calibration_run_20260909_CRASHED_ATTEMPT_1` —
   `CRASHED_UNTYPED_NON_GOVERNING` (exit: SVD LinAlgError after
   untyped training divergence; no report).
2. `m4_v5_calibration_run_20260909` —
   `VERIFIER_RESULT_RECORDED_NON_GOVERNING_NUMERIC_GUARD_GAP`
   (exit: execute completed all 3,328 units and its report, the
   terminal verifier refused on my descriptor-accounting bug,
   then the root verified standalone under `f3a4c6a3`; the
   productive overflow warning fired from the descriptor cast).
3. `m4_v5_calibration_run_attempt3_20260909` — the governing
   attempt (below), executed AFTER the pushed amendment.

**Incident PRE (`m4_c31a_c31f_pre_2026_09_09.py|.out`, commit
`67eb0d4f`)**: eight routes frozen OPEN on the pre-amendment
code (f64-out-of-f32-range passes the guard with the exact
attempt-2 overflow warning; SVD failure escapes untyped;
nonfinite singular values yield a valid-looking rank; anomalous
arms enter dispersion; 1-2 seeds average as complete; no
attrition gate; invalid descriptors reach M2 rows; no
planned/complete/incomplete accounting); routes 1/5/6/7 verified
already corrected at `3f432a0b`.

**C31B/C31C (commit `71695521`, PUSHED BEFORE attempt 3)**:
descriptor validity at the declared float32 serialization
boundary (typed `NUMERICALLY_INVALID_DESCRIPTOR`; infinity
bytes never compressed; SVD failure and nonfinite singular
values the same typed invalidity, never zero-rank; the fresh
verifier re-derives the same validity/values from the original
float64 parameters arm-by-arm; the valid-case record layout is
unchanged so the committed DEVELOPMENT root re-verifies
BIT-IDENTICAL — proven). Complete primary pair = both primary
arms in admitted endpoint states with valid descriptors;
generator means require EXACTLY the three sealed seeds
(`INCOMPLETE_PAIRED_GENERATOR` otherwise, kept in the
denominator); dispersion publishes planned/complete/incomplete
per cell and estimates only at ≥ max(3, ceil(16×0.8)) = 13
complete under the sealed 20 % attrition, else
`CALIBRATION_INCOMPLETE` never supported; the ladder consumes
only complete quartets with valid descriptors and reports every
exclusion with reasons.

**C31D**: 12-route incident battery on productive code
(`tests/test_m4_numeric_incident.py`, 12 passed; 32 with the v5
battery). **Incident POST**
(`m4_c31a_c31f_post_2026_09_09.py|.out`): all eight open routes
typed on the corrected code, and the four new guards each BITE
alone (f32-range off → overflow bytes compress; SVD-typing off
→ untyped escape; complete-pair off → anomalous arm pairs;
exact-three off → partial seeds average).

**C31E**: sealed append-only amendment
`M4_V5_NUMERIC_VALIDITY_AMENDMENT_1_2026_09_09.json`
(`43e0804e…`, amends v5 `d7280a92…`): trigger = the runtime
warning + code-path inspection, NO scientific effect; exactly
the four changed fields mapped; everything else byte-untouched;
committed and pushed at `71695521` BEFORE attempt 3.

## C35 — bounded CALIBRATION execution (GOVERNING: attempt 3)

Attempt 3 executed once in a fresh root
(`<state_root>/m4_v5_calibration_run_attempt3_20260909`) AFTER
the pushed amendment `71695521`: **3,328 units, 1,011.75 s,
10,603,900 updates, terminal fresh verification `verified:
true`** (5,352 descriptor evaluations = 4 × 1,338 valid units).
The update count equals attempt 2 exactly — executable proof of
inter-attempt determinism. Six units typed
`NUMERICALLY_INVALID_TASK_TRAINING` stay in the denominator.

Governing adjudication (committed:
`M4_V5_CALIBRATION_ADJUDICATION_ATTEMPT3_GOVERNING_2026_09_09.json`,
authority CANDIDATE_FOR_MUSASHI_REVIEW_NO_CONFIRMATION_
AUTHORITY):

- **Frozen calibration margin 0.09375**; eligibility under the
  PROPOSED rule (labeled for your review): 83/116 screen cells;
  **21/28 confirmatory slots ELIGIBLE**, the seven INELIGIBLE
  typed (parity4 both widths, discontinuity clean/white both
  widths, state_space::clean::w64).
- **Complete-pair / three-seed / attrition accounting (C31C,
  live on real data)**: every cell publishes
  planned/complete/incomplete — complete generators 14-16 of 16
  per cell, **2 generators typed INCOMPLETE_PAIRED_GENERATOR**
  (kept in denominators), **zero CALIBRATION_INCOMPLETE cells**
  (all ≥ the 13-complete attrition floor).
- **Dispersion**: every ELIGIBLE cell has UCB95 ≤ 6 —
  **precision SUPPORTED on eligible cells**; the fixed
  48-generator reservation stands.
- **Ladder EXECUTED** (224 groups; **1 quartet excluded with
  its reason published** — the C31C filter biting real data):
  integrated Brier M0 0.00327 / M1 0.00994 / M2 0.42977; paired
  M2−M1 gain **−0.41983** — **M2 does NOT advance** (the frozen
  descriptors actively hurt under the sealed λ; M0 remains
  best). Honest negative, descriptor costs in the accounting.
- CONFIRMATION: nothing generated, loaded or scored; 1,344
  reserved slot ids stand.

## C36 — counts and runtime

- v5 battery 20 passed; incident battery 12 passed;
  intervention battery 13 passed; M4
  mechanics battery 19 passed (all at the final code tip).
- Full suite at final code tip `71695521`: **3,228 passed, 2
  failed, 5 skipped (30:04)** — ONLY the inherited D1-anchor
  pair (the branch-inherited c4-race and the weekly flake both
  passed this run; the kill_10 double-flip found at `1869856e`
  was fixed test-only and is green). The adjudication + packet
  commits atop `71695521` are docs-only.
- B4/T2 read-only at start: both `active`, `NRestarts=0` (B4
  86 °C under its own guard). At return: **BOTH campaigns
  COMPLETED — B4 v7 `inactive/Result=success` (exit 0) and the
  T2 successor campaign `inactive/Result=success` (exit 0),
  zero restarts each, untouched by me throughout**; their
  results await their own adjudications (T2 is Musashi's; B4's
  is its runtime's).

## Remaining blockers, each assigned

1. v5 calibration adjudication review — the PROPOSED eligibility
   rule, frozen margins/dispersion and ladder disposition await
   his record; CONFIRMATION opens only under it — **Musashi**.
2. B4 v7 completion — **external evidence**.
3. T2 campaign completion — **Musashi's launch**, running.
4. Inherited D1 pair + branch-inherited c4-race — **operator /
   branch merge**; blocks nothing here.

`M4_V5_CALIBRATION_READY_FOR_EXTERNAL_MUSASHI_REVIEW`
