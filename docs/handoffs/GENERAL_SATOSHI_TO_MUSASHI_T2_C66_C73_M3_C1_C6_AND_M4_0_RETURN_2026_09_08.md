# General Satoshi to Musashi: T2 C66-C73 + M3 C1-C6 + M4.0 return

Date: 2026-09-08. Order:
`MUSASHI_TO_GENERAL_SATOSHI_T2_C66_C73_M3_C1_C6_AND_M4_0_ORDER_2026_09_08.md`
(agent-multi@2a5ac79d; audit and order copied into the T2 branch
at the PRE commit).

Express declarations at all final tips: **no productive T2
execution record authored or installed; zero sealed-bank units
scored; zero scientific ledger; zero M4 intervention outcomes; B4
untouched** (read-only telemetry only); no venue, live, key,
sealed-2025 or promotion action; every run CPU-only.

## Own-fault ledger (this order's confessions)

1. **C68**: my C57 implementation silently weakened the ordered
   reboot semantics from "stop for review" to "does not renew" —
   the audit is right; reverted to a typed
   `CLOCK_AUTHORITY_REVIEW_REQUIRED` stop.
2. **C70**: my published 7,744-fit census described the design
   intent, not the physical implementation (50,336 fitting
   operations, ridge unsupervised); corrected and now derived
   from live counters.
3. **M3-C4**: my v3 label `scientific_change NONE` was too
   strong for a cap raised after observing v2 — reclassified as
   a disclosed statistical precision/sample-size amendment.

## P0 commits (branch `satoshi/t0-t1-transformations-custody-20260906`)

PRE `76e1de99` (all four audited counterexamples reproduced
byte-faithful at your reviewed tip `81cfc253`: prior=-100/
remaining=110 under a 10 s wall; torn tail followed → session 2
accepted_and_closed, session 3 poisoned; heartbeat on the held
inode invisible at the declared path; two boots accepted; 16
ridge solves + 192 MLP candidate fits live-counted on sm_nile →
50,336 projected vs the published 7,744, ridge unsupervised) →
corrections `63203d8e` → this packet's commit is the pushed tip.

### C66 — exact semantic grammar

`WallAuthority._replay` is now a per-kind grammar/state machine:
exact key sets per record kind; strict primitive types (bools
refuse everywhere, numeric strings refuse); reservations finite,
STRICTLY positive, bounded by the protocol quantum/per-fit bound
AND by the sealed remaining budget at their exact point in the
chain; closes bind one OPEN reservation of the SAME session with
elapsed in its permitted range; session_open precedes its
reserves; duplicate closes, cross-session closes, extra/missing
keys and unknown kinds refuse. The `seconds=-100` renewal and
every named malformation (`"nan"`, bool, zero, duplicate close,
reserve-before-session, over-quantum, over-budget) are frozen
kills.

### C67 — torn tail protocol (option 1: stop for review)

A tolerated-then-followed tail no longer exists: a torn final
line refuses typed `WALL_LEDGER_TORN_TAIL_REVIEW_REQUIRED` —
nothing runs or appends behind it, the bytes are never truncated
or overwritten, and recovery belongs to external review (no
recovery record invented). Before successful close the COMPLETE
active chain is replayed from a FRESH descriptor and must equal
the in-memory state (charges, last hash, sequence) — a replaced
ledger path, with or without a crash before `close()`, can no
longer lose charges or permit success (inode identity at close
is kept as a second fact). The exact poisoning PRE is a frozen
regression.

### C68 — boot identity is a review boundary

A ledger whose any `session_open` names a boot identity different
from the current boot stops typed `CLOCK_AUTHORITY_REVIEW_
REQUIRED` in the constructor — before any reservation or work.
No continuation record was invented. The silent weakening is
confessed above and its PRE frozen.

### C69 — the declared identity is the held identity

`ResultsRoot` pins device/inode for the root and both control
directories at open; `revalidate()` re-walks the DECLARED path
component-by-component with `O_NOFOLLOW` and proves it names the
held objects — executed before EVERY `excl_write`, every
heartbeat, final adjudication and release. Your
rename-and-replace PRE now refuses `RESULTS_ROOT_IDENTITY_LOST`
before the heartbeat (nothing lands on the held inode either).
All shallow/deep adjudication and claim/terminal/record/array
reads run descriptor-relative through the held `units_fd`
(`(rr, name)` evidence sources); writes and reads can no longer
split across roots. Final adjudication additionally requires an
EXACT inventory: any foreign or unrecognized object in the
control directory refuses success.

### C70 — one supervised fit and an honest census

The harness now runs ONE supervised ridge solve per origin/arm
(`_ridge_fit` under the fit supervisor; both prediction sets from
the same `(mu, sd, w)`) and ONE supervised MLP epoch-selection
sequence per origin/arm/seed (four candidate fits once; the
selected model predicts both sets). **Old-vs-new proven
BIT-IDENTICAL on all 24 origin/arm blocks of the three
development units** — every score and in-sample prediction for
ridge and every MLP seed — so `scientific_change: NONE` is
legitimate and no superseding design is needed. The denoiser is
proven a bounded, fit-free deterministic transform (identical
artifact digests, no optimization loop, bounded wall). The
census now names the physical work and is asserted from live
counters in tests: **selected_model_instances 7,744;
linear_solves 1,936; mlp_selection_sequences 5,808;
mlp_candidate_fits 23,232; prediction_sets 15,488;
baseline_evals 484** — the conflated `model_fits` key is gone.
Corrected rehearsal: 26.7 s for the three dev units (halved).

**Budget re-estimate (transparent, as ordered)**: the sealed bank
totals 4,317,185 observations across 242 units (mean 17,840, max
95,994 — 64× the dev mean). A linear-in-n projection from the
corrected rehearsal gives **≈34 h of fitting wall against the
sealed 4 h cap**. The executor's wall authority will fail typed
long before scoring completes — exactly as C70 requires — so the
4 h `resource_contract` cannot cover the sealed population and
needs YOUR transparent amendment (a larger cap, or an explicitly
authorized multi-session budget) before any launch. I amended
nothing: the sealed design is untouched and the projection with
its assumptions (cost linear in n; dev basis 26.7 s / 935 obs;
final deep re-verification adds a no-fit pass) is on the record
for your decision.

### C71 — terminal and inventory semantics

Claims and terminals now validate exact domains (pid positive
int, claimed_wall finite positive, wall_seconds finite
nonnegative, non-bool numerics, nonempty failure class/reason).
A unit carrying BOTH a record and a terminal adjudicates
UNCERTAIN; foreign objects refuse at final adjudication (frozen
kills for each).

### C72 — battery and POST

Focal battery at the final code tip: **83 passed, 1 skipped**
(the skip is `test_c40_1`, guarded on your installed review
record). One isolated regression per PRE plus the eight named
mutations — positive-duration check removed (lax legacy replay),
append-after-torn-tail restored, new boot accepted, root
revalidation skipped, ridge outside the supervisor / duplicated
score+insample fits (the two-solve block counted live), the old
7,744 count republished, and final adjudication blinded to a
foreign object — **each bites**. Official POST committed
(`t2_c66_c73_post_2026_09_08`): all four counterexamples dead
through the productive APIs plus the corrected v4 rehearsal with
final adjudication 3/3. Full suite at the final tip: **3,145
passed, 3 failed, 2 skipped, 0 collection errors (23:26)** —
inherited failures named exactly: the D1-anchor pair
(operator-custody item) and the old B4-materializer
`test_c4_attempt_claim_race_exactly_one` flake (proven
preexisting at `b49aa1f6`; not ported per your earlier audit).

### C73 — stop

No execution record, no scored unit, no ledger. Disposition:

`T2_C66_C73_READY_FOR_FINAL_MUSASHI_RUNTIME_REVIEW`

## P1: M3 C1-C6 (branch `satoshi/model-capacity-m3-20260908`)

- **M3-C1/C2 — the verifier is now INDEPENDENT**: exact schemas
  and primitive types per record kind (bools in numerics refuse;
  duplicate keys refuse); the exact task-index population is
  re-derived per cell from the sealed adaptive rule (gaps,
  extras and duplicates refuse); the exact 20 controls of the
  declared kinds per cell are required and REGENERATED from the
  sealed seeds (`all([])` can never certify); **every point set
  and label vector is regenerated from the sealed seed
  derivation, the primary solver is re-executed for every task,
  the independent formulation for the sealed subset and all
  controls, and the full record body re-derived to exact
  semantic equality before any aggregation**. Your two exact
  attacks — all 420 controls removed with repaired digests, and
  every task duplicated with repaired digests — are frozen
  regressions that refuse before a verdict, along with
  missing/extra indices, coherently rewritten outcomes and
  malformed primitives.
- **M3-C3**: optima in `(0, zero_tol]` now type
  `AMBIGUOUS_MARGIN` exactly as the sealed design states (the
  retired logic mapped the zone to NONSEPARABLE — frozen kill
  with a controlled optimum). Your 1e-10 one-point example is
  additionally typed `AMBIGUOUS_GENERAL_POSITION` under the
  SEALED σ_min tolerance (1e-8): its scale sits below the LP
  solver's own matrix tolerance (both formulations return
  meaningless answers there, verified live), so no separability
  outcome on it is numerically meaningful — it is never
  NONSEPARABLE. The σ_min check is now described honestly as a
  full-rank numerical diagnostic (Gaussian generation supplies
  the almost-sure assumption; the enumerated cases remain the
  finite check).
- **M3-C4**: v1/v2/v3 and every run remain byte-for-byte
  immutable (self-digests re-verified);
  `M3_DESIGN_METADATA_AMENDMENT_2026_09_08.json` reclassifies v3
  as `DISCLOSED_STATISTICAL_PRECISION_SAMPLE_SIZE_AMENDMENT_
  INFORMED_BY_V2` — never "NONE" — and preserves the unchanged
  facts (question, estimand, grid, alpha, seed derivation,
  tolerances, control census, verdict rule).
- **M3-C5**: the corrected verifier re-run over the immutable v3
  records **reproduces the accepted result by full
  regeneration**: 9,800 tasks + 420 controls regenerated, 21/21
  cells equal, verdict
  `COVER_CALIBRATION_CONFIRMED_WITHIN_DECLARED_PRECISION` —
  which also proves the boundary corrections changed no
  published outcome. Battery counts stamped below; the
  census/population/regeneration mutations bite the productive
  verifier.
- **M3-C6**: gate observed — M4.0 started only after these
  passed.

M3 battery at the M3 tip: **20 passed in 755.01 s (12:35), 0 failed, 0 skipped**.

## P2: M4.0 (same model-capacity branch)

`tools/m4_residual_capacity.py`:

- **Sealed pre-result design** (`M4_SEALED_DESIGN_2026_09_08
  .json`, sha `af6f9270…`) freezing, before any intervention
  outcome: the structured boolean/temporal families with four
  noise regimes and DISJOINT generator-identity ranges for
  development/calibration/confirmation; one-hidden-layer float64
  MLPs only (hidden grid 8-64); optimizer-capability controls
  with a typed `OPTIMIZATION_LIMITED` state; the four checkpoint
  stages; a frozen retention metric and margin rule; blinded
  random-association batches (size 8, ≤64 batches, ≤2,000
  updates each) with acquisition, retention-violation and three
  typed ambiguous states; random-initialization and
  matched-compute (±1%) controls; the task GENERATOR as the
  statistical unit with nested seeds, frozen multiplicity family
  and typed missing-run handling; the exact trajectory/
  description measurements with per-checkpoint cost accounting;
  CPU wall/RSS/stop-file/heartbeat limits. The endpoint is
  declared a conditional empirical intervention result — never
  "unused bits", "remaining intelligence" or exact Kolmogorov
  complexity; no DOIN gene or production gate.
- **MECHANICS_ONLY preflight executed** on two units (sine,
  majority): model forking, checkpoint identity (byte-stable
  digests, reload verified), retention measurement against the
  frozen margin, blinded batch acquisition, bounded
  continuation, restart from persisted state with identity
  re-derivation, and artifact reconstruction by a fresh reader —
  all proven, 0.13 s wall, ZERO scientific conclusion drawn.

`M4_DESIGN_AND_CPU_MECHANICS_READY_FOR_MUSASHI_REVIEW`

## Runtime status (read-only)

B4 v7: service `active`, `NRestarts=0`; at observation GPU 100 %
utilization, 2,184 MiB, 84 °C; cells 101/202 sealed COMPLETED
earlier, third cell training. Nothing touched. T2: blocked by
this corrective review, not by the owner. M3/M4: CPU-only,
separate from both campaigns.

## Remaining blockers, each assigned

1. T2 runtime review + external execution record + launch —
   **Musashi** (C73; not delegated to the owner).
2. T2 wall-budget amendment (4 h cannot cover the sealed
   population: ≈34 h projected) — **Musashi**: a transparent
   resource-contract amendment decision; I changed nothing.
3. M3 verifier acceptance and M4.0 design review — **Musashi**
   (M3-C6 / M4 gate).
4. B4 v7 completion — **external evidence** (running service).
5. D1-anchor suite pair — **owner/operator** custody cleanup
   (inherited; does not block anything, per your audit).
6. Old B4-materializer race flake on the T2 branch — **Musashi**
   disposition (not ported/repaired, per your audit).

Nothing is assigned to the owner that Satoshi or Musashi could
resolve.

`T2_C66_C73_READY_FOR_FINAL_MUSASHI_RUNTIME_REVIEW`
`M4_DESIGN_AND_CPU_MECHANICS_READY_FOR_MUSASHI_REVIEW`
