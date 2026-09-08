# Musashi to General Satoshi: T2 C66-C73, M3 C1-C6 and M4.0

Date: 2026-09-08

Authority: the owner requested that executable lanes remain moving. This order
authorizes bounded CPU work only. B4 remains under its existing service and
must not be modified, restarted or used as a source checkout. This order grants
no venue, live, key, sealed-2025, checkpoint-promotion or additional GPU
authority.

Read first:

- `docs/audits/MUSASHI_AUDIT_T2_C57_C65_AND_M3_2026_09_08.md`;
- T2 candidate `81cfc25300195cb61bd36377e27184524ca6ab35`;
- M3 candidate `5009e15116ad4b74ec5ae0f5a64e4bdcff4f67ad`; and
- `docs/research/model_capacity/M3_M6_CONFIRMATORY_DESIGN_DRAFT_2026_09_07.md`.

Priorities are P0 T2, P1 M3 verifier correction, then P2 M4.0 design and
mechanics. Keep the T2 and model-capacity worktrees separate.

## 1. P0: T2 C66-C73

### C66: exact semantic grammar for wall authority

Freeze the negative-reservation PRE from the audit. Every wall-ledger record
kind must have an exact schema, exact primitive types and finite domains before
it affects state:

- sequence and reservation references are positive integers, never bools;
- durations are finite real numbers, never bools or numeric strings;
- every reservation is strictly positive and bounded by the sealed remaining
  budget and protocol quantum;
- every close names one open reservation from the same session and has a finite
  elapsed value in its permitted range;
- session, boot, pid and generation fields are canonical and exact; and
- extra, missing, reordered-state or semantically impossible records refuse.

Implement a grammar/state machine, not a collection of `float()`/`int()`
coercions. Acceptance: a self-consistent `seconds=-100`, `"nan"`, bool, zero,
duplicate close, reserve-before-session and cross-session close all refuse.

### C67: torn tail and replacement recovery

A tolerated torn final line must not be followed by records in the same file.
Choose one fail-closed protocol:

1. stop for external review; or
2. create an immutable successor segment that binds the complete predecessor,
   the exact torn bytes and an external recovery record.

Never truncate, overwrite or append behind the torn fragment. Before successful
exit, replay the complete active ledger chain from a fresh descriptor and
require equality with the in-memory state. Replacing the ledger path at any
point, including followed by a crash before `close()`, must not lose charges or
permit success.

Freeze the exact PRE: session 2 currently runs and closes, while session 3 can
no longer replay the ledger.

### C68: boot identity is a review boundary

Restore the ordered semantics. If a prior ledger segment names a different boot
identity, stop with a typed `CLOCK_AUTHORITY_REVIEW_REQUIRED` before any new
reservation or work. If continued operation after reboot is desired later, it
requires a separate external record; do not invent that record in this order.

### C69: results-root identity remains the declared identity

Pin device/inode facts for the results root and every control directory. Before
every side effect and before final adjudication, re-open the declared path
component-by-component and prove it names the held objects. A rename-and-replace
must refuse before heartbeat, claim, arrays, record, terminal, release or
successful exit.

Move shallow/deep unit adjudication and all claim/terminal/array reads to
descriptor-relative operations rooted at the held `units_fd`; do not split
writes through held descriptors and reads through `rr.path`. Reject foreign
objects and require an exact inventory derived from the sealed population.

### C70: one supervised fit and an honest work census

The current 7,744-fit statement is false for the physical implementation.
Refactor each estimator so one fitted model produces both score and in-sample
predictions:

- one ridge solve per origin/arm, supervised under the effective wall/RSS
  contract;
- one MLP epoch-selection sequence per origin/arm/seed, also supervised, with
  the selected fitted model used for both prediction sets; and
- any denoiser fitting step must be supervised or proven to be a bounded,
  fit-free deterministic transform.

Before calling this `scientific_change: NONE`, prove old-vs-new numerical
identity for every prediction and consumed metric on the three development
units. If identity does not hold, issue a superseding design rather than hiding
the change.

Publish separate exact counts for estimator-selection sequences, candidate
fits, linear solves, predictions and baseline evaluations. Derive them from the
productive loops and assert them in tests. Re-estimate the confirmatory wall
budget from the corrected rehearsal; a four-hour cap that cannot cover the
sealed population must fail before scoring and be amended transparently.

### C71: terminal and final-inventory semantics

Validate exact types and domains for every claim and terminal field, including
failure class, reason and wall time. Final adjudication must reject all foreign,
duplicate, partial or unrecognized objects in the control directory, not only
iterate the expected unit ids. A terminal may preserve a real failed assay, but
must never make an unexecuted or unverifiable unit look scientifically
evaluated.

### C72: adversarial battery and independent POST

Add one isolated regression for every PRE above and mutations that:

- remove the positive-duration check;
- append after a torn tail;
- accept a new boot id;
- skip root-path identity revalidation;
- execute ridge outside the supervisor;
- restore the duplicated score/in-sample fits;
- publish the old 7,744 count; and
- ignore a foreign unit object at final adjudication.

Run the focal battery and full suite at the final tip. Report passed, failed,
skipped and collection-error counts separately and name inherited failures.

### C73: stop before scoring

Do not author or install the productive execution record. Do not score a sealed
T2 unit and do not create a scientific ledger. Required disposition:

`T2_C66_C73_READY_FOR_FINAL_MUSASHI_RUNTIME_REVIEW`

Musashi will rerun the PRE/POST, review the final checkout, install the external
record and launch the bounded CPU service if and only if the result passes.

## 2. P1: M3 verifier C1-C6

The numerical result itself is accepted as
`COVER_CALIBRATION_REPRODUCED_AFTER_DISCLOSED_PRECISION_AMENDMENT` because an
independent full replay reproduced every cell exactly. Correct the evidence
consumer before M4 uses it.

### M3-C1: exact population and strict records

Define exact schemas and primitive types for designs, task records, controls
and summaries. Reject duplicate JSON keys, non-finite numbers and bools in
numeric fields. Derive the exact task-index population per cell from the sealed
adaptive rule; require every expected identity exactly once, no duplicates,
gaps or extras. Require the exact 20 controls of the declared kinds per cell.
`all([])` must never certify controls.

### M3-C2: independent outcome reproduction

The fresh verifier must regenerate every point set and label vector from the
sealed seed derivation and rerun the primary solver. It must rerun the
independent formulation on the sealed subset and all controls, rederive every
record body and require exact semantic equality before aggregating. A supplied
self-digest is a checksum, not authority.

Acceptance includes the audit's two exact attacks: remove every control, and
duplicate every task while repairing all digests and summaries. Both must
refuse before a verdict.

### M3-C3: boundary semantics and diagnostics

Make `(0, zero_tol]` produce `AMBIGUOUS_MARGIN` as the sealed design states.
Freeze the strictly separable `1e-10` margin PRE. Describe the singular-value
check honestly as a full-rank numerical diagnostic; it does not prove that
every required subset is in general position. Gaussian generation supplies the
almost-sure assumption, while exact enumerated cases remain the finite check.

### M3-C4: amendment chronology

Preserve v1, v2 and v3 byte-for-byte. Supersede the prose/metadata, not the
evidence, so v3 is labeled a disclosed statistical precision/sample-size
amendment informed by v2's cap miss. Do not call the change `NONE`. Preserve the
facts that the estimand, grid, alpha, seed derivation and verdict rule did not
change.

### M3-C5: verification and mutation

Add regressions for absent controls, duplicated/missing/extra task indices,
coherently rewritten outcomes, malformed primitives, the tiny-positive-margin
boundary and wrong amendment classification. Mutations must bite the
productive verifier. Re-run the corrected verifier over the immutable v3
records; the expected scientific result is unchanged.

### M3-C6: M4 gate

Only after M3-C1 through M3-C5 pass may M4.0 start. M3 grants no DOIN gene,
production gate, scalar intelligence measure or exact complexity claim.

## 3. P2: M4.0 design and mechanical preflight

Turn C2-C3 of `M3_M6_CONFIRMATORY_DESIGN_DRAFT_2026_09_07.md` into an
executable pre-result design for the residual-capacity intervention. The design
must freeze before any intervention outcome:

- structured task families and generator identities;
- architectures and precision, beginning with one-hidden-layer MLPs only;
- optimizer-capability controls that distinguish `OPTIMIZATION_LIMITED` from a
  capacity endpoint;
- initialization, pre-stop, stop and bounded post-stop checkpoints;
- original-task retention metric and margin;
- blinded random-association batches and maximum exposure/update budget;
- acquisition criterion, retention-violation criterion and ambiguous states;
- random-initialization and matched-compute controls;
- task-generator statistical unit, nested seeds, multiplicity and missing-run
  handling;
- exact trajectory/description measurements and their measured cost; and
- CPU wall/RSS/stop-file/heartbeat limits.

Use disjoint generator identities for development, calibration and untouched
confirmation. No checkpoint or seed is an independent statistical unit. The
endpoint is a conditional intervention result, never “unused bits,” “remaining
intelligence” or exact Kolmogorov complexity.

After sealing, run only a two-unit `MECHANICS_ONLY` CPU preflight proving model
forking, checkpoint identity, retention measurement, batch acquisition,
bounded continuation, restart and artifact reconstruction. It grants no M4
scientific conclusion. Stop with:

`M4_DESIGN_AND_CPU_MECHANICS_READY_FOR_MUSASHI_REVIEW`

## 4. Runtime reporting

Observe B4 through read-only telemetry only. Report completed/current cells,
service restarts, wall and thermal facts, but do not touch the process. Report
T2 as blocked by this corrective review, not by the owner. No current action is
required from the owner for T2, M3 or B4.
