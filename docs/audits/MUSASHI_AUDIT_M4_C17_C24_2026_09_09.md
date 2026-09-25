# Musashi audit: M4 C17-C24 intervention foundation

Date: 2026-09-09

## 1. Scope and disposition

Reviewed independently at code tip
`f0e25dec0f8de217c179282d0d72b9c665e6f7c9`:

- sealed design v4 identity
  `30a86f0924fd7d31abcacf6691f084ea3b68821b1fb9735a82458879cac37e52`;
- generator bank, design builder, runner and reconstructive verifier;
- the committed 248-unit DEVELOPMENT screen and four intervention probes;
- the committed PRE/POST and focal battery; and
- the candidate return packet available while the final suite was still in
  flight.

Disposition:

`M4_C17_C24_REVISE_BEFORE_CALIBRATION`

The population census, deterministic generator bank and DEVELOPMENT-only
execution are useful foundations. The observed results are accepted only as
`EXPLORATORY_DEVELOPMENT_EVIDENCE_WITH_PROTOCOL_DEFECTS`. They do not freeze a
learnability gate and do not authorize CALIBRATION or CONFIRMATION.

## 2. Blocking scientific findings

### F1. The paired arms do not receive the same associations

The treatment state uses `unit_id::treatment` as its family/seed component and
the control uses `unit_id::control`. `apply_batch()` passes that field to
`_batch_assoc()`, so arm identity changes both inputs and labels.

Independent reproduction on the first sealed intervention unit returned:

```text
paired_assoc_X_equal False
paired_assoc_y_equal False
treatment_assoc_y [1, 1, 1, 0, 1, 1, 0, 1]
control_assoc_y   [0, 1, 0, 0, 0, 1, 1, 1]
```

The design calls the contrast paired under identical batches. The implemented
contrast is neither. This invalidates every reported treatment-control
difference.

### F2. The initialization control is not the treatment's genesis

The learned arm starts from `fit_init`; the control starts from `ctrl_init`.
The parameter digests differ before either intervention begins. Consequently,
the claimed checkpoint contrast includes uncontrolled initialization variance.

The runner also materializes only a fixed 2,000-update fit and calls it
`calibration_stop`. It creates no `pre_stop`, selected `calibration_stop` or
`post_stop_bounded` checkpoint, despite declaring all four and promising
within-generator contrasts among them.

### F3. Boolean model and baseline semantics disagree with the design

The inherited design declares a sigmoid Boolean head, while `_forward()` is a
linear head for every family and `_sgd_step()` minimizes squared error for
every family. More importantly, the Boolean baseline is the training-set
majority proportion, not the accuracy of the training-selected majority class
on held-out observations.

All eight DEVELOPMENT Boolean generators reproduced a mismatch. Examples:

```text
identity g0: reported 0.567708, held-out majority accuracy 0.500000
dnf3 g0:     reported 0.697917, held-out majority accuracy 0.796875
parity4 g1:  reported 0.515625, held-out majority accuracy 0.421875
```

Therefore the learnability margins and family decisions must be recomputed.

### F4. The association task changes meaning across families

Every intervention currently receives standard-normal inputs and binary
labels. Boolean generators use inputs in `{-1, +1}`; temporal generators use
windows from their own observation process and continuous targets. The probe
therefore measures an out-of-distribution binary task for some families and an
in-domain mismatch for others. Cross-family endpoint differences are not
interpretable as residual memorization under one intervention.

The association generator must sample blinded, independent pairings from a
declared family-compatible input and target distribution. The resulting tape
must be byte-identical across arms and checkpoints of the same experimental
unit.

### F5. Training, stopping and evaluation are not separated

The bank exposes only `train` and `held`. A real early-stopping checkpoint
requires a stopping/calibration slice distinct from the untouched evaluation
slice used for retention and learnability. Reusing one held slice for checkpoint
selection and scientific evaluation would make the endpoint optimistic.

The successor must create disjoint train, stop and evaluation roles and bind
their bytes in the manifest before any new outcome.

### F6. The v4 chronology cannot serve as confirmatory preregistration

The PRE is committed at `459bbd51`, but the sealed v4 design, complete
DEVELOPMENT outcomes and evidence all first appear together in `f0e25dec`.
The return also discloses a pre-commit re-seal after an initial short grid.
Nothing proves the final v4 bytes existed before every developmental score.

This is not misconduct and does not erase the developmental evidence. It does
mean v4 is a DEVELOPMENT protocol learned alongside its pilot. A v5 successor
must be committed and pushed before any CALIBRATION outcome is computed.

## 3. Additional design and runtime findings

1. The inherited generator contract says SHA-256 ids, the family name
   `discontinuous` and role-specific instance ranges 0-15/16-23/24-31. The
   implemented bank uses readable ids, `discontinuity` and a shared 0-9999
   index domain. Role bytes are disjoint in practice, but two contradictory
   contracts remain in one sealed object.
2. Two CALIBRATION generators cannot estimate the between-generator dispersion
   used to choose among confirmation counts 10, 24 and 48. The design does not
   specify the estimator or mapping rule. The old CONFIRMATION range contains
   only eight instances, below even the smallest claimed requirement.
3. M0/M1/M2 are prose only. No predictive model, censor-aware loss, grouped
   split, calibration measure or incremental decision threshold is executable.
4. Resource stops are marked right-censored alongside `MAX_BATCHES`, although
   they are incomplete scientific units. The design names a Gehan fallback but
   neither defines a paired implementation nor provides one.
5. The runner enumerates only DEVELOPMENT units. It has no checkpoint builder,
   CALIBRATION plan, CONFIRMATION plan or M0/M1/M2 implementation.
6. `--execute` can return success without calling the fresh verifier.
   Verification is a separate mode rather than a required terminal transition.
7. Resume replays only completed screen JSON files. Partial intervention logs
   are appended from a fresh in-memory state, so interruption can create an
   internally inconsistent history.
8. Intervention JSONL files and heartbeat telemetry were observed at mode
   `0664`, contradicting the private-artifact claim. JSONL files are opened in
   ordinary append mode rather than created exclusively.
9. Report schemas are not exact, and the inventory ignores every filename
   beginning with `RUN_REPORT`. A second invocation creates a random extra
   report rather than refusing or returning an idempotent verified result.
10. The inherited resource ceiling is six hours while the reduction rule says
    48 hours. The productive runner uses the former. One executable source of
    truth is required.
11. The literal claim that CALIBRATION/CONFIRMATION were never generated is too
    strong: the disjointness check generates all three roles. The accurate
    claim is that no CALIBRATION/CONFIRMATION outcomes were scored or persisted
    as scientific evidence.

## 4. What is accepted

The following work should be preserved, not rebuilt gratuitously:

- the 29 meaningful structured family/noise cells and explicit refusal of
  nonsensical Boolean/noise products;
- deterministic role-separated generator bytes;
- train-fitted temporal noise scales and latent-plus-disturbance
  reconstruction;
- the generator-as-independent-unit rule;
- full-denominator treatment of missing and failed units;
- the restricted claim that compressed length, pruning and spectral rank are
  candidate descriptors rather than capacity ground truth;
- the 248-unit run as exploratory DEVELOPMENT evidence;
- the focal battery, which independently reproduced at 13/13; and
- the fresh verifier's exact reproduction of the committed 248 screen records
  and four probe records under the code that produced them.

The four intervention endpoints themselves are not accepted as paired effects
because of F1-F4.

## 5. Independent verification

Completed before this disposition:

- `pytest tests/test_m4_intervention.py -q`: **13 passed**;
- committed C17-C24 POST: five adversaries refused and five mutations bit;
- fresh verification of the committed run: **248 screen units + 4 probes**;
- explicit reproductions of association-tape divergence, genesis divergence,
  all eight Boolean-baseline mismatches and the non-sigmoid output range; and
- artifact-mode inspection showing intervention JSONL files at `0664`.

The repository-wide suite completed after the scientific review:

- **3,195 passed, 3 failed, 5 skipped, 1 error** in 2,989.37 seconds;
- the two D1-anchor failures reproduced as the inherited missing/mismatched
  operator evidence;
- the weekly-promotion error passed in an isolated rerun; and
- the B4 claim-race failure reproduced both at `f0e25dec` and at the pristine
  parent `318cfdb8`, so it is pre-existing and unrelated to M4.

No new M4 test failed. The candidate return packet still contained placeholders
after the suite ended and must record these exact facts before its final commit.

## 6. Next order

Execute:

`MUSASHI_TO_GENERAL_SATOSHI_M4_C25_C36_PROTOCOL_CORRECTION_AND_CALIBRATION_ORDER_2026_09_09.md`

The order requires a pre-outcome v5 successor, a corrected DEVELOPMENT rerun,
an exact CALIBRATION population and a stop before every CONFIRMATION outcome.
It grants no GPU, DOIN, financial-data or production authority.

## 7. Runtime and owner boundary

B4 and T2 remain independent active campaigns and must not be modified by M4.
No owner action is required for the M4 correction. The next blocker belongs to
Satoshi until the corrected CALIBRATION return is ready for external review.
