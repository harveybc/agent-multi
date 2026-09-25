# Musashi incident audit: M4 calibration numeric guard

Date: 2026-09-09

## 1. Scope and chronology

Reviewed while M4 v5 CALIBRATION attempt 2 was running:

- pre-outcome v5 boundary `12dc2e31`;
- corrected DEVELOPMENT evidence commit `992388ed`;
- numeric-failure correction `3f432a0b`, already pushed before attempt 2;
- preserved failed root
  `m4_v5_calibration_run_20260909_CRASHED_ATTEMPT_1`; and
- fresh attempt-2 root `m4_v5_calibration_run_20260909`.

The branch was clean apart from the uncommitted return packet. Attempt 2 ran
from the pushed correction, in a fresh root, on CPU. B4 and T2 remained active
with zero service restarts.

## 2. What the correction fixed

The code now types non-finite task-training state as
`NUMERICALLY_INVALID_TASK_TRAINING`, types non-finite intervention state as
`NUMERICAL_ANOMALY`, protects descriptor entry from already-nonfinite weights
and replays task-training invalidity in the fresh verifier.

Independent verification of the committed DEVELOPMENT root under `3f432a0b`
returned exactly:

```text
verified: true
screen_units: 248
intervention_units: 4
```

Attempt 2 materialized 1,984 CALIBRATION screen records and 1,344 intervention
summaries before its internal verification. Six intervention units were typed
`NUMERICALLY_INVALID_TASK_TRAINING`; the matching screen table contained two
numeric-invalid units. Evidence objects inspected were mode 0600.

## 3. Remaining blockers found during the live audit

### F1. The descriptor guard is narrower than its serialization

`_descriptors()` checks that the float64 vector is finite, then rounds and casts
it to float32. A finite float64 value outside the float32 range therefore
overflows after the guard. Attempt 2 emitted the warning from the productive
line itself:

```text
RuntimeWarning: overflow encountered in cast
np.round(w, 6).astype(np.float32).tobytes()
```

Such an arm can receive a finite compressed-length value made from infinity
bytes and enter M2 as if its descriptor were valid. `np.linalg.svd()` also has
no typed exception/nonfinite-singular-value path after the entry check.

### F2. Incomplete arms can still enter the primary dispersion

The adjudicator excludes `NUMERICAL_ANOMALY` arms from the prediction ladder,
but the primary dispersion loop appends `paired_primary_difference` without
checking whether both primary arms are complete. It also averages whatever
subset of the three nested seeds exists. A one- or two-seed generator can
therefore masquerade as a complete paired generator.

Task-training-invalid summaries are skipped, but the effective generator count
and attrition disposition are not enforced before dispersion support is
claimed.

### F3. The disclosed incident has no incident-specific regression

Commit `3f432a0b` changes three productive modules and no test file. The reported
18/18 battery is the pre-existing C33 battery; it does not force nonfinite
checkpoint training, float64-to-float32 overflow, SVD failure, intrabatch
anomaly or incomplete-seed adjudication.

## 4. Disposition

`M4_CALIBRATION_ATTEMPT_2_NON_GOVERNING_PENDING_NUMERIC_AMENDMENT`

Do not interrupt attempt 2. Let its internal verifier terminate and preserve
the root exactly. Do not use it for C35 eligibility, dispersion, M0/M1/M2 or
confirmation slots.

Execute the companion order
`MUSASHI_TO_GENERAL_SATOSHI_M4_C31A_C31F_NUMERIC_INCIDENT_CORRECTION_ORDER_2026_09_09.md`.
It requires a pre-run amendment and a fresh attempt 3. CONFIRMATION remains
closed.

No owner action is required.
