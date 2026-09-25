# Musashi to General Satoshi: M4 C31A-C31F numeric incident correction

Date: 2026-09-09

Priority: P0 continuation of M4 C25-C36. Execute immediately after CALIBRATION
attempt 2's internal verifier exits. Do not terminate the running verifier.

Authority basis:

- pre-outcome v5 boundary `12dc2e31`;
- finite-path DEVELOPMENT commit `992388ed`;
- partial numeric correction `3f432a0b`;
- v5 design identity
  `d7280a92047d98898418fb7cd750b22c506a621eb381d9847e0fe926b7df69b9`;
- independent DEVELOPMENT verification at `3f432a0b`; and
- audit `MUSASHI_INCIDENT_AUDIT_M4_CALIBRATION_NUMERIC_GUARD_2026_09_09.md`.

Recorded disposition:

`M4_CALIBRATION_ATTEMPT_2_NON_GOVERNING_PENDING_NUMERIC_AMENDMENT`

This is a correction inside the active order. It grants no CONFIRMATION, GPU,
DOIN, financial data, live action or production authority.

## 1. M4-C31A: preserve and classify both attempts

Let attempt 2 finish its internal verifier. Preserve attempts 1 and 2 byte for
byte and publish their exact exit states. Classify:

- attempt 1: `CRASHED_UNTYPED_NON_GOVERNING`;
- attempt 2: `VERIFIER_RESULT_RECORDED_NON_GOVERNING_NUMERIC_GUARD_GAP`, or
  its exact failed-verifier state if it fails.

Do not run C35 adjudication against either root. Do not inspect aggregate
eligibility, dispersion, ladder comparisons or proposed confirmation slots
before the correction design is pushed.

## 2. M4-C31B: descriptor numeric domain

Keep the existing finite-path descriptor bytes unchanged. Before the float32
cast, require every rounded parameter to lie within the finite float32 range.
If not, return a typed `NUMERICALLY_INVALID_DESCRIPTOR` with every derived
descriptor unavailable. Never compress infinity bytes.

Guard SVD explicitly:

- catch decomposition failure;
- require every singular value finite;
- type failure as `NUMERICALLY_INVALID_DESCRIPTOR`; and
- never replace it with zero rank, zero cost or another apparently valid
  measurement.

Suppressing a warning is not a correction. The productive returned status and
the fresh verifier must derive the same invalidity from the original float64
parameters.

## 3. M4-C31C: complete paired generators only

Define a complete primary pair as both `initialization` and
`calibration_stop` ending in one of the endpoint states admitted by the sealed
restricted-endpoint contract. `NUMERICAL_ANOMALY`, invalid task training,
invalid descriptor, resource stop, missing record and uncertain state are not
complete pairs.

For each `(cell, generator)` require exactly all three sealed model seeds before
forming the generator-level mean. If one seed is incomplete:

- mark the generator `INCOMPLETE_PAIRED_GENERATOR`;
- keep it in the cell denominator and attrition accounting; and
- do not average the remaining one or two seeds as if the generator were
  complete.

Dispersion must publish planned, complete and incomplete generator counts. It
may be estimated from complete generators only if the predeclared 20-percent
attrition allowance and minimum count are both satisfied. Otherwise the cell
is `CALIBRATION_INCOMPLETE`, never precision-supported.

The prediction ladder must likewise consume only complete quartet records with
valid descriptors, while reporting every excluded unit and reason. A missing
descriptor cannot be imputed from outcomes or silently drop one arm.

## 4. M4-C31D: incident-specific regression battery

Freeze the exact failed routes before editing:

1. nonfinite parameters during checkpoint training crash instead of returning
   `NUMERICALLY_INVALID_TASK_TRAINING`;
2. finite float64 parameters outside float32 range pass the descriptor guard;
3. an injected SVD failure escapes untyped;
4. nonfinite singular values produce a valid rank;
5. intrabatch divergence is not replayed as `NUMERICAL_ANOMALY`;
6. a producer claims numerical invalidity while fresh replay is finite;
7. a producer claims finite completion while fresh replay is invalid;
8. one anomalous primary arm enters paired dispersion;
9. two of three seeds are averaged as a complete generator;
10. attrition beyond the sealed allowance still reports precision support;
11. an invalid descriptor enters M2; and
12. an incomplete unit disappears from the denominator.

Each test must call productive code. Add guard-specific mutations for the
float32-range check, SVD typing, complete-pair check and exact-three-seed check.
A broad upstream refusal does not count as a bite.

## 5. M4-C31E: pre-run amendment and attempt 3

Create an append-only v5 numeric-validity amendment, or an exact v6 successor,
that changes only:

- descriptor validity at the already-declared float32 serialization boundary;
- typed SVD failure;
- complete-pair/three-seed requirements; and
- explicit attrition accounting.

The amendment must state that the trigger was the runtime warning and code-path
inspection, not a scientific effect. Map every changed field and prove the
finite DEVELOPMENT path remains bit-identical. Commit and push code, tests and
amendment before attempt 3.

Then execute CALIBRATION once in a third fresh root. Its runner must finish with
its own fresh verifier. Preserve all prior roots. No reuse, copying or editing
of attempt-2 scientific records is allowed.

## 6. M4-C31F: resume the original return boundary

Only after attempt 3 verifies may C35 derive candidate eligibility, dispersion,
M0/M1/M2 and typed confirmation slots. Then complete the original C36 packet
with:

- all three attempt identities and exact dispositions;
- incident PRE/POST and mutation outputs;
- counts of invalid task, intervention and descriptor units;
- planned/complete/incomplete generator counts by cell;
- the exact attrition and precision decision; and
- final focal and repository-suite counts from the final pushed tip.

Stop at the original external-review boundary. Do not generate, load or score
any CONFIRMATION outcome.
