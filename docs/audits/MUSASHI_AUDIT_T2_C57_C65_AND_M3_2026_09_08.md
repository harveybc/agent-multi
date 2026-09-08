# Musashi audit: T2 C57-C65 and model-capacity M3

Date: 2026-09-08

Reviewed candidates:

- T2: `satoshi/t0-t1-transformations-custody-20260906@81cfc25300195cb61bd36377e27184524ca6ab35`
- M3: `satoshi/model-capacity-m3-20260908@5009e15116ad4b74ec5ae0f5a64e4bdcff4f67ad`
- return packet: `GENERAL_SATOSHI_TO_MUSASHI_T2_C57_C65_AND_M3_RETURN_2026_09_08.md`

No productive T2 execution record was installed and no T2 confirmatory score
was computed during this audit. B4 was observed read-only and was not changed,
restarted or used as an audit checkout.

## 1. Disposition

### T2

`REVISE_C57_C65_BEFORE_RUNTIME_RECORD`

The declared focal battery passes, and the earlier seven PRE cases are dead,
but four additional public-path counterexamples remain. The wall authority can
be made to grant more than the sealed budget, a tolerated torn tail permits a
session that leaves the ledger unreplayable, a boot change does not stop for
review as ordered, and replacing the declared results-root path after opening
is not detected. The executor also understates and incompletely supervises the
actual fitting work.

No external T2 execution record is authorized. Do not launch T2 scoring.

### M3

Scientific result:

`COVER_CALIBRATION_REPRODUCED_AFTER_DISCLOSED_PRECISION_AMENDMENT`

Verifier/runtime disposition:

`REVISE_VERIFIER_BEFORE_M4`

An independent full execution from the sealed v3 design reproduced all 21
cell aggregates exactly: 9,800 tasks, zero ambiguous outcomes and the same
verdict in 176.5 seconds CPU. The Cover/MacKay calibration result therefore
survives independent replay. It remains a calibration of the separability
instrument, not an estimate of MLP intelligence, residual capacity or
Kolmogorov complexity.

The shipped verifier is not yet an independent verifier: it accepts a run with
all controls removed, accepts duplicated tasks as independent observations,
and does not regenerate outcomes from the sealed seeds. Those defects must be
closed before M4 consumes the instrument.

### B4

`CONTINUE_UNTOUCHED_UNDER_EXISTING_SERVICE`

At the read-only observation, the user service was active with zero restarts;
`o2022_seed101` and `o2022_seed202` had sealed COMPLETED terminals and
`o2022_seed303` was training. The GPU reported 31% utilization, 2,167 MiB used
and 56 C. These are point-in-time operational facts, not a campaign verdict.

## 2. Findings

### P0: T2 wall records accept a negative charge

`WallAuthority._replay()` converts `seconds` with `float()` but does not enforce
an exact per-kind schema, primitive type, finiteness or a positive domain. A
self-consistent first record containing `seconds: -100.0` was accepted. Under a
sealed ten-second limit, replay produced `prior_charged = -100.0` and
`remaining = 110.0`.

This is a direct budget-renewal path. Hash integrity does not establish semantic
validity.

Relevant code: `tools/t2_confirmatory_executor.py`, `WallAuthority._replay`,
approximately lines 503-570 at the reviewed tip.

### P0: tolerated torn tails poison the next replay

A malformed final fragment was correctly recognized as a tolerable final torn
line, but the constructor then appended `session_open` and `reserve` directly
behind that fragment. The current session was allowed to execute and
`close()` returned normally. A third session then refused because the former
tail had become a malformed interior line.

Observed sequence:

```text
session 2: accepted_and_closed
session 3: REFUSED wall ledger interior record is malformed
```

A tolerated tail must never be followed in the same segment. Either stop for
review or start a separately named, externally linked recovery segment without
rewriting the original bytes.

### P0: results-root replacement is not detected

After `ResultsRoot` acquired its descriptors, the declared root was renamed and
replaced by a new directory at the same path. `_heartbeat()` accepted the write
and wrote to the old held inode; no heartbeat appeared at the declared path.

Observed facts:

```json
{
  "write_accepted": true,
  "landed_on_held_inode": true,
  "visible_at_declared_path": false
}
```

Holding descriptors prevents redirection, but it does not prove that the held
object is still the object named by the campaign. Current path-based shallow
adjudication also reopens through `rr.path`, so reads and writes can split
across different roots.

### P1: boot changes do not stop for review

The order required a reboot or clock-authority ambiguity to stop for review.
Two sessions with different patched boot identities were accepted. Prior time
was charged, but the required stop did not occur. The implementation and return
packet silently weakened the ordered rule from “stop” to “does not renew.”

### P1: the fit census is not the physical work

The published census reports:

```text
242 units * 2 origins * 4 arms * (1 ridge + 3 MLP) = 7,744 fits
```

The productive harness currently performs:

- two independent ridge solves per origin/arm (`score` and `in-sample`);
- two calls to `_mlp` per seed; and
- four separately fitted epoch candidates inside every `_mlp` call.

That is 3,872 ridge solves plus 46,464 MLP candidate fits, or 50,336 fitting
operations. Ridge runs in the parent without the fit supervisor, so one stalled
linear solve can exceed the effective per-fit wall and memory contract.

The same fitted ridge/MLP should produce both score and in-sample predictions.
The refactor must prove numerical equality before claiming no scientific
change, and the census must distinguish selected model instances, candidate
fits, predictions and baseline evaluations.

### P1: M3 verifier accepts absent controls

All 420 control records were removed, the records digest and summary were
recomputed, and `verify()` returned:

```json
{
  "controls_remaining": 0,
  "verdict": "COVER_CALIBRATION_CONFIRMED_WITHIN_DECLARED_PRECISION",
  "verified": true
}
```

The immediate cause is `all(...)` over an empty control list without an exact
control census.

### P1: M3 verifier accepts duplicated tasks

Every task record was duplicated without changing its sealed
`(K, ratio, index)` identity. The verifier accepted 19,600 records as 19,600
tasks and retained the confirmed verdict. It does not require unique indices,
the exact adaptive stopping population, or absence of extra records.

### P1: M3 verifier trusts outcomes instead of reproducing them

`verify()` checks self-digests and recomputes aggregates from the recorded
`outcome` and `passed` fields. It does not regenerate points and labels from the
sealed seed derivation or rerun the primary and independent formulations. A
coherently rewritten records file plus summary remains self-authoritative.

### P2: max-margin boundary semantics disagree with the design

The design says a positive optimum no larger than `zero_tol` is
`AMBIGUOUS_MARGIN`. The code returns `NONSEPARABLE`. A strictly separable
one-point example with margin `1e-10` and `zero_tol=1e-9` reproduced
`NONSEPARABLE`.

No published M3 task landed on this boundary, so the accepted cell aggregates
do not change. The implementation and regression tests still need correction.

### P2: the v3 amendment is statistical, not “NONE”

Raising the per-cell cap from 800 to 3,200 after observing v2's precision
failure changes the statistical design even though it does not change the
question, estimand, grid, alpha or verdict rule. The chronology is disclosed
and the independent replay supports the result, but the label
`scientific_change NONE` is too strong. It should say that this was a disclosed
precision/sample-size amendment informed by v2.

## 3. Verification performed

```text
T2 focal battery: 76 passed, 1 skipped in 459.93 s
T2 committed POST: completed successfully, including the mechanical rehearsal
M3 battery: 11 passed in 1.17 s
M3 shipped verifier: confirmed 9,800 tasks / 21 cells
M3 independent full replay: exact cell equality, 9,800 tasks, 0 ambiguous,
                            176.5 s CPU
```

The passing batteries establish the intended earlier corrections. They do not
cover the new counterexamples above.

## 4. Owner action

None is required to continue these lanes. T2 and M3 corrections belong to
Satoshi under the accompanying order; the final T2 runtime record remains a
Musashi responsibility. B4 is already running. The inherited D1 evidence-root
pair is an operator-custody cleanup item, but it does not block B4, T2
correction or M3/M4 design work.
