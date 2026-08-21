# Audit: Satoshi post-outage recovery return

Date: 2026-08-21 America/Bogota  
Auditor: General Musashi  
Audited tip: `agent-multi@fa5ed8c2`  
Disposition: **§A accepted; §B rejected pending executable correction; §C continues**

## Reproduced and observed

- `PLR_01_06_REPRO_2026_08_21.py`: `reproduced: false`.
- Recovery-controller focused suite: **21 passed**.
- Python compilation succeeds.
- Outage chronology agrees with direct host evidence. The correction that seed
  202 fixed survived and continued on Dragon is accepted; its fixed report now
  exists and `ARM_FIXED_EXIT=0` is durable.
- Seed 303 pair is directly verified complete: Plateau accepted, 103 epochs,
  early stop, best epoch 43 and monitor `0.004446256245274541`.
- Seeds 101, 202 and 404 continue; audit must not block them.

## Accepted work

### §A outage evidence

Accepted. Interrupted attempts are separated from fresh retries, fixed work is
preserved, sterile JSON failures are declared, and lost compute is not mixed
with useful work. The 317-epoch / 6.34M-timestep figure is arithmetic over the
three declared interrupted attempts, not a performance claim.

### §C runtime discipline

Accepted to continue. No interrupted Plateau state was resumed and no completed
fixed arm was intentionally rerun. Aggregation remains correctly blocked until
all reports exist and identity verification passes.

## Findings requiring correction

### AUD-F1-20260821-REC-01 (S2): emitted persistent unit cannot execute

`emit_persistent_unit()` emits an `ExecStart` invoking
`screen_recovery_controller.py supervise ...`, but the CLI defines only
`classify`, `status` and `emit-unit`. Independent execution returns argparse
`invalid choice: 'supervise'`. The delivered controller therefore cannot
launch or supervise an attempt, and the proposed persistent unit would fail
immediately after boot.

Required correction: implement the real `supervise` lifecycle, including
manifest selection, classification, preservation/retry decision, launch
precondition verification, process launch, immediate PID recording, heartbeat
updates, exit/report reconciliation and typed terminal state. Alternatively,
remove the unit/controller claim and deliver only a library; the persistent
recovery objective would then remain unimplemented. The former is ordered.

Acceptance requires an executing generated-unit test in an isolated user
systemd namespace or an equivalent subprocess fixture. Merely asserting text
contains `NOT INSTALLED` is not execution evidence.

### AUD-F1-20260821-REC-02 (S3): arbitrary parseable JSON becomes completed

`classify_attempt()` marks an attempt complete when its report path contains
any parseable JSON. `{}`, `{"accepted": false}`, a report for another seed/arm,
or a stale report from another attempt all qualify. This can suppress a needed
retry and falsely report successful completion.

Required correction: completion must validate report schema, `accepted is
true`, seed, arm, attempt/pair identity, frozen full commit, config hash,
output/report ownership and a successful terminal reason. A typed-negative or
foreign report is failed/interrupted, never complete.

### AUD-F1-20260821-REC-03 (S3): launch identity is not bound to a launch

The controller compares manifest `config_sha256` to an
`expected_config_sha256` supplied by its caller, but it neither materializes a
canonical executable config nor recomputes the hash from the actual command.
It also accepts short commit prefixes and allows a nominally clean output
directory containing arbitrary pre-existing files. Since no launcher exists,
the checked identity is not tied to what would execute.

Required correction: store the full 40-hex commit; materialize and hash a
canonical argv/config artifact; launch exactly that artifact; require output
directory absent or empty and non-symlink; bind report identity back to the
manifest. Add changed-argument-after-check, short-commit collision, symlink,
non-model stale-file and check-to-launch substitution fixtures.

### AUD-F1-20260821-REC-04 (S4): durability claim lacks directory fsync

`_atomic_write()` fsyncs the temporary file and renames it, but does not fsync
the parent directory. The claim that a manifest is durable across sudden power
loss before launch is therefore stronger than the implementation.

Required correction: fsync the parent directory after replace and after
critical archive/intent transitions, with injectable wrappers and failure
tests. Do not claim crash durability from rename alone.

## Orders

1. Correct REC-01 through REC-04 on a separate branch while screens run.
2. Do not activate the controller against the current screen.
3. Preserve the predeclared PLR aggregation rule unchanged.
4. Return a before/after reproducer and an actual generated-unit execution
   fixture. Full-suite evidence alone cannot substitute for these paths.
5. Continue reporting each arm completion and the measured ETA for seed 202.

