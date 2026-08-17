# Audit of Satoshi Finding-277 Policy-Behavior Delivery

Date: 2026-08-17
Auditor: General Musashi
Implementation reviewed: `agent-multi@055ac32e`
Branch: `satoshi/finding-277-policy-behavior-20260817`
Parent runtime source: `agent-multi@3d2bf3f4`

## Verdict

**REJECTED AS COMPLETE; CORRECTION REQUIRED.** The delivery contains a useful
taxonomy nucleus and a read-only diagnostic prototype, but it does not yet
satisfy WP0-WP5 of the accepted order. It must not be used as stopping,
aggregation or promotion authority.

The active campaign was not mutated by this audit. At 2026-08-17T23:20:59Z,
all four workers were running identity `f9379f596e80fda4` with fresh
heartbeats and zero systemd restarts.

## Findings

### AUD-F1-20260817-281 — S2 — open

`classify_policy_behavior()` labels a time-varying action series
`STATE_RESPONSIVE_ACTIVE` without receiving observations, a model identity or
an intervention result. Action variation proves mapped activity, not state
responsiveness. An open-loop pattern, stochastic artifact or trace/model mixup
can receive the only class currently marked promotable.

Independent counterexample: `[-0.2, 0.2, -0.2, 0.2]` at threshold `0.1`
returns `STATE_RESPONSIVE_ACTIVE` and
`promotable_as_learned_activity=true`, although the API has no observation
evidence at all.

### AUD-F1-20260817-279 — S2 — open

The sidecar's stated role and custody boundaries are not executable. It checks
only `path.name` and a free-text CSV `split`; it does not resolve the trace's
`.meta.json` or the nested-split manifest. A trace below a
`sealed_test_2025/` parent with filename `evaluation_return_trace.csv` and
split `evaluation` is accepted. The resulting measurement contains only the
trace path/hash, not role, experiment/config/model hashes, observation
contract or attempt-bound checkpoint.

The real seed-101 sidecar run also rendered `evaluation` even though the
trace metadata binds its data file to `nested_splits/train_monitor.csv`. This
is evidence that role information exists and is being discarded.

Additional fail-open behavior in the same boundary:

- malformed action, position, equity, trade and cost values are filtered;
- missing costs become `0.0`;
- invalid positions reduce the numerator while the denominator remains the
  full row count;
- a mutable trace is read and hashed without a stable before/after snapshot;
- checkpoints are listed at cell level, not bound to each measurement.

### AUD-F1-20260817-278 — S2 — open

The shared classifier silently drops malformed, NaN and infinite actions,
then classifies the surviving subsequence. A four-value sequence containing
two invalid values was accepted as a two-value
`CONSTANT_DIRECTIONAL_EXPOSURE`, contrary to the module contract that
non-finite actions are `UNAVAILABLE`.

Threshold-zero crossings are also wrong: exact zero is counted as a crossing
although `map_action(0, 0)` returns HOLD. Deterministic and stochastic
`[0.0, 0.0]` is consequently mislabeled `STOCHASTIC_ONLY_ACTIVITY` instead of
`CONSTANT_HOLD`.

### AUD-F1-20260817-280 — S3 — open

The delivery is neither production-integrated nor repository-clean:

- `055ac32e` explicitly contains only `WP0+WP1+WP5(1-3)`;
- `tools/p1lr_actor_probe.py` is untracked;
- neither new tool is declared in the engineering-surface registry;
- no production caller uses `_policy_behavior.py` outside the sidecar;
- stopping, aggregation and promotion therefore still do not share it;
- WP2 disposition, WP3 2x2x2 diagnostic, discrete baseline, WP4 successor
  comparison and the remaining WP5 acceptance work are absent.

The full suite result is **1 failed, 1,611 passed**. The failure names both
new executables as `unclassified_new_executables`.

## Independent Evidence

Reproducer:

`docs/audits/evidence/MUSASHI_FINDING_277_ADVERSARIAL_REPRO_2026_08_17.py`

Command:

```bash
python docs/audits/evidence/MUSASHI_FINDING_277_ADVERSARIAL_REPRO_2026_08_17.py \
  --implementation-root \
  /home/harveybc/Documents/GitHub/.worktrees/agent-multi-satoshi-277-20260817
```

Observed: exit 1; all nine acceptance checks false. The fixture uses only a
temporary directory and does not access a campaign identity.

Focused tests supplied by Satoshi: **26 passed**. Full suite:

```text
1 failed, 1611 passed, 2 warnings in 137.65s
```

Real read-only prototype run over omega's current identity found one cell and
three traces: two `CONSTANT_HOLD`, one
`STATE_RESPONSIVE_BELOW_THRESHOLD`, zero promotable. These are useful
diagnostic observations but are not accepted scientific facts until findings
278, 279 and 281 are corrected.

## Verified Non-Findings

- The implementation did not mutate the active identity in this audit.
- The sidecar refuses an output path lexically inside the identity root.
- Basic constant/directional/below-threshold taxonomy tests pass.
- All four campaign services were active, fresh and advancing; low
  instantaneous utilization on dragon was not a hang because its timestep
  counter advanced between log samples.

## Disposition

Finding 277 remains open and becomes **correction required**. Findings
278-281 are assigned to General Satoshi for implementation. General Musashi or
Retsu must independently reproduce the returned packet; the implementer may
not close any finding.

The detailed executable order is:

`docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_FINDING_277_CORRECTION_ORDER_2026_08_17.md`
