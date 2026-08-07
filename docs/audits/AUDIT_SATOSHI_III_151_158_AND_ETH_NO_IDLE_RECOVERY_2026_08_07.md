# Audit: Satoshi III 151-158 and ETH No-Idle Recovery

Date: 2026-08-07 America/Bogota
Auditor: General Musashi
Runtime scope: ETH curriculum decision D1
Decision authority: project owner

## 1. Verdict

The 151-158 packet is partially accepted. The focused correction tests for
the rolling-origin, pause, configuration and decision contracts pass, but the
packet did not prove a real second-host decision run and its proposed
`full-v3` launch was neither started nor the next approved experiment.

The approved next job is now running: four concurrent ETH curriculum seeds,
each executing `N14`, `EN4_10` and `E4` sequentially under the frozen D1
contract. The abandoned `full-v2` chain remains archived and is not resumed.

## 2. Independently Reproduced Findings

### AUD-F1-20260807-159 (S2): genuine remote replicas failed locally

`validate_arm_record()` treated a remote path as a local `Path` and required
it to exist on the producer. A genuine Dragon/Gamma replica therefore failed
only after an expensive arm completed. Corrected at `agent-multi@94bebe2a`:
the record validates the independent remote observation and its path/hash
bindings; final collection re-probes the authority over SSH.

### AUD-F1-20260807-160 (S2): replica namespaces collided

The remote directory used only `path.parent.name`, so equal arm labels from
different seeds could overwrite one another. Corrected at `94bebe2a` with
`run/seed/arm/artifact` namespaces and a regression test.

### AUD-F1-20260807-161 (S3): untested handover parameter broke tests

The uncommitted `handover_bars` change raised `AttributeError` in two contract
tests and accepted non-positive values. Corrected at `94bebe2a`; the value is
bound into run identity and the CLI rejects values below one.

### AUD-F1-20260807-162 (S2): idle fleet plus wrong queued experiment

All four GPUs were idle. Only paused `full-v2` supervisors were alive.
Uncommitted `full-v3` drafts repeated the zero-activity contract, contained an
unsafe `git add -A` launcher and did not establish shared lineage or unique
work. The drafts were removed before execution. D1 was launched instead.

### AUD-F1-20260807-163 (S2): CUDA ordinal reversed Gamma assignments

The first bounded launch put seed 303 on the 5090 and seed 404 on the 5070 Ti
because CUDA ordinal order differed from `nvidia-smi` index order. The launch
was stopped before one complete epoch. `agent-multi@46ce057b` binds each
worker to its exact GPU UUID and status proves the service PID on that UUID.

### AUD-GEN-20260807-164 (S3): correction probe is stale

`docs/audits/evidence/repro_runs/correction_probe_v2.py` currently reports
`all_pass=false`: cases 140 and 145 are harness errors after API/schema
changes. This does not invalidate D1, but it cannot be cited as acceptance
evidence until repaired and rerun before/after.

## 3. Verification Evidence

- Focused contracts: 122 passed.
- Decision/rolling/tool-index follow-up: 85 passed.
- Replica decision contract alone: 31 passed.
- Engineering surface index: `OK`, 82 tools, zero structural problems and
  zero unclassified executables.
- Preflight packet:
  `~/.local/share/agent-multi/eth_curriculum_decision_20260807_v2/fleet_preflight.json`.
- Launch packet:
  `~/.local/share/agent-multi/eth_curriculum_decision_20260807_v2/fleet_launch.json`.
- Runtime status packet:
  `~/.local/share/agent-multi/eth_curriculum_decision_20260807_v2/fleet_status.json`.

Exact frozen revisions on every participant:

- `agent-multi@46ce057b2dafe712ca098e99dd19cec5bc8f4628`
- `gym-fx@efa491600bdc9fee10efdfbe251474d63284a28b`
- `doin-node@5bd6d3966df37e98e0de6fb904d0ec81566866a6`
- `doin-core@e05a3325625a9ad497b56866485c7606024e3681`
- `doin-plugins@8c959a611d63dce8a67e9cf838b130cd1f3f1bad`
- `trading-contracts@cd050834406c68d14cde72986b99e5db34425e4e`

## 4. Live D1 Assignment

| Worker | Seed | GPU | Current arm | Direct proof |
| --- | ---: | --- | --- | --- |
| Omega | 101 | RTX 4070 Laptop | N14 | service PID equals CUDA PID on bound UUID |
| Dragon | 202 | RTX 4090 Laptop | N14 | service PID equals CUDA PID on bound UUID |
| Gamma | 303 | RTX 5070 Ti Laptop | N14 | service PID equals CUDA PID on bound UUID |
| Gamma | 404 | RTX 5090 | N14 | service PID equals CUDA PID on bound UUID |

At the last audit sample all four processes were runnable at approximately one
CPU core each, CUDA memory was allocated, temperatures were 43-57 C, and the
global idle watchdog reported `ACTIVE`, streak 0. Early N14 epochs reported
zero trades; that is an observed normal-arm baseline, not a claim about the
later easy arms. `EN4_10` and `E4` must demonstrate their own activity.

Per-host one-minute guardians are installed. They do nothing while the bound
worker is active, stop after a complete `seed_packet`, and start an idempotent
recovery service if a worker disappears before completion. Recovery uses the
same root, seed, budget, GPU UUID and replica authority.

## 5. Orders for General Satoshi III

Work in a separate worktree/branch while D1 runs. Do not commit or pull in the
three running worktrees because every arm checks revision invariance.

1. Repair `correction_probe_v2.py` so cases 140 and 145 execute current APIs;
   produce immutable before/after JSON. Do not reinterpret exceptions as
   passes.
2. Materialize D2 as the next no-idle job, parameterized by the D1 winner, so
   it can start immediately after independent aggregation. Do not assume the
   winner before D1 completes.
3. Add complete per-arm GPU telemetry and preserve all raw action/execution/
   cost diagnostics exposed by the simulator. Explicitly label unavailable
   cost components; never infer zero.
4. Convert the runtime recovery units and completion/collection operation into
   reviewed, tested, declarative deployment artifacts after D1. Prove no
   duplicate worker can start during recovery.
5. Prepare an automatic completion transaction: collect four packets on
   Omega, re-probe every remote replica, aggregate, notify the owner, and queue
   D2. A failed validation must alert and retain the GPUs for a bounded
   diagnostic/recovery job rather than silently idling.

No finding is closed by this document. Findings 159-164 require independent
correction evidence and owner disposition under the existing audit protocol.
