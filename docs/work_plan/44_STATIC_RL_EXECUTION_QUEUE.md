# Static RL execution queue

## Scope and method

2026-10-19 continuation of `RL_STATIC_METHOD_STATE.json`, not a replacement
method or admission authority. The canonical methodology file is absent in
this checkout; the method skill's portable checklist applies. Discovery and
unit-test design are complete for this bounded wrapper. Main owns source-clock,
replacement data, admission, full environment acceptance and consolidation.
Only this document, `tools/curated_rl_campaign.py`, its unit test and the
explicitly authorized `tools/rl_warehouse.py` helper are owned by this change.
This document does not issue data admission or authorize a financial launch.

Reuse: governed replay config merge and metric collector; registered env,
agent and `rl_pipeline`; SAC policy-tensor identity; engineering runner atomic
status helper; data-gov `TerminalOutbox.put`, `collect_metrics` and
`collect_artifacts`. No new trainer, evaluator,
service, warehouse schema, DOIN path or delegation. The optional direct-report
mode reuses predictor's existing `tools.i6dmo.outbox.Outbox` and `HttpWarehouse`.

## Requirements and test design

| ID | Requirement | Evidence in test_curated_rl_campaign.py |
|---|---|---|
| Q1 | Existing pinned receipt and full population before model/env | admission and mutation refusals |
| Q2 | Sequential flat/temporal SAC plus flat/passive long/short/random | tiny real registered trainer integration |
| Q3 | Static validation, no learn, immutable checkpoint tensors | inference train/learn traps and hash parity |
| Q4 | Per-stage configs/checkpoints, bounded subprocesses | queue artifacts, timeout and failure tests |
| Q5 | Resume skips completion, refuses drift, status is read-only | resume, custody, lock and status tests |
| Q6 | Existing governed outbox, no service required | enqueue failure/retry without rerunning |
| Q7 | Explicit real-development scope, sealed CPU/CUDA execution | device/evidence refusals, controls CPU |
| Q8 | Pinned gym namespace and registered real environment | both SAC encoders, real Env.make_env |
| Q9 | Physical predictor outbox without fake Flow-v3 registration | offline idempotent SQLite enqueue |

Synthetic receipts in tests describe generated fixtures only. They do not
admit any real population. No TEST role is opened; this first queue handles
TRAIN and VALIDATION only. A sealed test release requires a separate protocol.

## Execution contract

The externally supplied JSON has schema `curated_rl_campaign.v1`,
`classification: NON_GOVERNING`, unique integer `seeds`, a flat `common_config`,
and exactly `splits.train` and `splits.validation`. Each split supplies
`input_data_file`, `curated_input_manifest`, `curated_input_manifest_sha256`,
`feature_columns` and `curated_input_split`. Existing admission validation
checks the digest, lineage, features, support and bytes, not just a status flag.
Common config must seal positive `total_timesteps`, `training_transition_cap`
and `evaluation_transition_cap`, `train_freq=1` (the existing default),
`action_space_mode=continuous`,
`continuous_action_contract=target_exposure_hysteresis_v2` and
`max_entries_per_day=1`. It must not enable optimization, warm starts,
oracle fitting or remote logging. This engineering queue refuses a governing
classification rather than inventing release acceptance.

The legacy `publication` mode (omitted or `FLOW_V3`) supplies an existing `campaign_sha256`, `deliveries` by split,
and an externally registered `unit_ids` map keyed by `seed/ARM/split`.
These are externally issued identities, not grants created by the wrapper.
Each terminal uses the existing governed schema and metric names; publication
means durable local enqueue, NOT warehouse acknowledgment or scientific
acceptance. Existing data-gov recovery owns later send/readback. No automatic
network retry or monitoring loop is introduced. Live Flow-v3 deliveries must
belong to each registered unit and satisfy the campaign's full dataset set;
shared split deliveries are not a substitute for per-unit registration.

### Real-data development option

Keep `classification: NON_GOVERNING`; set
`scientific_scope: REAL_DATA_DEVELOPMENT` and
`execution_backend: local_subprocess`. This scientific scope is not a
governance enum or a release claim. Main owns the concrete design and
materializer; the minimal additional fields are:

```json
{
  "scientific_scope": "REAL_DATA_DEVELOPMENT",
  "classification": "NON_GOVERNING",
  "execution_backend": "local_subprocess",
  "seeds": [7],
  "arms": ["NO_TRADE", "PASSIVE_LONG", "PASSIVE_SHORT", "RANDOM", "SAC_FLAT", "SAC_PER_FEATURE_CONV1D"],
  "execution": {
    "device": "cuda",
    "gym_root": "<absolute-pinned-gym-checkout>",
    "gym_code_commit": "<40-hex-commit>",
    "task_timeout_seconds": {"train": 14400, "validation": 3600},
    "max_attempts_per_task": 2,
    "gpu_evidence": {
      "path": "<absolute-existing-heartbeat>",
      "sha256": "<64-hex-digest>",
      "assigned_gpu_uuid": "<explicit-assigned-GPU-UUID>",
      "max_age_seconds": 86400
    }
  },
  "publication": {
    "mode": "WAREHOUSE_REPORTS",
    "predictor_root": "<absolute-pinned-predictor-checkout>",
    "predictor_code_commit": "<40-hex-commit>",
    "code_commit": "<agent-commit>",
    "experiment_set_key": "rl-raw-eurusd-static",
    "actor": "canonical-rl-curation",
    "lake": "curated-core-prices",
    "resources": {"train": "raw-rl-eurusd/train.csv", "validation": "raw-rl-eurusd/validation.csv"},
    "pump_between_stages": true,
    "warehouse_url": "<operator-configured-existing-warehouse-tunnel>",
    "token_env": "DATA_GOV_LAKE_TOKEN",
    "pump_timeout_seconds": 300
  }
}
```

Existing schema, common config and admitted split fields above remain required.
CPU execution is explicit `device: cpu` and needs no GPU evidence. For CUDA,
the parent must expose exactly the assigned UUID. Existing pinned readiness
evidence and the worker's actual PyTorch build, device UUID and model parameter
device must agree; failures never fall back to CPU. Controls always use CPU
children with CUDA masked. No readiness collector or GPU probe runs in the
queue parent. The fixed 100000 budget is SAC environment transitions, not a
claim of 100000 optimizer updates; no early stopping or optimization is added.

Gym Python sources must match the pinned commit. The worker preserves agent
`app.main`, binds the pinned gym `app` namespace before environment loading,
and verifies actual gym and registered plugin module origins. Each attempt
overrides `session_flatten_custody_root` to its own `session_custody` directory.
Main supplies the analytical instrument fingerprint and admitted calendar
identity; the wrapper does not manufacture broker identities or calendars.

`WAREHOUSE_REPORTS` requires no Flow-v3 campaign, registration or delivery IDs.
Its campaign identity is the exact contract file SHA; optional `unit_ids` are
explicit local labels, otherwise task IDs are used. A separate interpreter
pins and imports predictor's existing publisher without polluting agent module
namespaces. The operational output root must be ext4 and physically contains
`warehouse-outbox.sqlite`, not a symlink to a shared outbox. Main can mirror
the closed root later. No per-row prices are uploaded. Reports preserve all
finite numeric summary metrics, known undefined trading metrics as null,
compute evidence and nested execution diagnostic counters. Units distinguish
USD, percent, fraction, ratio, count, transitions and seconds. Tags include
input policy, cost scope, RAW feature-selection incompleteness, contract,
config, summary and checkpoint hashes. Dataset lineage is explicitly
UNVERIFIED in the warehouse protocol, not a false Flow-v3 verification claim.

Enqueue is local and offline. Optional HTTP pumping occurs only between stages
or via the separate `pump` command, never in a learning step. Tokens are read
only by the existing HTTP publisher when pumping; this change never reads an
operator EnvironmentFile. `WAREHOUSE_STATUS.json` contains last-known pump
evidence; queue ENQUEUED state is not remote acknowledgment.

The legacy queue order per seed is SAC_FLAT, SAC_PER_FEATURE_CONV1D, NO_TRADE,
PASSIVE_LONG, PASSIVE_SHORT, RANDOM, each with train then validation.
Real-development defaults to NO_TRADE, PASSIVE_LONG, PASSIVE_SHORT, RANDOM,
SAC_FLAT, SAC_PER_FEATURE_CONV1D. Explicit `arms` must contain exactly all six.
Controls' train hook is the existing no-op initialization/save interface.
Both evaluation roles use the same continuous-action contract and execution
constraints. Passive short is a sign adapter on registered buy_hold_agent;
it does not replace execution rules. The random agent's action space is seeded
explicitly. Training-pipeline evaluation is retained as TRAIN diagnostics,
never validation selection. No checkpoint ranking or automatic promotion.

Resume is stage-level, not partial SB3/replay-buffer continuation: interrupted
training retries in a new attempt directory; completed training is never
relearned to retry validation or publication. An OS advisory lock serializes
writers; its descriptor is inherited by the worker so parent death cannot
unlock a still-running child. It releases after the last holder exits.
A subprocess timeout records a typed
failure. Real-development timeouts are sealed per role; CLI timeout overrides
are refused. At most two attempts are admitted: initial execution plus explicit
resume after failure. No daemon, automatic worker restart or monitoring loop
is introduced.
Each finished attempt retains `EXECUTION.json`, including failures and cost,
even when a later attempt succeeds. Failed/interrupted attempts are local
engineering evidence; only completed stages are enqueued as result terminals.
Resume also refuses runner-source, contract, checkpoint, summary, trace and
effective-config drift. The checkpoint's policy identity is its tensor digest;
the archive's byte digest is only a transport-integrity check. Input files must
remain immutable while consumed. Context-overlapping receipts are refused
until a separate scored-origin protocol exists. The pre-existing 2025 sealed
support firewall is checked before training, without granting a waiver.

The `status` action reads only the local journal, never probes a process or
service, and returns typed UNAVAILABLE for absent/unreadable state. Enqueued
means last-known local custody, not a claim about remote acknowledgment.

## Commands

These commands require externally supplied admitted receipts and appropriate
publication configuration. They are interface examples, not authorization to run
a financial campaign. Use a new private output directory outside the checkout
and live campaign roots; do not point at existing operational state.

```bash
python -m tools.curated_rl_campaign run \
  --contract <existing-engineering-contract.json> --output-root <private-run-root> --timeout 120
python -m tools.curated_rl_campaign resume \
  --contract <same-contract.json> --output-root <same-run-root> --timeout 120
python -m tools.curated_rl_campaign status --output-root <same-run-root>
```

For real development omit `--timeout`; role budgets come from the sealed contract.
The separate direct-report pump is:

```bash
python -m tools.curated_rl_campaign pump \
  --contract <same-contract.json> --output-root <same-run-root>
```

The legacy default executor starts a bounded, CPU-isolated child per stage.
Set `DATA_GOV_CHECKOUT` to the existing data-gov checkout for its shared
publication helpers. No server is required to enqueue. Legacy mode has no auto-send.

## Consolidation gate

Main integrated the method delta into `RL_STATIC_METHOD_STATE.json` and
registered the tool in `tools/TOOL_DECLARATIONS.json`. Financial execution remains blocked
on the existing admission/environment/cost/schedule requirements, not on a
new boolean approval synthesized by this wrapper.

## Verification and next action

Method delta: S9 component implementation, S10 synthetic integration and S11
bounded queue recovery verified. Financial alpha/release remains NOT STARTED.
The test-first PRE run failed collection because the wrapper did not yet exist.
Latest focused run: **79 passed**, no skips, CPU-only (30.06 seconds). This
includes 42 queue tests and existing input, temporal-SAC and engineering
runner regressions. Test-first real-development cases initially failed for
missing execution/report interfaces. Existing worker tests invoke real SAC on
synthetic test environments; two additional worker integrations invoke real
registered `Env.make_env` from the pinned gym checkout on small generated
source fixtures, for both SAC encoders at four training timesteps. Evaluator
train/learn traps and saved/reloaded tensor parity remain covered. Full market
acceptance remains main's separate evidence, not a claim from these tiny tests.
Real data-gov collectors/outbox are exercised locally, including a write that
succeeds before losing its acknowledgment, followed by idempotent resume.
Actual predictor Outbox enqueue is tested offline with a physical SQLite file,
duplicate report identity and zero HTTP attempts. No service, GPU, financial
data fit, remote worker, credential read or DOIN operation was used.

Reproduce with the existing trading-stack interpreter and data-gov checkout:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
DATA_GOV_CHECKOUT=<existing-data-gov-checkout> \
GYM_FX_CHECKOUT=<pinned-gym-checkout> \
PREDICTOR_CURATION_RUNTIME=<pinned-predictor-checkout> python -m pytest \
  tests/unit/test_curated_rl_campaign.py tests/unit/test_curated_rl_input.py \
  tests/unit/test_temporal_sac_features.py tests/unit/test_static_engineering_checks.py -q
```

Next action: main consolidates and launches its admitted raw-development
contract in its isolated runtime, then verifies actual control metrics and
warehouse acknowledgment. Direct reports avoid per-unit Flow-v3 registration;
they do not confer governing classification, sealed TEST release or broker-cost
certification. Implementation is frozen for the first launch. Ruff is not installed in this isolated worktree;
no lint-clean claim is made.
