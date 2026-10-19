# Static RL execution queue

## Scope and method

2026-10-19 continuation of `RL_STATIC_METHOD_STATE.json`, not a replacement
method or admission authority. The canonical methodology file is absent in
this checkout; the method skill's portable checklist applies. Discovery and
unit-test design are complete for this bounded wrapper. Main owns source-clock,
replacement data, admission, full environment acceptance and consolidation.
Only this document, `tools/curated_rl_campaign.py` and its unit test are owned
by this change. No financial training is authorized by this document.

Reuse: governed replay config merge and metric collector; registered env,
agent and `rl_pipeline`; SAC policy-tensor identity; engineering runner atomic
status helper; data-gov `TerminalOutbox.put`, `collect_metrics` and
`collect_artifacts`. No new trainer, evaluator,
service, warehouse schema, DOIN path or delegation.

## Requirements and test design

| ID | Requirement | Evidence in test_curated_rl_campaign.py |
|---|---|---|
| Q1 | Existing pinned receipt and full population before model/env | admission and mutation refusals |
| Q2 | Sequential flat/temporal SAC plus flat/passive long/short/random | tiny real registered trainer integration |
| Q3 | Static validation, no learn, immutable checkpoint tensors | inference train/learn traps and hash parity |
| Q4 | Per-stage configs/checkpoints, bounded subprocesses | queue artifacts, timeout and failure tests |
| Q5 | Resume skips completion, refuses drift, status is read-only | resume, custody, lock and status tests |
| Q6 | Existing governed outbox, no service required | enqueue failure/retry without rerunning |

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

`publication` supplies an existing `campaign_sha256`, `deliveries` by split,
and an externally registered `unit_ids` map keyed by `seed/ARM/split`.
These are externally issued identities, not grants created by the wrapper.
Each terminal uses the existing governed schema and metric names; publication
means durable local enqueue, NOT warehouse acknowledgment or scientific
acceptance. Existing data-gov recovery owns later send/readback. No automatic
network retry or monitoring loop is introduced.

The queue order per seed is SAC_FLAT, SAC_PER_FEATURE_CONV1D, NO_TRADE,
PASSIVE_LONG, PASSIVE_SHORT, RANDOM, each with train then validation.
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
failure. No daemon, GPU, worker restart or live campaign root is used.
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

These commands require externally supplied admitted receipts and registered
publication identities. They are interface examples, not authorization to run
a financial campaign. Use a new private output directory outside the checkout
and live campaign roots; do not point at existing operational state.

```bash
python -m tools.curated_rl_campaign run \
  --contract <existing-engineering-contract.json> --output-root <private-run-root> --timeout 120
python -m tools.curated_rl_campaign resume \
  --contract <same-contract.json> --output-root <same-run-root> --timeout 120
python -m tools.curated_rl_campaign status --output-root <same-run-root>
```

The default executor starts a bounded, CPU-isolated child process per stage.
Set `DATA_GOV_CHECKOUT` to the existing data-gov checkout for its shared
publication helpers. No server is required to enqueue. There is no auto-send.

## Consolidation gate

Main integrated the method delta into `RL_STATIC_METHOD_STATE.json` and
registered the tool in `tools/TOOL_DECLARATIONS.json`. Financial execution remains blocked
on the existing admission/environment/cost/schedule requirements, not on a
new boolean approval synthesized by this wrapper.

## Verification and next action

Method delta: S9 component implementation, S10 synthetic integration and S11
bounded queue recovery verified. Financial alpha/release remains NOT STARTED.
The test-first PRE run failed collection because the wrapper did not yet exist.
Latest focused run: **63 passed**, no skips, CPU-only. This
includes 26 queue tests and the existing input, temporal-SAC and engineering
runner regressions. Both worker-entry integration cases invoke real registered
SAC training at four timesteps/two gradient updates on a synthetic environment;
the suite checks evaluator train/learn traps and saved/reloaded tensor parity.
The environment is a test double, not new full gym-fx accounting acceptance.
Real data-gov collectors/outbox are exercised locally, including a write that
succeeds before losing its acknowledgment, followed by idempotent resume.
No service, GPU, financial data fit, remote worker or DOIN operation was used.

Reproduce with the existing trading-stack interpreter and data-gov checkout:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
DATA_GOV_CHECKOUT=<existing-data-gov-checkout> python -m pytest \
  tests/unit/test_curated_rl_campaign.py tests/unit/test_curated_rl_input.py \
  tests/unit/test_temporal_sac_features.py tests/unit/test_static_engineering_checks.py -q
```

Next action: main supplies actual admitted
split/delivery/unit identities only after closing its source/replacement and
full environment acceptance gates. Governing classification, sealed TEST
release, broker-cost certification and warehouse send/readback remain outside
this wrapper's authority. Ruff is not installed in this isolated worktree;
no lint-clean claim is made.
