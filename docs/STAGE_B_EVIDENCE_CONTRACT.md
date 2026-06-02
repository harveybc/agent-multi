# Stage B Return-Trace Evidence Contract

`agent-multi` is the *single* producer of Stage B per-bar return traces.
financial-data's DSR/PBO/Reality-Check evaluator is the *single* consumer.
This file pins the on-disk contract between them.

## Schemas

- Per-trace CSV: `stage_b_return_trace_v1` — defined by
  `pipeline_plugins._return_trace.TRACE_FIELDNAMES`.
- Per-trace metadata sidecar: `stage_b_return_trace_v1`.
- Run-level evidence index: `project3_return_trace_evidence_v1` —
  `pipeline_plugins._return_trace.EVIDENCE_SCHEMA_VERSION`.

Heldout boundary: `2025-01-01`. A trace containing any timestamp `>=`
this date is refused unless both `final_stage_c_evaluation` and
`stage_c_acknowledged` are true on the run config. Final Stage C mode
is *not* implemented here.

## Trace CSV column order (fixed)

```
step, timestamp, asset, timeframe, split, episode_id, run_id, seed,
bar_index, price, action_raw, position, reward, gross_return,
net_return, equity, pnl, commission_paid, slippage_paid, trade_cost,
trades
```

Allowed `split` values: `train`, `validation`, `test`, `train_epoch`,
`validation_epoch`, `evaluation`, `stage_b_validation`.

### Column semantics

- `step` is 1-indexed within the episode.
- `position` is the post-step position held during the next bar
  (`>0` = long, `<0` = short, `0` = flat). Exposure fraction in the
  evaluator is computed as the share of rows with `|position| > 0`.
- `reward`, `gross_return`, `net_return`, `pnl` are per-step values
  (NOT cumulative). The evaluator sums or compounds these as needed.
- `equity` is the absolute account equity at the end of the step
  (cumulative state, not a per-step delta).
- `commission_paid`, `slippage_paid`, `trade_cost` are per-step
  realized costs.
- **`trades` is CUMULATIVE within the episode.** It is sourced from
  `info["trades"]`, which gym-fx's `bt_bridge` populates as a
  monotonically non-decreasing counter (`trade_count += 1` on every
  `notify_trade(isclosed=True)`, reset to `0` at episode start).
  Consumers must take the last row's value as the final trade count
  for the episode — *not* the sum of the column. The financial-data
  evaluator's `infer_trade_count` auto-detects this by checking the
  series is non-decreasing and falls back to summing only for legacy
  per-step traces.

## Path layout

When the run config sets `return_trace_dir` (validation pipeline) or
`return_trace_file` (single-rollout pipeline), the following files are
emitted next to one another:

```
<run_dir>/return_traces/
    <split>_return_trace.csv          # one per split, fixed column order
    <split>_return_trace.csv.meta.json # sha256, config_hash, boundaries
    evidence.json                      # run-level index, schema above
```

`evidence.json` is the only file downstream tooling needs to discover
to reach every trace. Each entry in `evidence.json["traces"]` carries:

```
split, trace_file, trace_file_sha256, metadata_file, row_count,
first_timestamp, last_timestamp, contains_heldout_rows,
stage_c_authorized, episode_id
```

`trace_file_sha256` is taken over the raw bytes of `trace_file` at
write time; the evaluator re-hashes on load to detect tampering.

## Backward compatibility

A run config that omits `return_trace_dir` and `return_trace_file`
produces *no* trace, *no* metadata sidecar, and *no* evidence file.
This is the legacy default and is covered by
`tests/unit/test_return_trace.py::test_legacy_config_emits_no_evidence`.

## Locked run plans

`tools/project3_stageb_run_plan.py` expands a reference Project 3
config into one locked config per `(role, seed, cost_scenario)` cell.
Every emitted config sets:

```
_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED: true
_project3_stage_b_lock:               true
stage_c_access:                       "DENIED"
final_stage_c_evaluation:             false
stage_c_acknowledged:                 false
return_trace_dir:                     <run_dir>/return_traces
return_trace_file:                    <run_dir>/return_traces/evaluation_return_trace.csv
```

Default seeds: `0,1,2,3,4` (minimum 5 paired; override via
`--allow-too-few-seeds-for-smoke-test` marks the manifest
non-promotable). Default cost scenarios: `base`, `plus_50pct`,
`plus_100pct`. The legacy `pessimistic` scenario remains accepted for
old packets, but it is not sufficient for the pragmatic Stage B
diagnostic matrix.

Known deterministic baseline configs are wired to real no-training agent
plugins:

```
no_trade     -> no_trade_agent
buy_and_hold -> buy_hold_agent
random       -> random_agent
momentum     -> momentum_agent
reversal     -> reversal_agent
```

Unknown baseline names remain `template_only` and add the
`BASELINES_TEMPLATE_ONLY` promotion blocker.

The manifest lists `expected_evidence_file` per cell so a Stage B
reviewer can verify the contract before any run is launched.
