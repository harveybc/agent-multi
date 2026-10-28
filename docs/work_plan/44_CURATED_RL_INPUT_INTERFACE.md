# Curated RL input interface

This receipt is supplied by the data-preparation/admission pipeline, not by
the training agent. No receipt for the recovering financial population has
been issued. The adapter adds content checks; it does not certify a source
clock, a holiday calendar or a feature-selection method from a boolean flag.

The existing `env_plugins/gym_fx_env.Plugin` calls
`app.curated_rl_input.validate_curated_rl_input` before constructing the
environment. Set `require_curated_rl_input=true`, `curated_input_manifest`,
`curated_input_manifest_sha256` (from admitted inventory), `curated_input_split`,
`input_data_file` and ordered `feature_columns` in the effective flat config.
The existing governed runner preserves these fields and never truncates an
admitted input to its legacy 384-row replay default.

Receipt fields:

| Field | Meaning |
|---|---|
| schema | `curated_rl_input.v1` |
| status | `ADMITTED`, issued only after curator acceptance |
| availability_verified | true only after point-in-time verification |
| open_session_coverage_verified | true only after required calendar support reconciliation |
| input_sha256 | SHA256 of the prepared CSV bytes |
| split | Exact role matching configuration, e.g. train or validation |
| feature_columns | Ordered, unique selected numeric column names |
| selection_sha256, calendar_sha256, inventory_sha256 | Producer's immutable upstream identities |
| timestamp_semantics | `completed_bar_available_at` |
| period_start, period_end | Timezone-aware interval `[start,end)` including declared context |
| row_count | Actual required full population size |
| expected_timestamps_sha256 | SHA256 of concatenated expected UTC timestamps, each `datetime.isoformat()` followed by LF |

CSV includes DATE_TIME (or configured date column), OPEN/HIGH/LOW/CLOSE and
the selected feature columns. Timestamps must be timezone-aware, unique and
strictly increasing. Numeric inputs must be finite; OHLC values must be positive
and internally ordered. No fillna, interpolation or padding repairs a bad file.
The expected timestamp digest is generated from required admitted support,
not from the potentially incomplete delivered file merely to make it pass.
For context before a scored split, declare the context-inclusive receipt
interval; the evaluation driver must separately gate scored origins.

On admission failure no environment or model is built. Successful identity
travels in the ordinary RL summary and governed terminal, with existing data-gov
publication. A config pin protects against changed bytes, not a malicious
producer able to rewrite the trusted config and receipt together. Files must
remain immutable during consumption; this is not a snapshot filesystem.

The governed replay now accepts `replay_mode=inference` for fixed-policy
evaluation with `load_model`, instead of unconditionally invoking train mode.
Its existing CPU/compute limits remain in force. Scientific GPU dispatch and
separate train/validation scoring are still campaign work, not implied by this
small runner change. No new warehouse schema, secret or service is required.

## Metric transport

The existing collector now includes profit, equity, return, drawdown, Sharpe,
closed-trade count, wins/losses, win percentage, profit factor, closed-trade
costs, submitted/filled entries and maximum filled entries in a business day.
Null or nonfinite ratios are not sent as fake numeric zeros. The retained
summary keeps their nulls and authoritative trade-stream provenance.
Sharpe annualization, Sortino, turnover and total costs including open positions
still require the timestamped equity/accounting acceptance step; this change
does not claim those are newly certified.

## Reproduce component tests

Use the dedicated trading-stack environment, CPU and one BLAS/OpenMP thread.
In agent-multi:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  tests/unit/test_curated_rl_input.py tests/unit/test_temporal_sac_features.py \
  tests/unit/test_sac_warm_start_expansion.py tests/unit/test_sac_normal_finetune.py \
  tests/test_governed_input_is_consumed.py tests/test_governed_replay_custody.py \
  tests/test_compute_contract_publication.py -q
```

In the matching gym-fx checkout:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  tests/test_daily_entry_quota.py tests/test_strided_observation.py \
  tests/test_feature_window_preprocessor.py tests/test_protected_order_execution.py \
  tests/test_shared_execution_envelope.py tests/test_trading_metrics.py \
  tests/test_session_exposure_env.py tests/test_trade_reconciliation_stream.py -q
```

These are synthetic engineering tests, not financial scores. Do not run a legacy
sample config on the affected data to substitute for an admitted experiment.

Additional cross-checkout integration (two arms, real gym-fx + SB3, four
synthetic gradient steps, save/load and unchanged policy tensors throughout
subsequent evaluation): set `GYM_FX_CHECKOUT` to the matching checkout and add
`tests/integration/test_curated_temporal_gym_bridge.py` to the agent-multi test
command. The child process checks it loaded that checkout, not an installed
older environment. On 2026-10-18: 62 agent-multi and 191 gym-fx focused tests
passed, including these two integration cases. No GPU or financial fit.
