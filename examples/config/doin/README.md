# DOIN Trading Experiment Configs

These files are canonical experiment configs owned by `agent-multi`. They are
portable and contain no host, GPU, credential or customer-account settings.

The same experiment can run locally:

```bash
agent-multi --load_config examples/config/doin/trading_asset_solusdt_4h_sac_v1.json
```

For a machine-specific run, resolve the runtime overlay before constructing
the plugins:

```bash
agent-multi \
  --load_config examples/config/doin/trading_asset_solusdt_4h_sac_v1.json \
  --runtime_overlay configs/runtime/omega.json
```

The DOIN node later references this file from its domain's
`optimization_config` and selects the external `doin-plugins` entry points:

```text
optimization_plugin = trading_asset
inference_plugin = trading_asset
synthetic_data_plugin = trading_scenario
```

`default_optimizer` remains the local optimizer plugin. `trading_asset` is the
DOIN adapter around it, not a replacement or a second hidden optimizer.

The SOL file is a research vertical-slice seed, not a production champion. Its
data/config/artifact paths must pass the runtime overlay and real-cell smoke
gate before any node is launched.

The bounded local gate uses:

```bash
agent-multi \
  --load_config examples/config/doin/trading_asset_solusdt_4h_sac_smoke_v1.json \
  --runtime_overlay configs/runtime/omega.json
```

The smoke profile uses a chronological prefix, two candidates and one short
generation. It verifies integration, metric evidence and model portability;
its performance is not research evidence and must never enter selection.
