# Phase 1 Asset-Policy Results

Generated models, histories, logs and resolved configs are ignored by Git.
This tracked directory documents their stable logical layout. Physical output
is rooted at each machine overlay's `ARTIFACT_ROOT`, preventing concurrent
Gamma 5070 Ti and 5090 workers from sharing writable files:

```text
baseline/<asset>_<timeframe>_<policy>/
smoke/<asset>_<timeframe>_<policy>/
optimization/<asset>_<timeframe>_<policy>/
inference/<asset>_<timeframe>_<policy>/
```

Each optimization directory receives a candidate-history CSV, resume JSON,
statistics JSON, optimized-parameters JSON, exact champion checkpoint and final
resolved config/manifest. Consolidated comparable metrics belong in DOIN OLAP;
these local files are reproducibility evidence and resumable working state.
This directory intentionally excludes raw training logs, temporary checkpoints,
and large candidate histories. It retains only compact, reproducible selection
evidence such as `*_promotion_candidates.json`.

`solusdt_4h_promotion_candidates.json` is a validation-only chain
reconciliation snapshot. It is not a protected-test report, a promoted model,
or a release manifest.
