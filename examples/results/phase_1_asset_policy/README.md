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
