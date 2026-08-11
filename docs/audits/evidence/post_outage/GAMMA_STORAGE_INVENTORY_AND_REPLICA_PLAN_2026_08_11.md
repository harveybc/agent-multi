# Gamma Storage Inventory + Replica Plan (order §3.3)

Date: 2026-08-11 · Read-only inventory of
`/home/harveybc/Documents/GitHub/_pre_trading_stack_20260713` (219 GB,
dominant consumer of gamma's 89%-used root; 42 GB free). NOTHING was
deleted, pruned or rewritten.

## Inventory (read-only; nice/ionice; raw listings on gamma at `~/.local/state/agent-multi/storage_inventory_20260811/`)

- 249,434 files enumerated with size + mtime (`files.tsv`), du depth-2
  (`du_depth2.txt`).
- By repository: `financial-data` 178.3 GB (experiments 105.2,
  features 72.4, market_data 0.5), `agent-multi` 40.2 GB
  (experiments 36.2, .git 3.8), `doin-node` 0.24 GB, `gym-fx` and
  others < 0.2 GB.
- By artifact class: data 180.6 GB (157,092 files: csv/parquet/json/
  npz), models 32.9 GB (12,970 files: zip/pt/h5/keras), other 4.4 GB,
  logs/docs 0.25 GB, source 272 files, databases 5 files (0.02 GB).
- OLAP/chain databases located: 1 significant —
  `doin-node/examples/results/phase_1_daily/blockchain/chain.db`
  (14.7 MB, 2026-03-27): a LEGACY BLOCKCHAIN — byte-for-byte
  preservation mandatory (order §9 / non-negotiable 6).
- Historical experiment trees (`financial-data/experiments`,
  `agent-multi/experiments`) and model artifacts are treated as
  potentially unique until the replica is proven and the owner
  reviews; git-history representation analysis is part of the
  post-replica digest pass (the tree predates the trading-stack split,
  so most data/model artifacts are expected to be absent from current
  Git history).

## Replica plan (EXECUTING)

1. `rsync -a --bwlimit=25000` (25 MB/s ceiling — does not saturate
   live-trading connectivity) from gamma to
   `dragon:~/replicas/_pre_trading_stack_20260713_gamma_replica/`;
   dragon has 542 GB free ≥ 219 GB needed. Launched 2026-08-11 evening
   under nice/ionice; log at
   `~/.local/state/agent-multi/storage_inventory_20260811/replica_rsync.log`
   on gamma. ETA ≈ 2.5 h at the ceiling.
2. After completion: deterministic manifests computed INDEPENDENTLY on
   both sides (`find -type f | sort | xargs sha256sum` streamed to a
   manifest file), compared; any mismatch re-syncs and re-proves.
   Content digests therefore happen once per side, bound to the
   replica proof.
3. Only AFTER the proven replica: cleanup candidates limited to
   reproducible package/download caches (pip/conda caches, .git object
   packs are NOT candidates); any deletion of historical experiments,
   models, databases or the tree itself requires EXPLICIT OWNER
   AUTHORIZATION — none is requested here.
4. Dispatch disk-budget gate: the GPU readiness probe (P0 §3.2
   package) carries a typed `HOST_DISK_INSUFFICIENT` classification
   derived from expected job artifacts + reserve; insufficient space
   blocks that host only and alerts once. WP4 screen artifacts on
   gamma (8 cells × ~20 MB models + traces ≈ low GB) fit inside the
   42 GB free with the gate armed.
