# 377: CPU component and policy observer

Owner asked to use spare resources while374/376/367 train. All four GPUs have
active workers; low sampled utilization and free VRAM do not mean an idle
training slot. Use one capped CPU process instead of duplicating fits.

NON_GOVERNING diagnostic. At hourly intervals for at most96 hours, authenticate
copied histories and tensor snapshots of the current374 runs and376 branches.
Take the first completed epoch, multiples of20 and latest completed epoch,
independent of return. On a fixed256-window chronological TRAIN-only panel,
measure actor margins, action changes, entropy, twin-critic disagreement and
component drift. Inspect stable rank per weight matrix. Existing TRAIN/VAL
metrics are read from authenticated histories; no account is re-evaluated and
no weights, source files or live services are modified.

| Requirement | Test or runtime guard |
|---|---|
| No GPU/training | CUDA hidden, CPU model construction, no model.update, bounded CPU/RAM |
| No winner-picked epochs | Fixed first/multiples20/latest selector |
| No torn or altered tensors | Snapshot digest from copied HISTORY; identity checks; atomic output |
| Comparable probes within run | Same ordered256 TRAIN windows and support hash |
| Real module classification | Use module_key including post_fusion attention |
| Correct matrix metrics | Zero, rank1, identity, scaled and batched matrix tests |
| No false causal diagnosis | Rank is not capacity; time-associated drift is not proof of cause |
| No repeated expensive analysis | Per-history and code-keyed cache, immutable snapshot copies |

This is not an activity-matched control and cannot promote a model, close
phase2 or prove lack of data. Its purpose is to prepare component-local
evidence while the paired learning-rate treatment and history curves finish.
Each run uses its first available completed snapshot as reference;376's first
post-fork epoch is102, not the pre-fork101. Cross-arm intervention conclusions
belong to376, which preserves the common101 anchor.

## First execution

Completed2026-11-19T00:03Z:17 checkpoints across four available runs, four
plots, zero collection errors. The fifth configured run is the queued376
treatment and is explicitly WAITING_FOR_FIRST_COMPLETED_EPOCH, not a failure.
Five focused tests and correctness lint pass. Enabled CPU-only service:
75% of one core quota,3GiB memory cap, hourly refresh,96-hour maximum.
Observed first-pass CPU time about11.3 seconds and memory about692MiB.

Illustrative descriptive findings on256 fixed TRAIN windows per run:

| Run | Epochs | Critic/actor disagreement | Validation return |
|---|---|---|---|
| six months,seed19 |20 to101|23.0% to7.8%|-25.98% to-56.48%|
|48months,seed19|9|66.0%;median top-two probability margin0.0058|-22.38%|
|376 control|102 to108|10.2% to10.9%;8.2% action changes|-55.56% to-55.36%|

Greater actor/critic agreement on TRAIN is not evidence of out-of-sample
utility. Small probability margins in the early48-month run warrant tracking
decision sensitivity, but do not establish the cause of later divergence.
No training policy or checkpoint was changed from this descriptive reading.
