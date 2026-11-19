# 389: matched native SAC augmentation

Reset every arm to the same 299 bundle; preserve its selected71 fixed-TRAIN
input, grouped branches and fusion. Freeze through fusion in actor/online Q;
train the post-fusion attention, core and heads. Preserve the source SAC loss,
three actions, native costs and optimizer state. No ZigZag/control input.

Real environment collection remains online tick by tick. In +4/+8/+16
replace cumulatively 10% of sampled transitions with native synthetic TRAIN
transitions drawn uniformly from the nested 387 paths. Materialize each
synthetic path with the same stochastic frozen behavior policy and random
stream, independently reset its account; cache genuine next state/reward/
elapsed steps/terminal. This is off-policy augmentation, not a synthetic
online account or an expert demonstration. Static banks have equal behavior
and sampling rules across augmentation arms. Real-only has no replacement.

Fixed 80 complete real epochs per arm, seed 7 primary, 19 replication, same
update cadence, warm-up, batch and check frequency. Best eligible checkpoint
on real validation including epoch zero, report all epoch curves; fixed
budget rather than stopping different arms on different epochs. Compare both
seeds, never select the more favorable seed. No changes to TEST/live.

Acceptance: verify cohort and file identities, exact transition packing,
no crossing a synthetic episode, equal counts at common epochs, initial
native accounts, recovery at epoch boundaries, training only downstream of
fusion, model held constant on disk, and per-host RAM admission.

## Launch

The four original paths yielded 24,051 transitions each, with 2,283,
2,279, 2,267 and 2,276 filled trades respectively under stochastic collection.
Those are bank generation receipts, not the profit or activity of a trained
arm. Real refits use actual online native account states. Synthetic samples
are gathered from fixed market windows using a path-local successor index.
Unit tests verify mixture counts, actions, successor/reward/duration/terminal,
and RNG recovery. All 58 downstream trainable tensors (16,431 scalars)
are present in the original optimizer. Code upstream of fusion stays frozen;
it is evaluated from the actual 71-channel windows for actor and twin Q,
not by substituting actor fusion tensors for critic fusion tensors.

The restartable coordinator admits workers using **system RAM**, reserving
the unused part of sibling worker limits and 2 GiB of headroom. Each worker
also has a hard cgroup RAM cap. Durable owner, progress and final comparison
are written separately from the model. Failure on a live boot remains visible
instead of silently substituting a seed, resetting weights or restarting forever.
Primary seed 7: all four workers launched; seed 19 is queued automatically.
