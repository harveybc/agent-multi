# History397: frozen-prefix cache qualification

Scope: speed experiment, not a new policy or a feature-selection result. The
reference is epoch 1 of history397, not an epoch after replay warmup. Do not
replace a production model or infer a change in trading quality from this test.

| Requirement | Check | Evidence |
|---|---|---|
| Same starting model and TRAIN support | Original bundle loader, canonical freeze flags, row mapping restricted to valid TRAIN windows; no validation cache | 397_SAC_LOSS_CACHE_GATE.json |
| Separate actor, online twins, target twins | Five tables of fused features before attention. For Q, both the data and parameter axes are mapped by twin. Both targets retain their own table. | tools/loss_cache397.py, logits and twin/target deltas |
| No change to stochastic learning | Paired RNG, online dropout on, targets in evaluation mode, native SAC update. Compare logits, individual Qs, losses, gradients and model state across 16 successive paired batches. | 397_SAC_LOSS_CACHE_GATE.json |
| No change to accounting or learning budget | Run original epoch driver. Replay stores row IDs instead of transmitting duplicate windows; native position engine and strategy are unchanged. Evaluate raw TRAIN and validation without lookup. | EPOCH_PARITY.json and COMPARISON.json in the experiment output |
| Negative controls | Detect swapped buy/sell counts, changed loss with unchanged metrics, slower run when setup is included | tests/test_loss_cache397_receipt.py |

The epoch receipt requires exact equality of all reported metric fields, update
counts, mean and last update diagnostics, plus a wall-time saving even after
charging the full cache setup to the first epoch. It does not certify identical
orders tick by tick or bitwise equality of the final historical weights (the
reference did not retain an epoch-1 checkpoint). The two experiments run on the
same GPU at different times, not concurrently.

The cache is valid only for these windows, group order, scaler, and unchanged
frozen prefixes. It is rebuilt in-process, never reused across model or data
identities. Frozen state is hashed again after the epoch. This experiment's
row-ID replay is not interchangeable with raw-window replay. For another
recipe, a thawed or stochastic prefix, or changed target parameters, run a
fresh qualification; do not enable this path by default.
