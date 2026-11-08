# Selected Weight-Preserving Refinement

Owner priority2026-11-08: primary RTX5090, preserve the successful frozen model
rather than relearn all weights. The primary GPU was idle; no unrelated worker
was interrupted. The secondary replication queue continues unchanged.

## Source and Contract

Use the76-input epoch6 checkpoint from structural run
`8145612ef9f01c9de669fcb31189744cac20f538eafd2b6803ebba4a8a432f72`, with the70-input
selection `9be13b93b3d77efe644571313aa744e591bdc1ab8e6ccb47f216294540ada02b`.
Reference validation:13.80849% return,2.09706 Sharpe,206 orders,57.76699% wins,
3.62493% maximum drawdown. Do not initialize from the failed fresh70-input run.

All28 groups survive. Remove the six nulled input connections by slicing only
the input-channel axis of the first convolution in affected branches. Copy every
other tensor exactly, including actor, twin critics, delayed critics and log_alpha.
Remap index buffers to the70-column input recipe. Fusion slots, positional
attention, core, heads and retained scaler constants are unchanged. If an entire
group disappeared this adapter would refuse rather than silently delete a branch
whose biases can produce a nonzero constant response.

## First Experiment

- Freeze actor and online-critic extractors, fusion, attention and core; freeze
  transferred temperature. Train actor and twin-critic heads only. Verify every
  frozen parameter remains byte-identical after each epoch.
- Delayed critics inherit the original delayed values and retain standard soft
  updates. This is distinct from resetting them to the online critics.
- Learning rate0.00003, ten times below the source. Fresh AdamW moments and fresh
  replay65536, batch128, update every4. First4096 replay transitions use the
  transferred stochastic policy; updates begin after that fill. No random-policy
  replacement, no synthetic observations and no state feedback.
- Full TRAIN2020-2023 and VALIDATION2024,13 observations/stride2, unchanged native
  strategy, execution costs, risk and calendar. No TEST access.
- Maximum2000 epochs, validation return/week, min_delta1e-8, patience20 with the
  existing first20-epoch patience grace. Store an eligible epoch0 as initial best.
  Only checkpoints with strictly>2orders/week in both periods can replace it or
  reset patience. A higher return with no activity is explicitly rejected.
- Preserve the original reference, transferred epoch0, per-component tensor
  archives, best inference bundle and recovery state. No automatic unfreezing or
  second architecture experiment; evaluate this isolated head-only refinement first.

## Tests and Admission

| Requirement | Check |
|---|---|
| Exact retained weights | tensor-by-tensor copy/projection assertions |
| Coherent SAC state | actor, both online Q and both delayed Q compared on projected inputs |
| Same initial strategy | full-support policy actions identical and native TRAIN/VAL accounts exact |
| No false winner without orders | eligibility before selection/patience, tested inactive counterexample |
| Epoch0 never discarded for degradation | synthetic training fixture preserves eligible epoch0 |
| Frozen means unchanged | trainable partition test and epoch-level byte equality |
| Reusable weights | full bundle reload test; separate role/component archives |

CPU verification before deployment:36 passed,6 fixture-dependent skips. CUDA
acceptance, full real-data transfer and account parity are required by the worker
before any optimization step; their receipts are stored with the run. Numerical
logit parity allows atol1e-5/rtol1e-5 but requires zero action disagreements; Q
parity allows atol1e-6/rtol1e-5. Retained tensor values themselves are exact.
