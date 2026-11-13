# Native Critic Head Capacity Contrast

Owner authorizes launch after319. Four new fits:318 folds A/B x seeds19/43,
base weight decay0.01 only. Reuse the four sealed318 controls without training
them again. Source actor299,71 inputs, annual labels, splits, MSE, LR, schedules,
batch128,4096 maximum updates,64-update checks and patience20/delta1e-8 unchanged.

## Single Factor

Native architecture: one208->8->3 head per critic twin,3398 total parameters.
Control: three such heads per twin averaged,10194 parameters. Construct the
same initialized three-head ensemble as318, verify its digest against the
sealed control, then retain head0 independently in each twin. Do not simply
construct count1 with the same seed: that changes twin1's RNG position.
Common tensors match exactly; initial functions differ because an average of
three heads is not head0. Record that difference. Optimization geometry also
changes with averaging; results cannot isolate parameter count from it.

Reconstruct each control checkpoint's metrics with the same cached embeddings
before fitting. Reject mismatched source, labels, support, folds, initial
tensors, schedules or control metrics. Select native checkpoints using only
the original FIT stopping block. Audit only after both endpoints are fixed.
Save selected and fixed checkpoints, cold-reload parity, all metrics and curves.

All four stopped audit regrets must improve over their matched controls and
executable-opening regret must not worsen to call this consistently helpful.
Also report fixed-budget endpoints and the preserved actor. No automatic actor
update, phase2 pass, live deployment or representation/feature change.

## Test Design

| Requirement | Evidence |
| --- | --- |
| Common initial weights | Native output equals ensemble head0 for each twin; exact source digest |
| No dormant parameters | Exactly2 heads; parameter count is one third; both receive gradients |
| Source immutable | Copy ownership test, source tensors and full TRAIN actor outputs exact |
| Reuse authentic controls |318 seals, reconstructed endpoints and schedule identities |
| No temporal selection leak |318 blocked folds and tested RegretStopper unchanged |
| Autonomous execution | Four sequential resumable arms, lock, progress, failed/completed receipts |
