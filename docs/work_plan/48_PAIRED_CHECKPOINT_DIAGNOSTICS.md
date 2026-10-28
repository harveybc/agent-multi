# Frozen-Support Checkpoint Diagnostics

Owner approved 2026-10-19: inspect existing full TRAIN and full VALIDATION
checkpoint trajectories. Never choose a different year or replace the official
checkpoint. No fit, TEST, learner changes or new temporal experiment.

Stage: requirements and unit/integration test design complete before code.
The analyzer uses stdlib only. It replays the declared mean objective and
patience, verifies artifact hashes, recomputes growth from complete equity
traces and enforces unchanged timestamp support and input identity per role.
Complete pairs of an active campaign form a dated partial snapshot; publication
uses a new diagnostic identity and the existing governed outbox/readback.
It writes only outside source campaigns and retains a second physical copy.

| Requirement | Evidence |
| --- | --- |
| D01 fixed full support | timestamp digest, input identity and row/day counts |
| D02 exact objective | hand-computed reset, threshold and patience examples |
| D03 distinguish prior step/incumbent | synthetic nonselected intermediate point |
| D04 numeric/artifact integrity | nonfinite, timezone, order and corrupt-hash rejection |
| D05 preserve source and chosen model | read-only source, new result directory, official best replay |
| D06 provenance and persistence | source hashes, script hash, OLAP readback and mirror |

For every reset compare validation to both the preceding evaluated checkpoint
and the previously selected incumbent. Report both counts; they answer different
questions. Deterioration is a numerical directional observation, not a
significance test. No fitted trend, hindsight selection or causal conclusion.
The daily unit is the existing monitor's observed New York calendar date,
including observed partial-session dates; do not silently redefine weekdays.

## Executed Return

2026-10-19 18:40 UTC: 48 complete checkpoint pairs verified and diagnosed,
20 baseline + 20 dropout + 8 L2 (partial through40000). All paired growth,
support, selected checkpoints and patience decisions replayed.17 focused
tests cover arithmetic, threshold, prior-vs-incumbent distinction, source
immutability, corrupted summaries, forged patience and resigned changed support.
An initial invalid OLAP experiment-key format was rejected before publication;
fixed with deterministic valid keys and a regression.48/48 diagnostic reports
then delivered with readback, no pending/mismatch; closed SQLite backups and
JSON artifacts copied to the second physical machine with matching content.

Baseline: 5 of9 resets after the first checkpoint worsened VALIDATION versus
the previous selected incumbent; nevertheless the final100000 checkpoint was
also the best observed validation return (+0.751879%). This is not evidence
that stopping earlier would have improved this run. Dropout:2 of3 such resets;
0/20 checkpoints profitable on full validation. L2:2 of2;0/8 profitable so far.
These are observed trade-offs, not causal attribution or significance tests.

Reproduce with CPU only, external output directory:

```bash
CUDA_VISIBLE_DEVICES='' python -m tools.paired_checkpoint_diagnostics \
  --root <campaign-root> --output <diagnostic-root>
```

Add `--publish` only in the configured publication environment. It uses the
existing warehouse client and a separate outbox; source artifacts are read-only.
An active campaign yields a content-addressed snapshot of completed pairs.
The tool does not train, select a new champion, alter timers or consume TEST.
