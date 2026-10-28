# V2 behavior audit and next contrast

Owner authorized continuation after the completed v2 results. All 24 fits
completed first attempt; 72 reports were authenticated and published/read back.
All four synthetic GPU controls passed. No three-seed arm met the joint
activity/behavior requirements; this does not invalidate the recorded metrics.

## Questions and frozen scope

1. When do directions/close choices collapse across the four stored checkpoints?
2. Do logits vary over identical market support, and respond to temporal order?
3. Are voluntary exits economically useful versus retaining the SAME open trade
   until its native SL/TP/forced exit, with its original units and costs?

Read only TRAIN and INNER_A trading evidence and model probes. Do not choose a
checkpoint, read TEST, retrain, modify the closed campaign, add feedback or
change input transformations. The new audit tool runs outside the frozen
runtime, which authenticates itself against the campaign spec. Results identify
both the tool digest and original artifacts. OBS_2024 is not used for this
diagnostic decision.

## Tests before implementation

- Aggregate reconciles gross minus explicit entry/exit costs with net; an empty
  population gives undefined ratios, not zero evidence.
- Refuse inconsistent trade accounting, timestamps and out-of-period indices.
- Native long/short favorable/adverse fixtures: replay the actual close exactly,
  hold until protection or force-flat, demonstrate both beneficial and harmful
  voluntary exits. Do not mutate the original engine/trades.
- Shared support hashes bind checkpoint probes; archive hashes and native reports
  authenticate before use. Probes are CPU diagnostics, not bitwise GPU replay.
- Temporal reversal is only a sensitivity probe, never a model input treatment
  or a causal performance estimate.

## Interpretation

Per-trade HOLD counterfactuals overlap and ignore the subsequent entry sequence.
Their deltas are NOT a portfolio return or an implementable oracle. Actual
market-policy metrics remain unchanged. A single-seed result is not robustness.
Select the next narrow contrast from the diagnostic evidence, holding the
admitted observations/business fixed. No fit is authorized merely by a favorable
counterfactual or by forcing a balanced number of BUY and SELL commands.

## Measured first phase and next frozen contrast

24 units/96 checkpoints audited; 75 evidence files copied and hash-verified.
The pure native close replay passes for all recorded voluntary exits. Cases
with 100% one-hour trades have costs exceeding positive gross returns. For
INNER, DSR seed 7 has 28 beneficial/41 harmful early exits, mean close-minus-HOLD
-9.16 USD per compared trade. Mixed seed 7: 35/43, -9.49 USD. These are local
counterfactuals, not alternate portfolio profits.

Collapsed argmax choices coexist with nonzero categorical entropy (for example
fixed-risk seed 43 always LONG on 256 common INNER windows, entropy 0.854).
The next contrast is therefore inference-only, before another training sweep:
all three fixed-risk checkpoints, each with (a) exact argmax replay, (b) sampling
its trained conditional categorical policy, and (c) a feature-blind control
using route-specific mean TRAIN probabilities from the pinned common-support
probe. Sampled arms use three fixed evaluation seeds (101/202/303), coupled by
bar timestamp. Same positions, sizing, SL/TP, costs and business restrictions.
No balancing trades, reward bonus, temperature or feedback feature is added.
The blind control tests whether any benefit is simply random trading rather
than useful market conditioning. Controls are not promoted as strategies.

This is 21 evaluation units / 42 TRAIN+INNER rollouts, zero fits and no 2024
selection. Deterministic trade-file hashes and metric values must reproduce
the parent before comparison. Every unit is resumable through authenticated
completion; the sequence is finite. Early-close behavior and costs remain
reported rather than inferred from activity alone.

## Execution evidence

Behavior audit completed: 24 units, 96 checkpoint probes and 12543 native
voluntary-close accounting replays (absolute monetary tolerance 1e-8 USD),
using TRAIN and INNER only. Independent copy: 75 files match the close manifest.
Scope is per-trade accounting and probes, not full GPU trajectory replay.

Inference audit source `211542e7`, collector `53345121`. First actual GPU
argmax replay reproduced the parent's trade-file hashes and native metrics
exactly for TRAIN and INNER. At 20:50 UTC, 3/21 evaluation units complete;
the next unit dispatched automatically. About 46--57 seconds per measured
model-based unit; blind controls not yet timed. Worker/reboot timer and
independent mirror/warehouse collector enabled; first mirror successful.
The collector publishes only the sealed completed audit, separately tagged
as inference diagnostics. No new training, source campaign changes or TEST.

Tests first failed on absent diagnostic modules, then 5 tests passed including
eight native favorable/adverse directional fixtures. Publication seal tests
reject changed bytes and path escape; final fast subset 6 passed. No full
repository-suite claim. The audit remains descriptive: no isolated seed or
checkpoint is promoted and no claim that all early exits should be disabled.

Next decision: compare conditional sampling against its matched TRAIN-marginal
control across all three training seeds and all three sampling seeds. If they
are indistinguishable, do not attribute extra trades to learned timing. If
conditional sampling adds value, quantify the deterministic-inference loss
before changing training. Further training-budget, gradient-interference or
credit-horizon experiments remain subsequent, not silently added here.
