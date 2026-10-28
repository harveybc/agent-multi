# Multiscale selected inputs: proposal, not deployed

Date: 2026-10-28. The user approved continuing the exit-control repairs and
requested discussion of multiscale inputs. This document records that proposal;
it does not replace the baseline or authorize an unmeasured financial result.

## Current representation

The pinned baseline uses PyTorch, grouped feature extractors, a channel-fusion
layer and a residual causal TCN. Input: 13 observations at offsets
[-24,-22,...,-2,0] on admitted observed H1 rows. Closures occupy no slot.
Consequently the span is 24 observed-row intervals, not a fixed wall-clock day.
The synthetic exit diagnostic fills three causal price channels and zeroes
the other channels; it is not a fit on all selected financial inputs.
The inherited feature manifest is not a claim that the model-specific floating
selection has completed: that stage still follows strategy/learning qualification.

## Two different experiments

1. Sample existing encoded H1 features at several strides. Feature identity and
   indicator definition remain H1; the model sees several temporal views.
2. Aggregate completed OHLCV bars to each clock/session timeframe, then recompute
   indicators with explicit parameters. For example RSI(14) on 6-hour bars is
   not an H1 RSI(14) sampled every sixth row. These are new derived features;
   profile, encoding, TRAIN scaling and selection must be established for them.

Begin with (1). Do not silently relabel a selected feature or globally smooth
the inputs. A five-stride tensor does not require storing five copies of the
feature lake, but it increases input size/compute and may require more history.

## History and alignment

Stride in observed market rows and true elapsed-hour resolution are different
JSON options, not aliases. Initially preserve the existing observed-row clock.
With a fixed span of 24 row intervals, strides 1/2/6/12/24 yield 25/13/5/3/2
observations including t. Two samples do not expose a rich temporal pattern.
With 13 observations per branch, spans are respectively 12/24/72/144/288
row intervals. The largest branch requires 289 base H1 rows, before any
additional indicator/filter warmup. No layer can recover absent history.
Computing a whole sequence of coarse summaries at earlier fusion query times
also requires their earlier history: 13 samples at stride 24 for a query at
t-24 reach t-312, requiring 313 base rows rather than 289. The exact buffer
requirement is derived from the whole graph, not merely the largest stride.

Each sampling path is right-aligned to t. Calendar closures are not inserted,
interpolated or counted as observed market ticks. Real missing in-session
quotes remain explicit data-quality failures. For true multi-hour bars,
availability is their close time, not their start, and partial current bars
must not be passed off as complete. Daily/macroeconomic levels follow their
release clocks; sample-and-hold and source availability are not fabricated
market quotes.

## Proposed configurable structure

```text
Admitted selected H1 features; fixed TRAIN encoding/scaling
  -> shared chronological base buffer, ending at t
  -> per-scale right-aligned sampler (no learned weights initially)
  -> per-(feature profile, scale) small multichannel causal extractor
  -> explicit temporal-alignment adapter or per-branch causal summary
  -> learned channel fusion
  -> small causal core
  -> existing action heads and business controller
```

Group identity is (profile, scale); channels within a profile are processed
jointly unless an independently measured per-column extractor is selected.
The chosen receptive field must be recorded in both branch steps and base-row
units. A kernel-1 extractor remains an explicit option, not a temporal detector.

Do not concatenate branch index j as if it represented the same timestamp at
every scale. If fusing sequences on the baseline 2-row grid, use only coarse
states available at each query time, by backward as-of routing plus explicit
availability masks. This routes representations, not invented raw price bars.
Derive each target's feature entirely from its causal prefix. Using interpolation
between past and later coarse states would leak into earlier query positions.
An alternative is one causal summary per scale at t: simpler, but temporal
patterns must already have been encoded before that summary, and the final
fusion is no longer a temporal sequence. Compare these as different designs.

Another exact common-grid construction computes each scale's summary separately
at each fine query q: encode [x(q-(L-1)s), ..., x(q-s), x(q)] for stride s.
Concatenate those summaries at the same q before the causal core. No upsampling
of raw inputs is necessary. Batched gathers or explicitly dilated causal
convolutions can implement this; cost and receptive field must be measured.
This is not equivalent to blindly averaging every branch to 13 bins.

The initial recommendation is to preserve the baseline fine temporal path and
attach coarse context to it. Do not downsample every fine signal to the coarsest
clock; that could erase the local reversal cues needed for early closure.
No claim that all scales or all groups benefit is made.

## Available operations

- Direct gather/indexing: exact configurable decimation, no averaging or weights.
- `torch.nn.AvgPool1d`: averaging plus stride; an explicit transformed-input arm.
- `torch.nn.Conv1d` with stride: learned reduction, explicit left-only causal
  padding and right-edge alignment required. Dilation enlarges receptive field
  without itself reducing output length.
- `torch.nn.AdaptiveAvgPool1d`: fixed output length, but equal tensor lengths
  do not establish equal physical timestamps. Not an automatic alignment fix.

Averaging, maximum pooling and filtering have different semantics. None is a
universal default across signed returns, circular calendar encodings, release
flags, categorical regimes and prices. Decimation can alias high-frequency
signals; a causal anti-alias filter is a separately declared contrast, not a
hidden repair. Preserve the unfiltered fine branch in that comparison.

Sources: PyTorch official [AvgPool1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AvgPool1d.html),
[Conv1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv1d.html) and
[AdaptiveAvgPool1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AdaptiveAvgPool1d.html).
Runtime inspected locally: PyTorch 2.13.0; online stable docs currently 2.14.
No framework upgrade is part of this proposal.

## Experiment order and required tests

First close the fixed-delay counterexample and causal learnability tests from
plan 110. Then compare the unchanged 2-row baseline to baseline plus one coarse
branch at a time. Keep group definitions, costs, action contract and seeds fixed.
Measure same-context versus expanded-context arms separately. Before combining
winning branches, include a parameter-budget-matched single-scale control,
common eligible rows and matched training budget; report compute and memory.
Use predeclared internal selection, not repeated tuning on external validation.

Required tests: right edge includes t for every history length; no future
perturbation changes earlier outputs; closure/calendar handling; real-gap
rejection; same eligible origins across compared arms; correct time alignment;
per-profile pooling semantics; encoding/scaling fixed from TRAIN; branch removal
and actual parameter counts; checkpoint replay. No rolling z-score or hidden
clipping; no agent-state feedback in neural inputs.

JSON surface to design: sampling clock, scales, observations per scale,
sampler/filter kind, enabled profile-scale groups, extractor kind/width/kernel/
dilation, weight sharing, fusion mode/grid, pooling, warmup and mask policy.
These are proposed fields, not an already supported runnable configuration.

## Parallel repair progress

The fixed-delay acceptance regression was implemented while this proposal was
discussed. `tools/rescore_exit_delay.py` authenticates the closed study and its
parent, fits the delay on TRAIN before reading confirmation, rebuilds each
recorded net outcome and re-scores all 13 policies without training. Result:
delay 6, zero qualified pure-RL policies. Historical producer receipts remain
unchanged. The executable gate now requires this control; the frozen deployed
runtime was not edited. The next repair is the no-clock-shortcut causal corpus,
not an unmeasured multiscale architecture switch.

Repair verification: 35 focused tests passed, including the real CLI's resume
and checkpoint-tampering check; targeted Ruff correctness checks passed. The
no-refit re-scoring report SHA256 is
`da335eaedd9769a6366f7c46ea7f050ac1caf8c4b71299581d6121a529475b71`.
