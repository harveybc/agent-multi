# 368 - Preserved Champion Fine-Tuning

## Contract And Test Design

Owner priority: refine selected299 (+18.36994575% VALIDATION) without stopping
the running365/367 experiments. This is a separate continuation, not a scratch
fit, a new selection pass, or an automatic promotion.

| Requirement | Implementation contract | Verification |
|---|---|---|
| Preserve the original policy | All71 inputs,28groups, actor/attention/critics/targets/temperature and named Adam moments retained | Exact transfer, cold reload, native epoch-zero accounts |
| Original LR (owner amendment before launch) | Original3e-5 for actor, critics, temperature; not LR/10 | Check original effective LR and all optimizer groups |
| Enough refinement time | ES80, minimum100, maximum2000 epochs | Frozen design and patience tests |
| Four wider reductions | Every20 rejected epochs, factor0.2, floor4.8e-8, no cooldown | Events at20/40/60/80; all groups reduced |
| Restore champion at every actual drop | Full model and Adam moments; preserve reduced rates, replay, RNG, scheduler and global counters | Rollback identity and interrupted/uninterrupted parity |
| Consistent champion | Previously accepted balanced_min monitor on native return/week; activity >2/week in both periods | One acceptance decision for checkpoint, ES and LR scheduler |
| Autonomous recovery | Pinned runtime and enabled service; full epoch recovery, no duplicate worker | Runtime receipt and nonzero optimizer updates |

The balanced score is `min(TRAIN return/week, VALIDATION return/week)` with
absolute improvement >1e-8. It is not a promise that validation never falls:
the weaker period can improve while the stronger one declines. Both native
accounts remain visible. Zero-activity candidates cannot replace the source.

The owner amended the initial request before launch: start at the original
LR3e-5, not its tenth3e-6 (already used in365/367). The new experiment changes the
initial LR, patience, reduction factor, floor and reduction scope (all three roles instead
of actor only). It is a combined refinement policy, not an isolated causal
estimate for any one of these changes.

No random head reset here. Prior evidence concerns **critic heads**, not
attention or actor heads, and its benefit was conditional. This contrast
preserves the already-trained source and its optimizer moments; resetting and
recalibrating critics would be a different initialization experiment.

TRAIN2020-2023 and VALIDATION2024 stay identical; no TEST read, synthetic data,
rolling z-score, agent-state inputs, feature removal or window reduction.
The source export has no replay/RNG recovery from its original training;
this new run starts a seeded replay and RNG. Subsequent recovery preserves
them exactly. Do not describe it as continuation of the original replay.

Expected no-improvement LR sequence:
`3e-5 ->6e-6 ->1.2e-6 ->2.4e-7 ->4.8e-8`.
Twenty warmup epochs suppress ES counting, not champion saving; the minimum100
budget permits at least20 epochs after the fourth drop. Improvements reset
patience, so completion time is not fixed. Keep the source if no eligible
candidate improves the declared monitor.

## Execution Evidence

Test design recorded before implementation.39 focused CPU tests passed,
including full-model transfer at both rates, shared monitor, all-role rollback,
interrupted/uninterrupted native-engine parity, rate sequence and floor.
Correctness lint passed on all six changed Python files.

The first test run exposed an all-role resume metadata mismatch and a
rounding-only fifth reduction. Both were corrected before deployment; old
actor-only behavior remains covered and the pinned365/367 runtimes are unchanged.

Pinned runtime `ea59a964`, archive SHA256
`a86293256b1007503c1178d5e44e569e521a066e04c9e7f3dea7e3e18f418cf3`;
Python-code identity
`d65705139fbc3f2c01be358bf742b09e66527f833d60a37e343c1b14594fff94`.
Run identity
`0c9ef06f6471523f4a8a3ee7ae2af728f9c3c9a6f29ea3d3e024a0dae5686606`.

An enabled persistent worker started on the otherwise idle RTX5070Ti Laptop.
The assigned device had no compute worker and host available RAM exceeded8GB.
Existing training workers were not interrupted. Worker memory high/max limits
are4/5GiB; CPU quota200%, numerical libraries one thread. Restart on failure
resumes epoch recovery; three starts/hour limit prevents a runaway failure loop.
Linger is enabled, so no interactive session is required.

Cold model and optimizer reload matched exactly; input shape13x71, all three
initial role LRs `2.9999999999999997e-05`. Native epoch-zero qualification:

| Period | Return % | Trades | Orders/week | Sharpe |
|---|---:|---:|---:|---:|
| TRAIN | 16.05822355000215 | 745 | 3.5645933014354068 | 0.7819691259276358 |
| VALIDATION | 18.369945750000117 | 207 | 3.8333333333333335 | 2.690712133251012 |

These are reproduced source results, not improvement from fine-tuning. The
first epoch started after qualification. Runtime output lives in the private
`champion-refinement368-20261117` run directory; deployment, service and copied
qualification receipts are in the operator's private `refinement368-deploy`
directory. No host identifiers or credentials are part of this report.

## Read-Only Weight And Objective Probe

After the first natural rollback at epoch20, copied snapshots20,21,25 were
authenticated against their history SHA256 values. CPU inference used the same
128 TRAIN windows (13x71), with no updates or changes to the running workers.
Evidence: [probe](generalization/refinement368_weight_probe.json).

The epoch20 snapshot is **before** rollback. The receipt's post-rollback full
weight digest equals the original source exactly. CPU versus saved GPU source
logits differ by at most9.06e-6; do not claim bitwise cross-device logits.

| State | Actor NOP/BUY/SELL on128 TRAIN windows | Twin-min Q argmax NOP/BUY/SELL | Action changes from source |
|---|---|---|---:|
| Source / restored champion | 116/4/8 | 2/29/97 | 0 |
| Epoch20 before rollback | 86/17/25 | 73/28/27 | 42 |
| Epoch21 after one continuation epoch | 3/20/105 | 27/38/63 | 113 |
| Epoch25 | 103/9/16 | 87/17/24 | 16 |

For the source, the analytic gradient of the categorical SAC actor objective
with respect to logits points toward increasing SELL in120/128 windows.
This is an instantaneous fixed-Q logit-space diagnostic, not a claim that every
network parameter update has that effect or that SELL is economically correct.
Source mean Q advantage SELL-minus-NOP is0.13856 against alpha0.005313.

At epoch21, relative L2 weight displacement from source is9.26% actor core,
9.90% actor fusion,11.00% actor head and11.35% critic head, yet88.28% of probe
actions changed. Across the full validation support, SELL is5877/6207 signals
(94.68%), not a percentage of executed orders. Native returns are TRAIN-77.33%,
VALIDATION-21.48%. This is substantial functional drift, not proof of a single
bad layer, insufficient model capacity or an invalid source checkpoint.

The evidence narrows the diagnosis: the good deployed actor and its restored
critics are not aligned in these states, and the current actor objective can
push away from that actor immediately. Full rollback also restores temperature
and Adam moments while retaining the newer replay, so replay/critic/temperature
interaction remains a competing explanation. Epoch25 partly recovers the probe
actions but native VALIDATION remains-12.57%; matching actions on128 windows is
not enough to establish account recovery.

Next diagnostic: freeze the source actor, assess critic action ranking against
realized own-policy TRAIN continuations on matched replay, and compare before
and after critic-only calibration before another joint update. No blind actor
or attention reset and no new causal conclusion from weight magnitudes alone.
Current training continues unchanged; this probe does not promote a candidate.
