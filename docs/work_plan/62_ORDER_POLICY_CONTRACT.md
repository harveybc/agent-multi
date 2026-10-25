# Order Policy Contract: Diagnose Before Optimizing

Owner authorized implementation and unattended execution, 2026-10-25.
The preceding R3 campaign completed both arms. Its final grouped policy traded
22 times in OBS_2024; all exits were agent closes. That is an observation, not
proof that target-position semantics caused the result.

## Scope and Fixed Contract

Reuse recipe R3, its 120 frozen TRAIN-scaled inputs, 13 observations at stride 2,
calendar, costs, support, grouped actor/critics and native SAC optimizer. No
expert labels, imputation, clipping of observations, rolling scaling, feedback
features, new feature selection, early stopping or TEST. Keep seed 7 and 100000
decisions with four fixed checkpoints. Costs remain assumed, not broker verified.
FIT is 2020 through 2023H1; INNER_A is 2023H2; 2024 is development already read.
No checkpoint or decoder is selected by the diagnostic or by 2024.

Each order has SL, TP and a declared percentage of rel_volume * current balance
as margin budget (leverage converts this to notional). The present controlled
contrast keeps that percentage at 1.0, rel_volume 0.1, leverage 50. Learned
percentage sizing is a separate later contrast, not silently introduced here.
Every bar retains protections, costs and the immutable weekend/day-entry rules.
SL/TP enforcement may adjust proposed distances under the existing recipe;
these adjustments are logged, and are not clipping of input features.

## Ordered Automation

1. CPU alpha: action truth tables plus the real pinned engine on synthetic
   prices. Verify next-open fills, long/short, explicit hold/close, invalid
   commands, TP/SL including ambiguous bars and gaps, daily cap, reopen delay,
   forced flat, no cross-week position, sizing and reward/accounting telescoping.
2. Diagnose all four saved grouped checkpoints on TRAIN and INNER_A. Authenticate
   the parent RESULT, files and policy hashes. Cache raw deterministic actions
   at their actual decision bars, plus seeded stochastic-action samples for
   distribution only (not a second trading score). Replay target deadbands
   0.15, 0.25, 1/3, 0.45 over identical support. The 1/3 control must reproduce
   parent numerical metrics exactly on the original GPU; no tolerance is added
   after seeing failures. Report durations, proposals versus fills, refusals,
   costs, protection adjustments and action occupancy conditional on position.
   Changing a saved policy's decoder is a diagnostic, NOT a newly trained policy.
3. Fresh grouped TARGET control and COMMANDS5 treatment, sequential on the
   primary GPU. Everything except action semantics stays fixed. Both terminate
   at 100000 even if an intermediate checkpoint looks better. The fresh TARGET
   policy tensor hash must reproduce the parent final policy exactly; otherwise
   the dependent COMMANDS5 fit is not started as if the comparison were paired.
4. Copy artifacts to the second machine, publish metrics with readback and close
   the queue automatically. A failed integrity/alpha/replay gate stops fits with
   a typed reason; no unattended repair changes scientific settings.

## Explicit Action Semantics

The first scalar remains in [-1,1]. TARGET preserves the existing 1/3 deadband:
zero means target-flat (close if open); opposite target may close-and-reverse.

COMMANDS5 uses five equal-width bins with these exact boundaries:

| First scalar | Command | Applicable position | Otherwise |
| --- | --- | --- | --- |
| [-1,-0.6) | CLOSE_SHORT | short | recorded invalid-command HOLD |
| [-0.6,-0.2) | OPEN_SHORT | flat | recorded invalid-command HOLD |
| [-0.2,0.2] | HOLD | any | keep existing protections |
| (0.2,0.6] | OPEN_LONG | flat | recorded invalid-command HOLD |
| (0.6,1] | CLOSE_LONG | long | recorded invalid-command HOLD |

The remaining two scalars retain their original TP/SL mapping. Existing orders
are never resized or have SL/TP silently changed. Reversal requires close then
a subsequent eligible open. Positions are used only by the execution harness,
never concatenated into the actor or critic observation.

This is an action-encoding package contrast, including removal of atomic
reversals, NOT proof that one individual mechanism explains an improvement.
Both remain partially observed: explicit commands do not give the policy its
inventory, entry price or remaining daily capacity. State-routed specialists
would explicitly expose position via routing and are a later declared study,
not a disguised no-feedback baseline. SAC's continuous action aliases also
remain; a discrete/hybrid agent is another separate hypothesis.

## Tests Defined Before Implementation

| ID | Structural and behavioral evidence |
| --- | --- |
| O01 | Truth table for all commands in flat/long/short, exact bin boundaries |
| O02 | Target decoder equals legacy on boundaries and random action/position pairs |
| O03 | Observations unchanged by position; no state tensor, missing data still HOLD |
| O04 | Real engine protects positions on HOLD/invalid command, next-open close |
| O05 | Required SL/TP/sizing, costs, SL-first, gaps, day cap and weekend flatten |
| O06 | Undiscounted reward sums reconcile terminal equity/net profit, both sides |
| O07 | Parent integrity, cached support, full numerical replay at original threshold |
| O08 | Gate failure cannot launch fit; bounded retries and reboot reuse authenticate outputs |
| O09 | Final reporting retains all arms/periods and diagnostics, no score-driven selection |
| O10 | Native CPU and GPU SAC updates with new decoder and exact policy reload |

Next optimization is deliberately deferred until this evidence exists: inspect
activity mechanisms, then small joint learning-parameter searches, followed by
capacity/context and eventually periodic refitting. No large DEAP campaign now.

## Verification Before Deployment

56 focused tests passed on CPU, including 16 signed real-engine synthetic
scenarios and a native SAC commands5 update/save/reload with missing observation
intervals. The truth-table test first failed because the new module did not yet
exist. Two fixture issues (pandas timestamp unit and read-only synthetic arrays)
were corrected before the passing engine run; no real data were edited.
Runtime witnesses are re-executed by the deployed queue before diagnostics.

## Deployment

Scientific code b117dce0, spec
79c4b87eb64d7285583614afe624a196dbf7c632f0351d2d0f4aa2c4130bfd5e.
At 2026-10-25T02:37:30Z alpha passed16 scenarios on the deployed environment;
the 25k TRAIN replay was exact (78.5s including deterministic/stochastic action
capture). Checkpoint50k TRAIN diagnosis was underway. Sequential reboot and
collector timers enabled; first copy/pump succeeded with no reports expected.
The operational collector normalizes the new duration metrics' metadata to
hours, without changing native values or scientific runtime.

Automatic handoff verified 2026-10-25T02:42:24Z: all8 exact parent metric
replays and32 fixed-decoder cases completed; GPU admission passed; fresh TARGET
was training at1750 decisions/749 gradient updates. COMMANDS5 is queued after
the control. No intermediate score determined this dispatch. Native receipts
and generated status are authoritative for subsequent progress.
