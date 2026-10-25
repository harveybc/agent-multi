# Frozen order-policy diagnosis: measured return

Completed 2026-10-25T06:55:14Z, 228.263 seconds, producer `0998df03`.
Parent spec `79c4b87eb64d7285583614afe624a196dbf7c632f0351d2d0f4aa2c4130bfd5e`.
Report SHA256 `faf136ecabda9282a2fd39327c19235e6b11db7709c992a421ef6d0e5dcd639b`.
32 manifested files verified in a second physical copy. No fitting, policy
selection, TEST scoring, new observation features or parent-file modification.

## Results

All ten TRAIN/INNER_A checkpoint-period replays reproduced native metrics and
order counts exactly on the original GPU. Parent results/artifacts authenticated
again after inference. 31 focused tests passed before deployment, including
weight/RNG/mode preservation using a native SAC instance without training.

| COMMANDS5 checkpoint | TRAIN orders | INNER_A orders | INNER_A profit USD | Observed behavior |
| ---: | ---: | ---: | ---: | --- |
| 25000 | 277 | 38 | +733.80 | Every deterministic decision is OPEN_SHORT |
| 50000 | 0 | 0 | 0 | Every deterministic decision is CLOSE_LONG while flat |
| 75000 | 68 | 6 | -206.27 | Predominantly CLOSE_SHORT; few entries |
| 100000 | 0 | 0 | 0 | Every deterministic decision is CLOSE_LONG while flat |

At 25k, repeatedly requesting OPEN_SHORT while already short is converted to
HOLD. The positive internal-period result is therefore compatible with an almost
constant short-opening rule plus existing SL/TP/calendar controls, not evidence
that the actor learned to select individual entry opportunities. This is a
behavioral finding, not a replacement or selection of the 25k checkpoint.

At 100k, all 21220 TRAIN and 2804 INNER_A deterministic decisions are invalid
CLOSE_LONG commands while flat. The existing engine converts them to HOLD and
does not create a phantom trade. The parent TARGET policy remains exactly
reproduced; its final TRAIN/INNER orders are 265/17.

## Frozen actor and critic probes

Sample: 256 evenly spaced decision observations per period, same bar hashes
across all checkpoints/arms, independent of outcomes. Per observation: 32
seeded Gaussian samples, eight critic action probes at fixed policy TP/SL, and
16 training-mode repetitions. These distributions are not additional trading
backtests. Labels attached to the five-bin probes describe COMMANDS5; TARGET's
actual command semantics remain those in its decoder and order-count report.

Final COMMANDS5, INNER_A:

- Deterministic first action: mean 0.967713, temporal SD 0.004719, entirely in
  CLOSE_LONG. The analogous TRAIN mean/SD are 0.968960/0.006535.
- Stochastic action samples: 8179/8192 CLOSE_LONG (99.8413%), 13 OPEN_LONG;
  TRAIN 8180/8192 CLOSE_LONG. Hence the inactive deterministic result is not
  merely a mean-action artifact of a broadly exploring final Gaussian policy.
  These are samples from the saved policy, not recovered historical replay.
- The minimum of the two Q estimates prefers +0.99 (CLOSE_LONG) in all 256
  observations in each period. INNER_A mean estimates: HOLD -0.099575,
  OPEN_LONG -0.085004, CLOSE_LONG center -0.072882, +0.99 -0.071127.
  The network favors an action that is equivalent to HOLD in the actual flat
  execution state. Q values are learned soft estimates, not dollars or measured
  counterfactual returns. Hidden-state mixtures can contribute to this mismatch.
- Entropy coefficient alpha is 0.000566786, target entropy -3. Its value alone
  does not establish why the policy became concentrated.
- Switching to training mode changes no command in 4096 repetitions per
  period at the final checkpoint. Dropout changes numerical estimates but does
  not immediately undo the final action collapse. This does not rule out its
  effects during learning.
- The actor's two-dimensional temporal embedding has nonzero temporal SD
  (INNER_A 0.149067/0.086065). It is not an identically constant observation
  representation. This does not establish sufficient information or capacity.

## What this establishes, and what it does not

Established: exact reproducibility; repeated changes in learned action preference;
final critic preference consistent with the actor's concentration; deterministic
and sampled final policies both concentrated; correct no-position handling by
the tested execution path. Inactivity occurs on TRAIN too, so describing it only
as a TRAIN-to-validation generalization gap misses the observed mechanism.

Not established: a unique causal explanation, Bellman calibration, historical
exploration coverage, superiority of another algorithm, or that removing dropout,
increasing capacity, increasing training duration or changing entropy will fix it.
The replay buffer was not retained. No claim that all business-engine paths are
certified follows from these replays.

## Next experiment design, not silently deployed

Keep the recipe, splits, costs, calendar, no-feedback observation and action
contract fixed while testing learning stability. First bounded contrast: remove
dropout from the critic extractor only (including the corresponding target
extractor), leaving actor dropout, weight decay and every other setting fixed.
Freeze this configuration and verify initialization parity before fitting. This
isolates one possible source of noisy action-value gradients; it is a hypothesis,
not a promised remedy. Report all fixed checkpoints and action distributions,
including invalid-command occupancy, not only net profit.

Subsequent separate options: entropy control, then categorical/hybrid action
representation with a maintained implementation. Do not silently add inventory
features, position routing, action masks, invalid-action penalties or learned
sizing: each changes the agent's information/objective/business interface and
must be an explicit contrast. No large genetic search or feature-selection redo.

This diagnostic service ended successfully. No successor fit was launched by it.
The prior 24/24 financial reports remain in the warehouse; the new diagnostic
arrays and Q probes are in the separately authenticated evidence collection,
not misreported as new financial scores.
