# 376: Paired temporal actor learning-rate continuation

## Scope and decision frozen before execution

Owner approved fixing idle GPU admission and using the free device for the
attention/core contrast suggested by375. Preserve running374 history curves
and367. No new floating pass, broad DEAP, synthetic data or live work.

The available full-state anchor is the completed **six-month seed19 endpoint,
epoch101**, not its validation-selected epoch4. Earlier model tensors exist,
but their full replay/Adam/RNG recoveries were not retained. Do not silently
substitute weights alone. Epoch101 is chosen by the already fixed training
budget; this is **recovery after overfitting**, not prevention from an early
checkpoint. Negative results cannot reject the prevention hypothesis.

Two sequential arms resume this same loss/target state, named Adam moments,
replay, RNG and counters. Both use the same optimizer grouping. Control keeps
3e-5 everywhere; treatment uses6e-6 for actor attention and actor core only,
keeping all other parameters at3e-5. No weight/head reset, no LR scheduler,
no early truncation:20 complete TRAIN passes,6006 updates/pass,128 batch,
71 inputs,13 observations,stride2. This modifies the effective AdamW step,
including its decoupled decay; it is not a gradient-only intervention.

The existing2024 monitor is descriptive, already used in preceding research;
there is no new untouched internal validation claim. No candidate promotion.
TEST is not loaded. Source subset producers/scaler, market and risk remain fixed.

## Acceptance and evidence map

| Requirement | Check before/while running | Artifact |
|---|---|---|
| Correct GPU admission | Use physical UUID in the live supervisor; restart only an idle parent with no children | private service/queue receipts |
| Source authenticated | Hash every transferred recovery and declared store/market file | CONFIG / PREFLIGHT |
| Only requested LR changes | Parameter-name partition, complete coverage, finite factors, nonempty attention/core; unchanged moments and weights | FORK / optimizer unit tests |
| Grouping control preserves updates | Compare original versus grouped factor1 with identical samples/RNG | unit tests / QUALIFICATION |
| Resume exact | Same-device cold reload then identical sampled updates for each arm | QUALIFICATION |
| Full budget and data | Assert endpoint121 and120120 additional updates; preserve all inputs and support | RESULT |
| No selection trick | Endpoint is primary; best monitor only descriptive; all trades/costs and net/trade reported | SUMMARY |
| Activity confound visible | Publish trade counts and net/trade; do not infer discrimination from return alone | SUMMARY |
| Durable automation | Independent arm outputs, atomic recovery, finite sequential queue | QUEUE_STATUS / RESUME |

Both endpoint and full learning curves are required. If fewer trades accompany
better returns, that alone is not an activity-matched proof. Any positive result
requires a subsequent independent-seed/activity-controlled confirmation; this
single-seed diagnostic cannot close phase2 or justify deployment.

## Orchestration correction

The original CUDA numeric ordering differed from the management utility's
ordering. Unit files had already been changed to physical identifiers, but
running parents retained old numeric environments. The idle parent is now
restarted with its installed physical identity. A bounded guard waits for the
other parent's next idle boundary and applies the same correction only with
no child worker. Neither active training worker is interrupted.

## Execution

At2026-11-18T21:37Z the finite GPU queue is running the control at epoch102
(first of20 additional passes). The treatment follows automatically.
Runtime revision:c4f28a64; code identity:
`8c14613b1a42debbbf244a81eb31387c97c14911a87372f5194da34ac3b688d8`.

-51 focused tests passed; correctness lint passed.
-All71 inputs retained:2781 valid TRAIN windows and6208 validation windows.
-Full-state fork preserves model tensors, named Adam moments, replay and RNG.
-Both forks reproduce source native accounts exactly on the new device:
 TRAIN+349.3972773%,748 trades; validation-56.4827730%,1471 trades.
-Four successive sampled updates reproduce exactly after cold reload for
 each arm; grouped factor1 control is identical to the original optimizer.
-25 parameter tensors (actor attention/core) receive the treatment factor;
598 others retain their rates. No target/critic/head reset.
-One initial admission attempt stopped before qualification because the
 desktop browser appeared as a GPU compute client. A narrowly declared exact
 executable exception plus GPU/RAM memory floors corrected this; other
 compute clients remain disallowed. No experiment results were produced by
 that stopped launch.

Both374 devices and367 continue; the idle supervisor correction is verified
by real worker progress, not just service status. The guard for the other
parent remains armed until its idle boundary. Runtime configuration, device
identifiers, source copies, qualifications and recovery stay outside the public
repository. Source+18.37% model is not replaced and phase2 remains open.
