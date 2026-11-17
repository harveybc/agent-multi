# 366: Rollback Activity Diagnosis

## Scope And Test Design

CPU-only analysis of copied365 A, D and control checkpoints. No change to live
workers, data, rewards, scheduling, checkpoint selection or feature pool.
All 71 columns and full native TRAIN/VALIDATION accounts remain unchanged.
TEST is not read. Validation is descriptive, not an intervention fitting set.

| Requirement | Predeclared test | Evidence |
|---|---|---|
| Authenticate snapshots | Collection hashes, weight receipt hashes, source support | Result input identities |
| Explain disappearance of trades | NOP/BUY/SELL, commands, refusals, side counts | Full native account replay |
| Localize functional drift | Restore one actor component from best at latest | Fixed probes and native accounts |
| Avoid fabricated causal attribution | Unchanged and whole-actor restoration controls | Exact CPU replay controls |
| Measure critic pressure | Analytic actor-objective logit gradient vs autograd; common-Q-offset invariant | Unit tests and fixed probes |
| Distinguish collection from deployment | Frozen categorical versus greedy, three common streams | Native accounts; not historical replay |
| Verify rollback | Receipt after-weight digest equals selected checkpoint | Per-event checks |
| Preserve training | No optimizer steps; copied weights/RNG unchanged after diagnostics | Runtime assertions |

Probe panel is fixed before inspecting results: 512 equally spaced valid TRAIN
windows and 256 VALIDATION windows. Epochs are initial, restored-best, immediately
before/at/after first rollback where available, and the latest copied epoch.
Epoch snapshots describe PRE-rollback weights. Restoration receipts describe
POST-rollback weights. Do not interchange them.

Actor objective derivative holds critics fixed. It measures local pressure under
the implemented SAC objective, not whether critics predict true returns. Component
swaps are diagnostic hybrids, not selected or promoted models. One seed per
architecture cannot establish a general architectural effect.

## Completed Evidence

2026-11-16 local date. Fifteen copied checkpoints, 21 component/identity
interventions, and 24 frozen-policy evaluations: 60 paired TRAIN/VALIDATION
replays (120 account simulations). Seven observed rollback receipt digests
match their selected-best weights exactly. Six whole-actor/unchanged controls
pass. No optimizer step, source checkpoint mutation or live worker alteration.
Sixteen focused tests pass, including the native-engine rollback/restart test.

Artifacts: [summary](generalization/rollback366_summary.json) and
[plot](generalization/rollback366_activity.png). Full evidence remains in
`/var/tmp/rollback-activity366-20261116/analysis/RESULT.json`; its SHA256 is
recorded in the summary, along with analysis source digests. The plot connects
sampled checkpoints, not every intervening epoch or immediate post-drop state.

### Activity Concentration

These are the latest COPIED PRE-rollback states, not current recovery states,
not promoted candidates, and not final results of the still-running campaign.
Returns are percentages on the full respective historical account.

| Arm | Epoch | TRAIN return | VAL return | VAL trades | VAL trades/week | VAL NOP share | VAL buy/sell | VAL early closes |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| A, one branch residual block | 25 | 14.4867 | 1.7088 | 74 | 1.3704 | 97.229% | 7/67 | 4 |
| D, per-timestamp dense branches | 53 | -5.7992 | 3.4358 | 54 | 1.0000 | 97.632% | 0/54 | 0 |
| Original control | 10 | 8.5892 | 0.4870 | 57 | 1.0556 | 98.324% | 22/35 | 2 |

All three fail activity eligibility. D's positive validation return does not
make it an admissible replacement. The lack of early closing in D coincides
with zero BUY actions in validation, so a short position receives no opposite
signal. No native daily-cap, position-occupied, risk-budget or level rejection
explains this collapse; those rejection counts are zero. Calendar restrictions
still reject some proposed entries and remain enforced.

### Critic And Actor Geometry

On the fixed TRAIN probes, minimum-twin Q prefers NOP on 86.72% (A), 84.38% (D)
and 83.20% (control) of latest points. Initial fractions were 2.73%, 0.78% and
1.56%. This shift accompanies policy concentration; it does not prove the
critics are correctly calibrated to realizable returns.

The actor objective actually minimizes
`sum_a pi(a|x) * (alpha * log(pi(a|x)) - min(Q1,Q2)(x,a))`.
The analytic logit gradient was checked against autograd, including invariance
to a common Q offset. At the endpoints its mean NOP gradient is near zero or
positive rather than universally negative: the snapshot is already close to
the entropy-regularized preference, not receiving a universal immediate push
toward more NOP. Do not mistake Q preference for proof of one-step causation.

The >2 orders/week rule is a selection gate, not a constraint in this objective.
It correctly rejects inactivity but does not itself train a policy to become
eligible. This is an objective/selection mismatch to investigate, not evidence
that the gate or execution engine is broken.

### Frozen Sampling Versus Deployment

Three predeclared common uniform streams (101/202/303), same frozen weights,
same bars, costs, strategy and complete TRAIN account:

| Arm / snapshot | Greedy TRAIN | Sampled TRAIN range | Greedy trades | Sampled trades range |
|---|---:|---:|---:|---:|
| Original control, epoch0 | +16.0582% | -49.8452% to -43.8252% | 745 | 2275-2339 |
| Original control, epoch10 | +8.5892% | -42.8809% to -28.0967% | 411 | 1534-1598 |
| A, epoch25 | +14.4867% | -34.3223% to -16.8871% | 287 | 1199-1252 |
| D, epoch53 | -5.7992% | -51.8865% to -43.7957% | 238 | 1835-1892 |

Online collection samples categorical actions while checkpoint scoring is
greedy. The disparity is large and appears even in the good source policy.
SAC intentionally optimizes a stochastic entropy-regularized policy; this
finding alone is not an implementation bug. These are frozen-policy accounts,
not reconstructions of the changing-policy training episode. Increasing entropy
or replacing greedy deployment with sampling is NOT justified by these losses.

### Component Restorations

Restore only the named actor part of control epoch10 from control epoch0;
all other actor and critic weights remain at epoch10. These are diagnostic
hybrids, never optimized on validation or promoted.

| Restored part | TRAIN return | VAL return | VAL trades |
|---|---:|---:|---:|
| None (identity) | +8.5892% | +0.4870% | 57 |
| Branches | -0.2335% | +1.1311% | 124 |
| Fusion | +12.0229% | -1.1992% | 61 |
| Attention | +8.5227% | -0.0339% | 54 |
| Core | +1.1554% | -5.9497% | 158 |
| Head | -4.2661% | -12.4911% | 289 |
| Entire actor | +16.0582% | +18.3699% | 207 |

The source function survives intact when the entire actor is restored. A
single-part restoration does not recover it; head replacement alone is harmful.
This supports joint functional drift/coadaptation, not an isolated bad head or
attention layer. Relative L2 drift from best in control: actor encoder12.33%,
actor head24.54%, critic encoder8.34%, critic head30.22%. D has a much smaller
actor drift (encoder1.75%, head2.90%) yet also loses activity. Norm size alone
does not diagnose useful learning or establish that the network is too large.

### Restoration And Numerical Exception

All seven observed receipt after-state digests match the selected source state.
A's activity increases again after the first restoration, then contracts again.
Restoration itself is not failing. It restores temperature and Adam moments too;
replay, random streams and global counters intentionally continue.

For example A's alpha grows from0.0050126 at best epoch5 to0.0059867 at epoch25,
then full restoration returns it to the best value. This resets some entropy
adaptation too. Also only actor LR is reduced: at D epoch53 the critic and
temperature LR are16 times actor LR (3e-6 versus1.875e-7). These are verified
mechanics, not demonstrated causes; the initial activity deterioration predates
the first LR drop, so LR imbalance cannot explain its onset alone.

All15 copied checkpoints reproduce native return and trade count exactly on
CPU. One does NOT reproduce every signal: D epoch0 validation changes one NOP
to BUY, rejected before operational start. Return and trades remain identical;
the extra refusal is recorded. Thus the strict all-signal parity flag is FALSE
(14/15 pass), not silently accepted. Device/numerical attribution needs a paired
logit witness; it is not established merely by guessing rounding. All hybrids
compare CPU with CPU and their identity controls pass exactly.

## Next Action

Keep365 running unchanged and preserve selected299. Phase2 remains open; no
broad DEAP, new floating score, live change, or STL restart follows this report.
The next bounded diagnosis is to authenticate a full recovery and hold the
actor fixed while comparing critic targets/action rankings under the sampled
continuation versus greedy continuation on identical TRAIN origins. Separate
flat/open-position cases in analysis, without adding agent-state network inputs.
Measure whether the objective recommends abandoning known profitable greedy
decisions, then qualify a paired update intervention before retraining broadly.
Also obtain an exact device-logit witness for the one initial D discrepancy.

Do not solve this by inventing an activity bonus, scaling logits from validation,
resetting a head in isolation or declaring missing market signal. The current
evidence identifies reproducible failure modes, not one proven root cause of
all training variability or the exact number of additional data years needed.
