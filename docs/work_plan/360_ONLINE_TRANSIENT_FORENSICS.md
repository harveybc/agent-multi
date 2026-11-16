# Online Transient Forensics

Owner requested2026-11-16: inspect359's current transients without interrupting
its three training queues. CPU-only analysis of copied, committed epoch weights;
no optimizer step, new feature, strategy change, validation fitting or promotion.

## Questions And Predeclared Checks

1. Is the initial decline associated with numerical failure, large gradients,
   representation drift, actor-head drift, critic drift or entropy control?
2. Does lower critic loss correspond to better policy decisions? Losses measured
   on different evolving replay distributions are NOT directly comparable errors.
3. At fixed market probes, how do actor entropy, action margins, min-twin Q
   preferences, Q common level/action spread, twin disagreement and alpha change?
4. Separate actor representation versus head by two diagnostic swaps: initial
   head/current representation and current head/initial representation. Full native
   TRAIN/validation accounts evaluate the two hybrids; do not select a winner or
   feed it back into the running experiment.

Use epochs1,5,10,20,24 and the last COMPLETE epoch available when copied.
The three seeds currently have different treatments: report within-run changes,
not a causal LR ranking. Full-policy account parity anchors CPU analysis to the
recorded GPU metrics. All market probes use identical rows across snapshots.
TRAIN/validation are descriptive; no claims about unseen deployment or convergence.

| Requirement | Test/evidence |
|---|---|
| Original runs untouched | Copies only, SHA256 against committed HISTORY snapshots |
| No hidden training | CPU-only, no update/optimizer calls; state/RNG invariant around probes |
| Swaps exact | Only actor head or actor encoder keys replaced; unknown/missing shapes rejected |
| Q diagnosis meaningful | Common-level shift changes level but not action margins or Boltzmann target |
| Drift connected to function | Weight changes alongside action flips, margins and full native accounts |
| No false optimization | Hybrids diagnostic only, source299 remains selected |
| Durable outcome | JSON, plots, report and method-state entry; queues remain unchanged |

Interpretation limits: weight displacement does not itself establish a cause.
Encoder/head swaps break co-adaptation, so either swap can worsen performance
without proving that module is intrinsically defective. Low entropy on one
uniform time probe does not prove entropy failure on the replay distribution.
Q agreement is not calibration against true future returns. Existing no-agent-state
interface remains fixed; possible observation aliasing is not silently repaired.

## Additional Frozen Collection Witness

Code inspection confirms training samples categorical actions whereas native
monitoring uses greedy actions. Before observing this witness, compare source299
and each last copied checkpoint using BOTH greedy and three independent common
uniform streams101/202/303. Same bars, costs, sizing, calendar and weights.
Since the actor sees market only and all three symbols are available, action
probabilities can be computed once on the full series; native position-dependent
commands and account transitions still execute sequentially. Preserve the true
distribution entropy when overriding sampled actions. This is NOT replay of the
historical changing-policy collector; it isolates the execution policy difference.
Tests: exact categorical intervals/frequencies, unchanged inputs and invalid-row
rejection. No change to running collection, entropy target or deployment policy.

## Outcome

Completed20 checkpoints,12 actor component swaps and16 frozen-policy accounts.
See [measured results and device exception](360_ONLINE_TRANSIENT_RESULTS.md).
The original strict CPU parity gate failed at control48 and stopped the first
analysis. A second descriptive run explicitly recorded that exception; it did
not relax the gate or mark it passed. Original-device replay and two matched
attention-dispatch witnesses separately explain it. No running worker changed.
