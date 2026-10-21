# Fixed-Budget Temporal Context Study

Owner clarification: expensive native-SAC early stopping remains disabled.
The ten-cell BC/DAgger monitor study is finite, cheap, and a separate learning
algorithm. It does not establish that native SAC suffers from overtraining.

## Frozen Design

Reuse the authenticated native SAC control: 13 observations, stride 2 observed
open-market bars, seed 7, 100000 transitions. Run stride 1, then stride 4, with
the same 13 observations, selected 15 inputs, causal normalizer, architecture,
optimizer, business envelope, execution costs and TRAIN 2020-2023 / VALIDATION
2024. TEST stays closed. Sampling includes the completed current bar; missing
weekend bars are not invented. Context is 12, 24 or 48 observed-bar intervals,
not necessarily elapsed wall-clock hours. Trade execution remains hourly.

Keeping tensor shape constant keeps parameter counts constant. This contrast
tests resolution/context jointly, not their separate causal contributions.
No window-size or architecture search is authorized by this design.

Four frozen diagnostic evaluations remain at 25k/50k/75k/100k transitions;
the final 100k policy is used, with no patience, checkpoint selection or best
restore. Final TRAIN and VALIDATION summaries retain profit, drawdown, Sharpe
and its convention, trades, win rate and profit factor. Compare daily growth
and the TRAIN-VALIDATION gap, not raw profits over unequal period lengths.

The incumbent must improve validation daily growth by >1e-8, with nonworse
generalization gap and validation drawdown and nonzero activity. If either
stride qualifies, compare the selected stride with stride 2 on fresh paired
seeds 17 and 29. Otherwise close after two fits. Maximum six new fits, with
at most two fresh-root attempts per fit. Development selection is not an
independent estimate; no promotion to trading follows automatically.

## Automation

One sequential heavy process on the preferred GPU. A pinned predecessor gate
waits for the finite monitor study to finish before fresh resource admission.
Publication and independent collection are separate durable timer roles;
publication delays never cause a refit. Preserve prior attempts, weights,
effective configuration, input/selection/calendar hashes and all reports.
Unknown/integrity failures stop with a concrete typed diagnostic; unavailable
resources wait and are retried by the timer without consuming fit attempts.
No DOIN change, no secondary GPU on the same low-RAM host, no live trading.

## Traceability / Test Design

| Requirement | Test / evidence |
| --- | --- |
| Fixed shape; only stride and seed may change | strict v10 request and full common-config equality |
| Exact 1/4 screening; conditional paired replication | complete synthetic queue and no-promotion closure |
| Do not overlap predecessor | pinned spec and completion gate, changed predecessor rejection |
| Native temporal semantics | existing real preprocessor origin/prefix/stride tests |
| Fixed final budget, no early stop | existing final-100k outcome verifier |
| No intervention at stage boundaries | durable worker/publication/collection timers |
| Stable metrics and artifact custody | inherited authenticated close/readback collector |

PRE retained in session evidence: three expected context-contract failures.
POST: 125 focused regressions including native monitor integration passed;
120 focused CPU checks including real preprocessor stride/prefix probes passed.
The two sets overlap and are not 245 independent tests. Current stage:
integration verification passed. Sixteen final context tests and 35 worker
tests passed. Runtime `9aa461dd`, spec `faab4d09`: worker, publisher and independent
collector timers enabled. Initial dispatch waits for the pinned predecessor;
its latest observed progress was 8/10 without failures. No new native SAC result
is claimed at this deployment snapshot. The scientific runtime stays pinned
while these operational notes are updated.
