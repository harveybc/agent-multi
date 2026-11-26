# 489. Frozen actor: information versus continuation target

Follow-up to [488](488_sac_objective/README.md). CPU session simulation on two
hosts, followed by CPU ExtraTrees action-advantage regression. This is not a
SAC retrain. The weights, temporal feature pipeline and deployed action
interface are untouched.

| Stage | Evidence |
| --- | --- |
| Freeze | Selected299 checkpoint and feature support verified against experiment 369. Same frozen logits sent to both workers. |
| Panel | Full Sunday-to-Sunday weeks within each TRAIN year, at least 12 valid observations; omit weeks containing any 369 anchor. Up to 12 decision rows sampled per position role per week using a date-derived seed, without inspecting outcomes. No labels cross the weekly endpoint. |
| Sample | 149 weeks; 2,026 fit decisions (2020-22), 795 monitor decisions (39 weeks in 2023). Monitor: 468 flat, 46 long, 281 short. On 119 monitor rows all first actions tie economically. |
| Counterfactual | Clone the same native state; force each of 3 first actions; continue to weekly terminal settlement, once with the greedy actor and with 4 paired stochastic tapes. Reconcile greedy actor account and continuation accounting. Separate weekly starting accounts, not a contiguous four-year equity simulation. |
| Two factors | Stochastic soft vs realized greedy discounted reward; frozen market embeddings vs the same embeddings plus one-hot position, signed distances to protection/target in entry-sigma units, bars held, and scheduled time to force-flat. Only the diagnostic critic sees the latter. |
| Fit | Per-state centered three-action labels; 128 ExtraTrees, minimum 8 samples per leaf, all dimensions available, seeds 7/19/41. Same decision rows, weighting and settings for all arms. Inputs to the two original critic heads are captured during the original inference path; their minimum Q must reproduce the sealed surface exactly. |
| Decision | Negative delta means smaller USD regret than the frozen actor. Average within week, then over weeks. For each seed require improvement in each of four monitor quarters and negative upper 95% paired week-bootstrap bound (5,000 resamples). Pass all three seeds. Not a familywise confidence claim; even a passing diagnostic requires a separate joint policy test. |

Raw reports are kept under `/var/tmp/critic489`: `frozen/spec.json` seals the
dates/settings, `weeks/*.json` contains the paired trajectories' action values,
and `RESULT.json` contains quarter, role and seed contrasts. The session
records are authenticated against the manifest and resume independently.
Use [the queue](../../tools/role489_queue.sh) with the second worker's SSH
alias on a host with the archived 369 runtime and Selected299 data bundles.
The queue recovers missing remote weeks locally before analysis.

Uncertainty limits: previously unseen sessions for this diagnostic, not
unseen by the original actor's training. The weekly causal intervention
asks about one action on the original actor's visited states, not the account
return of a new policy. Position frequencies differ substantially. Three
forest seeds are not three independently trained SAC actors. Failure by these
readouts does not exclude all state signals or all possible critic designs.

## Outcome: no actor update

In 39 monitored weeks the actor's mean weekly average best-action shortfall
was 20.57 USD per sampled state. The original minimum Q added 3.24 USD
(paired week-bootstrap interval +0.73 to +5.87). Forced NOP on the first
decision followed by the original actor gave 20.42 USD; this is not an
always-flat policy. Retrospective ranking by observed greedy reward gave
0.006 USD, versus 2.64 USD with the four sampled soft continuations.

| Diagnostic readout | Mean delta vs actor (USD; average of seeds) | Range over three seeds |
| --- | ---: | ---: |
| Soft, market only | +2.83 | +2.40 to +3.14 |
| Soft, with position | +2.58 | +1.98 to +2.89 |
| Greedy, market only | +2.21 | +2.05 to +2.38 |
| Greedy, with position | +2.51 | +2.37 to +2.71 |

All 48 arm-seed-quarter deltas were positive (worse). Runtime Q parity
error was zero; 2,821 sampled states produced 42,315 continuations.
Do not promote any of these readouts or restart the joint SAC with them.
Investigate next whether fit-vs-held action differences can be learned from
the frozen representation at all (time-blocked learning curves, role
stratification, permuted-target negative control, planted positive control).
See [full results](489_critic_result.json) and [sealed session list](489_critic_spec.json).
