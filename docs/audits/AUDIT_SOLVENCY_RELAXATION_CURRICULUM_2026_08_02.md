# Solvency-Relaxation Curriculum Collision Test

Date: 2026-08-02
Auditor: General Musashi, temporary independent auditor and research lead
Origin: owner NEAT/gym-fx observation, formalized by Lieutenant Satoshi II
Runtime mutation: none
Campaign interference: none

## 1. Verdict

The owner's historical observation is technically credible and closely aligned
with curriculum and relaxed-constraint reinforcement-learning research. It is
a strong prior and a worthwhile falsifiable research line. It is not evidence
that the active `USDCAD@4h` campaign is currently starved by insolvency.

The proposal is accepted as research line P20 with three corrections:

1. instrument termination causes before attempting the proposed measurement;
2. separate train-only solvency dynamics from immutable realistic
   train-tail/validation/test evaluation;
3. test synthetic recapitalization/chronological continuation instead of
   assuming that merely allowing negative balances preserves useful trading.

Job 0 remains untouched. Job 1 remains unchanged and is not delayed by this
line. A future domain may adopt the axis only after the bounded diagnostic
changes a decision.

## 2. Code Reconstruction

The deployed behavior is real:

- `gym-fx/app/env.py:131` defaults `min_equity` to 1% of initial cash;
- `gym-fx/app/env.py:343` terminates at that threshold or when the bridge is
  otherwise terminated;
- `gym-fx/app/bt_bridge.py:181-195` uses the same bridge flag for insolvency
  and natural data exhaustion;
- `gym-fx/app/env.py:756-809` exports no termination reason;
- `agent-multi/pipeline_plugins/rl_pipeline_with_validation.py:670-725`
  records episode length but not the cause;
- the training, train-tail and validation environments are all made from the
  same effective config. A naive `min_equity` change would therefore relax
  validation too, contrary to the proposal's hard boundary.

There is a second implementation constraint. In
`gym-fx/strategy_plugins/direct_atr_sltp.py:479-499`, position size is derived
from current cash. Negative cash produces a non-positive raw size and therefore
no new risk-increasing order. Lowering `min_equity` alone would create an inert
post-insolvency tail, not reproduce the owner's historical continued-learning
regime.

## 3. Active-Campaign Evidence

Both v2 job templates inherit `min_equity=100` from `initial_cash=10,000`.
Job 0 uses leverage 1 and evolves `rel_volume` from 0.01 to 0.25.

Omega's current job-0 logs contain 12,966 deterministic train, train-tail and
validation evaluation summaries. Their observed final balances range from
9,971 to 10,078. No visible evaluation approaches the 100 insolvency floor.
The current champion's validation evidence reports final equity 10,027.44 and
maximum drawdown 0.04899%.

This does not prove that stochastic training never terminates early: training
episode causes are not emitted. It does establish that insolvency relaxation
is not a credible blocker for the current campaign and that job 1 should not be
delayed for it.

## 4. Finding

**AUD-F1-20260802-059 (S4, open; blocks P20 measurement only):** the bridge
collapses insolvency and natural data exhaustion into one boolean and neither
the environment summary nor the training telemetry emits the cause. The
proposal's first measurement cannot be reconstructed from current logs or OLAP.

Required correction owner: Lieutenant Satoshi II as temporary technical lead.
This finding does not block job 0, job 1, champion archival or live-demo work.

Minimum emitted facts:

- `termination_reason`: `data_end`, `min_equity`, `external_stop`,
  `safety_limit`, or `unknown`;
- `episode_start_bar`, `episode_end_bar`, `episode_length`, eligible bars and
  observed fraction;
- equity/cash at termination, minimum episode equity and insolvency-event count;
- split, mode, seed, candidate, epoch, config hash and code revisions;
- separate training and evaluation aggregates in candidate/OLAP evidence.

## 5. Correct Decisive Experiment

Use paired seeds, the same asset/config and equal training timesteps. Do not
require positive profit. Primary outcomes are realistic-validation fitness,
episode coverage, activity, insolvency events and learning-curve stability.

| Arm | Training dynamics | Evaluation dynamics | Purpose |
| --- | --- | --- | --- |
| A | terminate at realistic floor; reset from normal start | realistic | deployed control |
| B | forced liquidation, full penalty, synthetic recapitalization, continue at next chronological bar | realistic | test credit/coverage starvation without impossible negative-cash trading |
| C | terminate at realistic floor; reset to a deterministic randomized chronological start | realistic | distinguish continuation benefit from repeated-prefix starvation |

Arm B must expose the insolvency event and recapitalization state to the policy,
must not erase realized loss from reward/fitness evidence, and must never be
available in live/demo execution. Arm C is important because the present reset
path restarts the same historical prefix after every failure; the owner's
effect may partly be coverage, not negative equity itself.

Run the diagnostic first on CPU or in a declared accelerator window with a
small fixed policy/training budget. Promote only if B or C improves paired
realistic validation across seeds without increasing validation ruin,
drawdown, inactivity or action collapse. If both fail, retire or narrow P20.

## 6. Prior-Art Collision

The broad mechanism is not novel by itself:

- Bengio et al., *Curriculum Learning* (ICML 2009), frames curricula as a
  continuation method for difficult non-convex optimization:
  https://icml.cc/Conferences/2009/papers/119.pdf
- Shperberg, Liu and Stone, *Relaxed Exploration Constrained Reinforcement
  Learning* (AAMAS 2024), explicitly relaxes constraints during training while
  requiring the learned policy to satisfy them at deployment:
  https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/shahaf_shperberg_AAMAS_2024.pdf
- Turchetta et al., *Safe Reinforcement Learning via Curriculum Induction*
  (NeurIPS 2020), uses reset controllers and curriculum selection to prevent
  unsafe termination while preserving learning:
  https://papers.nips.cc/paper/2020/hash/8df6a65941e4c9da40a4fb899de65c55-Abstract.html
- Pardo et al., *Time Limits in Reinforcement Learning* (ICML 2018), shows
  that termination semantics and incorrect bootstrapping can destabilize
  learning:
  https://proceedings.mlr.press/v80/pardo18a.html

The potentially useful Project 3 contribution is narrower: insolvency-aware
chronological continuation for evolutionary RL, crossed with visible execution
cost curricula, with selection on immutable realistic weekly risk-adjusted
validation and complete termination telemetry.

## 7. Owner Decision Needed

None now. The promotion of Satoshi II to Lieutenant is recorded. The next
owner decision is requested only if instrumentation shows material starvation
or the bounded A/B/C changes realistic validation enough to justify a new
domain.

