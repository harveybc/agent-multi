# Context-adaptive selector: recommendation review

2026-10-29. Owner supplied external recommendations while plan134 runs.
This addendum records useful hypotheses and necessary distinctions; it does
not change any pinned source, objective, gene, report or scenario in that run.

## Mathematical interpretation

Nash means resistance to unilateral improvement under specified payoffs and
allowed strategies. Inefficiency is possible, not an inevitable consequence
of using short-horizon agents. For a potential game, a global maximizer of
the potential is a Nash equilibrium; the converse need not hold. These are
properties to verify for the actual utility, not a reason to label every
competitive allocation inefficient. See [MIT Nash lecture](https://www.mit.edu/~6.7980/nfgs_nash.html)
and [MIT potential-game notes](https://live.ocw.mit.edu/courses/14.126-game-theory-spring-2024/mit14_126_s24_lecture_slides_potential_games.pdf).

Account Pareto efficiency means no feasible alternative improves one chosen
account objective without worsening another. It does not necessarily maximize
profit and minimize drawdown simultaneously, choose a unique policy, or make
every player better off. The vector of account return and drawdown is distinct
from the vector of individual players' utilities. The empirical archive in134
does not certify global optimality. See [Boyd, vector optimization](https://see.stanford.edu/materials/lsocoee364a/transcripts/ConvexOptimizationI-Lecture07.html).

A real-time score is not itself a Pareto certificate. We optimize complete
policies offline, then execute a causal decision rule. Independently picking
the best policy per regime does not guarantee the combined switching policy
is best: positions, balance, funding and opportunities cross regime boundaries.

## Disposition of the recommendations

| Recommendation | Disposition |
| --- | --- |
| Compare performance by market context | Retain as a diagnostic and paired ablation hypothesis. |
| Short agents win in ranges; long agents win in trends | Test both propositions and their opposites; do not encode them as facts. |
| Reserve the slot for a long agent during trends | No categorical lock. All safety and qualified early-close decisions remain enabled. |
| Avoid destructive short-scale noise | Already has uncertainty, margin and persistence controls; measure delayed helpful exits as well. |
| Add ATR/ADX | Candidate context descriptors only, not installed inputs or a proven regime classifier. |
| Account for missed opportunities | Replay full alternative trajectories with common exogenous randomness. Do not add fictitious missed profits to cash. |
| Compare Sharpe/Sortino and recovery | Secondary diagnostics with sample counts and explicit undefined cases. Preserve return and drawdown separately. |
| Resolve short versus long ownership | Compare against short-only, long-only, all-profile and NOP controls; no fairness quota forcing each specialist to trade. |

In particular, short-horizon win rate need not be higher, long-horizon net
profit need not be larger, and rapid rotation need not compensate for costs.
No model is assigned any of these characteristics without measurements.

## What is actually implemented now

Python controller and DEAP search; native absolute quote-side execution.
Current context is signed six-completed-bar efficiency:
`(close[t] - close[t-6]) / sum(abs(diff(close[t-6:t+1])))`.
Zero variation yields zero. It is a causal statistic, not an HMM, an ADX rule,
or a certified regime detector. A single optimized coefficient can favor or
disfavor an aligned horizon; its sign is not forced to favor long profiles.
No ATR/ADX feature or new feedback channel has been added to a learner.

Volatility variation is present in the synthetic scenario panel and report
dispersion, but there is no explicit volatility-conditioned horizon-weight
coefficient in the current arbiter. Synthetic regime labels are reserved for
test analysis; they are not passed into `choose`.

While flat, all eligible direction/profile candidates compete with NOP.
While occupied, position-conditioned HOLD/CLOSE applies regardless of the
entry horizon. A new short-scale request cannot directly close or reverse an
order; nor can a long-profile label prevent a justified early close. Native
protection and mandatory weekly flattening always retain authority.

## Next comparison, after the frozen baseline closes

1. Inspect existing native replays by synthetic pattern, volatility direction,
   mirror, entry profile and side. Report trades, wins by side, net return,
   drawdown, holding time, early-close/SL/TP/forced-flat counts and costs.
   Retain full-account totals; drawdowns and Sharpes do not add across groups.
2. Compare no-context allocation with the existing efficiency-conditioned
   score, holding reports, account rules, risk geometry, scenarios and budget
   fixed. First use paired parameter ablation, then matched reoptimization to
   avoid confusing a parameter removal with an optimized baseline.
3. Add explicit volatility context only as a separate, documented ablation.
   Fix estimator and support before evaluation; fit any thresholds on the
   permitted calibration/TRAIN partition. Do not introduce rolling z-score,
   clipping, future-smoothed regimes or hidden changes to neural observations.
4. Test uncertain and delayed regime changes, false breakouts, sharp genuine
   reversals, unprofitable high rotation and short profitable sequences that
   beat a long hold. Also include the opposite outcomes. Preserve the known
   early-close acceptance fixtures from131/133.
5. Re-evaluate the integrated switching policy on entire trajectories. A
   hindsight best-regime selector is an explicitly labelled upper control,
   never the deployable policy or the source of live opportunity-cost values.

These are follow-up experiments, not silently enqueued replacements for134.
Existing daily/weekly rules, costs and single-position accounting are shared.
No additional approval is needed to finish134; this review does not interrupt
its automated search, confirmation or report generation.
