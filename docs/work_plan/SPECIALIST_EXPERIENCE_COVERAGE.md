# Four action specialists: experience coverage before training

Status: design and evidence inventory, **not** qualification for training.
The admission checklist below takes precedence over the initial proposals.

Storage retirement (2026-11-20): prior single-actor run receipts, plots and
exports are historical diagnostics, not four-expert scores or resumable
training runs. Large replay buffers and epoch recovery states were pruned;
the per-host manifests are under
`~/.local/state/experiment-cleanup/retired-artifacts-20261120-v{1,2}/`.
The historical prefix and pinned 61-input source bundle were retained. Only
the declared representation may be reused; do not resume the retired policy
or rank these receipts among the four independent role fits.

## Measured baseline, not the new training set

Protocol 385, FIT only (2020-21). Verified all four NPZ SHA-256 digests in
DATA.json. For each role count distinct market timestamps, then signs of
the native paired advantage in bp (ties within 1e-8). Flat columns 1/2
compare BUY/SELL to waiting; long/short column 1 compares CLOSE to HOLD.
The two openers use the same timestamps. Counts are not independent paths.

| Role | Unique ticks | Act better | Wait/hold better | Tie |
|---|---:|---:|---:|---:|
| OPEN_BUY | 9707 | 4563 | 5104 | 40 |
| OPEN_SELL | 9707 | 4689 | 4785 | 233 |
| CLOSE_BUY | 8559 | 4492 | 4061 | 6 |
| CLOSE_SELL | 8613 | 4305 | 4300 | 8 |

These were economic targets for one flat classifier and two closers, not
four independently trained specialists. Positions sampled beyond the first
eligible post-entry decision: zero by construction. Horizon profile: 6 h.

## Required inventory for the replacement

1. Inventory 2020-23 and 2016-23 separately on identical supported columns,
   clocks and execution rules. Record eligibility, refusals, native barrier
   hits and expiries separately from discretionary decisions. Do not turn a
   missing observation or a forced exit into a learned NOP or CLOSE.
2. Cross role, risk-profile horizon, trend direction/strength, volatility
   level/change, temporal speed and regime transition. Include quiet/no-edge,
   acceleration, deceleration, false breakout, reversal, range, shock, spread
   widening, gap, reopening and weekly cut. For closers also cross position
   age, signed excursion, barrier proximity and time left. Freeze numeric
   bands from TRAIN before inspecting validation. Mark unreachable cells
   with an engine witness or rule; never silently count them as covered.
3. Exercise every declared level and pair of levels, and explicitly require
   higher-order dangerous combinations (e.g. aged long/short + near SL +
   adverse acceleration, profitable long/short + false reversal + later TP,
   long-horizon position + short-lived shock + liquidity deterioration).
   Pair coverage does not certify untested higher-order intersections.
   Mirror controlled scenarios, validate the native order ledger, and include
   cases where waiting is optimal and where costs reverse the gross ranking.
4. Preserve separate IDs for market tape, real timestamp, synthetic path,
   parent regime/segment, entry origin, position and decision. Store paired
   outcomes, costs, source and seed. Report unique paths, origins, weeks,
   ticks and position-ticks; replay draws, if reported, have their own column.
   Counterfactual positions at one tick share a market outcome. Overlapping
   windows, compressed time copies and minibatch repeats are not new histories.
5. Generate synthetic *paths*, not copies of rows, to fill weak cells; verify
   requested vs realized regime, speed, action outcome and feature coherence.
   An accelerated path must pass through the native bar clock, fill, spread,
   barriers and deadlines. Ambiguous intrabar event order is not an oracle
   label without finer data or a declared execution rule. Separate calibrated
   plausible paths from deliberately extreme stress fixtures.
6. Publish role x cell counts (including zeros), outcome sides and source
   diversity before GPU training. Set budgets separately for common families,
   sparse critical cells and deterministic boundary tests. Hundreds of
   thousands of rows are a scale target, not proof of independent coverage.
   Check learning curves by increasing distinct trajectories, clustered
   uncertainty, and economic regret by role/cell on held-out trajectories.
   No synthetic siblings across folds; keep the common real 2024 evaluation
   unchanged and at natural frequencies. No promotion by training count.

## Role learner comparison (design, not a run receipt)

Freeze four independently trained experts OPEN_BUY, OPEN_SELL, CLOSE_BUY and
CLOSE_SELL. Each may start with an identical copy of the qualified market
encoder; no shared mutable parameters between fits. Fix the profile and native
continuation used for paired action outcomes and record their identities.
For openers let D = net equity after opening minus net equity after waiting;
for closers D = net equity after closing minus net equity after holding.
Compare at the same terminal mark/time, including subsequent opportunities.
Use a common account unit for arbitration. Never conflate this attainable
target with retrospective best ZigZag timing. Close experts produce market
surfaces queried by the controller at actual barrier distances and time left.

- Comparator: constant FIT estimate, then a binary classifier with a separate
  chronological calibration block: P(D > 0). Keep ties. Publish Brier/log
  loss and calibration by role alongside account results; the probability
  of a win does not quantify its size or model uncertainty.
- First candidate: supervised distributional advantage per role. Estimate
  conditional mean D and conditional quantiles; measure realized quantile
  coverage, and measure disagreement across separately fitted seeds as a
  diagnostic, not a guaranteed error bound. Check for crossed quantiles.
  Compare on paired economic regret; in the account, wait if neither entry
  clears the frozen admission rule. Close vs hold uses its own rule.
- Challenger: binary discrete SAC per role, with native sequential rewards,
  current/next role routing and the other policies frozen. An opener's action
  cannot keep acting as an opener inside a position. Evaluate actual signed
  action advantage, not actor entropy as a calibrated confidence. Require
  equal market support, splits, costs and representation. Record unique
  states, environment transitions, optimizer updates and wall time separately.
- Convergence comparison: propose cap 2000 epoch-equivalents, a minimum
  of 100 before patience may stop a healthy run, then patience 80 monitored
  evaluations. Evaluate once per declared epoch. Define one supervised epoch
  as a full pass over the training pairs; for SAC report full chronological
  collection passes and updates per collected transition (not the same unit).
  Save early checkpoints without calling them convergence. Stop earlier for
  invalid data, nonfinite loss or broken routing, with an explicit failure
  receipt. Select on internal chronological blocks; reserve real 2024 for
  the declared comparison, not scheduler decisions. Run at least three paired
  seeds, report last vs best and block/seed dispersion. Qualify weekly refits
  separately by training to each cutoff and scoring the next real week.
- Integration: freeze the other three experts and change one role at a time;
  then replay the four together from the same starting account. Router must
  abstain on conflicting tied entries, never reverse on close, never call the
  opposite closer, and always enforce native risk and calendar protections.
  Test missing/nonfinite/out-of-support reports: prohibit new risk, retain
  mandatory exits/protection, and log a controller fallback independently of
  learned NOP. Report by side, harmful early exits and missed beneficial exits,
  net after costs, drawdown and qualifying activity. Do not promote a method
  merely because validation once peaked or because it ran to the epoch cap.

## Adversarial review: admission checklist

1. **Define what the expert can actually know.** Freeze the four role IDs,
   ordered feature hashes, window, timing, native fee and fill convention,
   risk profiles and clock. A different horizon changes the order: predict
   a separate indexed output per eligible profile or fix one declared profile
   for the comparison; do not pool different profiles under the same scalar.
   The encoder sees market only. The controller may query indexed close
   surfaces with causal order coordinates. Check whether the same observable
   coordinates still hide materially different decisions (e.g. carry cost
   or executable time); report irreducible regret, not just aggregate accuracy.
   Geometry, position eligibility and broker exits must come from the engine,
   not a learned head. Publish an ideal-output truth table before fitting.
2. **Do not train the old policy back into the new one by accident.**
   [381](381_SHORT_TARGET_RESULTS.md) found 30/135 sign reversals on the
   general short panel when changing continuation at fixed 24 decisions.
   For identical snapshots archive the local order P&L, later account P&L,
   sign, horizon and continuation ID; repeat under an alternative legal
   frozen reference. Report near-zero ties separately from material flips.
   Unresolved reference-sensitive cases are not unconditional ground truth.
   Keep the optimal ZigZag exit score as a separate TRAIN-only measure.
3. **Compare account assignments, not two independent yes/no votes.**
   On verified 385 FIT flat data (bp columns 1,2; threshold 1e-8 bp),
   neither action beat waiting on 818 ticks, buy only on 4200, sell only on
   4326, and both on 363. Mean advantages were -5.59 and -4.64 bp.
   Archive paired WAIT/BUY/SELL on the same snapshot and settlement clock.
   Score regret of the one selected action including abstention, not just
   marginal accuracy or max probability. For multiple profiles compare at
   a common account clock with reinvestment, not different exit dates.
   Include mirrored buy/sell, both-positive, both-negative, ties and cost
   reversal. Shuffle candidate enumeration and require identical action.
4. **Do not confuse a simulator outcome with an executable choice.**
   Keep forced close, execution refusal, native intrabar stop, missing quote,
   and truncated suffix as separate outcomes with denominators. In 385 the
   endpoint could truncate at a split boundary and the closer sampled only
   surviving first decisions (direct_data385.py). Do not pad these cases,
   silently drop them or inject an already stopped position into the closer.
   For assessments needing resolved future pivots/suffixes, purge unresolved
   origins at split boundaries. Replay paired branches from cloned identical
   account/RNG states and reconcile to ledgers. Require a perfect-information
   routing witness and a no-information/no-edge control on prescribed paths.
5. **Match train populations to the intended account.** Maintain separate
   ledgers for reachable baseline positions, constructed reachable stress
   positions, and positions actually visited by the assembled candidate;
   report coverage/support and regret for each. On TRAIN, replay the assembled
   policy to identify newly visited states. Do not repair or tune against the
   confirmation weeks. In [382](382_SHORT_FACTORIAL_RESULTS.md) a head's
   lower local regret did not guarantee better account net. Compare entry
   pair and exit pair replacements as well as single roles and the full four.
6. **Chronology applies to inherited weights and features too.**
   [383](383_PREFOLD_UPSTREAM_INVENTORY.md) found zero qualifying pre-fold
   71-input bundles in its census; do not extrapolate that count to 61 inputs.
   Verify cutoffs for feature selection, producers, scaler, ZigZag/risk fit,
   encoder, generator, reference policy and calibration for *each* fold.
   A fixed retrospective recipe comparison is useful but not an independent
   weekly refit test. Keep the established 2024 comparison explicitly
   historical where it informed upstream selection. For a generalization
   claim require genuine later unused weeks or prospectively frozen scoring.
   Separate 8 real, nested 4 real, and 4 synthetic prefix + 4 real arms.
7. **Do not let oversampling buy false certainty.** Retain source frequencies
   and sampler weights. Train coverage and stress distributions may differ
   from the market; calibrate and assess reports on later real blocks at
   natural prevalence. Check interval coverage/width, tail losses, action
   rate and regret by role, scenario and support. No generated parent path
   or donor segment may cross folds. Explicitly test a withheld scenario
   family and a shuffled-target control; no universal-coverage claim.
8. **Convergence and admission must be operational, not aspirational.**
   Before an expensive run persist a machine-readable experiment manifest:
   four role-specific trainable parameter inventories, frozen module hashes,
   data/label/engine identities, split dates, reference policies, budgets,
   controller fallback and arbitration, and monitor/restore definitions.
   Predeclare practical effect margins against controls, minimum activity,
   tail-risk limits, support minima and block-interval rule using business
   limits or TRAIN estimates, not the winning run. Until populated these
   fields are unresolved, not defaults. Check one optimizer update changes
   only the intended expert; a cold reload must reproduce a decision trace.
   Record last/best/restored epochs, LR and reason; compare across multiple
   weeks/seeds. Smoke tests only test plumbing. A long run or an isolated
   peak proves neither convergence nor reproducibility. Do not start floating
   selection or broad DEAP before the composite passes the account criteria.
