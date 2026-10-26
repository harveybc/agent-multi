# Learning mechanism diagnosis

Owner accepted the ordered investigation: reward chronology, close observability,
learning gradients, then a narrow paired training contrast. This is PPO hybrid,
not SAC. Preserve the admitted 120-column recipe, 13 observations at stride 2,
fixed TRAIN constants, action mask only, costs and business rules. No expert,
new account inputs, forced directional balance, trade bonus or TEST access.

## Closed predecessor

Inference audit completed 21 units / 42 TRAIN+INNER evaluations in 586.43 s;
42 reports delivered/read back without discrepancies. Conditional sampling beat
its matched feature-blind control in profit in 4/9 INNER comparisons. Mean
INNER profit difference +25.46 USD and Sharpe difference +0.091 do not establish
robust learned timing. No new model selected. Original metrics remain intact.

## Tests designed before implementation

| Requirement | Independent witness | Acceptance |
|---|---|---|
| Action precedes reward | Native engine trajectories, both directions | Reward uses before/after equity of the executed transition |
| Costs neither disappear nor double | Sum equity rewards, native trades, terminal balance | Reconcile independently at gamma=1; log telescopes separately |
| Terminal versus rollout cut | Nonzero bootstrap with variable durations | True terminal excludes next value; nonterminal cut includes it |
| Missing observations do not erase P&L | Automatic HOLD interval | Discounted per-bar sum and exact elapsed duration |
| Position ambiguity | Same market tensor and mask, distinct valid SL | Demonstrate an opposite close/HOLD preference without adding inputs |
| Gradient geometry is meaningful | Aligned, opposed, zero analytical gradients | Exact cosine/sign; zero has undefined cosine |
| Instrumentation is non-mutating | Weights, existing .grad and RNG before/after | No optimizer step or changed parameters/state |

Native fixtures are engineering evidence, not new financial data. Financing is
currently debited at close: disclose timing separately from accounting accuracy.
No historical checkpoint or report is rewritten.

## Finite real diagnostic

Authenticate the closed parent and three CAT_FIXED final checkpoints. For each,
collect 512 fresh on-policy transitions at three fixed TRAIN anchors (0, 1/3,
2/3 of the admitted decision span), starting flat with original capital. These
are diagnostic restarts, not portfolio evaluations. Compute the actual PPO
surrogate/value/entropy gradients on that rollout without applying updates.
Record routing frequencies, rewards, duration, value residuals, gradient norms
and cosine on the shared encoder; source, support and checkpoint hashes.
No INNER/2024 fitting, no best-seed selection. Nine diagnostic units, zero fits.

At the old policy, ratio=1 and KL=0 by construction. These cannot diagnose
update magnitude; use stored training logs for that and instrument subsequent
fits if a contrast is warranted. A gradient conflict is evidence on sampled
rollouts, not proof of general causation. The three anchors do not cover all TRAIN.

## Conditional next contrast

First resolve any failed causal/accounting test. If shared-gradient interference
is observed, compare shared versus independent actor/critic encoders using the
same branch architecture, identical copied initial encoder weights, same seeds
7/19/43 and budget. Explicitly report extra parameters in the independent arm.
Do not change gamma, reward scaling, architecture width and update budget in
the same contrast. If there is no relevant conflict, select the next experiment
from measured value/credit-horizon evidence instead. No speculative large grid.

Financial evaluation at spaced checkpoints, not costly per-update early stopping.
Report TRAIN and INNER separately with profit, Sharpe, drawdown, trades, win rate,
long/short counts and voluntary/protective exits. More trades alone is not success.

## Diagnosis measured and bounded contrast authorized

Native reward/terminal/aliasing and analytical gradient tests: 13 passed. The
synthetic financing witness handles both debit and credit according to the
declared side-specific rates. Gradient audit: nine units, 18 batches, 4608 fresh
TRAIN decisions, zero parameter updates, independent 12-file copy verified.
Actor/critic cosine is negative in 14/18 batches, including all six batches of
seed 19 (-0.67 to -0.33). Weighted critic/actor gradient norm ratios range about
0.12--1.74; no universal critic dominance. Value explained variance is generally
near zero. This supports measuring encoder separation, not declaring it a fix.

Frozen next queue: four new independent-encoder synthetic learning controls
(seeds 7/19/43 at current signal; seed 7 lag 6), same 131072-decision protocol
as the parent's accepted shared encoder. All must pass nine contexts and exact
reload before financial training. Then six fresh fits, paired by seed:
SHARED, INDEPENDENT x 7/19/43, CAT_FIXED equity reward, 102400 decisions each.
All original training parameters retained. Only encoder sharing changes; the
independent critic is an exact deepcopy at initialization (extra parameters
reported). Compare initial predictions/values and each common parameter exactly.
Checkpoints at 25600/51200/76800/102400, financial scoring only at the fixed final
checkpoint on TRAIN and INNER; no new 2024 read or selection. Instrument training
logs and sparse gradient probes without perturbing RNG, then publish all six.
Two fresh attempts maximum per unit, durable failures retained, reboot-resumable
queue; a synthetic failure prevents financial fits. No intermediate human gate.

## Deployment

Code `e31e7d8d`; spec
`5e655081d3297f4c0e2c4ee881bd338f4d5ca4ae462ec7a59eb49df29f78c738`.
At 23:57:43 UTC the first synthetic control had reached 5632/131072 decisions
on the primary GPU, service RAM about 2.5 GB. No financial fit started at that
observation. Worker and independent copy/publication timers enabled; first copy
and warehouse pump succeeded (zero scientific reports expected yet).
20 focused tests pass, including bounded qualification failure before finance,
receipt corruption rejection and exact instrumented/uninstrumented training.
Neither repository-wide tests nor a live broker execution is claimed.
Parent shared financial fits took 363--366 s; new synthetic throughput suggests
about 45 minutes for four controls. Whole queue estimate 90--120 minutes is
provisional: independent financial fits have not yet been timed.

## Mandatory trading breakdown

Owner requests per-direction trading evidence, not just aggregate profit. The
read-only `tools/trading_ledger_report.py` authenticates stored receipts and emits
run, direction, exit-reason and ledger-check CSVs. Count a completed trade once;
BUY/SELL shares use all trades; each directional win percentage uses only that
direction and strictly positive NET P&L, including all costs. No trades means
undefined win percentage, never zero accuracy. Keep every seed and split apart.
Include gross/net, four cost components, win/loss size, profit factor, expectancy,
durations, streaks, protective/voluntary exits, SL/TP distances and sizing ranges.
Native Sharpe/DD retain their original definitions. Original CAT_FIXED and the
exact ARGMAX replay are explicitly duplicates, not independent observations.

The first report covered 114 stored period reports (72 original, 42 inference),
with zero detected arithmetic/protection/count/position-overlap discrepancies.
This is NOT independent calendar/data or broker-cost certification. Behavioral
limitations remain: fixed-risk seed 43 is long-only with zero voluntary exits;
seed 19 misses the mean weekly activity requirement on INNER. No metric or
strategy is silently redefined to make those runs eligible.
