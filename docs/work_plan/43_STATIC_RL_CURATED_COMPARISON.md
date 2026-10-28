# Static RL on curated inputs

Owner directive: 2026-10-18. Implement engineering now; no financial fit before
data admission and selection recovery. Satoshi owns the independent source and
literature tasks already assigned. This is not an accepted scientific campaign.

## Experiment

First compare the existing SAC with direct observations against the same SAC
with small per-feature causal Conv1D branches and a temporal Conv1D core.
No periodic refitting or online gradient updates during evaluation. TRAIN,
VALIDATION and TEST stay separate; hyperparameters use development only.
Do not reuse affected weights, selections or results.

Initial hypothesis: 13 observed bars spaced by two admitted open-market hours,
ending at the current completed bar. Missing open-market hours must be repaired
or excluded explicitly upstream, not treated as scheduled closures. No smoothing,
interpolation or invented weekend ticks. Context can cross a known closure.
Source/availability and calendar admission remain prerequisites.

Temporal arm: one causal Conv1D per feature, four filters, kernel 3; concatenate
channels at equal timestamps, causal core Conv1D with eight filters, kernel 3.
No temporal pooling, Dense layer, BatchNorm, dropout or positional sinusoid in
this extractor. Flatten only at the policy-head interface, retaining all 13
positions. Actor/critic heads [32, 32] for both arms. These are initial settings,
not established optimal sizes; capacity and window sweeps come later.

Execution uses the existing protected-order strategy and session overlay.
At most one entry submission per business day (America/New_York initially);
the fill day is also charged, including delayed/partial fills. A cancelled
entry consumes that day's submission allowance. Protective children and exits
do not consume allowance. One position; reversal first closes, never opens
another position in the same action. Keep the 12h close/open exclusions.
Quota is enforced by the execution plugin, not learned by the agent.

SAC uses a thresholded scalar direction interface whose magnitude can also
control entry size: the shared envelope supports portfolio-fraction sizing.
It holds same-direction positions unchanged. The separate raw fractional path
can continuously rebalance and needs additional weekly risk-increase masking.
Do not describe SAC versus a discrete agent as identical action expressivity.
Before deployment certify session timing and next-bar fills on the real path.

Owner's later question: compare daily caps 1, 2, 4 after the static baseline,
without requiring any minimum number of trades. Keep the baseline at 1 pending
the proposed contrast; do not triple full training now. Train each cap under
its own environment (changing only evaluation caps creates a distribution
shift), keep initial equity, daily loss/exposure limits, costs, seeds and support
matched. Choose by development net profit/risk and stability, not trade count
or win rate alone. If 4 does not materially beat 2 after costs, prefer 2.
The quota applies per account/instrument in this single-asset experiment;
a multi-asset account needs a shared quota authority, not independent counters.

FINRA Notice 26-10 replaces old PDT requirements effective 2026-06-04, with
broker transition through 2027-10-20. This is not a universal daily-count rule
for spot EURUSD or automatic confirmation of a broker account's adoption.
Source: https://www.finra.org/rules-guidance/notices/26-10 .

## Metrics and governance

Net profit, end equity (initial USD10,000), return, maximum drawdown, Sharpe,
Sortino, entry submissions, filled entries, closed trades, net winning-trade
percentage, profit factor, costs, turnover, exposure and constraint violations.
Risk ratios use the full timestamped equity series, including flat intervals,
with explicit annualization. Undefined ratios remain null, not fabricated zero.
Compare no-trade, business-constrained passive long/short and random controls
on exactly the same data/calendar/cost support. Old SAC results are not a valid
baseline until their data and protocol identities are established.

Reuse governed_run/data-gov and warehouse outbox; do not invent a second cube.
Bind prepared file bytes, ordered features, admitted selection, source/calendar
identities and split to each run. Configured pins provide integrity, not proof
that a malicious producer's admission statement is true. Bulk artifacts stay
off Git; operational SQLite/locks stay on ext4. No new service is launched here.

## Test design and traceability

| Requirement | Required test / evidence |
|---|---|
| Temporal geometry | Output T unchanged; future perturbation leaves prefix unchanged; t affects final output |
| Policy integration | Real SB3 build, forward, save/reload for both arms; default legacy behavior unchanged |
| Window alignment | Stride includes step-1 (bridge count of completed bars); no future reads; no strict-mode padding |
| Daily cap | Same-day reentry blocked; next business day allowed; close/reversal allowed; delayed and partial fill charged once |
| Data identity | Wrong digest, split, features or non-admitted input rejected before constructing the environment |
| Static evaluation | Existing evaluator does not learn; later acceptance must prove frozen tensor hashes on real episodes |
| Business/equity | Existing session/accounting tests plus real-path acceptance before any financial fit |

Architecture: extend SAC configuration, use its SB3 feature-extractor hook;
extend gym-fx preprocessing/execution ownership, not predictor or DOIN.
Unit-test design is complete; component implementation and synthetic integration
may proceed. Scientific data acceptance, full environment acceptance and an
autonomous campaign launcher are not conferred by these tests.

## Literature candidate

CrossQ (ICLR 2024) is a useful later continuous-control challenger, not a proven
best FX trader. Its maintained SB3-Contrib implementation uses carefully placed
critic BatchNorm, and currently accepts Box but not Dict observations. Its large
default critic must not silently replace the compact comparison above.
Sources: https://arxiv.org/abs/1902.05605 and
https://sb3-contrib.readthedocs.io/en/master/modules/crossq.html .
Satoshi's bounded comparison is received. Its weekly-first/80-run proposal is
not adopted: this static comparison still comes first. Before training, close
the real-engine action/fill/reward transition and financing/accounting findings
recorded in gym-fx `RL_PREPARATION_REVIEW.md`. Synthetic encoder integration
does not certify those separate contracts.

## Execution repair status

Curated inputs now require `next_open_v2` + Backtrader + the shared execution
envelope. The same path is used in the flat and temporal SAC integration tests;
the policy's weights remain fixed during evaluation. Engine acceptance details
are in gym-fx `NEXT_OPEN_TRANSITION.md`. Financing can debit via an explicitly
declared analytical model/rate; actual broker rollover is not yet verified.
Absent/nonfinite returns and drawdowns cannot rank as zero-return/risk-free
checkpoints. Undefined risk diagnostics remain null in final reports.

Bounded automation (no service or financial fit):

```bash
python -m tools.verify_rl_static_engineering \
  --gym-root <matching-gym-checkout> --output-root <new-private-report-directory>
python -m tools.verify_rl_static_engineering \
  --output-root <same-report-directory> --status
```

Latest consolidated run: gym 256/256, agent 65/65, no skips. The additional nine
engine tests cover native trade lifecycle after direct settlement, repeated
closes, zero net PnL and a hand ledger with slippage/commission/financing.
Runner unit tests: 7/7. PRE regressions:
12/13 metric-integrity tests failed before the score correction. Reports/logs
are private generated artifacts, not committed datasets. This is engineering
acceptance only: data/selection admission, broker costs, full risk/accounting
acceptance and the financial campaign's sealed schedule remain open.
