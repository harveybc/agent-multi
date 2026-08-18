# Audit: Explicit Early Close and Order Routing

Date: 2026-08-18
Auditor/implementer for this correction: General Musashi
Runtime mutation: invalid P1LR workers and risk-increasing live decision runners
were stopped; existing broker-native SL/TP protection and observers remained
active. No real-capital authority exists.

## Verdict

The pre-2026-08-18 experiment and live controller were not suitable for the
owner's business question. The action could request long or short but could not
explicitly target flat, and live runners stopped inference whenever exposure
was present. The observation also included a backtest-only episode countdown
and could not compute true unrealized PnL after raw prices were removed.

Those defects are corrected in `gym-fx@1a606df`, `lts@1c16d2d` and the
agent-multi delivery commit containing this report. The replacement P1LR
identity is new and starts from new zero-update 2,660-input genesis artifacts.
No prior P1LR terminal is promotion-eligible.

## Findings

| ID | Severity | Finding | Disposition |
| --- | --- | --- | --- |
| EC-01 | S2 | SAC had no explicit close-to-flat action | corrected with `target_exposure_hysteresis_v2` |
| EC-02 | S2 | Paper/Demo runners skipped inference while a route had exposure | corrected; infer then close/monitor |
| EC-03 | S2 | agent state depended on episode length and unrealized PnL was not live-reconstructible | corrected with position/equity/true PnL/holding-duration state |
| EC-04 | S2 | MT5 exposed 60 H4 bars, insufficient for nested feature warm-up plus 256-row scaling | corrected source requires at least 800; EA recompilation remains a human deployment step |
| EC-05 | S3 | adaptive market/limit/stop behavior came from plugin defaults rather than the P1 causal contract | P1 fixes market and TTL zero; routing moved to document 39 |
| EC-06 | S2 | active venue seats used labeled linear canaries rather than a corrected SAC champion | unresolved by design until corrected smoke/decision emits a valid artifact; SAC route is implemented and fail-closed |
| EC-07 | S3 | finite warm-up gaps could have been converted to zero in live observation | corrected; incomplete derived history refuses inference |
| EC-08 | S2 | Alpaca's signed short quantity was negated again from `side=short`, reversing perceived exposure | corrected in `lts@ed3cf67`; exact negative-short regression added |
| EC-09 | S2 | an Alpaca model close outside regular hours cancelled protection before its market flatten could fill | corrected in `lts@ed3cf67`; defer and preserve protection while closed |
| EC-10 | S2 | close-first allowed a reverse entry on the next daemon tick of the same bar | corrected in `lts@1c16d2d`; durable model-close fact consumes the full bar on all venues |
| EC-11 | S3 | an unavailable IBKR historical-data session could hold the runner inside one request while emitting errors every two seconds | corrected in `lts@1ec2889`; bounded request timeout restores degraded-heartbeat/backoff behavior |
| EC-12 | S4 | SIGTERM during an unsuccessful IBKR reconnect could execute `None.close()` and leave the unit failed | corrected in `lts@1ec2889`; absent runner closes are guarded |
| EC-13 | S2 | the installed `p1lr-decision@` template still defaulted to the v1 contract and old screen gate, so merely unmasking it after the v2 screen could run or verify the wrong identity | corrected operationally on all three hosts with an explicit immutable-runtime drop-in binding v2 in `ExecStartPre` and `ExecStart`; upstream installer hardening and independent effective-unit reproduction assigned |
| EC-14 | S4 | normal cells currently label per-checkpoint source as `easy_training_epoch`, although phase/config/hash facts prove normal dynamics | do not restart or reinterpret the 16-cell screen; correct the source enum before the next experiment identity and retain the current raw facts |

## Reproduced Evidence

- historical parity: 18,085 timestamp-matched rows, all 83 features matched the
  frozen model-ready file within `rtol=2e-5`, `atol=2e-6`, zero mismatched
  columns;
- `gym-fx`: 91 tests passed;
- `lts`: 736 tests passed after live-control and IBKR continuity corrections;
- `agent-multi`: 1,647 tests passed; four sibling-repo lookup failures caused
  only by the isolated `/tmp` worktree, then all 10 affected tests passed with
  the real `doin-node` sibling mapped into the test root;
- corrected zero-update artifacts: four distinct 2,660-input policy tensors,
  one per seed, each bound to the new observation-contract digest.

## Residual Gates

1. Run the one-pass mechanics smoke and reject constant/dead/genesis-equal
   selections before full spending.
2. Compile and attach the updated MT5 EA with `InpClosedBarHistory=800`.
3. Publish a SAC manifest only from a loadable corrected trained artifact with
   exact config, observation and parity evidence.
4. Reconcile direct broker facts before restarting any risk-increasing runner.
5. Treat limit/stop/stop-limit as a later execution experiment. Current P1 uses
   market so routing cannot confound curriculum attribution.
6. Before decision dispatch, reproduce the effective systemd unit on every host
   and prove runtime revision, v2 contract, v2 screen gate and seed/GPU binding;
   a loaded generic template is not sufficient evidence.

## Runtime Addendum

The corrected mechanics screen was deployed from one detached runtime commit,
with one seed per physical GPU. During the first live replay, MT5 directly
closed a protected short on an opposite model signal; the pre-addendum code
then opened long on the same H4 bar. That long was directly observed with
native SL and TP. The corrected runner now reports
`model_close_pending_same_bar` with `orders_submitted=0`; this is measured
evidence for EC-10, not a hypothetical counterexample.

Alpaca likewise exposed EC-08 with a real Paper short. The erroneous close
became a queued market flatten while the equity market was closed. The runner
is now corrected and bar-gated; the existing Paper transition must be followed
to direct terminal facts rather than relabeled as success.

IBKR retains a protected USD.CAD short but cannot currently obtain 51 H4 bars:
TWS reports that the trading session is connected from a different IP address.
The runner is observably degraded and submits no order; native broker
protection remains the direct safety layer until the market-data session is
reconciled.

## Screen Acceptance and Decision Dispatch

The replacement screen completed 16/16 and was sealed with tree digest
`61a59e2121b158239d8cd31d94167796b9eb301463228e8aee2cf0717fe48594`.
Dragon loaded all 16 terminal models. The typed verdict is
`SCREEN_VIABLE_REGION`, with all gates true, 7 active cells, 9 measured inactive
cells and six viable handoffs. No screen return is promoted as performance.

The first decision start reproduced EC-13: per-seed operator environment files
overrode the newly pinned gate with the old v1 path. All four `ExecStartPre`
checks refused the foreign contract hash before any trainer existed. The units
were stopped; the operator environment override was removed from the effective
v2 drop-in; gate/verdict and contract hashes were matched on all hosts; and the
four effective units were re-inspected before restart.

At 2026-08-18 08:09 America/Bogota, all four decision units were active under
identity `ac0941e7bdb1a163`, each with assigned UUID equal to
`CUDA_VISIBLE_DEVICES`, v2 contract in both preflight and runner, and the sealed
v2 verdict. EC-13 remains assigned for upstream installer correction and
independent reproduction; the runtime incident itself submitted zero broker
orders and created zero duplicate training writers.
