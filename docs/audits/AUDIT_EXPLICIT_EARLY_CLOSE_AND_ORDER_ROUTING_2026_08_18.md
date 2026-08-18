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

Those defects are corrected in `gym-fx@1a606df`, `lts@d93cdda` and the
agent-multi delivery commit containing this report. The replacement P1LR
identity is new and starts from new zero-update 2,660-input genesis artifacts.
No prior P1LR terminal is promotion-eligible.

## Findings

| ID | Severity | Finding | Disposition |
| --- | --- | --- | --- |
| EC-01 | S2 | SAC had no explicit close-to-flat action | corrected with `target_exposure_hysteresis_v2` |
| EC-02 | S2 | Paper/Demo runners skipped inference while a route had exposure | corrected; infer then close/monitor, no same-bar reversal |
| EC-03 | S2 | agent state depended on episode length and unrealized PnL was not live-reconstructible | corrected with position/equity/true PnL/holding-duration state |
| EC-04 | S2 | MT5 exposed 60 H4 bars, insufficient for nested feature warm-up plus 256-row scaling | corrected source requires at least 800; EA recompilation remains a human deployment step |
| EC-05 | S3 | adaptive market/limit/stop behavior came from plugin defaults rather than the P1 causal contract | P1 fixes market and TTL zero; routing moved to document 39 |
| EC-06 | S2 | active venue seats used labeled linear canaries rather than a corrected SAC champion | unresolved by design until corrected smoke/decision emits a valid artifact; SAC route is implemented and fail-closed |
| EC-07 | S3 | finite warm-up gaps could have been converted to zero in live observation | corrected; incomplete derived history refuses inference |

## Reproduced Evidence

- historical parity: 18,085 timestamp-matched rows, all 83 features matched the
  frozen model-ready file within `rtol=2e-5`, `atol=2e-6`, zero mismatched
  columns;
- `gym-fx`: 91 tests passed;
- `lts`: 728 tests passed;
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

