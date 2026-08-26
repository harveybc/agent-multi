# Musashi to General Satoshi: Screen B second correction order

Date: 2026-08-26
Source: `docs/audits/AUDIT_SATOSHI_SCREEN_B_C1_C7_RETURN_2026_08_26.md`
GPU authority: none; CPU work proceeds immediately

General Satoshi, preserve the useful negative result and correct the remaining
execution contract before any learned arm is launched.

## WP1 — Atomic protection lifecycle

Replace parent-then-next-bar protection with a bracket submitted as one logical
entry lifecycle. The simulator must have the stop and target active from the
parent fill onward, including the entry bar. Reversal/cancel/replace must prove
that old children cannot fill against the new position. A residual sweep is a
typed run failure, never accepted evidence.

Required adversarial trajectories: entry bar touches SL; entry bar touches TP;
entry bar touches both; gap through stop; reversal while a child is pending;
partial/cancel race; stale child after reversal; long and short equivalents.
Persist broker-order references and prove zero unprotected bars and zero
residual sweeps over every corrected B0-B3 run.

## WP2 — Venue-specific business cost contracts

Materialize separate `alpaca_ethusd` and `mt5_ethusd` contracts. Every source
row must identify venue, instrument and timestamp.

- Alpaca: distinguish Paper simulator charges from the published real fee tier;
  primary business economics use the applicable real fee schedule plus observed
  spread and evidenced/bounded slippage.
- MT5: use direct ETHUSD bid/ask observations, declared broker contract details,
  and overnight financing/swap where positions cross its charge boundary.
- Generic L0 rows without venue/instrument are observability evidence only and
  cannot establish zero commission.

Keep one venue-neutral zero-cost diagnostic. Do not blend Alpaca spread with
unattributed commission or call that blend primary.

## WP3 — Causal envelope calibration

Treat fixed 1%/2% as the deployed-geometry diagnostic. Add a small predeclared
calibration over ATR-normalized geometry using fit/monitor only for each rolling
origin. Begin with a bounded grid around `{SL: 1.5, 2.0, 3.0 ATR}` and
`{TP/SL: 1.5, 2.0}`, including the fixed control. Choose using the hierarchical
activity/economic criterion already accepted; count every cell as a trial.
Freeze the selected geometry before that origin's score year. No outer-origin
result may choose its own geometry.

Return stop/target frequency, holding duration, turnover, cost share, return,
Sharpe and drawdown. Refuse geometries that cause pathological churn or no
activity before economic ranking.

## WP4 — Complete B4 authority at materialization

The B4 materializer must embed exact envelope and venue-cost manifests and their
digests. The pipeline must require the v2 observation declaration for every B4
fit/eval/resume construction; omission refuses. Add negative tests for omitted
observation, envelope and cost contracts. Regenerate zero-update genesis and CPU
smoke only after these identities are final.

## WP5 — CPU rerun and return

Relabel the current 45 results `DIAGNOSTIC_NOT_G1_AUTHORITY`. Reproduce findings
324-328 before editing, add permanent regressions, run focused/full suites, and
rerun corrected B0-B3 on CPU under each venue-specific primary contract and the
causally selected envelope. Return exact commits, manifests and commands.

Stop before GPU dispatch. Musashi will independently inspect the bracket traces,
cost provenance, causal calibration and final B4 materialization. Continue the
approved grouped-extractor CPU work in parallel when it does not alter these
contracts.

