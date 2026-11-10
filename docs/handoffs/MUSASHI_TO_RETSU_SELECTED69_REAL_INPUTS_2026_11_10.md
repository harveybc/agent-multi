# Retsu: exact selected69 actor and real hourly inputs

2026-11-10. Bounded handoff; Musashi resumes phase2 selection after this delivery.
This is an acquisition/reproduction contract, not an assertion that the live
producer is operational. No broker service or existing position was changed.

## Model to integrate

The next selection starts from the preserved **69-input, 28-group** actor,
not the experimental close fields and not the degraded fine-tuned actor.

Local authoritative bundle: `/var/tmp/sac69-weight-bundle-20261108/bundle/`.

| Artifact | SHA256 |
|---|---|
| model.pt | `8ab11f607dc18c53ce45b34dc628e21272767702d60e0bfbe272620761de4e40` |
| MANIFEST.json | `69f2c7e48a0028c9695a65dafaf3b9d5a1d7c4e33542340eb062fbea3bc5d424` |

Take `RECIPE.json`, `MODEL_CONFIG.json`, `PROBE.npz`, weights and manifest
together. `RECIPE.json:list.columns` is the exact ordered list;
`tensor.columns` describes upstream versus consumer transformations;
`scaler.features/mean/std` are the fixed TRAIN constants. Do not estimate new
constants or apply LOG/ASINH/DIFF twice. Inspect manifest checksums before loading.
Use the existing native `app.structural_refit.load_bundle` inference path.

The 70-input model you integrated (`079bef0e...`) remains a distinct reference.
The 69-input bundle physically removed `RAW_CLOSE_LOG_RETURN` with retained
weights preserved. It is not sufficient to give 69 columns to the old weights.
Step225 records full-support action/account parity and reload verification.
No further selection winner is automatically promoted into the live pool.

Window: 13 observed closed H1 bars, offsets -24,-22,...,0, not 13 clock hours.
Market closures occupy no invented slots. Upstream warmup is longer than this
window. Infer in eval mode, categorical greedy action; no agent-state inputs.
Historical TRAIN/VALIDATION boundaries must not truncate the forward feed.

## Exact historical reference and producer evidence

Use the immutable copies already consumed by the actor:

- `/var/tmp/sac70-warm-v3-20261108/data/store.parquet`
- `/var/tmp/sac70-warm-v3-20261108/data/manifest.json`
- `/var/tmp/sac70-warm-v3-20261108/data/bars.parquet`
- `/var/tmp/sac70-warm-v3-20261108/data/closures.json`
- `/var/tmp/sac70-warm-v3-20261108/SUPPORT_RECIPE.json`

These are historical reference inputs, **not current market feeds**. Their
checksums are in that run's `SPEC.json:files`. The selected69 recipe is the
consumer contract; do not substitute a mutable similarly named full-pool file.

Producer lineage is also recorded under `/var/tmp/features-candidates/`:
`eurusd_h1_layer1_pool_r4`, `eurusd_h1_layer2_pool`,
`eurusd_h1_full_pool_s_a1`, `eurusd_h1_activity`, and the cross-asset stores.
Their manifests identify source digests. They are recovery/reference locations,
not permission to use any alternate revision. In particular,
`eurusd_h1_activity/ACTIVITY.json` contains historical metrics/features and
`P_HIT.parquet` contains scores; neither is an exported fitted estimator.

Your existing `docs/forward_model_pool/PRODUCER_MANIFEST.json` records frozen
GMM/Kalman recovery and upstream warmups; retain its receipts. Preserve the
exact state and causal initialization for both across process restarts.

## Recovery and acquisition by family

| Current inputs | Real parent / recovery location | Required action before hourly use |
|---|---|---|
| `act.p_hit` | Activity HGB; historical `eurusd_h1_activity` scores and producer provenance | Recover original fitted estimator, ordered inputs, transforms, fit boundary and package versions. Metrics/parquet alone cannot reconstruct it. Search producer/export archives. If absent, recover by reproducing the original fit from the exact archived source/config/data, then require historical score parity before accepting the recovered state. Do not fit a replacement on recent data and call it the same feature. |
| `fac.fx_f1.ret_4`, `ret_24` | Frozen first FX component over GBPUSD, USDCHF, USDJPY, EURGBP, EURJPY, AUDUSD | Recover full-precision components, means/scales, sign/order, return definition and fitted window from the cross-asset producer. Rounded loadings are insufficient. A deterministic reconstruction must reproduce the sealed historical factor, including downstream returns, before admission. |
| `cot.eur.net_long`, `net_share`, `net_z156` | CFTC EURO FX code `099741`; same report type/fields as original producer | Obtain actual released reports and preserve vintages/reference date/publication time. Verify overlap against sealed columns, keep the original denominator and 156-release history. Never replace the release clock with the Tuesday reference date. |
| `wv.usd.A4` | Signed USD-strength returns from GBPUSD(-), AUDUSD(-), USDCHF(+), USDJPY(+) | Collect all actual cross bars on the admitted clock, reproduce original strength and causal wavelet window/endpoint. Missing leg is not zero. Check historical prefix before processing new observations. |
| `hb.eur.inst_freq` | EUR-strength returns from EURGBP and EURJPY | Same real aligned cross bars; original strength convention and trailing Hilbert window128. Preserve initialization and phase/frequency calculation; no full-series noncausal replacement. |
| `dly.us10y.chg_1d`, `chg_5d` | FRED `DGS10` release observations | Retain your verified overlap receipt and observation-based changes. Missing FRED observations remain missing, not zero or interpolated. Preserve training availability semantics and record actual retrieval/publication evidence. |
| Remaining 59 selected columns | Admitted EURUSD bid OHLC, calendar/events, deterministic transforms, frozen Kalman/GMM | Recompute the exact producer chain, verify complete historical-prefix values and then extend causally. Event clocks use their actual timezones/DST, not guessed fixed offsets. Recover necessary warmup/state; the model window is not enough. |

Count check: 59 + 2 bond + 8 other external/learned = 69. Your 70-column report
described the remaining 60 as bid/calendar while listing the two bond inputs
separately. They must be accounted for separately: the old model had 60
bid/calendar/frozen-producer columns, two bond columns and eight others.

The attached report establishes that the pinned public FX revision stops in
May2026. It cannot deliver today's hour. This handoff has not verified a current
continuation endpoint. An actual current bid/cross feed must be admitted with
clock, symbol, price-side and aggregation parity; a different provider is a
source change requiring measured overlap, not a transparent substitute.
MT5 execution quotes are not automatically identical to training bid candles.
Do not use synthetic extension, backfill with unrelated assets, or claim that
paid calendar/crypto services supply these FX features. Report actual vendor
coverage and any credential needed, without publishing secrets.

If original activity/PCA state cannot be recovered with parity, the current
actor cannot be deployed unchanged. Keep that failure explicit. A separately
trained model without those inputs is a new candidate, not a live-feed fix.

## Freshness, completeness and restart

At each scheduled hourly decision require the latest completed market bar and
a computed feature vector available no more than one hour behind the decision.
Store event time, available-at and observed-at separately. Reject duplicate,
out-of-order, partial or nonfinite windows. No null, zero, interpolation or
synthetic substitutes for missing cells. Zero naturally produced by an actual
feature calculation is valid; a missing-value zero is not.

COT and daily rates are not hourly releases. Their last genuinely available
published observations remain usable only under the exact training as-of
policy; calculate the hourly feature vector from those real observations and
record their age. Requiring a new COT report every hour would be impossible and
would silently redefine the feature. Do not forward-fill a market-feed outage.
Closed-market periods generate no fabricated H1 bars.

Prefix validation must cover every retained column, the final scaled window,
logits/actions, and restart from saved producer state. Perturbing future rows
must not change earlier outputs. Reuse the native probe, then historical full
account replay; a correct final action alone does not establish tensor parity.
Separate `CODE_VERIFIED`, `HISTORICAL_REPLAY`, `SHADOW_OBSERVING` and
`PAPER_DEMO_EXECUTING`. Real freshness is required for the last two.

## Native strategy and pool succession

Do not copy stale inherited expert metadata from the recipe as executable rules.
Use `env_plugins/three_signal_native.py` and the pinned run's `RISK.json`:
flat NOP/BUY/SELL; an opposite signal closes an existing position without a
same-tick reversal; SL/TP and forced-flat protection retain priority.
One position at a time, leverage1:50, margin cap `q * 0.1 * balance(t)`;
no old one-entry/day limit. Entries begin12h after weekly open; forced flat12h
before weekly close. Protection is not postponed until the next inference.

Keep the original reference and every admitted candidate in parallel virtual
accounts on the same real timestamps, quotes/spread, costs and calendar.
Follow step221: at least one complete common forward week and valid coverage
before weekly comparison; record both common-window and lifetime performance.
Freeze the existing ranking/risk/activity rule before looking at new results.
One authorized writer per paper/demo route; queue succession until flat.
The opening model retains close responsibility for its existing position.
Historical replay does not count as a forward week. No live-capital activation.

Deliver back the artifact hashes, column-level prefix/restart checks, source
freshness evidence, complete-window probe and actual model hash in the heartbeat.
Do not replace EURUSD with the old SPY/ETH linear runners. Broker/symbol support
must be verified independently for each paper/demo route.

## Ownership and research resume point

Retsu owns acquisition, state recovery and the live pool. Musashi now resumes
step261's phase2 work: preserved69 baseline, ordered deletion, cumulative
phase2 reinsertion inventory, component weight analysis, exact export/parity.
No phase1-invalid inputs are automatically restored. No failed critic threshold
calibration is imported into this actor. Live integration does not block the
historical phase2 experiment, and a phase2 result does not by itself authorize
live promotion.
