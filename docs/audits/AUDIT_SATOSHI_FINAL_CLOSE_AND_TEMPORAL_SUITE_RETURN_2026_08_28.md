# Audit: final close hardening and temporal-information suite

Date: 2026-08-28  
Agent-multi: `48a40c9b`  
Gym-fx: `b5d3fdb`  
Verdict: **CLOSE AUTHORITY PARTIALLY ACCEPTED; TEMPORAL PASS VERDICTS REJECTED**

## Findings

### 1. Critical: the phase-randomization control fails but is excluded from the verdict

The order requires phase-randomized surrogates to lose phase-dependent predictive performance. The report shows:

- returns/momentum: surrogate R2 0.8452 > pretrained R2 0.8348;
- trend/level: surrogate R2 0.3153 > pretrained R2 0.1958.

Nevertheless, both families receive `PASS` because `temporal_gate` never consumes `phase_surrogate_next_bar_r2`. This violates the predeclared acceptance logic and converts contrary evidence into a pass.

Required: phase preservation must be a typed gate where the target actually depends on phase. If a control is non-identifying for a family, classify it inconclusive rather than excluding its result after execution.

### 2. Critical: shuffled-time control also destroys sample/target alignment

The tool globally permutes the entire series, builds embeddings, then scores those embeddings against targets from the original unpermuted series. This destroys both temporal order and the identity of each sample. Near-zero R2 is therefore expected even for a time-blind but value-sensitive encoder and cannot establish that temporal order is useful.

Required controls:

- shuffle bars within each causal window while preserving that window's target and value multiset;
- compare an order-sensitive encoder with an explicitly order-invariant pooled baseline;
- use identical train/validation indices for every treatment;
- retain a global sample-misalignment negative only under a separate name.

### 3. High: four PASS results rely only on one trivial synthetic task

The suite uses one deterministic sinusoidal channel and a next-bar linear ridge probe. Random encoders already obtain R2 values up to 0.79, and volume reaches 1.0. A single seed and tiny positive deltas over random initialization are called passes without paired uncertainty or minimum effect.

The original order also requires real-data probes, per-feature reconstruction, frequency-band spectral error/coherence, lagged cross-correlation, quantile/volatility/barrier probes and fused-representation tests. None are present in this return.

Required:

- classify current output `SYNTHETIC_MECHANICS_DIAGNOSTIC`, not a temporal acceptance suite;
- run at least four signal seeds and four encoder seeds;
- report paired differences and confidence intervals;
- require a predeclared practical effect, not merely `>`;
- add real causal train/calibration/monitor probes for quantiles, realized volatility and barrier hit;
- test every family and the fused representation;
- add reconstruction and spectral/phase measurements by family.

### 4. High: lagged-memory R2 is in-sample

`lagged_correlation_preservation()` fits and scores least squares on the same observations. Its 0.86-0.9999 values are training reconstruction, not out-of-sample memory retention, and can be inflated by dimension and autocorrelation.

Required: chronological fit/validation split, ridge selected only on fit/calibration, naive autoregressive baseline, random encoder and shuffled-within-window controls.

### 5. High: Backtrader close economics use a stale exit price

`notify_trade()` sets `exit_price` from `bridge.price`, which is published in `_publish_obs()` and may still refer to the prior observation when the close callback fires. It then derives size as `gross / (exit_price - entry_price)`. Thus gross/net PnL can be correct while the recorded exit and size are mutually fabricated from a stale price.

Required:

- capture completed closing-order execution price, size, commission and order/trade lineage in `notify_order()`;
- join that immutable fill evidence to `notify_trade()`;
- derive gross/net from fills and reconcile against Backtrader's trade values;
- reject mismatch beyond tolerance;
- test gap fills, flat-price/breakeven closes, partial fills if supported and reversal close/open sequences.

### 6. Medium: exact duplicate identity is episode-local but not durable

The in-memory episode-scoped index is sufficient for one uninterrupted simulation episode. It does not by itself prove idempotence across process retry. Keep the claim explicitly episode-local; durable attempt identity remains the responsibility of the outer runner.

### 7. Medium: parameter conservation correction is accepted with wording limits

The new union accounting is improved, but the report must distinguish extractor-wide unique parameters from sums of overlapping submodules. This is mechanics accounting only and says nothing about appropriate model capacity.

## Accepted

- Strict event-field validation, PnL identity checking and typed conflict refusal are accepted.
- Exact duplicate payload replay is idempotent within an episode.
- Summary statistics no longer silently default malformed economics to zero.
- Architecture manifest v2 remains accepted as mechanics evidence.
- Structural controls for future immutability, newest-bar sensitivity, reversal sensitivity and save/load parity are accepted on the tested synthetic probes.
- Volume-flow is correctly classified inconclusive rather than passed.

## Dispatch ruling

Do not redispatch the long paired SAC campaign until finding 5 is corrected and reproduced. Continue the temporal-information suite immediately, but withdraw the four `PASS` labels and replace them with `SYNTHETIC_MECHANICS_ONLY_PENDING_REAL_PROBES` until findings 1-4 are resolved.

