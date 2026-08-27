# Pretraining Objectives — Formulas, Units, Ranges, Causal Diagrams

Order: post-transfer objectives 2026-08-27 WP1. Anchor convention: the
executing window of step `t` covers rows `[t-32, t)`; the last observed
bar is `a = t-1`. Purge between partitions = max forward horizon across
ALL objectives (mechanical). 2024 development_outer and sealed 2025 are
structurally absent from the fit slice.

## 1. Masked-patch reconstruction (accepted lineage)

Encoder input: runtime tensor with contiguous temporal spans zeroed.
Target: the (policy-transformed, target-side only) runtime tensor at
MASKED steps. Loss: MSE over masked steps. Range: z-scored units,
loss ~O(1).

    past ──[w_{t-32} … w_{t-1}]── t
              ▲ mask spans          │ no future access
              └── reconstruct ◄─────┘

## 2. Multi-horizon quantile (accepted lineage)

Target `y_h = log(C[a+h]/C[a])`, h ∈ {1,3,6,12}; pinball loss at
q ∈ {0.1, 0.5, 0.9}; head monotone by construction. Units: log return;
typical |y| < 0.1 on H4.

    [window ends at a]──►(a, a+h]: strictly forward log return

## 3. Hierarchical contrastive

Views: anchor window vs causal in-window smoothing at declared scales
s ∈ {2,4,8}: average-pool along time at s, upsample back (repeat),
left-pad with the oldest smoothed value — every value originates
INSIDE the window. InfoNCE per scale with declared temperature 0.2,
mean over scales. Projection head (excluded adapter): MLP dim→dim→32,
L2-normalized. Negatives: other train windows in the deterministic
seeded batch; declared false-negative policy: anchors within
exclusion_steps = 12 (the max horizon) are EXCLUDED. Loss range:
[0, log(batch)]. Diagnostics: embedding/projection std (collapse),
effective negatives, per-scale losses.

    window ──► encoder ──► z_anchor ─┐
       │ pool(s)+repeat (in-window)  ├─ InfoNCE(τ) per scale
       └─► encoder ──► z_view ───────┘   negatives: train batch minus
                                          |Δstep| ≤ 12 neighbors

## 4. Volatility (declared estimator, never a default)

Estimator `realized_vol_close_to_close`:
`r_i = log(C[a+i]/C[a+i-1])`, i = 1..h; `vol_h = sqrt(mean(r_i²))`;
annualization DECLARED ("none" here); target `log(vol_h + ε)`,
ε = 1e-8. Horizons {3,6,12}. Units: log of per-bar close-to-close
realized std. Typical range on H4 ETH: log-vol ∈ [-6, -2.5]. Loss:
MSE. Head: Linear(dim → H), excluded adapter.

    [window ends at a]──►(a, a+h]: forward returns ONLY
                └── trailing data never enters the target

## 5. Barrier hit (past-only scale, prospective barriers)

Scale at anchor: trailing realized vol over lookback = 64 bars ENDING
at `a` (past-only; lookback ≤ warmup 256 validated). Barriers:
`upper = C[a](1 + 2.0·scale)`, `lower = C[a](1 − 2.0·scale)`.
Labels over (a, a+h], h ∈ {6,12}: 0 = first upper hit, 1 = first lower
hit, 2 = neither/censored at horizon. Same-bar collision rule DECLARED
`conservative_adverse_first` (adverse/lower wins; with close-only
labeling a collision is structurally impossible — enforced for a
future OHLC upgrade). Class weights `total/(3·count)` FROZEN from the
CALIBRATION partition only (absent class → 1.0, recorded). Loss:
per-horizon 3-class weighted CE. Head: Linear(dim → H·3), excluded
adapter. Observed real-data label mix (bounded screen): all three
classes present.

    past ──[lookback 64]──► a ──►(a, a+h]: first-touch walk
           scale (past only)      barriers fixed at a, then prospective

## Adapter exclusion (all objectives)

Encoder artifacts (`branch_<family>_encoder.pt`) and head artifacts
(`branch_<family>_heads.pt`) are disjoint by construction; the runner
refuses key overlap, and the transfer loader's strict key-set equality
refuses any head/optimizer/calibration key. The exact key inventory is
published with the WP2 screen evidence.
