# 419: TRAIN state cuts and OPEN_SELL target alignment

Two separate experiments. Frozen 417/418 representation, closers and opening
stream. No 2024 data or promotion.

**State resolution.** Fifteen empirical quantiles per bounded coordinate,
pooled across 2019-22 training positions, roles and horizons, with ties
assigned to the right bin. Cuts saved before replay, including inverse
transforms to risk units and hours. All 16 bins occupied per coordinate.
Quartiles SL: 0.75/1.02/1.32 R; TP: 0.83/1.19/1.53 R; time: 23/45/65 h.
Quantized scalar improved fixed-entry mean payoff against the native curve in
all seeds, but sequential return lost against the curve in seed 43
(-19.43% vs -16.27%). No tested encoding passed the existing combined rule.

**Sell labels.** Freeze each head's 90th percentile from H1 probabilities;
evaluate its independent high-score offers in H2 of 2022 and 2023. Compare
passive training advantage and managed evaluation advantage on the exact
intersection of timestamps. Each H2 excludes 37 otherwise eligible origins;
2,459 and 2,192 matched origins remain respectively. Reference controls are
the expected random selection with the same count in each week/horizon.
These are hypothetical offers, not nonoverlapping account trades.

| H2 SELL cells, 3 seeds x 4 horizons x 2 years | Passive training label | Managed label |
|---|---:|---:|
| Positive mean advantage vs waiting |14/24|1/24|
| Higher win fraction than week-matched control |19/24|22/24|
| Higher mean advantage than week-matched control |16/24|18/24|
| Median mean advantage (bp per offer) |+1.35|-2.72|

**2023 paired bridge.** Matched high-score sells were replayed with passive
exit and the stored 404 standalone net matched within 0.0001 bp per offer.
The 404 waiting reference uses a passive reference trade of the same
horizon; 408/411 use a common learned-close 6h opportunity. With the
sale's passive exit unchanged, substituting the reference changes its
relative advantage. Only then is the sale's learned exit enabled. Medians
of the 12 cell means: +1.82 bp (original label), -1.67 (common reference,
passive sale), -2.60 (common reference, managed sale). Medians of paired
changes: -3.30 and -1.09 bp; medians need not sum. No claim about 2024
or the viability of an always-passive policy.

Decision: keep state curve; use a single explicit reference and the intended
managed exit contract when fitting and evaluating a specialist. Next useful
controlled contrast is to refit the *same specialist*, changing only its
training labels (not the downstream arbitration), versus a paired old-label
refit, then evaluate both on managed account outcomes. A label mismatch is
measured; profitability after correction is not established.

Artifacts: `/var/tmp/role419-cuts-sell/{SELL.json,SELL_BRIDGE_2023.json,CUTS.json,DONE.json}`
plus per-seed results and weights. Reproducers: `tools/role419_cuts_and_sell.py`
and `tools/role419_sell_bridge.py`, in the 413 pinned runtime.
