# 420: does changing the SELL training target improve managed outcomes?

Two paired copies of the native market-only OPEN_SELL head (208 frozen
coordinates, hidden width 8, SiLU, four binary logits for 3/6/12/24h).
Old target: sign of the passive trade's advantage vs a passive same-horizon
reference. Aligned target: sign of the learned-close trade's advantage vs
the shared 6h managed reference. The two changes are bundled, not separately
identified. No synthetic data, new close models, BUY arbitration or 2024.

Use 2020 onward where both labels exist; intersect all eight alternative
targets (four horizons times two definitions) by decision timestamp and
market row. For a score year Y, fit through June Y-1 with a session-end
purge; H2 Y-1 monitors managed-label BCE for **both** arms, H1 Y fixes each
head's 90th-percentile threshold, H2 Y scores with native fees and closer.
Both arms receive identical initialization and permutations, batch of 512
origins with four horizon labels each, AdamW LR 3e-4, decay .001, rollback
on validation plateau (20 grace, patience 20, /5, LR floor 4.8e-7).
Retain each best monitor checkpoint; train the pair for the same number of
epochs until both patience counters reach 80 (counting from epoch 100) or
the 2,000-epoch cap. Not a replication of the old checkpoint or the best
classifier for the old label. This isolates the label under a shared deployment
endpoint; the optimizer states and best epochs may diverge as a consequence.

One-account replay permits only SELL or NOP when flat. Choose the horizon
with the highest H1 percentile among those strictly exceeding their H1
90th-percentile probability. The engine still has a single position and
uses the native closer. A separate descriptive comparison takes the same
number (top tenth, minimum one) of matched H2 origins per week/horizon in
both arms, and evaluates signed managed advantage; it cannot route live orders.

All three seeds (7/19/43) in both H2 2022 and H2 2023 must beat the paired
control and zero on account return and matched mean bp, with >2 actual
trades per native scoring week, to qualify for replication. Years and seeds
are not independent markets. No promotion or feature removal.

Runtime: 413 CPU environment, `tools/role420_sell_refit.py`. Artifacts:
`/var/tmp/role420-sell-paired` (per-epoch histories, champions, account
metrics, terminal report and decision).
