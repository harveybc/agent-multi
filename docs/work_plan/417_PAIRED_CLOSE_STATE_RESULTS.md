# 417: paired close advantage with position state

Decision: keep the existing state-conditioned curve. Neither challenger passes
the paired, fixed-entry and sequential-account gates across three seeds.
Exploratory TRAIN only, not an independent confirmation of the selected encoder.

Fit on purged 2019-22 labels; score July-December 2023. Each BUY/SELL closer
and horizon has a separate fit. Inputs are the frozen closer's eight market
logits and the three contemporaneous state coordinates used by the native
controller. Direct: affine readout of logits and coordinates. Adapter: adds
24 products of logit and coordinate. Native curve: original trilinear lookup
and class-to-money conversion. No recurrent predictions or new market inputs.

Fit objective: tick-balanced squared error of signed close-now advantage
over holding to a barrier/mandatory exit, including the labeled reference
opportunity, plus one equally weighted cross-position difference objective
per conflicted tick. Within each such tick average over every positive-negative
pair. Ridge penalty 0.1 per fit tick, intercept unpenalized; exact linear
solve, no stochastic epochs. The sufficient-statistic implementation is
checked against explicit Cartesian pairs on a fixture. 1,221,675 fit pairs
across eight cells; overlapping hypothetical positions, not independent trades.

Regret in bp on 4,083 conflicted tick/side/horizon groups, pooled with equal
group weight. Each triplet is ordered curve / direct / adapter.

|Seed|True state regret|Swapped state regret|Same entries, mean net bp|Sequential return (%)|
|---:|---|---|---|---|
|7|17.103 / 16.964 / 17.007|20.363 / 20.332 / 20.297|-4.067 / -4.215 / -4.245|-20.83 / -19.42 / -20.68|
|19|17.084 / 17.061 / 17.116|20.280 / 20.560 / 20.459|-3.622 / -4.131 / -4.028|-23.12 / -20.80 / -19.08|
|43|17.097 / 17.107 / 17.166|20.386 / 20.565 / 20.546|-3.219 / -3.200 / -3.240|-16.27 / -19.35 / -17.99|

Swap rotates position coordinates only within a tick, side and horizon. Static
regret uses the passive continuation in the label; deployed decisions recur
on subsequent ticks. Fixed-entry replays reset capital and use exactly the
same 1,201 / 1,318 / 1,277 origins. Net bp there excludes a second opportunity.
Sequential accounts share the proposed opening stream, but a still-open trade
can block a later proposal. Native counts and net returns matched 412.
The swap advantage does not identify which coordinate matters or prove all
other feedback architectures ineffective. Keep the close weights unchanged.

Reproducer: `tools/role417_paired_close.py` in the 413 CPU environment.
Per-cell metrics, wins, counts, Sharpe, drawdown, challenger coefficients
and terminal verdict: `/var/tmp/role417-paired-close/`.
