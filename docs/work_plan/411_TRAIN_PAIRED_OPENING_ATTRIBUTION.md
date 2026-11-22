# 411: Paired attribution of managed opening advantage

Frozen 409 ridge on 2022, diagnostic on 2023 inside TRAIN. No threshold
changes, 2024 scoring or new network training. Three closer seeds on the
same market, 4,716 origins x eight side/horizon combinations x three
seeds = 113,184 matched hypothetical offers, not independent account trades.
Inputs, trained heads, origin prices, size, SL and TP were checked before
attribution. For each identical opening replay one native account with the
learned close enabled and another holding until a barrier or mandatory exit.

Decomposition (bp of initial capital): passive gross + change in gross from
the learned exit - managed spread, slippage, commission and financing +
incremental reference payoff = managed advantage over NOP. The reference
is the same 408 weekly 6h reference opportunity, not all possible forgone
trades. Its effect includes both availability and capital at the later fill.
Costs are ledger attribution on actual fills, not an alternate frictionless
market path. Max absolute per-offer discrepancy against sealed 408 labels
was below 1e-12 bp. The 409 selected-offer counts and mean payoffs match.

|Seed|Selected|Passive gross|Change in gross|Managed costs (signed)|Reference|Managed advantage|Passive advantage|
|---:|---:|---:|---:|---:|---:|---:|---:|
|7|2555|-2.099|+0.603|-2.980|+0.194|-4.283|-4.396|
|19|2795|-2.354|+1.446|-2.906|+0.150|-3.664|-4.557|
|43|2681|-1.908|+0.843|-2.967|+0.144|-3.888|-4.201|

The incremental **net** benefit of allowing the learned close (including
cost and continuation differences) was +0.11/+0.89/+0.31 bp. In contrast,
the actual selected openings had negative mean gross even before accounting
costs, on both native passive and managed exit paths. The two clocks are
paired interventions; this does not establish that the underlying market
was unpredictable or that different barrier profiles would behave the same.

BUY 6h illustrates why the selected population matters. In its own top
decile (472 per seed), net close benefit was +1.15/+1.25/+0.82 bp; where
BUY 6h won the eight-way arbitration (103/111/117), it was -7.39/-1.93/
-0.69 bp. These are different subsets; do not disable the closer based on
the latter. Next diagnostic: on the same tick compare the winner against
the runner-up and a fixed alternative, reporting prediction margin, realized
paired advantage and no-trade separately. A hindsight best-of-eight is an
oracle ceiling, not a deployable control.

Reproduction: run `tools/role411_payoff_parts.py` in the 409 CPU environment
then `tools/role411_summarize.py`. Per-offer ledgers and `SUMMARY.csv`
are under `/var/tmp/role411-train-attribution/`.
