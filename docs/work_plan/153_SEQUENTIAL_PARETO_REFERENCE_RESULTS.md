# Exact Sequential Account Reference Results

2026-10-30. Retrospective synthetic calculation, not trading performance.

See [derivation152](152_TWO_PHASE_SEQUENTIAL_PARETO_DERIVATION.md).

Complete finite trees: 18; terminal accounts: 37578; nodes: 46256; cold witness replays: 126.
Maximum cash conservation error: 2.16004991671e-12 USD. Independent pairwise dominance checks passed.

All four saved horizon profiles, both directions, NOP and occupied CLOSE/HOLD are enumerated. The 5 decisions end at native weekly forced flat. Each case starts flat or with a 12h-profile BUY/SELL. Fees, bid/ask, funding, integer sizing and protections remain native. Drawdown is measured on hourly close equity, not unobserved intrabar equity extrema.

## Flat Initial State

The displayed member maximizes return within each finite tree; it is NOT the sole Pareto choice. Alternative return/DD points and conditional action frontiers are in REPORT.json.

| Scenario | Net return % | DD % | Orders | BUY / SELL | Early closes | Best <=1 entry % | Best single horizon % |
|---|---:|---:|---:|---|---:|---:|---:|
| smooth / +1 | 1.854088 | 0.000000 | 2 | 2 / 0 | 0 | 1.219505 | 1.854088 |
| smooth / -1 | 1.798867 | 0.000000 | 2 | 0 / 2 | 0 | 1.155105 | 1.771956 |
| oscillating / +1 | 1.928812 | 0.024876 | 3 | 1 / 2 | 1 | 1.087102 | 1.928812 |
| oscillating / -1 | 1.916259 | 0.024975 | 3 | 2 / 1 | 1 | 1.029694 | 1.916259 |
| reversal / +1 | 1.780639 | 0.024939 | 3 | 1 / 2 | 1 | 1.150155 | 1.762151 |
| reversal / -1 | 1.839086 | 0.024940 | 3 | 2 / 1 | 1 | 1.196857 | 1.832246 |

## Initially Occupied

Returns include the initial position, whose entry, 12h SL/TP and volume stay fixed. Subsequent entries can use any horizon. The two columns include optimal legal continuations after the first decision; they are not the isolated close-regret metric.

| Scenario | Position | Best return after CLOSE % | Best return after HOLD % |
|---|---|---:|---:|
| smooth / +1 | BUY | 1.356576 | 1.345917 |
| smooth / +1 | SELL | 1.359551 | 0.712568 |
| smooth / -1 | BUY | 1.300928 | 0.637103 |
| smooth / -1 | SELL | 1.303910 | 1.320944 |
| oscillating / +1 | BUY | 0.997469 | 1.567365 |
| oscillating / +1 | SELL | 1.000442 | 0.548542 |
| oscillating / -1 | BUY | 1.003205 | 0.554247 |
| oscillating / -1 | SELL | 1.006172 | 1.533978 |
| reversal / +1 | BUY | 1.502460 | 1.728932 |
| reversal / +1 | SELL | 1.505439 | 1.290929 |
| reversal / -1 | BUY | 1.560710 | 1.334122 |
| reversal / -1 | SELL | 1.563725 | 1.778347 |

## Scope And Next Integration

This calculates the exact frontier for these finite trees at fixed quality=1 and the saved risk recipe. It neither optimizes risk coefficients nor computes a four-year TRAIN frontier. Mirror rankings need not be identical: saved BUY/SELL geometry, funding and integer units differ. A real policy cannot choose using these known future paths; it needs a common action over indistinguishable report histories, then an explicit risk preference to select a Pareto member.

Next: retain all K=3 TRAIN endpoint alternatives in an opportunity graph and qualify bounded exact native subproblems there; then qualify non-anticipative scenario-tree allocation against this oracle and the running local-score DEAP baseline. No live search was restarted or edited; no floating selection is released.

Evidence: [REPORT.json](../audits/evidence/sequential_pareto_20261030/REPORT.json). The report identifies each complete terminal ledger file by digest for independent reconstruction.
