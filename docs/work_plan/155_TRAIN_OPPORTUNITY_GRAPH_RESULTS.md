# TRAIN opportunity graph and native prefix Pareto results

2026-10-30. Plan154 completed at its declared scope. No source result or live
search was edited. No validation, TEST, model training or floating selection.

## Immutable source recovered

Source K=3 identity:
`d54d51768a5ac257b7428a8e7afc8fb61bd467009dea77e8224a64c9ee824a9e`.
The original checkout's indicator digest had changed. Recovery found the
exact pinned blob at heuristic-strategy revision
`28ad98e82f4f945cd1c8729b4c75989710a2108d` and exported it to a separate
snapshot. The later difference is a docstring about inert Deviation, not a
changed indicator algorithm. Nothing was reverted in that checkout.

All original input, producer and record digests were authenticated. Candidate
pivot indices, selected endpoints and all selected net/MAE/drawdown/score values
reconstruct with maximum error **0.0**. The native engine uses its separately
identified current accounting version, not the old indicator dependency tree.

## Full TRAIN graph, not just winning edges

Every complete origin retains all K=3 chronological endpoints for both sides.
Alternatives are not deleted because they lose locally or have negative profit.

| Nominal h | Complete origins | Censored origins | Directional edges | Nonpositive edges | Nonwinning edges retained |
|---|---:|---:|---:|---:|---:|
| 3 | 19,620 | 259 | 117,720 | 61,454 | 78,480 |
| 6 | 19,108 | 771 | 114,648 | 59,127 | 76,432 |
| 12 | 16,191 | 3,688 | 97,146 | 49,469 | 64,764 |
| 24 | 12,539 | 7,340 | 75,234 | 38,214 | 50,156 |
| Total | 67,458 | 12,058 | **404,748** | **208,264** | **269,832** |

Origins overlap across horizons, sides and endpoints. These are NOT independent
samples or account trades. 365,072 directional edges satisfy the inherited
reference calendar filter. Native eligibility is checked again at execution.
Incomplete K=3 tails are censored; there is no filling, interpolation or joining
across absent open-session bars. Graph arrays remain outside git because they
contain source-derived price data; their hashes and reproducible builder are
published in [GRAPH.json](../audits/evidence/train_opportunity_graph_20261030/GRAPH.json).

## Quote and endpoint semantics repaired before native use

The admitted source is BID OHLC. Passing it directly as MID would shift quotes.
The new local adapter preserves the observed BID exactly and computes modeled
ASK = BID + declared spread. It does not claim observed historical ask quotes.
Spread, commission, slippage and financing remain the declared simulated costs.
Both directions have explicit cash-arithmetic tests.

A pivot extreme is a retrospective reference, not a guaranteed fill. A planned
pivot exit submits CLOSE after the pivot bar and fills at the next open. SL,
TP and weekly forced flat may end the trade first. Early CLOSE can override
the planned endpoint. Each opening retains its horizon, pivot slot and planned
pivot bar alongside the actual native ledger. The risk recipe is unchanged:
rho0.005 and quality1 are controls, not newly optimized/calibrated estimates.

## Bounded native proof on real TRAIN paths

The first complete session in each year was selected ONLY by calendar and bar
availability, before any scoring. In each case the first THREE hourly decision
points after the operational gate are exhaustively branched. Openings choose
side/horizon/endpoint; occupied actions choose HOLD/CLOSE, with planned exits
enforced. Afterwards no new entry is allowed in this proof; existing positions
continue to their planned exit, protection or weekly deadline.

This is an exact **restricted-prefix action library**, not an exact full-week,
annual or four-year Pareto solution. The 12,058 censored horizon-origins are not
invented or silently made usable. Nominal horizon is not a forced duration cap.

| TRAIN session opens (UTC) | Initial actions including NOP | Terminal accounts | Pareto vectors | Max-return member % | Its hourly-close DD % | Executed orders | BUY / SELL | Early closes |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 2020-01-02 00:00 | 15 | 385 | 4 | 1.293794 | 0.028960 | 2 | 1 / 1 | 1 |
| 2021-01-03 22:00 | 25 | 721 | 2 | 1.134378 | 0.295663 | 1 | 0 / 1 | 0 |
| 2022-01-02 22:00 | 25 | 1,063 | 15 | 1.227479 | 0.149256 | 2 | 1 / 1 | 1 |
| 2023-01-02 07:00 | 23 | 1,037 | 7 | 0.756562 | 0.039321 | 1 | 1 / 0 | 0 |

These returns use known future TRAIN paths and are NOT learned performance or
annual returns. They are oracle reference points, not deployable strategies.
Other return/DD tradeoffs remain in the report. There is no claim of statistical
significance from four hand-bounded sessions.

The max-return 2020 and 2022 paths close BUY before opening SELL; 2021 keeps a
24h-profile SELL until TP; 2023 keeps a 6h-profile BUY until TP. This demonstrates
that the library can represent differing decisions, not that a model learns them.
In 2020 the initial 24h K=3 tuple is censored and some other endpoints exceed the
execution boundary; in 2023 one 24h endpoint exceeds it. They are recorded in
`initial_support`, not replaced or called model failure.

## Verification

- 3,593 nodes; **3,206 terminal accounts**; independent pairwise nondominance.
- **28** frontier witnesses replayed cold with exact objective equality.
- Maximum cash conservation discrepancy: **1.3784529073745944e-12 USD**.
- **138 tests passed**, zero skips, including legacy graph/risk/game/native tests.
- Graph reports and all four NPZs match byte for byte on a fresh rebuild.
  Source Python bytecode caches are excluded from comparison (paths/timestamps).
- Native report and all four complete payoff files match byte for byte on a
  separate cold run. No source input changed during execution.

Evidence: [native report](../audits/evidence/train_opportunity_graph_20261030/NATIVE_REPORT.json).
The complete native prefix payoff files contain actions/objectives only, not
historical prices. They are published for independent dominance verification.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/build_train_opportunity_graph.py \
  --source <completed-k3-run> --output <new-graph-root>

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/verify_train_prefix_pareto.py --source <completed-k3-run> \
  --graph-root <new-graph-root> --engine-root <pinned-native-engine> \
  --output <new-native-proof-root>
```

## Next integration, without redefining this result

The full TRAIN opportunity graph is now available; a full historical account
frontier is not yet computed. Next expand bounded decision coverage and compare
a sequential planner against these exact prefixes, preserving balance, equity
peak and occupied-position state. A deployment planner must use non-anticipative
causal reports. Neither this oracle nor the current local-score DEAP substitutes
for that integration or authorizes phase2 floating selection.
