# P3 Future Work

P3 is `outline`/deferred; these lines define what must exist before drafting,
so future work here is pre-evidence by design. Format per line: limitation →
falsifiable question → prior-art state → required implementation/data →
cheapest discriminating experiment → decision metric (unit) → dependency /
kill condition → registry ID.

## 1. Cell-qualification pipeline

- Limitation: zero protected-v2 champions exist; the six-cell gate is unmet.
- Question: which asset/timeframe cells satisfy the qualification contract
  (activity, stability, artifact integrity) under identical protocol?
- Prior art: not applicable (internal gate).
- Required: the running campaign series; no new implementation.
- Experiment: apply the frozen qualification checklist per completed campaign.
- Metric: qualified-cell count (count) with per-cell evidence hashes.
- Dependency: P2 campaigns. Kill: fewer than 3+3 cells after the planned
  asset series — P3 defers again rather than lowering the gate.
- Registry: P3 gate (H0/H1 boundary).

## 2. Opportunity-gate calibration versus no-gate control

- Limitation: the rush/opportunity gate exists as design only; its claimed
  value is untested.
- Question: does a calibrated probabilistic gate improve portfolio RAP or
  drawdown versus the mandatory no-gate control at matched turnover?
- Prior art: first_pass baseline anchor (BOCPD, verified) via P9; gating lit
  unopened.
- Required: gate implementation over frozen cell action streams (replay, not
  live).
- Experiment: replay frozen cells with gate on/off; calibration measured
  before utility.
- Metric: Brier score (dimensionless) first; then RAP delta (fraction/week)
  at matched turnover.
- Dependency: item 1 cells; P9 narrowed data coverage. Kill: no calibrated
  signal beats BOCPD-class price-only baselines — gate becomes future work in
  the paper, not a claim.
- Registry: P9/P3.

## 3. Allocation cadence: static versus change-triggered

- Limitation: weekly cadence is a default (ADR-011), not evidence.
- Question: does change-triggered rebalancing beat fixed weekly at matched
  risk on frozen cells?
- Prior art: candidate_unverified (rebalancing lit seeded via DeMiguel row).
- Required: allocator replay harness over frozen cells.
- Experiment: fixed vs periodic vs triggered schedules on identical streams.
- Metric: protected-split RAP (fraction/year), turnover (fraction), drawdown.
- Dependency: item 1. Kill: no schedule separates from weekly within seed
  noise — ADR-011 stands and the paper reports it.
- Registry: P10.
