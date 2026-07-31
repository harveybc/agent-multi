# P4 Future Work

Format per line: limitation → falsifiable question → prior-art state →
required implementation/data → cheapest discriminating experiment → decision
metric (unit) → dependency / kill condition → registry ID.

## 1. Protected-canary parity experiment (M3)

- Limitation: the paper's results section is empty until canaries run; all
  current evidence is read-only observation.
- Question: do minimum-size protected brackets achieve confirmed server-side
  SL+TP attachment and restart-safe reconciliation on each eligible venue?
- Prior art: not applicable (venue-contract measurement).
- Required: user-authorized canary enablement per doc 22 M3; no new code
  claimed here.
- Experiment: paired long/short minimum-size canaries per venue in liquid
  hours with forced restart mid-lifecycle.
- Metric: protection-confirmation rate (%), duplicate-suppression (count=0),
  reconciliation divergence (count=0).
- Dependency: user enablement + observation review. Kill: any venue failing
  the protection contract is excluded from production routing and reported —
  that exclusion is itself a result.
- Registry: P4 core (H0).

## 2. Fault-injection matrix

- Limitation: disconnect/stale-data/duplicate-submission behavior is designed
  fail-closed but not demonstrated as experiment.
- Question: do the declared fail-closed invariants hold under injected API
  disconnect, stale quotes and duplicate submission on paper venues?
- Prior art: candidate_unverified (chaos-testing practice lit unopened).
- Required: injection harness at the adapter boundary (bounded Musashi
  packet).
- Experiment: scripted fault scenarios against paper endpoints with invariant
  assertions.
- Metric: invariant violations (count=0 target), recovery latency (s).
- Dependency: M2 complete per venue. Kill: none — violations are findings,
  not kill conditions.
- Registry: P4/P12.

## 3. Cost-model residual calibration

- Limitation: simulator cost assumptions versus paper fills are uncompared;
  P4's title question depends on this.
- Question: what are the distributional residuals (spread, slippage, rejection
  rate) between the Nautilus cost model and paper executions per venue?
- Prior art: candidate_unverified (Almgren-Chriss row seeded).
- Required: none new — the seven-day consolidated shadow (M4) produces the
  data.
- Experiment: paired sim-vs-paper fills over the M4 window.
- Metric: residual distribution parameters (bps) with coverage declaration.
- Dependency: M4. Kill: n/a — any result is reportable.
- Registry: P4 core (H0).

## 4. Functional-versus-liveness health probes (proposed new line P19)

- Limitation: the IBKR incident proved a green TCP probe can mask a totally
  failed observer for hours; the fix (authenticated-session freshness) is one
  instance of an unstudied class.
- Question: across venue adapters, what fraction of failure modes are visible
  to functional probes but invisible to liveness probes, and at what probe
  cost?
- Prior art: candidate_unverified (SRE practice row seeded; measurement study
  unopened).
- Required: probe-taxonomy instrumentation on existing adapters.
- Experiment: replay/insert the recorded failure classes; measure per-probe
  detection.
- Metric: detection coverage (% of failure classes) per probe type; probe cost
  (calls/day).
- Dependency: none — one incident corpus entry already exists. Kill: coverage
  difference <10 % — then liveness probes suffice and P5 reports that.
- Registry: proposed P19 (fills a demonstrated gap; shared with P5 item 4).
