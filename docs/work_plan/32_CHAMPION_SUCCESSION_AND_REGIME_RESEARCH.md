# 32. Champion Succession Doctrine and Regime-Specialist Research Track

Status timestamp: 2026-08-03 America/Bogota
Version: 1.0.0
Author: Satoshi III (Mujuro Utsutsu), successor technical lead
Authority: owner directive of 2026-08-03 (succession evaluation is
mandatory; the regime track is approved as staged research). Parameters
marked D1-D5 await General Musashi's disposition before implementation.
Non-authority: no LLM, Hermes or knowledge tool ever decides a
succession, a promotion or a metric verdict; every gate below is a
deterministic computation over persisted evidence, and the owner
overrides everything.

## Part I — Champion Succession and Divergence Doctrine

### S1. Succession gate (anti-churn, anti-overfitting)

A new simulation champion does NOT automatically take the live Paper/Demo
seat. It succeeds only when ALL hold:

1. **Statistical superiority:** challenger beats the incumbent on robust
   weekly RAP with a bootstrap confidence interval, **deflated for the
   number of campaign trials** (Deflated-Sharpe-style multiple-testing
   correction; the trial count comes from the DOIN campaign ledger, which
   already records every evaluated candidate). A sim-best that is best by
   luck must fail here.
2. **Minimum incumbent tenure:** 7 calendar days, waived only for an
   S-severity misbehavior of the incumbent (protection loss, divergence
   breach per S3, or owner command).
3. **Mechanics:** succession executes ONLY through the existing verified
   pipeline — hash-bound manifest replacement, drain of prior exposure,
   session reseed from actual post-close broker cash/equity. No new
   mechanism is created. The owner is notified with the gate evidence at
   every succession; live-capital seats (future) additionally require
   explicit per-event owner ratification (D5: whether Paper/Demo swaps
   also need per-event ratification, or gate-plus-notice suffices).

### S2. Tenure record and counterfactual replay (at every succession)

When a champion is replaced, freeze an append-only tenure record:

- realized live metrics over its full reign (S5 set);
- a **counterfactual replay**: the outgoing agent re-run in simulation
  over the exact realized period with modeled costs — the realized-vs-
  replayed gap is the **sim-to-real divergence**, our most diagnostic
  series;
- the incumbent then runs as a **shadow** (signals logged, zero orders,
  the existing shadow idiom) for at least one week beside its successor,
  giving a paired comparison on identical market data.

### S3. Daily and weekly cadence (honest sample-size split)

- **Daily — divergence tracing only, never verdicts:** realized vs
  modeled cost ratio (spread+slippage+fees), action-agreement rate under
  sim replay of live inputs, exposure overlap, fill-quality drift,
  calibration drift. Tiered thresholds (watch/warning/critical); a
  critical divergence is an S-severity event for S1.2.
- **Weekly — decision metrics:** weekly RAP with bootstrap CI, Sortino,
  Calmar, maximum drawdown, PSR for short windows, turnover, realized
  cost ratio.
- Storage: OLAP **views** over existing ledgers plus watchdog surfacing —
  no parallel database, no new source of truth.

### S4. Curriculum ablation (the owner's difficulty question)

Every champion carries an immutable training-stage tag:

- `easy_no_margin_call` — initial stage, explicitly WITHOUT margin call;
- `normal_realistic` — realistic fees AND margin call;
- `hard_pessimistic` (optional) — pessimistic fees, margin call and
  added adversity.

Hypothesis under test: **harder training reduces live divergence** (the
S2/S3 series), not necessarily raw return. Analysis: paired comparison of
divergence and decision metrics across successions grouped by stage tag,
at every succession and in a monthly rollup. This turns the easy → normal
→ hard curriculum from a belief into an audited measurement.

### S5. Metric definitions (single canonical list)

Primary: robust weekly RAP (owner-ratified, existing definition) with
bootstrap CI and trial-count deflation. Secondary: Sortino, Calmar,
maximum drawdown, PSR, turnover. Divergence: realized cost ratio,
action-agreement rate, exposure overlap, calibration drift, fill-quality
drift, backtest-to-live realization ratio. Exact formulas and window
lengths are fixed in the implementing commit and versioned with tests
(D2: Musashi disposition on the final set and thresholds).

## Part II — Regime-Specialist Research Track (R-track)

Motivation: prior experiments identified market states via hierarchical
unsupervised learning; states describe distinct dynamics of variable
duration. Question: do per-state specialists beat one weekly-fine-tuned
generalist? State of the art supports the direction (regime-conditioned
policy optimization, mixture-of-experts routing over expert policies,
jump-model regime identification with explicit transition persistence)
while flagging four risks: lagging regime inference at boundaries,
data fragmentation per specialist, the gater as new model risk, and
regime drift. The track is therefore staged and falsifiable:

### R0 — measured headroom before any training (cheap, first)

Partition the EXISTING champion evaluations (simulation and live tenure
records) by the unsupervised states. If inter-regime performance spread
is below threshold (D3), the specialist hypothesis has little headroom —
stop and record. Otherwise the headroom is a measured number. Deliverable:
one audited analysis document; zero training compute.

### R1 — two arms in simulation (only if R0 passes)

- **Arm A (owner's proposal):** k specialists trained only on
  regime-filtered windows, each window padded to INCLUDE transitions in
  and out of the regime (specialists must see boundaries).
- **Arm B (cheaper middle path):** one model with the regime posterior as
  a conditioning INPUT — no data fragmentation, no gater.

Both evaluated per-regime against the champion with trial-count-deflated
weekly RAP. Training runs as parallel DOIN domains on idle capacity
between campaign stages; jobs 0/1 are never preempted (Front 1 rule).

### R2 — standing shadow challenger at succession events

The winning R1 architecture becomes a permanent shadow challenger:
signals logged, zero orders, evaluated over the SAME tenure window at
every champion succession (the owner's "compare against the champion each
time a new one appears", formalized). Promotion rule: beat the champion
on the S5 primary metric across **two consecutive successions** → eligible
for an owner-gated live Paper pilot with soft gating (blend by regime
posterior), a generalist fallback, and reduced position limits whenever
regime confidence is low. Hard switching is never the first deployment.

### R3 — live pilot (future, separate owner decision)

Not specified here; requires R2 evidence and a Musashi audit.

### Resource and doctrine rules

Front 2 P0 and the active campaign always take precedence; R-track work
uses CPU or idle GPU windows only; the Front 5 second-domain conformance
question applies to every new plugin contract this track introduces;
all artifacts versioned; no acceptance claim exists only in chat.

## Part III — Dispositions Requested from General Musashi

- **D1:** succession-gate parameters (CI level, deflation method detail,
  7-day tenure, S-severity waiver list).
- **D2:** final S5 metric set, window lengths and daily alert thresholds.
- **D3:** R0 go/no-go threshold for inter-regime spread.
- **D4:** runtime placement of the R2 shadow challenger (which host, and
  its isolation from the live runners).
- **D5:** whether Paper/Demo successions require per-event owner
  ratification or gate-plus-notification.

## Part IV — References (state of the art consulted 2026-08-03)

- Bailey & López de Prado, The Deflated Sharpe Ratio (SSRN 2460551)
- Statistical jump models for regime switching (BS Capital Markets
  practitioner summary; arXiv:2410.14841 regime-switching factor
  allocation)
- Ensemble-HMM regime-shift voting framework (AIMS Press, 2026)
- Unsupervised regime detection via realized covariances (arXiv:2104.03667)
- FR-LUX: regime-conditioned policy optimization (arXiv:2510.02986)
- MoE deep-RL portfolio routing (Applied Intelligence, 2025)
- T2MIR: mixture-of-experts in-context RL (NeurIPS 2025)
- Champion/challenger and shadow deployment practice (FICO, DataRobot;
  NinjaTrader shadow-strategy mechanism; ML4Trading MLOps chapter on
  backtest-to-live realization monitoring)
