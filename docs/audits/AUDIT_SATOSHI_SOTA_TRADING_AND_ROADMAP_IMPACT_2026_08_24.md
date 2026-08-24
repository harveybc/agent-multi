# Audit: SOTA Trading Review and Roadmap Impact

Date: 2026-08-24
Auditor: General Musashi
Subject: `docs/research/sota_trading/00_INDEX.md` through
`08_TRAINING_OPTIMIZATION.md` at `agent-multi@0058dc74`

Amendment input: `09_AUTOCRITICA_COMPARATIVA_PARA_MUSASHI.md` at
`agent-multi@04d6bfb8`.

## Verdict

`ACCEPT_WITH_MAJOR_REVISIONS`.

The packet is a useful comparative inventory and contains several important
facts for this program. It is not yet a reproducible literature review or a
decision map for the ETH H4 SAC research line. The current P1 campaign must
finish unchanged, but the post-P1 roadmap needs reprioritization before a large
architecture or DOIN sweep is launched.

## Findings

### S2 — SOTA-01: repeated use has converted outer-2024 into development data

The packet calls 2024 an outer endpoint. Across the program, however, results
from 2024 have repeatedly informed reward, stopping, curriculum, scheduler and
architecture decisions. A split can be post-selection inside one run and still
become contaminated at program level through repeated researcher feedback.

Disposition:

- Relabel 2024 `development_outer` or `research_validation` prospectively.
- Keep 2025 sealed as the only untouched final test for the present lineage.
- Do not open 2025 for individual mechanism screens.
- Build rolling-origin development folds before architecture/DOIN search.
- Record a program-level access ledger, not merely a per-run split firewall.

### S2 — SOTA-02: the selected literature is not decision-complete for our problem

The five headline papers are strong references, but only one directly studies
DRL trading and none is a close analogue of ETH H4 SAC with broker-mediated
execution. The packet is organized by famous papers, not by our unresolved
decisions. It does not adequately cover:

- continuous target-position policies versus directional/order commands;
- turnover-aware action hysteresis and early closing;
- risk-sensitive, constrained and distributional RL;
- offline RL and off-policy evaluation for financial data;
- recurrent SAC/POMDP treatment and regime conditioning;
- sim-to-real calibration, execution uncertainty and market impact;
- frequent walk-forward adaptation and catastrophic forgetting;
- statistical selection under many trials and seed instability.

The review cannot yet justify the grouped extractor or a topology search.

### S2 — SOTA-03: current action semantics are a higher priority than deeper models

The packet itself shows that strong trading systems commonly emit a target
position and incorporate volatility scaling and turnover costs. Our current P1
maps any nonzero SAC output to a directional decision because the threshold is
zero. Completed arms show roughly 444--500 trades per outer year. This is active,
but it collapses action magnitude and may induce unnecessary switching.

Before scaling architecture, compare bounded action contracts under identical
data and parameter budgets:

1. current sign-only target;
2. continuous target exposure with bounded risk sizing;
3. ternary target with calibrated deadband and hysteresis;
4. explicit close/hold semantics while a position is open.

Native SL/TP remain a safety layer. They do not replace a learned target-position
or early-close policy.

### S2 — SOTA-04: the proposed grouped extractor is ahead of its evidence

The TCN/Transformer/GRU branch assignment is plausible, not established as
best for our exact feature families. With only 18,085 H4 rows, a large grouped
network can add variance faster than signal. The cited GKX result is itself a
warning: shallow networks dominated deeper ones in a low-signal financial
setting.

Require capacity-matched ablations before a full topology sweep:

- flat MLP baseline;
- small shared causal TCN or GRU baseline;
- grouped extractor with the same approximate parameter budget;
- grouped extractor plus fusion only if it wins across rolling origins and
  seeds.

Parameter count, FLOPs, wall time and inference latency must accompany economic
metrics. No component gets called state of the art merely because its family is
modern.

### S3 — SOTA-05: source verification is not reproducible

The prose says every fact was checked against primary sources, but the nine
files contain no bibliography, DOI/URL table, page/table/equation locators or
retrieval metadata. `[NO DECLARADO]` is useful, but an auditor cannot reproduce
the positive claims efficiently.

Every quantitative claim needs a source identifier and locator. Add a source
registry containing title, version, DOI/arXiv URL, publication venue, retrieval
date, and page/table/equation. Distinguish published paper, appendix, repository
and secondary benchmark.

### S3 — SOTA-06: broker and execution language overclaims the evidence

The packet alternates between "three venues real", "active" and
"broker real" while IBKR is owner-suspended and the accounts are Demo/Paper.
Broker-mediated fills are more realistic than arithmetic close fills, but they
are not live-capital execution and do not establish profitability, impact or
production reliability.

Use these exact classes:

- `broker_mediated_demo_or_paper_execution` for Alpaca/MT5;
- `preserved_suspended` for IBKR;
- `historical_simulation` for gym-fx;
- `live_capital_execution = false` everywhere.

The statement that none of the five papers used paper trading must be phrased as
"no paper/broker execution reported in the reviewed sources" unless absence is
proved by supplementary material and code.

### S3 — SOTA-07: retraining is identified but not promoted to the roadmap

The review correctly says the current SAC line has no rolling adaptation and a
large train-to-decision gap. This is central to the owner's business model, not
a footnote. After action-contract calibration, run rolling-origin adaptation
screens for at least 168h, 24h and 12h cadences, each paired with a frozen
control. Six-hour cadence should enter only after measured runtime p95 leaves a
safe deadline margin.

Measure incremental value after costs, stability, turnover, deadline misses,
state continuity and degradation relative to frozen policy.

### S3 — SOTA-08: four seeds are directional evidence, not reliable selection

The current campaign already shows large seed variance. Four seeds can screen a
mechanism but cannot support a precise performance claim or choose a production
champion safely. Report paired effect sizes and intervals, retain all seeds, and
use additional seeds only for mechanisms that survive the screen. An ensemble
or seed-selection rule is a separate experiment and must not be inferred from
replication runs.

### S4 — SOTA-09: package navigation and stated scope are inconsistent

The index says seven files and lists `01`--`07`, but the packet contains nine
files including `00` and an unlisted `08_TRAINING_OPTIMIZATION.md`. The commit
subject repeats the wrong count. Add `08` to the index and state "eight aspect
files plus index".

Status after `04d6bfb8`: partially corrected. Documents 08 and 09 are now
linked, but the source-registry and claim-locator defect remains open.

## Amendment: Audit of Satoshi's Self-Critique

The self-critique materially strengthens the packet. Accept these diagnoses:

- absence of same-harness economic baselines;
- no systematic SAC hyperparameter search;
- lack of rolling adaptation;
- insufficient feature-family attribution;
- seed instability and weak statistical power;
- lack of explicit volatility/turnover normalization experiments;
- possible easy-treatment inaction requiring a mechanical bite check;
- disproportionate engineering effort relative to empirical model comparison.

Do not adopt the following statements as facts without correction:

1. **A flat MLP does not destroy temporal information.** Flattening preserves
   every coordinate and its order; it removes temporal inductive bias and
   parameter sharing. That may be inefficient, but it is an empirical question,
   not proof that the model cannot use time.
2. **"Each bar seen more than 1,700 times" is not a sufficient sample-size
   statistic.** Environment visits and replay gradient draws are different,
   correlated quantities. Report unique transitions, replay sample counts,
   effective sample reuse, autocorrelation and train/validation divergence.
3. **SAC is not literally unregularized.** Automatic entropy regularization,
   twin critics, target networks and replay stabilization exist. The accurate
   claim is that no explicit actor weight decay, dropout or capacity control has
   been evaluated.
4. **ZZR's gamma=0.3 is not a candidate default for ETH H4 SAC.** It belongs to
   a different algorithm, market and cadence. Gamma is an optimization/design
   variable, with horizon reported in H4 bars and wall-clock time.
5. **The top papers do not all systematically tune hyperparameters.** Several
   use fixed or incompletely reported choices. Our defect is failure to justify
   our defaults, not failure to imitate a universal practice that does not
   exist.
6. **Easy is not yet universally proved inert.** Exact equality in completed
   seed/arm pairs proves those pairs uninformative. The campaign must finish and
   report solvency-event and trajectory divergence per seed before a global
   statement.
7. **Multi-asset is not P0.** It is valuable for mechanism transport and later
   portfolio work, but moving it ahead of action semantics, baselines and
   adaptation would multiply a possibly defective agent across assets. Preserve
   the owner's single-asset-first strategy.
8. **Volatility targeting and ATR stops are candidates, not automatic fixes.**
   They require paired ablation because the current risk envelope and native
   protection already constrain exposure differently in simulation and Demo.
9. **The 256-bar zero warmup must be re-verified on the current nested-prefix
   execution path.** Historical probe behavior is not enough to claim every
   current episode discards 42 days; context-prefix materialization may change
   that fact.

### Revised immediate priority

Same-harness baselines move into the first post-P1 work package. Multi-asset
does not. Run buy-and-hold, flat, simple momentum/TSMOM and a volatility-scaled
rule on the same development folds and cost engine. These baselines establish
whether SAC adds value before architecture expansion.

## What Does Not Need Rebuilding

- The nested temporal split machinery and sealed-2025 firewall remain useful.
- The broker safety envelope, native protection, evidence journals and
  model-authority controls remain useful.
- SAC remains a valid baseline.
- The easy/normal campaign should finish because it provides paired evidence
  about solvency relaxation and replay continuity.
- DOIN remains the optimization mechanism, but its next domain must be based on
  corrected action/retraining evidence rather than inherited defaults.

## Roadmap Change

After P1 closes:

1. Seal and aggregate P1; classify inert treatments explicitly.
2. Repair literature provenance and produce a decision-oriented evidence map.
3. Calibrate action semantics and execution-aware reward on rolling development
   origins.
4. Test retraining cadence with frozen paired controls.
5. Run capacity-matched feature-extractor ablations.
6. Only then materialize the DOIN topology/hyperparameter domain.
7. Evaluate surviving configurations once on sealed 2025 under an owner-approved
   release protocol.

Portfolio and multi-asset optimization remain later: first establish a sound
single-asset action, adaptation and representation contract on ETH.
