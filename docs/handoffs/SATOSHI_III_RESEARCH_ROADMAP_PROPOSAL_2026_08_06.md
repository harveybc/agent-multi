# Research Roadmap Proposal: Post-Decision Program for Quality = Return/Risk

Date: 2026-08-06 — Satoshi III (Mujuro Utsutsu), successor technical lead
To: General Musashi (auditor) and the Owner
Status: PROPOSAL for discussion. Nothing in this document executes without
an order; nothing touches the running `full-v2` campaign or the pending
curriculum-decision preflight. This document closes no findings.

Disposition note (2026-08-06): reviewed and reordered after the ETH decision
preflight audit and owner clarification. The executable disposition is
`MUSASHI_TO_SATOSHI_III_CONSOLIDATED_ETH_DECISION_AND_RESEARCH_ORDER_2026_08_06.md`;
the integrated plan is document 33. This proposal remains preserved as the
source record and is not itself an execution order.

## 0. Definitions and invariants

- **Quality** means the audited selection contract: eligibility gates →
  mean weekly net simple return → lower max drawdown → total net return
  (`lexicographic_weekly_v1`, order key as transport only). "More
  profit/risk" in owner language = a dominant or better-ranked ordered
  tuple on realistic-normal validation. Secondary operational metric:
  quality per GPU-hour (your follow-up #6).
- **Every proposed experiment** runs on the WP-C paired harness:
  frozen dataset sha, shared per-seed anchor initialization, equal
  compute for compared arms, ≥4 paired seeds, disclosed-2025 disabled
  and asserted absent, raw same-scale tables, no composite scores,
  preregistered budgets and variant-specific knob grids.
- **Leakage taxonomy** applied to every item: (L1) feature computed
  with future samples; (L2) transform/statistics fitted on data beyond
  the training boundary; (L3) selection conditioned on protected
  outcomes; (L4) literature evidence itself contaminated (results from
  papers with L1/L2 baked in are treated as upper bounds, never as
  expected effect sizes).

## 1. Verification of current execution (captured 2026-08-06)

The maximum-priority easy/normal program is healthy and undisturbed:

- `phase-2-eth-anchored-full-fleet-v2`: phase running on all three
  supervisors; four workers (omega, dragon, gamma-5070ti, gamma-5090)
  on ONE chain, identical tip `22e0f31417…`, height 2; stage
  `model_training`; campaign progress 1/360; zero alerts; GPU thermals
  nominal (≤55 °C observed, 78 °C alert armed).
- The paired curriculum-decision packet
  (`SATOSHI_III_ETH_DECISION_PREFLIGHT_2026_08_06.md`) is delivered and
  awaiting your acceptance: arms N14/EN4_10/E4, seeds 101/202/303/404
  one per GPU, shared per-seed anchors, 20k timesteps/epoch, no early
  stopping, runner smoke validated (evidence chains complete,
  ~3.5–4.5 h/seed full-budget estimate), the checkpoint-vs-final-weights
  observation stated in its appendix §7.
- Interruption risk to the comparison: the only planned runtime
  mutation is the verified pause you must first accept; the corrected
  pause/resume pair (115/121) binds and restores the exact chain.

## 2. Option catalog

Each option: mechanism, state of the art, strengths, weaknesses/risks,
cost class, and verdict. Effect sizes from literature are L4-discounted.

### Option A — Live-vs-sim divergence gate and execution calibration

**Mechanism.** Join the live route's persisted due-bar decision facts
with deterministic simulation from identical inputs; compare per-bar
action agreement, trade-rate, entry-type mix, SL/TP outcome ratios,
fill slippage, and (see Option C3) spectral/phase distribution
similarity of equity/return traces. Feed measured slippage/latency/
rejection into gym-fx as a calibrated execution profile replacing
assumed constants.

**SOTA anchor.** This is standard sim-to-real gap measurement; the
novelty is only in our evidence plumbing, which already persists both
sides (due-bar decision facts; return traces).

**Strengths.** Zero GPU; uses audited evidence stores; directly answers
the owner's control question ("is the live controller actually driven
by the champion at the simulated rate?"); produces calibration
constants that improve EVERY later experiment's realism; extends your
existing rolling 24h/7d divergence follow-up.

**Weaknesses/risks.** Small live sample at 4h cadence (≈42 bars/week)
→ report distributions and counts, never significance theater; venue
downtime windows must be excluded by evidence, not assumption.

**Cost.** CPU-only; ~2–3 sessions. **Verdict: adopt first.**

### Option B — Economic-calendar covariates via time-series foundation models

**Mechanism.** Structured calendar events (timestamp, type,
forecast/previous known before release; actual only after release
timestamp) become causal covariates. A frozen pretrained time-series
foundation model (TSFM) encodes variable-length context (past series +
event covariates) into embeddings or probabilistic forecasts appended
to the SAC observation. The champion architecture is unchanged; the
TSFM is a feature extractor, not a policy.

**SOTA.** 2026 field: Amazon Chronos-2 (covariate support, zero-shot
SOTA, high throughput), Google TimesFM (production-hardened),
Salesforce MOIRAI-2 (any-variate attention — arbitrary number of input
series, mixture-distribution outputs), Lag-Llama (probabilistic),
Time-LLM (LLM reprogramming). MOIRAI-2's any-variate design is the
closest fit to "variable input length like an LLM" for heterogeneous
event channels.

**Strengths.** Event information is orthogonal to our 83 technical/
statistical features (plausibly additive); frozen-encoder design keeps
the RL contract intact and the parity story auditable (encoder weights
hashed like any artifact); zero-shot embeddings need no training of the
TSFM itself.

**Weaknesses/risks.** (L1) actual values must be gated on release
timestamps — requires a point-in-time calendar source with revision
history; embedding drift if the TSFM is updated (pin weights by hash);
inference latency in live route (mitigated: 4h cadence makes even
seconds acceptable); crypto's calendar sensitivity is macro-mediated
(FOMC/CPI) — effect may be small; observation-dimension growth
interacts with buffer memory.

**Cost.** CPU for dataset build; one paired GPU experiment + small
preregistered grid (embedding dim, context length). **Verdict: adopt,
wave 3, after the causal event dataset exists (wave 2).**

### Option C — Decomposition (three distinct sub-options; they must not be conflated)

**C1. Causal decomposition INPUT features.** À-trous/left-aligned
wavelets, trailing-window multitaper, causal Hilbert (analytic signal
from trailing FIR Hilbert transformer), fracdiff — recomputed per bar
with warm-up discipline in the same engine that passed two-source
parity (`TechStatFeatureEngine`). Selection is delegated to the
existing full-genome GA feature-group genes (`feature_group__wavelet/
multitaper/hilbert/emd/fracdiff` already exist in the schema lineage;
they were dropped for ETH only because the dataset lacked the columns).
- Strengths: machinery exists; GA is exactly the owner's "filter useful
  from not"; no architecture change.
- Risks: (L1) is the entire risk — standard implementations use
  symmetric/centered windows; every function must ship with a
  shift-causality unit test (feature value at t invariant to appending
  future rows); EMD is intrinsically boundary-unstable and gets the
  strictest test or exclusion.
- Cost: CPU dataset regeneration + the campaign already planned.
  **Verdict: adopt, wave 2.**

**C2. Decomposition as TARGETS (forecasting-style) → translated to RL
auxiliary losses.** Literature basis: decomposition-ensemble
forecasting (CEEMDAN-Informer-LSTM; EMD-TI-LSTM claiming 36–39% error
reduction; multilevel wavelet networks; 2025 Computational Economics
review). Direct adoption is REJECTED: (a) it is a forecaster, not a
policy — grafting it forks the architecture; (b) much of this
literature decomposes the full series before splitting (L1/L4) and its
effect sizes are not causally trustworthy. The legitimate translation
for SAC: an auxiliary head on the shared encoder predicting CAUSAL
next-bar decomposition components (e.g., trend-band value), following
the RL auxiliary-task literature: "When does Self-Prediction help?"
(2024 — latent self-prediction dominates observation reconstruction),
action-conditional self-predictive frameworks, and 2025 multi-task
self-supervised trading RL.
- Strengths: representation shaping toward denoised structure without
  touching the reward; well-grounded in RL literature rather than the
  contaminated forecasting literature.
- Risks: aux-loss weight is a new sensitive hyperparameter; gradient
  interference with the critic; the owner's own warning applies —
  components without control value add optimization noise.
- Cost: GPU paired tests ×(1 + small grid). **Verdict: wave 3, ONE
  experiment per variant, latent self-prediction FIRST (strongest
  prior), causal-component head second, each with the declared
  structural handicap (§4, honesty tax).**

**C3. Decomposition metrics as DIAGNOSTICS (zero-training).** Hilbert
phase-distribution divergence, spectral band-energy distances,
distribution-similarity metrics computed post-hoc on COMPLETED traces:
(i) sim-vs-live divergence channels for Option A; (ii) fidelity gates
for synthetic data in Option F (stylized-facts checks: fat tails,
volatility clustering, autocorrelation decay). Causality is not a
constraint for post-hoc evaluation of closed windows.
- Strengths: maximum information per unit effort; zero GPU; no leakage
  surface; strengthens two audited gates.
- Weaknesses: none material. **Verdict: adopt immediately with A.**

### Option D — Self-supervised representation features (autoencoder family)

**Mechanism.** Frozen SSL encoder (hierarchical contrastive TS2Vec/
CoST family, or masked autoencoder Ti-MAE family — both currently
outrank plain VAE/VAE-GAN for time series) trained on the TRAIN SPLIT
ONLY, applied causally; embeddings appended to the observation.

**Strengths.** Complements hand-engineered features; SSL objectives
capture temporal structure the 83 features may miss; frozen-encoder
design mirrors Option B's auditable pattern.

**Weaknesses/risks.** (L2) any full-history training of the encoder
poisons validation — train-split-only, hash-pinned; representation
collapse must be checked (embedding rank/variance report); overlap with
Option C1 features (test AFTER wave 2 so marginal value is measured
against the enriched baseline, not the current one).

**Cost.** Encoder pretraining (hours, one GPU) + one paired experiment.
**Verdict: wave 3, after C1, sequenced behind B.**

### Option E — SAC internals: replay strategies and entropy stability

**Mechanism.** (i) Prioritized Experience Replay / Emphasizing Recent
Experience for SAC (the correct reading of the owner's "dynamic action
replay" intuition); (ii) entropy floor/range — your follow-up #2:
candidates drive `ent_coef` → 0 and collapse.

**Strengths.** Cheap, well-understood, directly targets an observed
pathology (entropy collapse epochs are already logged); replay changes
compose with any winner from waves 2–3.

**Weaknesses/risks.** PER's importance-sampling corrections interact
with SAC's twin critics — implementation must be tested against
published baselines first; entropy floor risks masking a symptom whose
cause is reward scale — pair with reward-scale sweep evidence.

**Cost.** GPU paired tests, smallest of all variants. **Verdict:
wave 3 (can interleave with B/D since it touches a different module).**

### Option F — Market-condition-controlled synthetic data (pretraining)

**Mechanism.** Conditional generation of OHLCV under specified regimes
→ same `TechStatFeatureEngine` → pretrain policy on synthetic, finetune
+ validate ONLY on real. SOTA: diffusion models now dominate TimeGAN/
QuantGAN for stylized-fact fidelity (DDPM-based generation, 2025
Quantitative Finance; GAN-diffusion hybrids 2026); **CoFinDiff (IJCAI
2025)** is precisely condition-controlled financial generation. Our
regime-classification tooling (feature-eng) supplies conditioning
labels; C3 metrics are the acceptance gate for generator fidelity.

**Strengths.** Addresses the true bottleneck (one ETH history = one
sample path); regime conditioning enables curriculum-style exposure to
rare conditions (crashes — finally exercising the solvency machinery
WP-C found dormant); the fixed-genome fixture is the cheap testbed.

**Weaknesses/risks.** Generator overfits → policy learns generator
artifacts (mitigate: fidelity gate + REAL-only validation, synthetic
NEVER in selection); compute-heavy; must come after the input contract
freezes or every earlier result invalidates the generator's feature
interface.

**Cost.** Highest. **Verdict: wave 4, gated on wave 2–3 freeze.**

### Option G — Portfolio expansion

Per your follow-up #1: second SAC asset with a reduced paired packet
before any SAC-wide claim; separate family before system-wide claims.
The per-asset pipeline (dataset contract → materializer → paired
campaign → parity → live gate) is now proven end to end on ETH.
**Verdict: wave 5, mechanical application of the pipeline.**

### Rejected / deferred (with reasons)

1. **Decomposition-ensemble forecasting architecture** — rejected
   (C2 rationale: architecture fork + L4-contaminated evidence base).
2. **NEAT-style topology evolution** — deferred: DOIN already evolves
   hyperparameters/features/stages; topology genes multiply the search
   space against our binding constraint (candidate evaluation cost);
   revisit only if wave-3 winners plateau.
3. **Free-text news/LLM sentiment** — deferred: unstructured, hard to
   point-in-time audit, and dominated in cost/benefit by structured
   calendar covariates (B). Reconsider after B reports.

## 3. Roadmap with branch plans

Success/failure criterion at every step: paired ordered-tuple
comparison on realistic-normal validation (direction consistency across
seeds + effect size), with quality-per-GPU-hour reported.

**Wave 0 (running).** EN/N decision program. — *Success:* curriculum
verdict for ETH/SAC; *failure/ambiguity:* explicit ambiguous state per
your §4.4; WP-D ablation decides mechanism attribution. Either branch
proceeds to wave 1 (independent).

**Wave 1 (CPU).** A + C3 diagnostics.
Tasks: (1) due-bar↔sim join tool; (2) divergence report (action
agreement, rate, SL/TP, slippage) + spectral/phase channels; (3)
calibrated execution profile PR into gym-fx cost model; (4) standing
24h/7d divergence page.
*Success:* divergence within preregistered bands → live gate hardened;
*failure* (divergence large): STOP-and-fix — tune simulator execution
model with the measured profile, re-run divergence before ANY wave-3
GPU spending (the owner's explicit tuning opportunity).

**Wave 2 (CPU + campaign).** Causal feature expansion.
Tasks: (1) causal decomposition functions + shift-causality unit tests
per function; (2) point-in-time calendar dataset + release-gated
covariates; (3) dataset regeneration + manifest; (4) two-source parity
re-run; (5) GA campaign with restored feature-group genes.
*Success:* enriched dataset becomes the new frozen contract;
*failure* (GA discards all new groups): negative result recorded,
contract unchanged, wave 3 proceeds on the lean baseline — no loss.

**Wave 3 (GPU, sequential paired tests on the WP-C harness).**
Order: B (event context) → E (replay/entropy) → C2 (aux heads) → D
(SSL embeddings), each: preregister grid → run 4 paired seeds → raw
table + paired differences → adopt/reject/ambiguous.
*Success* per item: variant enters the frozen input/agent contract;
*failure:* documented negative result; next item measured against the
unchanged baseline. Multiple-comparisons honesty: adopted variants are
re-confirmed jointly in ONE final combined-vs-baseline paired run
before the contract freezes.

**Wave 4 (GPU).** F, gated on the wave-3 freeze.
Tasks: (1) generator training (diffusion, regime-conditioned); (2) C3
fidelity gate; (3) fixed-genome pretrain/finetune fixture; (4) if
fixture positive, campaign-scale pretraining.
*Success:* pretraining lifts the paired tuple → adopt; *failure:*
generator archived as evidence; synthetic line closed until better
generators; wave 5 unaffected.

**Wave 5.** G, using whatever contract survived waves 2–4.

## 4. Cross-cutting risks and the honesty taxes

1. **Tuning-fairness tax (owner's point, adopted):** every variant
   competes against a baseline optimized for itself. We sub-search only
   variant-specific knobs on a preregistered small grid, hold the SAC
   basis fixed, and DECLARE the structural handicap in each packet. A
   variant that wins while handicapped is trusted; narrow losses earn
   at most one deeper look.
2. **Sequential-adoption drift:** wave-3 items adopted one by one can
   interact; hence the single joint confirmation run before freezing.
3. **Leakage:** every new feature ships with a shift-causality test;
   every encoder/generator is split-trained and hash-pinned; 2025 stays
   disclosed-and-disabled; a future untouched period (2026+) is the
   final protected comparison for the end-state system.
4. **Compute honesty:** all wave-3/4 packets report quality per
   GPU-hour; an expensive win must say it is expensive.
5. **Live boundary:** nothing in this roadmap touches venue authority;
   gates 8–9 remain governed by your standing orders.

## 5. Requests to the General

1. Ruling on the §7 appendix question in the decision preflight
   (final-weights evaluation recorded alongside best-checkpoint).
2. Ratification (with the owner) of the 112 quantization bounds.
3. Agreement that C3 metrics may be added to the rolling divergence
   evidence (extends, does not replace, your follow-up #5 contract).
4. Preregistration review of the wave-3 grids before any GPU spend.
5. Anything you would reorder — the waves are dependency-ordered, not
   dogma.

## 6. Reference index

TSFM: Chronos-2 (Amazon), TimesFM (Google), MOIRAI-2 (Salesforce),
Lag-Llama, Time-LLM. RL auxiliary tasks: arXiv:2406.17718,
arXiv:2406.02035; MTSSL portfolio RL (Appl. Soft Comput. 2025);
incremental RL + SSL trading (ESWA 2025). Decomposition forecasting
(L4-discounted): Computational Economics review 2025 (10.1007/
s10614-025-10899-z); CEEMDAN-Informer-LSTM (10.1016/j.asoc.2025);
wavelet multimodel frameworks (10.1016/j.bir.2025). SSL for TS:
TS2Vec/CoST/Ti-MAE/Series2Vec (arXiv:2312.03998). Synthetic:
DDPM financial series (arXiv:2410.18897; Quant. Finance 2025
10.1080/14697688.2025.2528697); CoFinDiff (IJCAI 2025); GAN-diffusion
hybrid (arXiv:2605.27113).
