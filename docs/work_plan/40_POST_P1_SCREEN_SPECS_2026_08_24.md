# 40. Post-P1 Screen Specifications (WP4, prepared NOT launched) — rev 2

Authority: Musashi SOTA correction and roadmap order 2026-08-24
(@6fef96ac) WP4, amended per
`AUDIT_SATOSHI_SOTA_RETURN_WP1_WP4_2026_08_24.md` (R01, R03, R04, R08)
and `MUSASHI_TO_GENERAL_SATOSHI_SOTA_RETURN_CORRECTION_ORDER_2026_08_24`
(C1, C3, C5). Status of every spec: **PREPARED / NOT LAUNCHED**. Launch
requires (a) P1 terminal seal, (b) Musashi design verification,
(c) explicit dispatch authorization for anything touching GPU. Nothing
below mutates the running campaign.

## Shared contract (all screens)

- **Data**: pinned ETH H4 dataset
  `ethusdt_4h_tech_stat_full_model_ready.csv`
  sha256 `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`.
- **Folds**: three rolling development origins — fit≤2021→score 2022;
  fit≤2022→score 2023; fit≤2023→score 2024. NEVER 2024-only. **Sealed
  2025 absent from every materialized config**, enforced by
  `tools/post_p1_screen_contract.check_sealed_absence` (refusal, not
  prose) with negative tests in
  `tests/test_post_p1_screen_contract.py`.
- **Causal eligibility (C1)**: every policy entering an origin's
  comparison must satisfy
  `check_causal_eligibility`: fit AND selection information strictly
  before that origin's score start. A model whose fit/selection
  timestamp reaches or exceeds the score start is REFUSED
  (negative-tested). Frozen P1 artifacts are admissible ONLY under the
  explicit `diagnostic_2024` label, on the 2024 diagnostic, and never
  enter the three-origin G1 claim.
- **Seeds**: paired {101, 202, 303, 404} across all arms; same seed =
  same init/data-order draw where the arm permits. Four seeds screen
  mechanisms; they never select champions (doc 38 §23.4).
- **Costs**: full cost engine every arm — taker fee + half-spread +
  configured slippage; identical cost-config hash across a screen's
  arms, bound into each arm return.
- **Per-arm evidence**: net return, net Sharpe, max drawdown, turnover
  (Σ|Δposition|), activity (trades/scored year), per-bar return series
  retained (statistics contract, doc 41), inference latency p50/p95,
  deadline evidence (decision inside the H4 bar), effective config
  hash, code identity (immutable worktree commit + launch-identity
  manifest with literal digest, finding-315 pattern), replay/env stats
  where RL is involved.
- **SL/TP coexistence**: native SL/TP remain the safety envelope in
  every arm, simulated identically; learned target-position/early-close
  acts INSIDE the envelope and never disables it. `envelope_close` and
  `policy_close` are logged separately; both enter turnover/activity.
- **Observation identity (F1/SOTA-C01)**: every screen materializer
  calls `check_observation_identity` BEFORE model construction, binding
  the exact ordered feature list, count, digest, window/price-window/
  agent-state flags and flattened shape against the screen's declared
  observation contract; any drift (e.g. the P1-executed 84-feature
  identity vs the declared 83) is REFUSED (negative-tested).
- **Trial accounting (C5)**: every effective option — each arm, seed,
  calibrated threshold, lookback — is one trial in the DSR/SPA ledger
  (doc 41). No option may be chosen after seeing results unless it was
  predeclared here as a comparison.

## Screen B — Same-harness economic baselines (CPU-only, first post-P1)

- **Question**: does a causally eligible SAC add value over mechanical
  rules under identical costs and folds?
- **Rule arms (fully bound, C5/R08)**:
  - B0 flat: position 0 always.
  - B1 buy-and-hold: position +1.0 always, no leverage.
  - B2a TSMOM-30d: position = sign(close_t−1 − close_t−181) (180 H4
    bars); B2b TSMOM-90d: same with 540 bars. BOTH run as separate
    arms; neither is selected post hoc; each is a trial.
  - B3 vol-scaled TSMOM: position = sign_B2a × min(1, σ_target/σ_real),
    σ_target = 15% annualized; σ_real = std of the last 180 per-bar log
    returns × sqrt(2190) (H4 bars/year), computed through bar t−1
    (lag 1); leverage cap 1.0; no other free parameter.
- **SAC arm (C1)**: B4 causal per-origin SAC — trained independently AT
  EACH origin using only that origin's fit data, under the frozen P1
  recipe (contract sha 2b31b7770f815b75 hyperparameters, fixed LR 3e-4,
  same stopping), selection restricted to data before the origin's
  score start. This is GPU work: Screen B's rule arms are CPU-only and
  can run first; B4 requires dispatch authorization and may follow.
- **Diagnostic annex (non-G1)**: frozen P1 champions, labeled
  `diagnostic_2024`, evaluated inference-only on the 2024 fold ONLY;
  reported in an annex; excluded from G1 by
  `check_causal_eligibility` (tested).
- **Decision rule (gate G1)**: SAC "adds value" only if B4's paired net
  Sharpe beats EVERY rule arm on ≥2 of 3 origins and ≥3 of 4 seeds, and
  the doc-41 SPA procedure over per-bar excess returns vs the best rule
  arm does not attribute the win to best-of-N noise.

## Screen A — Action semantics, staged (C3/R03)

- **Question**: which action semantics does the H4 problem pay for —
  decomposed so each stage varies ONE object.
- **Stage A0 — mapping only**: one scalar actor output a∈[−1,1], one
  common economic reward and sizing engine; three PREDECLARED mappings
  of the same output: (i) sign(a); (ii) continuous target = a;
  (iii) ternary with FIXED predeclared deadband θ_in=0.3, θ_out=0.1
  (hysteresis). No calibration in A0; the fixed thresholds are part of
  the spec, chosen before any result. Identical trunk, budget, data.
- **Stage A1 — calibration of the surviving mapping**: only if a
  thresholded mapping survives A0. Deadband/hysteresis grid
  θ_in ∈ {0.2, 0.3, 0.4} × θ_out ∈ {0.05, 0.1, 0.2}, calibrated on
  development folds only; EVERY grid point is a counted trial; the
  chosen pair is frozen before any later screen.
- **Stage A2 — mechanism change**: explicit close/hold head (doc 39
  semantics) as a SEPARATE capacity-matched comparison against the
  A0/A1 winner (±5% parameter tolerance, matched update budget). Runs
  only after A0/A1 conclude.
- **Compute**: GPU; each stage needs its own dispatch authorization.
- **Decision rule**: per stage, paired net Sharpe with turnover
  reported; a win from lower turnover at equal gross is valid.

## Screen R — Retraining cadence, causal (C3/R04, F2/SOTA-C02)

- **Question**: does rolling adaptation add net value, at what cadence —
  with BOTH the update method AND the total optimization budget frozen,
  so cadence is the only treatment.
- **Frozen update contract**: warm-start continuation (no reinit),
  optimizer state carried, replay carried and trimmed to the trailing
  window of 2,190 H4 bars ending at the refresh bar, batch and LR
  exactly the frozen P1 recipe values.
- **Equal TOTAL update budget (F2/SOTA-F02)**: every adaptive arm
  receives EXACTLY **260,000 gradient steps per scored year** via the
  deterministic quotient/remainder schedule
  `tools/post_p1_screen_contract.refresh_update_schedule` (the first
  `remainder` refreshes get one extra update; exact conservation is
  regression-tested): R168 = 5,000 × 52; R24 = 713 × 120 + 712 × 245;
  R12 = 357 × 120 + 356 × 610. R0 frozen control receives 0 (the
  no-adaptation reference, paired per fold/seed). Schedules are
  materialized constants, never tuned.
- **Arms**: R0 frozen; R168; R24; R12 — cadence is the sole factor.
- **Separate operational screen R-op (`cadence_plus_compute`)**:
  OPTIONAL, distinct spec and label — fixed 5,000 steps per refresh at
  every cadence, honestly measuring the OPERATIONAL bundle
  cadence+compute; compared on value per GPU-hour as well as economic
  outcome. No causal cadence claim may cite R-op, and no covariate
  language may claim causal isolation anywhere.
- **Deferred screen R2**: update METHOD at the winning causal cadence
  only — fresh reinit vs warm vs shrink-and-perturb (λ, σ predeclared
  in its own spec). Not part of Screen R.
- **Compute**: GPU; refresh runtime p50/p95 measured and published; 6h
  cadence enters only after p95 leaves a safe deadline margin.
- **Decision rule**: incremental net value vs paired frozen control,
  stability (variance of rolling Sharpe), deadline misses, state
  continuity, degradation-of-frozen curves.

## Screen C — Capacity-matched architecture (C5/R08)

- **Owner clarification (2026-08-26)**: the flat MLP may be used as an
  architectural baseline, not as the primary candidate. Reuse existing valid
  evidence first. Any new run must be bounded, matched on data/update budget and
  justified as necessary to estimate the value added by temporal structure.
  A deliberately weak shared-GRU control is not required.
- **Question**: which strong, data-compatible temporal architecture earns the
  right to enter DOIN optimization? Complexity is evaluated, but the campaign
  does not spend weeks rediscovering that flattened observations discard
  temporal and semantic structure.
- **Pre-screen C0 (CPU/single-GPU mechanics only)**: exact historical/live data
  availability, observation identity, tiny-fixture overfit, gradient arrival
  at every branch, causal masks, parameter/FLOP/latency report and executing
  pretraining artifact round-trip. This is not a performance comparison.
- **Strong arms**: C1 multiscale shared PatchTST/TFT-style encoder; C2 typed
  grouped extractor with PatchTST/causal-TCN, TFT/Transformer, TimesNet-style
  and GRU branches selected by input family; C3 C2 plus branch pretraining and
  gated cross-family attention. All receive matched update and wall-clock
  budgets; parameter count is reported and controlled within a predeclared
  band, not forced to equal a weak toy network.
- **Optional C4**: regime-routed mixture of the best temporal experts, only if
  per-regime headroom exists before training it.
- **Mandatory evidence**: parameter count, FLOPs/decision, wall time,
  inference latency, per-family ablations and representation diagnostics beside
  every economic metric.
- **Decision rule (gate G2)**: a strong arm must beat the best causal rule
  baseline and the other strong arms on paired net Sharpe across ≥2 of 3
  origins and ≥3 of 4 seeds. If none does, improve data/reward/action semantics;
  do not promote the archival MLP by default.

## Economic hurdle (distinct from model baselines)

- Record a predeclared **10% nominal annual COP opportunity-cost hurdle** as the
  owner's current CDT comparison point. It is not an architecture arm and does
  not replace flat cash, buy-and-hold, TSMOM or the MLP architecture baseline.
- Report strategy returns first in the venue/account currency after all costs.
  Also report a COP view using a timestamped USD/COP conversion series and state
  whether tax and inflation are included. Never compare a USD return directly
  with 10% COP without the currency bridge.
- The hurdle is a reporting threshold, not a reward term, fitness bonus or
  post-hoc promotion override. Its rate must be versioned by evaluation date;
  10% is the owner's current planning assumption, not a permanent market fact.

## Launch preconditions (every screen)

1. P1 sealed and aggregated; inert treatments classified per seed.
2. Musashi has verified this revision.
3. Explicit GPU dispatch authorization where applicable (Screen B rule
   arms are CPU-only).
4. Immutable worktree + launch-identity manifest, literal digest guard.
5. Materialized configs pass `check_sealed_absence` and every entering
   policy passes `check_causal_eligibility`; effective config + cost
   config hashes bound into every arm return.
6. Statistics contract (doc 41) predeclared parameters bound into the
   screen's aggregation config.
