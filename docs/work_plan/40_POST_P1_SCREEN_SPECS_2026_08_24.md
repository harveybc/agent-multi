# 40. Post-P1 Screen Specifications (WP4, prepared NOT launched)

Authority: Musashi SOTA correction and roadmap order 2026-08-24 (@6fef96ac),
WP4. Status of every spec here: **PREPARED / NOT LAUNCHED**. Launch requires
(a) P1 terminal seal, (b) Musashi design verification, (c) explicit dispatch
authorization for anything touching GPU. Nothing below mutates the running
campaign.

## Shared contract (applies to all four screens)

- **Data**: pinned ETH H4 dataset
  `ethusdt_4h_tech_stat_full_model_ready.csv`
  sha256 `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`.
- **Folds**: rolling development origins (document 38 §3.3 spirit, brought
  forward): three origins — fit≤2021→score 2022; fit≤2022→score 2023;
  fit≤2023→score 2024. NEVER 2024-only. **Sealed 2025 is absent from every
  materialized config** — asserted by the materializer
  (`sealed_test structurally unmaterialized`), not by prose.
- **Seeds**: paired seeds {101, 202, 303, 404} across all arms of a screen;
  same seed = same data order/init draw wherever the arm allows it. Four
  seeds SCREEN mechanisms; champion selection needs more (doc 38 §23.4).
- **Costs**: full cost engine on every arm — taker fee + half-spread +
  configured slippage; identical cost config hash across arms of a screen,
  bound into each arm return.
- **Mandatory per-arm evidence**: net return, Sharpe (net), max drawdown,
  turnover (sum |Δposition|), activity (trades/scored year), per-bar return
  series (retained for SPA/DSR), inference latency p50/p95 per decision,
  deadline evidence (decisions delivered inside the H4 bar), effective
  config hash, code identity (immutable worktree commit), replay/env
  statistics where RL is involved.
- **SL/TP coexistence (order clause)**: native SL/TP remain a safety
  envelope on every live/demo execution and are SIMULATED identically in
  every arm here; a learned early-close or target-position action operates
  INSIDE that envelope and never disables it. Protective closes are logged
  as `envelope_close`, agent closes as `policy_close`, and both enter
  turnover/activity accounting.
- **Statistics**: paired per-seed deltas with bootstrap intervals; IQM
  alongside mean; every arm x seed logged as a trial for DSR/SPA counting.

## Screen B — Same-harness economic baselines (CPU-only, first post-P1)

- **Question**: does any P1-surviving SAC policy add value over mechanical
  rules under identical costs and folds?
- **Arms (5)**: B0 flat (always out); B1 buy-and-hold (always long 1.0);
  B2 TSMOM sign of k-bar return, k ∈ {30d, 90d} in H4 bars, chosen ON
  DEVELOPMENT FOLDS ONLY and reported both; B3 volatility-scaled rule
  (TSMOM sign x target_vol/realized_vol, clamped [−1,1]); B4 the frozen P1
  champion policies evaluated inference-only on the same folds.
- **Compute class**: CPU (`CUDA_VISIBLE_DEVICES=""`); no training, only
  rule evaluation + frozen-policy inference.
- **Decision rule (gate G1, doc 38 §23.3)**: SAC "adds value" only if its
  paired net Sharpe beats EVERY baseline on ≥2 of 3 origins and ≥3 of 4
  seeds, and Hansen-SPA over per-bar excess returns vs the best baseline
  does not classify the win as best-of-N noise. Otherwise architecture work
  stays frozen and screen A becomes diagnosis, not scaling.

## Screen A — Action-contract screen

- **Question**: which action semantics does the H4 problem actually pay
  for? (SOTA-03: current sign-only threshold-0 may collapse magnitude and
  induce switching.)
- **Arms (4)**, identical SAC trunk, data and parameter budget:
  A1 sign-only target (current contract, control);
  A2 continuous target exposure in [−1,1] with bounded risk sizing and
  |Δposition|-cost internalized in reward (DMN-style);
  A3 ternary long/flat/short with calibrated deadband + hysteresis (enter
  |a|>θ_in, exit |a|<θ_out, θ calibrated on development folds only);
  A4 explicit close/hold head while a position is open (doc 39 semantics)
  under the same envelope.
- **Compute class**: GPU (training); requires dispatch authorization.
- **Decision rule**: winner = best paired net-Sharpe with turnover reported
  next to it; a win produced solely by lower turnover at equal gross is a
  VALID win (costs are real). Winner's contract is FROZEN before screen R.

## Screen R — Retraining cadence screen

- **Question**: does rolling adaptation add net value over a frozen policy,
  and at what cadence? (SOTA-07 — central to the owner's business model.)
- **Arms (4)**: R0 frozen control; R168 weekly refresh; R24 daily; R12
  twice-daily. Each refresh: bounded fine-tune on the trailing window under
  the frozen action contract; every Rx is paired with R0 on the same folds
  and seeds. Update variants within an arm (warm vs shrink-perturb warm
  start, doc 10 gap matrix D5) enter only as a nested sub-comparison if
  budget allows, else warm-start only.
- **Compute class**: GPU for refresh training; refresh runtime p50/p95
  measured and published — 6h cadence enters ONLY after p95 leaves a safe
  deadline margin (order clause).
- **Decision rule**: incremental value after costs vs paired frozen
  control, plus stability (variance of rolling Sharpe), deadline misses,
  state continuity, and degradation-of-frozen curves.

## Screen C — Capacity-matched architecture screen

- **Question**: does the grouped extractor earn its complexity at matched
  capacity? (SOTA-04: GKX warning — shallow dominated deep in low-signal
  finance; 18,085 rows.)
- **Arms (3+1)**: C1 flat MLP (control, current); C2 small shared causal
  temporal baseline (single TCN or GRU over all families); C3 grouped
  extractor (TCN/Transformer/GRU branch per family) at the SAME approximate
  parameter budget as C2; C4 grouped+fusion ONLY IF C3 wins across origins
  and seeds.
- **Mandatory extra evidence**: parameter count, FLOPs/decision, wall time,
  inference latency next to every economic metric. No component is called
  state of the art because its family is modern.
- **Compute class**: GPU; last screen before DOIN domain materialization.
- **Decision rule (gate G2)**: C3 must beat C1 AND C2 on paired net Sharpe
  across ≥2 of 3 origins and ≥3 of 4 seeds at matched capacity; otherwise
  the simpler winner feeds the DOIN domain.

## Launch preconditions checklist (every screen)

1. P1 sealed and aggregated; inert treatments classified per seed.
2. Musashi has verified the screen design (this document + any deltas).
3. GPU dispatch authorization explicit where applicable (B is CPU-only).
4. Immutable worktree + launch-identity manifest with literal digest guard
   (finding 315 pattern) on every sequential wrapper.
5. Materialized configs re-validated: sealed-2025 absent; nested-contract
   sha bound; effective config + cost config hashes in every arm return.
