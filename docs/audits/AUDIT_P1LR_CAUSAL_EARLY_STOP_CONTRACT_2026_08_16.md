# P1LR Causal Early-Stopping Contract Audit

Date: 2026-08-16
Auditor/implementer: General Musashi (Codex)
Scope: corrected-observation ETH SAC L1 comparison

## Verdict

The four-epoch phase-1 decision contract is rejected for the scientific
question. It could not exercise early stopping and mixed a fixed phase-2 LR
with one different phase-1 LR stratum. The in-progress identity was stopped
without deleting its files. It must not enter recipe selection.

The replacement contract is a paired 2x2 design. At a fixed seed and LR,
`normal->normal` and `easy->normal` differ only in phase-1 dynamics. LR is
constant across the two phases of a cell. The two LR strata remain explicit,
so LR and interaction effects are estimable rather than hidden.

## Configuration Audit

- Data: 11,509 fit rows through 2022; 2,190 scored 2022 monitor rows; 2,190
  scored 2023 inner-validation rows; 2,196 scored 2024 outer-validation rows;
  2025 sealed and unavailable to training/selection.
- Observation: 32 x 83 engineered features plus four agent-state values =
  2,660 inputs; rolling z-score 256; clip 10; raw price window disabled.
- Initialization: one zero-update genesis tensor per seed, shared by all four
  cells of that seed; distinct across seeds.
- SAC: explicit `[256,256]`, batch 256, replay 40,000, learning starts 1,000,
  train frequency/gradient steps 1/1, gamma 0.99, tau 0.005, fixed entropy 0.2.
- Execution: normal phase uses commission 0.0002 per side, spread 0.0001,
  leverage 1, relative volume 0.05, ATR SL/TP 2/3 and protected entries.
- Boundary: policy weights transfer; optimizer moments and replay transitions
  do not. This boundary is identical in control and treatment.
- Stopping: 1,000 epochs per phase, 20,000 timesteps/epoch, patience 60,
  floor 40 and min_delta 1e-4 in both phases. Activity stop is disabled.
- Selection: train-monitor plus inner-validation paired utility. Outer 2024 is
  one final truth evaluation. Sealed 2025 cannot be opened.

## Automated Invariants

1. Matched difficulty pairs differ only in `phase1_mode`.
2. `learning_rate == phase1_learning_rate == easy_learning_rate` in every
   corrected cell.
3. A runtime LR mismatch refuses before environment/model construction.
4. Both phase budgets and stopping knobs are asserted from the materialized
   config, not copied from prose.
5. Both phase endpoints persist epoch counts and stop reasons.
6. The observation contract is applied by materialization and revalidated at
   training/evaluation boundaries.

## Fleet Identity Incident And Correction

The first corrected fleet launch was stopped during its first minute when
distributed heartbeats exposed two experiment identities. Omega's canonical
`gym-fx` checkout contains local documentation-only edits while Dragon and
Gamma are clean. No canonical file was reverted, copied or discarded.

Fleet runs now bind `AGENT_MULTI_GYM_FX_ROOT` and `PYTHONPATH` to an immutable
clean `gym-fx` runtime worktree at the same commit on all hosts. Launch must
refuse unless every host independently derives the same screen and decision
identities before GPU training begins.

## Residual Research Questions

This experiment does not claim that either LR, `[256,256]`, replay 40,000,
window 32, entropy 0.2 or patience 60 is globally optimal. They are held fixed
or explicitly stratified to answer the difficulty question. Phase-specific LR
schedules, positive-easy handoff gating, longer easy allocation, topology and
feature selection remain separate future factors and must not be inferred from
this result.
