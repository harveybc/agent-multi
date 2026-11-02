# Batch-32 correction and clean relaunch

Owner correction, 2026-11-02: batch 32 was missing from the search space.
Options are now **32, 64, 128, 256**. Initial population starts with the
baseline configuration at batch 32, followed by the unchanged batch-128
baseline, then 18 random candidates. Population remains 20; two seeds per
candidate; maximum 2000 generations; combined TRAIN/INNER_A patience remains
10 with separate training/evolution counters. No other gene ranges changed.

During status verification a second defect was found: after synthetic
qualification, the worker called `configure` again, which tried to set
PyTorch interop threads after parallel work had started. The worker failed
before starting financial learning. It now uses the existing RNG-only
`configure_fresh_seed`, without changing the numerical thread configuration.

The old campaign coordinator and its workers were stopped. No completed
financial candidate existed. Old attempt evidence remains unchanged; no old
candidate receipt is silently relabeled or used as new optimization evidence.

Code `3a392cec`; new campaign identity:
`9299390af8d328ab812ecdc3e2b401d01378cf81b315d12b908e8b82d90c17b5`.
Replaces the launch in [188](188_SIMPLE_SAC_DEAP_LAUNCH.md).
Started 2026-11-02 06:44:34 UTC / 01:44:34 America/Bogota.

Verification: 51 tests pass, including all four exact batch decoding choices,
the seeded batch-32 configuration, and the qualification/financial reset
regression. Engineering smoke covers batch 32, baseline 128 and upper-bound
256: six real SAC updates, eighteen monitoring accounts and exact reloads.
Source/data/engine/evidence pins authenticate independently on both workers.
The wavelet evidence remains applicable: same recipe, source and store bytes.

The revised population is evaluated under a new immutable runtime. Full
floating selection still requires the prelaunch results briefing; this change
does not launch phase 2.
