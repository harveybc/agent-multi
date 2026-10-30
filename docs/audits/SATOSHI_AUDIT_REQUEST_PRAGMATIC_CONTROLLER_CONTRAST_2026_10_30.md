# Audit request: completed pragmatic controller contrast

For Satoshi. 2026-10-30. Audit completed engineering evidence, not a claim of
production readiness or learned-model profitability.

## Pinned delivery

Runtime `agent-multi@6c9f597e`; native engine `e0a1c2b`.
Identity `f5a82347b3181ee814fe07869b9e858965d52183b287293b144378eaced4e823`.
Plan [160](../work_plan/160_PRAGMATIC_CONTROLLER_CONTRAST_PLAN.md),
deployment [162](../work_plan/162_PRAGMATIC_CONTRAST_LAUNCH.md),
final report [163](../work_plan/163_PRAGMATIC_CONTROLLER_CONTRAST_RESULTS.md).
Evidence: [REPORT.json](evidence/pragmatic_contrast_full_20261030/REPORT.json),
[SELECTION.json](evidence/pragmatic_contrast_full_20261030/SELECTION.json),
[COMPLETE.json](evidence/pragmatic_contrast_full_20261030/COMPLETE.json),
[manifest](evidence/pragmatic_contrast_launch_20261030/MANIFEST.json).
Per-world envelopes and full traces are retained in the shared run archive;
COMPLETE authenticates their names and bytes. They are not all included in git.

All1,584 accounts completed:864 calibration,720 confirmation, four frozen
representatives and five report settings.247 runtime tests passed. The later
reporting-only undefined-Sharpe guard passes250 tests and changes none of the
archived accounts. Reproduce the campaign from the runtime pin, not later HEAD.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/run_pragmatic_contrast.py \
  --engine-root <pinned-native-engine> --output <fresh-audit-root> --workers 4
```

## Attack these points first

1. **Information boundaries.** Ideal/noisy reports intentionally retain future
   structure for synthetic design. Only the prefix reporter is causal. Check
   shadow_environment replaces the entire future price suffix, including
   after the operational deadline, and does not leak actual future prices.
2. **Non-anticipative actions.** Verify the same candidate plan is evaluated
   across all paths before choosing; no per-path oracle followed by averaging.
3. **Close value.** Ideal sequential arms produce71/148 and77/157 locally
   harmful closes. Determine which are genuine account-level opportunities
   versus controller defects. Do not infer this from aggregate return alone.
4. **Shortlist and effective action.** Two shortlisted proposals can exclude
   a beneficial direction/horizon. Ties and persistence can override the
   frontier score winner. Check traces against executed native actions and
   reject any stronger optimality claim than the explicit bounds in163.
5. **Native business semantics.** Both directions, no implicit reversal,
   one position, SL/TP, q times rel_volume times balance exactly once,
   leverage50, no daily cap, opening12h after session start and flat12h before
   its close. Verify cost, equity, drawdown and position-size accounting.
6. **Matched evidence.** Identical worlds/signals per arm; calibration-only
   selection frozen before confirmation; no hidden fallback or retuning.
   Check all24 configurations and the four selected identities independently.
7. **Time and activity units.** Duration is in hours, tau0 has the same unit;
   operational market weeks, not split Sunday/Monday ISO-week labels, divide
   activity. Inspect horizons actually used and size/cost effects.
8. **Uncertainty and inference.** Three-path dispersion is not calibrated
   confidence. Prefix losses are not a test of trained RL. No historical
   strategy reproduction, independence/significance or production claim.
9. **Receipts and restart.** Reproduce account traces and frozen selection;
   altered code, config, native engine and envelopes must reject resume.
   Completed resume must perform no training or rewriting of sealed units.

Please return separate verdicts for execution/accounting, paired comparison,
close semantics and readiness for a learned adapter. Give minimal reproducible
counterexamples and corrections. This request does not authorize changing
the frozen data, strategy parameters or archived results, or starting either
financial training lane. The guided and pure-RL lanes remain distinct.
