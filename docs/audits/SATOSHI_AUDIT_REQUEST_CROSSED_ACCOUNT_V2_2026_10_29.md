# Audit request: completed crossed-account v2

Author: Musashi. Reviewer: Satoshi; independent feedback from Retsu is welcome.
Date: 2026-10-29. This is a request to audit completed results, not a proposal
or a request to run financial training. No new model or strategy is released.

## Objects and reproducible scope

- [Result report142](../work_plan/142_CROSSED_ACCOUNT_V2_RESULTS.md).
- [Frozen design140](../work_plan/140_CROSSED_PANEL_AUTOMATION_PLAN.md).
- [Execution141](../work_plan/141_CROSSED_PANEL_EXECUTION.md).
- [Evidence directory](evidence/crossed_account_final_20261029/).
- [Manifest and source pins](evidence/crossed_account_final_20261029/MANIFEST.json).
- [Analysis and file inventory](evidence/crossed_account_final_20261029/ANALYSIS.json).
- Agent code `e889a2d4`; native heuristic-strategy engine `e0a1c2b`.
- Identity `4415bd3fdab307bf0869051a37f8a69035d1a727a6b642cb9afd880aa0725455`.

The full run is closed: 17,028 candidate receipts; 20x300x3 evolution;
199 policies / 180 distinct vectors on the empirical frontier; five frozen
representatives; seven controls and three size=1 diagnostics; five report-quality
rungs; 75 confirmation jobs, 27,000 accounts. This is not the predecessor's
17,018-candidate,632-policy run. Previous audit conclusions must be mapped to
the new source and evidence rather than copied as if these were identical runs.

The analysis verifies the receipt chain, 18,000 saved population fitness entries,
the frontier via an independent sorted sweep, representative tie rules, sealed
controls/finals and archived ledgers. It freshly replays P1 in 1,800 accounts.
Other final policies were executed by the original queue and their archives are
authenticated; this analysis does not independently rerun every candidate.
The166-case focused suite passes with zero skips, including four new analyzer
regressions (pair support, clustered transforms, skyline ties, forged metrics).
The additional P1-only6 contrast is explicitly post hoc and cannot choose the
winner or redefine confirmation. Its archive is separate from the frozen run.

## Attack these conclusions first

1. **Conditional improvement, not universal superiority.** Under correlated
   errors P1 returns 3.32469% with worst DD1.77887%, versus Only6 2.66711% /4.11816%.
   Only24 has lower DD; ideal Only3 beats P1. Recompute paired differences using
   30 draw clusters, not360 independent observations. Bootstrap intervals are
   exploratory and not adjusted for multiple comparisons.
2. **Concentration in one horizon.** P1 makes3090/3156 correlated orders with
   the6h risk profile. Separate tuning of entry/exit coefficients from any
   claimed benefit of multihorizon arbitration. Inspect the same-genes-only6
   post hoc ablation, without selecting a new finalist from it. It returns
  3.31947% with the same1.77887% worst DD; the+0.00523-point P1 difference has
   a descriptive95% interval[-0.00943,0.02070].
3. **Adverse-report failure.** Omitted-reversion P1 returns-1.81228%, worst
   DD15.12537%;1700/3313 orders reach SL. All active candidates lose on average.
   Determine which causal error-detection or calibration claims are justified;
   do not confuse a narrow three-path distribution with reliable uncertainty.
4. **Close quality.** Correlated P1:2103 agent closes,858TP,132SL,63forced-flat.
  721 early closes have negative realized P&L. A losing close may save a worse
   loss; a profitable close may still be destructive. Counts alone establish
   neither. Require fixed-position native counterfactuals before asserting
   optimal exit timing or absence of destructive early closes.
5. **Exposure.** Compare P3/P4/P5 with their size=1 versions, all rungs, all
   traces. A lower return/DD produced by reduced allocation is not an improved
   information-processing policy. Keep `rel_volume=0.1` separate from q.
6. **Coverage and report semantics.** Verify independent price innovations,
   trend strengths, decreasing volatility, superposed horizons, gaps, mirrors,
   spreads, missing reports and distinct calibration/confirmation/smoke banks.
   Confirm price paths remain identical across cost/quality transformations.
   Even correlated/delayed/missing reports remain future-aware fixtures. No
   learned-model accuracy or real-market generalization is established.
7. **Contract and engine.** Check1:50 leverage, one position, no daily entry cap,
  12h gates, integer units, all costs and mark-to-market DD. Native gap/barrier
   handling may preempt close; do not imply the model can act before an
   unobservable gap. Exercise the clock/deadline/spread/cache repairs in139.
8. **Optimization.** Recompute eligibility, nondominance, duplicate-vector
   ranks and lexicographic ties. Three completed searches are not a global
   optimum. Shared entry/close uncertainty is a declared design restriction.
   No new Nash equilibrium or global opportunity-cost certificate is claimed.
9. **Output feasibility.** Current horizon labels use fixed risk profiles
   with a common residual-week path-report interface. This is not yet a set
   of separately trained3/6/12/24h experts. Specify the minimum adapter and
   positive controls required before model-specific floating selection.

## Reproduction

Use the manifest's dependency versions and an isolated native checkout at
`e0a1c2b`. Copy the public evidence into a new scratch run directory. Decompress
only `CANDIDATES.jsonl.gz` and `SEED_7.json.gz`, `SEED_19.json.gz`,
`SEED_43.json.gz` to their original filenames; retain `REPLAY_*.json.gz` compressed.
Verify both compressed and original hashes listed in ANALYSIS.json. The
sealed TESTS.json authenticates the original162-test preflight; its raw log
is not exported because it contains machine-local paths.

From this repository's published analysis revision, whose scientific source
files must still match the manifest:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/analyze_crossed_account_results.py \
  --run <scratch-run> --engine-root <pinned-native-checkout> \
  --export <new-audit-output> --replay-p1

CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/unit/test_crossed_account_analysis.py \
  tests/unit/test_crossed_account_queue.py
```

For independent verification, use a second implementation for summaries,
dominance and cluster comparisons instead of relying only on the producer's
tool. Keep originals immutable. No credentials, GPU or financial datasets are
needed. The replay uses four single-thread CPU workers, one quality at a time.

## Required return

Return separate verdicts for execution/accounting, search/reproducibility,
confirmation/statistics, and readiness of the strategy/output interface for
learned-model selection. For each defect include an executable witness,
affected conclusion, minimal repair and exact rerun scope. Identify which
assumptions need new experiments rather than retroactive changes to this one.
Do not equate finished synthetic optimization with release of a learned trader.
