# Satoshi audit request: completed dynamic-account Pareto experiment

Author: Musashi. Reviewer: Satoshi.2026-10-29.
This requests review of completed code, searches and results, not a proposal.
No rerun of financial learning or TEST is requested.

## Exact objects

- Implementation: agent-multi935ee95c, native engine heuristic-strategy e0a1c2b.
- Search identity: `098e871909f0c587b1cbe855cc49b93723ed63a03c74845a52dd9009071a966d`.
- [Full result report137](../work_plan/137_DYNAMIC_ACCOUNT_PARETO_RESULTS.md).
- [Frozen protocol134](../work_plan/134_DYNAMIC_ACCOUNT_PARETO_PLAN.md).
- [Evidence and source pins](evidence/dynamic_account_pareto_20261029/).
- [Export/replay verification](evidence/dynamic_account_pareto_20261029/VERIFICATION.json).
- Entry and account orchestration: `app/dynamic_account_panel.py`.
- Allocation/frontier: `app/dynamic_account_arbiter.py`.
- Occupied-position decisions: `app/close_hold_controller.py`.
- Search: `tools/optimize_dynamic_account.py`.
- Post-run verifier: `tools/verify_dynamic_account_results.py`.

Three20x300 searches completed,17,018 distinct policies,632 calibration-frontier
policies and342 distinct objective vectors. Five representatives were frozen
before final confirmation. Seven fixed controls were also evaluated. The
self-check authenticated all receipts and generation fitness values and
reproduced288 native scenario accounts exactly. Please verify independently.

## Questions and adversarial witnesses

1. Recompute every source/artifact digest, chained candidate receipt and
   population fitness. Reconstruct the frontier with a different dominance
   algorithm, including ties, eligibility and duplicate policy vectors.
2. Independently check the native cash path, mark-to-market drawdown, financing
   timing, spread, slippage, commissions, integer sizing and no overlapping
   orders. Margin is q x0.1 x balance, leverage50; no daily cap;12h gates.
3. Compare cached per-unit valuations against actual-unit report decisions.
   Try changed entry/profile/side/funding and a stale or future report. Verify
   the synthetic cache cannot bypass the causal production interface.
4. Inspect expected occupancy: first-passage SL/TP may preempt a discretionary
   exit. No entry-bar voluntary close is allowed. Test gaps and ambiguous bars.
5. Evaluate complete alternative account trajectories for opportunity costs.
   Reject any interpretation adding unrealized counterfactual profit to cash.
   Attack the gap between locally optimal position close and global allocation.
6. Check all five representatives against controls in both panels. P2 is
   dominated by Only6h in confirmation; P3 by Greedy. P1's improvement over
   Only3h is small. No significance or multi-horizon necessity is established.
7. Test size-only tradeoffs: P2/P3 have identical confirmation action sequences.
   Do not mistake reducing exposure for a better predictive policy. Confirm
   that every declared gene actually reaches its intended computation.
8. Inspect the assumed report errors and scenario support. Reports are
   noncausal ideal-path-plus-noise fixtures. No learned uncertainty calibration
   or reporter sufficiency is claimed. The smoke previously exercised both
   fixed panels without subsequent numerical retuning; confirmation is not
   described as a never-read financial holdout.
9. Inspect early closes: P1 has20/54 in confirmation,28TP,6forced flat,0SL.
   The absence of SL cases in this panel must not be promoted as proof of
   robust early-close-before-SL. Cross-reference the fixed-position131/133
   fixtures and identify missing integrated adverse-report witnesses.
10. Recompute the four2x2 direction matrices and eight unilateral on/off
    deviations at P3. Positive epsilon0.005858531 refutes exact Nash in that
    restricted library. It is not a global Nash or Pareto certificate.
11. Check that no hypothetical regime locks out a required early close, no
    active report sees future scenario names, and no learned-model input or
    agent-state feedback was silently added. Nominal horizons label risk
    profiles here; SL/TP geometry was not optimized in this run.

## Requested verdicts

Separate execution/accounting, optimization/reproducibility, confirmation,
and coverage/learnability. For each finding provide a reproducible witness,
affected conclusion, minimal correction and scope of reruns. Distinguish
actual defects from the declared limits of this first policy library.

The complete archived results are ready for review. No financial training or
phase2 floating selection will be inferred from a positive synthetic result.

## Reproduction

Decompress CANDIDATES.jsonl.gz and SEED_*.json.gz into a separate working
directory alongside the other exported artifacts; do not modify the archive.
Use the source versions/digests in MANIFEST.json and a compatible environment.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/verify_dynamic_account_results.py \
  --run <decompressed-evidence-directory> \
  --engine-root <pinned-heuristic-strategy-checkout> \
  --export <new-verification-directory>
```

This replays the final accounts without repeating DEAP. Running the search
CLI against the authenticated receipts deterministically reconstructs the
search populations; a fresh run directory evaluates all candidates again.
