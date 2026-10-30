# Audit request: measured continuation gap

For Satoshi. Completed evidence, not a proposal-only or production release.

Review [results170](../work_plan/170_PRAGMATIC_PLAN_EXECUTION_GAP_RESULTS.md)
and the completed[attribution/resolution168](../work_plan/168_PRAGMATIC_CLOSE_AND_RESOLUTION_RESULTS.md).
Pinned diagnostic code `ab1c2778`; native engine `e0a1c2b`.
Original controller contrast `6c9f597e`; frozen one-tick intervention `de91e051`.
Diagnostic identity `b2e26ee997822fb67ca7ccddb9a8e0c71fe27653f197f87b917237465813933d`.

72 baseline accounts exactly reproduced;305 ideal-report closes;610 native
committed-plan replays; zero valuation discrepancy;150 reversals of the
original CLOSE/HOLD utility ranking under hourly policy continuation.
265 tests pass and completed resume authenticates all results.

Please attack:

1. Does the independent committed driver actually implement the same plan
   semantics, including entry timing, barriers, costs, second-order sizing,
   and forced flat? Check both direction fixtures and real recorded witnesses.
2. Is the one-tick HOLD branch the same frozen controller with preserved
   memory, rather than a conveniently chosen hindsight exit or another policy?
3. Do the two continuation shifts explain the whole utility difference?
   The reported identity residual is at most1.78e-15; check individual cases.
4. Are any claims stronger than the evidence? Ideal valuation parity is not
   general simulator certification; committed-plan optimality is not optimality
   of the actual policy;150 utility reversals are not148 profit-loss cases.
5. Does the proposed non-recursive frozen-reference rollout make the
   continuation assumption explicit without introducing hidden future access?
   Do not silently substitute perfect future report updates in a causal test.

Return separate verdicts on execution parity, interpretation of continuation
error and the correction direction. The current policy is not promoted and
no further coefficient optimization or financial training is authorized by
these results alone. Continue keeping the guided and pure-RL lines distinct.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/diagnose_pragmatic_plan_gap.py \
  --source <completed-pragmatic-contrast> \
  --attribution <completed-close-attribution> \
  --engine-root <pinned-native-engine> --output <fresh-audit-root> --workers 4
```

The result links include the manifests, hashes and a complete native witness.
All remaining per-world receipts are available in the shared sealed archive.
