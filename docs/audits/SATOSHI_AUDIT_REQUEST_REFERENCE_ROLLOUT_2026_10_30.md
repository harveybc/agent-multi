# Audit Request: Frozen-Reference Rollout

Review the completed experiment, not merely its proposal.
Code `8c7d0f22`, native engine `e0a1c2b`, full identity
`5930311292e4c6ef67eb2e9484bdaf92642ba85dfbe63657ee1619198f080af4`.

Entry points:
- [Plan and information contract](../work_plan/171_FROZEN_REFERENCE_ROLLOUT_PLAN.md).
- [Results and limitations](../work_plan/172_FROZEN_REFERENCE_ROLLOUT_RESULTS.md).
- [Complete table](evidence/reference_rollout_full_20261030/RESULTS.md).
- [Manifest](evidence/reference_rollout_full_20261030/MANIFEST.json).
- [Completion and raw receipt index](evidence/reference_rollout_full_20261030/COMPLETE.json).
- [285-test receipt](evidence/reference_rollout_full_20261030/TESTS.json).

All 1,512 accounts and 235 actual-policy close interventions completed;
zero service restarts, completed resume authenticated. The full native
account/case archive is available through the shared execution handoff;
COMPLETE lists exact names/hashes. Do not infer a raw replay from the summary.

## Attack First

1. Future-information boundary: `ContinuationWorld` carries suffixes of the
   same whole current ensemble. It must never fetch an actual future report
   or reveal the realized sample to the reference policy. Original
   observed-through time is retained; carried issuance is not a new model
   inference. Test both future-price perturbation and callback interception.
2. Native/account/controller state: references observe a real tick once;
   branches get independent copies. Compare snapshot close suppression with
   cold full replay; inspect persistence, native protection and financing.
3. Common action/reference selection: no sample-wise oracle maximization,
   recursive selector or two-order continuation cap. The native account
   `config` field is the shared runner configuration; the actual new policy
   specification is the arm/reference configuration in MANIFEST and its key
   in each unit receipt. Verify the complete binding.
4. Ideal adaptive failures: 85/235 closes hurt final profit and 86 hurt
   return-minus-DD. Review actual continuation, not just local order profit.
   Q of the reference is not Q of the adaptive selector. No claim of solving
   all continuation inconsistencies or global Pareto optimality is made.
5. Noisy turnover: Noise 1 joint has 944 orders, 876 early closes and 42.06%
   wins, versus simple control 301 orders, 200 early closes and 67.44% wins.
   Look for destructive re-entry, near-minimum holds and unstable report
   ranking before attributing a cause. Both directions are exercised.
6. Unchanged controls, paired worlds, costs, SL/TP, q/margin, weekly gates,
   horizon mix, activity, noise and causal losses. Verify fresh seed banks
   and that no coefficients were selected from these new results.

Give separate verdicts for execution/reproducibility, mechanism robustness
and readiness for a learned-output adapter. Suggest minimal discriminating
tests or corrections. Do not treat ideal synthetic profit as learned RL
performance, calibrated uncertainty or financial validation.
