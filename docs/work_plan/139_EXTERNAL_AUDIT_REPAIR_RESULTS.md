# External audit disposition and repair results

Date: 2026-10-29. Predecessor: agent-multi `3644ece9`, native engine `e0a1c2b`.
This addendum governs interpretation of reports 128, 133 and 137. Their archives
remain byte-identical. It is not a new DEAP run or a learned-policy result.

## Verdict received

- [Satoshi: dynamic account](https://github.com/harveybc/agent-multi/blob/satoshi/audit-musashi-dynamic-account-20261029/docs/audits/SATOSHI_AUDIT_VERDICT_MUSASHI_DYNAMIC_ACCOUNT_2026_10_29.md):
  execution, optimization and confirmation ACCEPT_WITH_CHANGES; coverage REVISE.
- [Satoshi: close/hold](https://github.com/harveybc/agent-multi/blob/satoshi/audit-musashi-close-hold-20261029/docs/audits/SATOSHI_AUDIT_VERDICT_MUSASHI_CLOSE_HOLD_2026_10_29.md):
  execution and rule ACCEPT_WITH_CHANGES; signal sufficiency not evaluated.
- [Satoshi: shared risk](https://github.com/harveybc/agent-multi/blob/satoshi/audit-musashi-shared-risk-20261029/docs/audits/SATOSHI_AUDIT_VERDICT_MUSASHI_SHARED_RISK_2026_10_29.md):
  formulas and synthetic replay corroborated; private TRAIN reconstruction not
  independently executed. The old multiple-of-sigma consumer is not suitable for
  absolute levels. This lane already uses AbsoluteBarEnv.
- [Retsu](../audits/RETSU_AUDIT_DYNAMIC_ACCOUNT_PARETO_2026_10_29.md): numerical
  reproduction passes; P1 superiority and sufficient scenario coverage do not.
  His original uncommitted report was copied without edits from his audit worktree.

Satoshi independently recomputed all 17,018 candidates in reverse order and found
no mismatches. That is his execution, not a new execution claimed by this repair.

## Implemented repairs

| Finding | Change | Regression evidence |
| --- | --- | --- |
| DA09 / F07 | Exact native TP-through arithmetic, both directions | Exact-threshold unit and native witnesses |
| DA10 / F08 | Required per-sample terminal spread, distinct from bar-open spread | Deadline spread jump, long and short, native cash equality |
| DA11 / F10 | Required external calendar deadline | Self-consistent shortened report rejected |
| DA12 | Independent engine clock in direct, cached and entry consumers | Stale report fails even with verify enabled and a warm cache |
| DA13 | Geometry checked on every call; cache binds position, report and funding | Entry mutation rejected; report/cost mutations recompute; differing execution config rejected |
| DA24 / F09 / F15 | Native boundary coverage | Gap SL preempts requested CLOSE; SL wins ambiguous bars; TP equality and deadline quotes |
| DA23 / F03 | Enumerated current test evidence | 148 passed, zero skips; node IDs and file hashes in repair receipt |

The new API intentionally requires migration: `PathReport.terminal_spreads`,
`CloseController.decide(..., expected_deadline=...)`, and a report plus external
deadline for the synthetic cached adapter. `prefix_report` also requires a
terminal spread estimate or known schedule value. Callers must not pass a report's
own timestamp/deadline as the independent clock/calendar. All in-repo callers
were migrated. A missing new field fails rather than silently using old semantics.

## Measured verification

- Eight initial regression cases failed before the patch (including both exact
  TP boundary counterexamples); the completed suite passes **148/148**.
- Eight native fixtures cover four execution modes in both directions. Their
  expected exit reasons are declared in the tests, not inferred from outcomes.
- **288/288 repaired accounts exactly equal the archived accounts**, including
  orders, complete action traces, cash, mark-to-market drawdown, costs, native
  Sharpe, directional counts and cached/direct parity. Original optimizer genes,
  risk configuration, worlds and native source pins were unchanged.
- Ruff passes on modified Python files; `git diff --check` passes.
- [Machine-readable receipt](../audits/evidence/external_audit_repairs_20261029/REPAIR_VERIFICATION.json)
  includes predecessor hashes, new source hashes, verifier hash and all test IDs.

This establishes the repaired behavior and numerical preservation for the archived
accounts. It does not claim a new execution of the 8,720-rollout component corpus
or of the full 17,018-candidate search under patched code. Those predecessor runs
were independently repeated by Satoshi; the repair scope is explicit.

Reproduction (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
STRATEGY_ENGINE_ROOT=<pinned-native-checkout> python -m pytest \
  tests/unit/test_external_audit_repairs.py \
  tests/unit/test_dynamic_account_arbiter.py tests/unit/test_dynamic_account_panel.py \
  tests/unit/test_dynamic_account_queue.py tests/unit/test_close_hold_controller.py \
  tests/unit/test_close_hold_reports.py tests/unit/test_close_hold_trace_audit.py \
  tests/unit/test_early_close_metrics.py tests/unit/test_absolute_exit_corpus.py \
  tests/unit/test_horizon_risk_rules.py tests/unit/test_zigzag_risk_profiles.py \
  tests/unit/test_protection_reference.py tests/unit/test_mechanism_games.py \
  tests/unit/test_mechanism_m0_runner.py --junitxml=<new-scratch>/tests.xml -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/verify_external_audit_repairs.py \
  --engine-root <pinned-native-checkout> --test-report <new-scratch>/tests.xml \
  --output <new-scratch>/repair
```

Use the version of this verifier attached to this repair: it explicitly limits
allowed source changes against the predecessor. The old verifier properly rejects
patched code under old source pins; that safeguard was not weakened.

## Paired P1 versus Only3h

Recomputed from authenticated confirmation accounts, then matched to the repaired
native replays. Units: percentage points of account return.

| Pattern | Volatility | Mirror | P1 | Only3h | Difference |
| --- | --- | ---: | ---: | ---: | ---: |
| trend | increasing | +1 | 1.795063 | 1.929959 | -0.134896 |
| trend | increasing | -1 | 1.922322 | 2.076749 | -0.154427 |
| trend | decreasing | +1 | 1.460713 | 1.567621 | -0.106908 |
| trend | decreasing | -1 | 1.658131 | 1.736993 | -0.078862 |
| reversal | increasing | +1 | 2.188876 | 2.198818 | -0.009942 |
| reversal | increasing | -1 | 2.728376 | 2.647803 | +0.080573 |
| reversal | decreasing | +1 | 3.282740 | 3.218124 | +0.064615 |
| reversal | decreasing | -1 | 3.833674 | 3.930058 | -0.096385 |
| overlap | increasing | +1 | 3.561464 | 3.595474 | -0.034010 |
| overlap | increasing | -1 | 4.036785 | 4.147442 | -0.110657 |
| overlap | decreasing | +1 | 4.799086 | 4.076322 | +0.722764 |
| overlap | decreasing | -1 | 4.871123 | 4.410603 | +0.460521 |

Mean +0.050199 pp, positive in 4/12 cells. Excluding the overlap/decreasing
mirrored pair: -0.058090 pp. This exclusion is a sensitivity diagnostic, not a
new selection criterion or a replacement evaluation. P1 is not promoted over
Only3h. P3 is also dominated by equal_guarded, not only by Greedy.

## Scientific corrections accepted

1. **DA15/16, Retsu 1/2/4.** Three independent price draws per panel, not twelve.
   Mirrors are paired transformations. Increasing volatility had a 3-pip spread;
   decreasing volatility had 0.5 pip. Noise was shared across worlds and its bias
   sign followed the mirror. These are confounds to remove in a new version.
2. **Early-close attribution.** Satoshi's counterfactual replay finds that 14 of
   P1's 20 confirmation early closes preempt a native SL. Therefore the absence
   of realized stops does not mean those opportunities were absent. The future-aware
   reports anticipate them. Across all 288 accounts there is one SL in the
   equal_guarded calibration control; the named P1-P5/Greedy/Only3/Only6 subset
   has none. This reconciles the two audit scopes.
3. **DA21/22.** Satoshi's no-early-close counterfactual changes P1 confirmation
   return/DD from 3.01/0.54 to 0.91/1.88 percent. This is mainly a demonstration
   of foresight-assisted closing, not proof of a robust allocation mechanism.
   The original adverse-bias test still preserved future path shape.
4. **DA04/18/19, Retsu 3.** Representatives were spaced by policy rank, not unique
   objective rank; equal-vector ties used a hash. P2/P3 perform the same actions
   with different exposure. Entry and close share one uncertainty-penalty gene.
   No historical representative or gene is changed after confirmation. Next
   versions must separate exposure from action diversity and declare tie rules.
5. **DA07.** Native Sharpe uses six calendar-day returns per world and sqrt(365).
   It is reproduced, not validated as a reliable annual performance estimate.
6. **DA20, Retsu 5.** Disabling P3's 12h-buy raises return 2.192742 to 2.239666
   and worsens DD 0.255156 to 0.296221. Scalar gain 0.005858531 is not Pareto
   dominance. Do not silently delete that specialist. Global allocation regret
   versus a joint scheduling oracle remains a new diagnostic, not a solved claim.
7. **Shared risk D03/D06/B03.** Modeled SL loss is not a bound on gap loss.
   Invalid sigma raises a typed error; an online consumer must record abstention,
   not coerce missing volatility to zero. The synthetic panel uses fixed finite
   sigma; it has no live sigma estimator. Old single-file pins do not certify
   all ZigZag code; private TRAIN evidence remains independently unverified.

## Integration with the guided lane

Satoshi's progress note adopts leverage 50, no daily cap, 12h entry wait and 12h
pre-close flattening. No new owner decision is required. His absolute-level
adapter, cost-reserved sizing and native parity tests address the old sigma-cap
consumer mismatch. This does not merge his learned ZigZag heads into the SAC lane.

Use his correlated-error ladder and fixed-delay control as designs to test, not
as already transferable calibrations. His AR(1) coefficient measured at H6 is not
a universal coefficient. His close-favoring tie band differs from our HOLD ties;
keep that an explicit future variant, not an undocumented repair. A full joint
path report cannot be replaced by marginal excursion curves without an adapter
and first-passage consistency tests. No general margin-closeout coverage is claimed
for arbitrary positions outside the present risk contract.

## Next action

Implement and verify the new, independently crossed scenario and report-quality
panel specified in [plan 138](138_EXTERNAL_AUDIT_REPAIR_PLAN.md), first with frozen
controls and forced-position early-close witnesses. Only then rerun optimization
under a new identity. No new search is running at this repair's close. Neither
financial training nor floating feature selection has resumed. The reason is the
specific remaining coverage/producer work above, not a conclusion that the inputs
cannot support trading.
