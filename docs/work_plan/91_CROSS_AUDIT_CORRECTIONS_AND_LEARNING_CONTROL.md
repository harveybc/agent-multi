# Cross-audit corrections and learning qualification

2026-10-26. Independent verdict integrated from Satoshi commit2fd4eeda without
editing it. Contract and reproduction accepted; decision use of feature-search
and DEAP outcomes rejected. This document does not reverse that disposition.

## Executed corrections

The isolated CPU tool `tools/output_audit_corrections.py` authenticated the
three original evidence collections and a fixed snapshot of130 completed model
receipts. It did not change original GRID/COMPARISON files, weights, services,
data or TEST. New derived files have separate identities and closure hashes.

| Finding | Correction and measured evidence |
|---|---|
| C1 | Derived GRID:81 cells with TP/SL distances, RR, cost/trade in pips, binary break-even reference, realized wins, exit reasons and financing separated from other costs |
| C2/M4 | Replayed LONG_HOLD, SHORT_HOLD and five seeded COIN_HOLD rules on TRAIN and INNER_A in each cost mode; compared trades/equity against every authenticated receipt, not just seed7 |
| C3 | Corrected comparison includes both controls; independently re-simulated SL endpoints; no SL or activity identification claim |
| M1 | Every INNER_A value is a reused selection-block measurement, not independent confirmation; no 2024 or TEST evaluation added |
| M2 | Re-evaluated strict executed-close eligibility for all130 receipts, both periods; zero changes relative to the old requested-command rule plus native validity/activity in this snapshot |
| M3/O13 | Retracted claim that the existing witness changed stops: it changed the future, held stops fixed and bypassed the policy |
| m1/m2/m3/m5/m6 | Corrected seed and margin text, TP slippage, margin-close costs, replay scope and calendar-day Sharpe convention |

Configured TCN receipts:124. Exactly LONG_HOLD trade records: **107/124 TRAIN,
114/124 INNER_A**. Satoshi's55/59 and58/59 describe his earlier snapshot, not a
discrepancy. Even though strict re-gating changes no decisions, equality with
blind policies invalidates interpreting this run as evidence of feature utility.
No pruning ranking from it is admitted. This is not evidence against the data.

The midpoint TRAIN control is asymmetric (TP lower bound.25 vs SL.5). For seed7:
TP68.1866 pips, SL74.5135, ratio.91509, cost3.05734 pips/trade, binary break-even
54.3593%, realized wins50.1006%; net -5869.59 USD, financing2135.76 USD and
other costs2791.06 USD. There are168 non-barrier exits among497 trades.

**Qualification of C1:** `(meanSL + meanCost)/(meanTP + meanSL)` assumes a binary
TP/SL payoff model. It is useful as a diagnostic reference, but variable sizes,
gaps, early/forced exits and duration-dependent financing violate that model.
It does not prove that the strategy must lose for every policy or signal.
The implemented ledger, not that approximation, establishes the observed loss.

DEAP ideal calibration: zero SL touches in7222 tapes. Four endpoint probes
(SL-long/short at0/1) around F2 seed7 reproduce metrics, commands and executed
trade fields exactly, excluding only the deliberately changed `stop_loss`
metadata. Thus SL is not calibrated. S~15, R~.13, A=.85417; the terms are not
balanced merely because coefficients equal1. In confirmation the controls
beat every finalist in Sharpe, and midpoint also beats them in profit.

## Reproduction and evidence

CPU, one thread;35 focused tests passed. Baseline generation:28 traversals
(2 cost modes x2 periods x7 blind rules); sensitivity:24 synthetic traversals.
Source evidence was hash-verified before use;130 source RESULT digests were
checked again at close. Public derived evidence is in
`docs/audits/evidence/output_cross_audit_corrections_20261026/`.

Producer SHA256: `b99142a615f63836243fd87a091822c5662a71707c4bb69c2912a93387edf817`.
Private v2 CLOSE SHA256: `3148155083e84cb44aab035b7f91672c77c869abd0fab629a0526232b40a3e76`.
Test receipt SHA256: `ac9080743f4db928693d39bc588c7ccaaa5b396bee9a020bdb8f9e33d4360a6d`.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.output_audit_corrections \
  --root "$CAMPAIGN_MIRROR" --study "$CONTRACT_STUDY" --deap "$DEAP_OUTPUT" \
  --engine-root "$PINNED_ENGINE" --data-root "$ADMITTED_DATA" \
  --output "$NEW_ISOLATED_OUTPUT"
```

The first sensitivity attempt compared full trade objects, including changed
SL levels, so it incorrectly rendered an execution difference. That evidence
is retained separately; a regression now distinguishes changed protection
metadata from changed fills/costs/reward, and the v2 run above supersedes it.
Neither attempt is a new fit or financial performance estimate.

## C4: next mandatory experiment

**Not executed.** Admission training of64 constant-observation steps and
oracle-to-engine tests do not demonstrate learning by the actual GroupedTCN.
The diagnostic must use the actual `tools.rl_subset_bridge.build_model`, native
market/rollout/optimizer, frozen budget102400, seeds7/19/43, grouped TCN,
categorical actions, configured costs and unchanged business restrictions.

1. Prepare an isolated recipe derived from the current one. Replace one
   existing market channel with an explicitly future-derived diagnostic
   signal; keep shape, architecture and every other channel unchanged. No
   account-state vector, moving z-score, clipping or expert demonstrations.
2. Plant the signed four-bar price movement after the decision, using the
   exact native bar/index mapping. Fit that channel's scale on TRAIN only.
   Admit only origins whose full lookahead is inside the same split. Freeze
   this common support for all diagnostic arms, including their replay windows.
3. Compare planted signal, permuted planted signal, and unmodified input on
   the same support, three fresh seeds each. Fix permutation and stopping
   rules before measuring. Do not load original campaign weights.
4. Evaluate TRAIN and INNER_A, labeled selection/development throughout.
   Require actual LONG/SHORT openings, AGENT_CLOSE both directions, NOP/HOLD,
   activity and positive INNER_A Sharpe in the planted arm; report each seed,
   blind-policy equality, costs, profit, DD and wins. A failure must remain
   a failed learning qualification, not be relabeled a lack of market signal.
5. Keep this deliberate-lookahead recipe impossible to admit as financial
   data. No 2024/TEST access; no diagnostic model or recipe reused by SBFS.
6. Only after reviewing this result fix the learner if needed, qualify the
   executed-close gate in its new evaluator, integrate the83+31 reversible
   prefilter, and create a new financial selection identity. No reuse of the
   rejected ranking as a feature decision.

The positive control has no measured runtime yet. No claim of complete audit
closure or permission to restart the heavy selection is made by this report.
Other open review items are the actual position-observability counterexample
and the low-equity reward termination path (m8); neither is silently changed.
