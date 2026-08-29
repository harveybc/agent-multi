# Audit: fill truth and temporal-information suite v2

Date: 2026-08-28  
Agent-multi: `6227e44b`  
Gym-fx: `e4ca6e8`  
Verdict: **REVISE CLAIMS AND FILL JOIN BEFORE LONG REDISPATCH**

## Findings

### 1. Critical: “branches carry value” is not supported by absolute performance

The three highlighted volatility probes beat random and within-window-shuffled controls by the predeclared margins, but all have negative monitor R2:

- returns/momentum: -0.0880;
- trend/level: -0.2173;
- volatility/distribution: -0.0978.

Negative out-of-sample R2 means each probe is worse than predicting the monitor mean. The evidence supports `RELATIVE_SIGNAL_DETECTED_VS_CONTROLS`, not “carries predictive value” and not branch selection for trading. Oscillators and volume-flow likewise cannot be described as containing no value; only this encoder/pretraining/probe combination failed to demonstrate incremental value.

Required:

- add an absolute skill gate against constant/seasonal/last-value baselines;
- require positive out-of-sample skill before `USABLE_PREDICTIVE_VALUE`;
- relabel current results without changing raw data;
- keep the three branches as candidates for the window/bottleneck screen, not winners;
- keep oscillators and volume-flow as unresolved alternatives, not permanently discarded families.

### 2. High: real-data decision lacks uncertainty across encoder/data seeds

The synthetic gate uses four signal seeds and four random encoders. The real-data results are single monitor-point estimates. A difference such as 0.0305 versus a random mean has no paired interval and can be sensitive to initialization, ridge selection or regime.

Required: evaluate pretrained and matched random encoders across at least four seeds and rolling causal origins; report paired effects, dispersion and confidence intervals. The sealed year remains untouched.

### 3. High: the fused representation shows no incremental pretrained value

The fused volatility R2 is -0.0893 versus -0.0891 for random, a margin of -0.0002. Quantile and barrier probes are also at baseline. Thus the actual representation consumed by SAC has not demonstrated that the useful relative signals survive random fusion.

Required:

- treat random fusion as a suspected information bottleneck;
- test frozen/pretrained fusion, supervised probe-trained fusion and branch concatenation under matched capacity;
- require fused positive skill before claiming successful end-to-end feature extraction;
- preserve branch-level diagnostics separately.

### 4. High: fill reconciliation treats economically identical but lineage-distinct candidates as unambiguous

The join collapses candidates to a set of `(price, abs(size))`. If two distinct completed orders have the same price and size and both reconcile with the trade PnL, `len(distinct) == 1`; the code silently chooses the last candidate. This is still ambiguous order lineage. Consumption prevents reuse but does not prove the chosen order belongs to that trade.

Required:

- ambiguity is based on candidate order identities, not distinct economic tuples;
- more than one unconsumed reconciling order must refuse unless an explicit parent/trade lineage proves the join;
- persist order ref in a dedicated event field rather than embedding it only in reason text;
- test two same-price/same-size closes and reversal sequences.

### 5. Medium: partial-fill semantics remain undefined

`notify_order()` stores one aggregate completed execution and the join assumes one completed fill closes one trade. If partial completion is possible in the chosen broker model, define aggregation and lineage explicitly; otherwise assert and document that the simulator emits only terminal aggregate fills.

### 6. Medium: real-data probes still do not constitute trading evidence

Quantile, volatility and barrier probes are representation diagnostics. They do not measure realized return, drawdown, turnover or execution costs. The existing `REPRESENTATION_DIAGNOSTIC` classification is correct and must remain.

## Accepted

- The v2 experiment was predeclared before execution.
- Within-window permutation preserves sample-target identity.
- Phase scrambling now participates in the synthetic rule.
- Chronological fit/calibration/monitor roles and out-of-sample lag probes are material improvements.
- Synthetic results support temporal sensitivity for three branches and correctly avoid passing volume-flow at a saturated ceiling.
- Active CPU/CUDA reconciliation and single-population conservation are accepted for the paths exercised.

## Ruling

Proceed to the bounded window/bottleneck and fusion screen after correcting labels and the fill ambiguity. Do not launch the long paired SAC campaign yet. The next screen should attempt to turn relative branch signal into positive out-of-sample fused skill before economic training consumes substantial GPU time.

