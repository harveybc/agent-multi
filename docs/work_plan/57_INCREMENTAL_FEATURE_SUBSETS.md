# Incremental Feature Subsets For SAC

Owner request2026-10-21: planned follow-up, not deployed. Keep the active fine
window sweep and its15-column identity unchanged. After it closes, before the
extractor/component refinements, test whether a compact subset generalizes
better. Periodic adaptation stays LAST; positive static profit is not a gate.

This is forward selection for the RL consumer, not a repeat of data admission.
The previous selection supplies candidates, not proof that the full set is best.

## Provisional Trial Priority

This is a hypothesis-based experiment order, NOT measured feature importance.
Attach existing selection metrics/period/support before dispatch; missing
importance measurements remain unmeasured. Fifteen current inputs are listed:

| Priority | Current policy features | Rationale / suspected overlap |
| --- | --- | --- |
| 1 | RAW_CLOSE_LOG_RETURN | Recent movement |
| 2 | RAW_LOG_RANGE | Intrabar amplitude |
| 3 | RAW_ELAPSED_HOURS | Actual elapsed time, including closures |
| 4 | hod_sin, hod_cos | Complete hourly cyclical group |
| 5 | dow_cos | Weekly seasonality; current subset lacks dow_sin |
| 6 | ATR | Local movement scale |
| 7 | rolling_std_24 | Contextual volatility; may overlap ATR/range |
| 8 | ADX | Trend strength |
| 9 | MACD_Histogram | Momentum at another scale |
| 10 | DI_minus | Direction; include ADX interaction rescue |
| 11 | RSI | Oscillation/momentum |
| 12 | Stochastic_D | Relative range position |
| 13 | WilliamsR | Possible overlap with Stochastic_D |
| 14 | HIGH | Test absolute-level contribution against transformed movement |

## Bounded Experiment

Start with return/range/elapsed. Add remaining features in this order, except
cyclical pairs enter together. Preserve canonical column order within subsets.
Keep a group only when it improves the predeclared validation/risk/gap rule;
otherwise defer it without asserting universal uselessness. Predeclare up to4
interaction-rescue trials (e.g. ADX+DI_minus, ATR+std). This greedy path is
order-dependent, not a global optimum. Confirm finalist versus full15 on shared
seeds7/17/29. Provisional budget up to12 forward fits,4 rescues and4 additional
confirmation fits; freeze the actual finite queue/configs before dispatch.

Maintain admitted data, full TRAIN2020-2023/VALIDATION2024, costs, business and
100k fixed budget/no costly ES/no refit. Compare the same time support and all
native metrics. Retain subset manifests, seeds, parameters, weights, config,
transform/data/code hashes, deltas and exclusion reasons through existing
warehouse/readback/independent copies. No TEST. No preferred validation year.
Changing the input count changes branch/head parameter counts: record this;
attribute information versus capacity only with a separate controlled contrast.
Check window/subset/encoder interactions on a bounded finalist shortlist.

## Transform And Observation Boundaries

Inventory both features and state: equivalent information may survive a feature
ablation through a state field. Mandatory execution OHLC/calendar/ATR, position,
risk and account state are not removed or deleted from the lake. New subsets
need explicit configuration/lineage and actual-shape reload tests; existing
v12 accepts only the original15 and must not be weakened in a running campaign.

Current calendar inputs are hod_sin/hod_cos/dow_cos. Plan a separate dow_cos vs
sin/cos weekday pair contrast, weekday0..6 with period7. Clock, DST, timestamp
meaning and availability come from the admitted root. Preserve observed bars,
including current completed bar; no synthetic closure bars. Hour pair period24.
Do not silently add dow_sin to the full15 reference. Compare unscaled cyclic
pairs in[-1,1] against existing causal rolling scaling in a separate treatment.

Current contract declares rolling_zscore256, no binary feature columns. Record
effective transforms per field. Fit learned transforms on TRAIN only; rolling
updates use only information available by t. No global TRAIN+VALIDATION fit,
future-support wavelets/centered filters, interpolation or synthetic padding.
Subset selection itself uses VALIDATION as declared development. Tests include
future perturbation, current-bar inclusion, exact subsets, cyclic groups,
unchanged business support, mandatory state, restart and full outcome custody.

Doctoral research hypothesis retained for later: different input families may
benefit from different preprocessing and extractors. Do not assume a universal
global transform/extractor, nor implement heterogeneous branches now. Separate
selection, transforms and extraction first; test their interactions afterwards.

Detailed Spanish design and importance rationale are in predictor master-v3
`program_v3/RL_INCREMENTAL_FEATURE_SUBSETS_2026_10_21.md`. Driver/configs for
this later study are NOT implemented or queued by the present documentation.
