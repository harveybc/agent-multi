# G noise-stress controls: declared before execution (2026-10-29, auditor)

Support: every 9th position of the frozen 1,744 specs (194 positions, deterministic, includes all modes and 2 gap cases).
Controller configs: frozen immediate (z0, m0, p1, no urgency) and guarded (z1, m.02, p2, urgent .25) from plan 132.
No parameter is tuned; nothing is selected after seeing results. Native engine e0a1c2b executes every decision.

Report families (all built from the harness future; test plane only, never observations):
- IDEAL, NOISY: the harness producers unchanged (paired baseline on the same 194 positions).
- BIAS_FAV_IDEAL(k), BIAS_FAV_NOISY(k), k in {0.5, 1}: every sample's O/H/L/C at step h (1-based) shifted by
  +direction * k * sigma_h, sigma_h = 1e-4 * sqrt(h) (the declared marginal sd of the noisy model's cumulative
  innovation). The bias is in the position's favour (report too optimistic).
- BIAS_ADV_NOISY(k), k in {0.5, 1}: same shift against the position (extra control, report too pessimistic).
- MISSING_DROP: NOISY report minus every sample whose own native-rule first passage ends at SL; weights renormalized.
  If every sample ends at SL the reporter has nothing left and emits no report: the controller HOLDs (counted).
- MISSING_UNDERWEIGHT: NOISY report with SL-ending samples' weights multiplied by 0.25, renormalized.
Metrics per family x config: early closes, destructive closes, missed optimal exits (missed_opportunities),
aggregate captured improvement (sum improvement / sum opportunity), mean regret/R; tolerance 1e-7 USD (plan 130).
Interpretation limits: synthetic, finite, retrospective stress; not evidence of learned-report robustness.
2026-10-29T05:57:57Z

## Addendum G2 (declared after G1 results, before G2 execution)
G1 observation: the declared-noise-scale bias (1e-4*sqrt(h), ~1-4 pips) is small against stop distances of
~9-17 pips, and MISSING_UNDERWEIGHT is degenerate whenever every sample ends at SL (renormalization restores
equal weights). G2 adds position-scaled bias, same support, same configs, nothing tuned:
- BIAS_SL_{FAV,ADV}_{IDEAL,NOISY}_k{0.25,0.5}: shift at step h = +/- direction * k * |entry - SL| * min(1, h/6).
2026-10-29T06:00:34Z
