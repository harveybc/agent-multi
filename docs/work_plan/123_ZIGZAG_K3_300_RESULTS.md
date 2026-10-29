# Three pivots: 30 versus 300 generations

Completed 2026-10-28. Selected K=3 under plan 122. Population 20, seeds
7/19/43, TRAIN 2020-2023 and all other scientific settings unchanged.
12 fresh searches completed 300 generations each in 839.041 seconds, excluding
post-run audits. No historical genome cache was adopted. All 360 historical
generation records (30 x 12) match the corresponding prefixes exactly.
Baseline file hashes are unchanged; no active queue remains.

[Results, comparison, scalar audit, parameter diagnostics and exact-grid control](../audits/evidence/zigzag_k3_p20_g300_20261028/RESULTS.json)

| Horizon | Best fitness at 30 | Best fitness at 300 | Change | 300-generation genes |
|---|---:|---:|---:|---|
| 3 h | 0.271176912 | 0.271176912 | 0% | 3 / 5.0 / 1 |
| 6 h | 0.237675675 | 0.237675675 | 0% | 4 / 5.0 / 2 |
| 12 h | 0.151035613 | 0.152872027 | +1.215882% | 7 / 25.0 / 6 |
| 24 h | 0.098117667 | 0.098117667 | 0% | 12 / 25.0 / 11 |

Genes: depth / deviation in pips / backstep. Deviation is not an identified
effective optimum: see the [parameter audit](124_ZIGZAG_PARAMETER_EFFECT_AUDIT.md).
These are oracle opportunity ratios, not account returns or Sharpe.

The new 12h champion was found in seed 7 at generation 62. The older champion
was depth 8 / deviation 21.1 / backstep 5. On their 15,695 common complete
origins, mean score improves from 0.185260299 to 0.187847783 (+1.396675%).
Coverage changes slightly from 81.6590% to 81.4478%, so the gain is not caused
by higher coverage. Other horizons have identical paired scores and coverage.

Five individual seed results improve: 6h/43 +7.6465%, 12h/7 +9.7757%,
12h/19 +3.4081%, 24h/19 +26.1667%, 24h/43 +15.0975%. Only the 12h improvement
raises the best-per-horizon score. Full per-seed histories are authenticated in
the run root; the public evidence retains the comparison and identities.

## Exact finite-grid control

The owner's adapter/range concern prompted a separate exhaustive evaluation
of all 2,004 legal depth/backstep pairs: depth 2..96 and backstep
1..min(24,depth-1), with deviation fixed at 5 pips and K=3. The score is
horizon-independent; horizon eligibility is evaluated separately, then each
selected result is rechecked with its own horizon evaluator.

All four grid winners equal the DEAP winners' fitness exactly. Runtime:
151.671 seconds. Thus the reported best scores are also maxima on this finite
two-parameter grid at deviation=5, under the existing horizon-band contract.
This does not claim optimality outside these bounds or for arbitrary deviation
semantics, other ZigZag implementations, costs or objectives.

## Verification

29 focused tests pass, including decoded-value forwarding and preservation of
all integer depth/backstep steps. Post-run audit verified 3,600 generation
records, 6,144 candidate paths and 2,048 selected scalar paths; maximum selected
reconstruction error is zero. Inputs, engine and software identities passed.
The parameter audit observed 13,232 distinct genomes and independently checked
6,688 real-engine calls. No neural training or validation/test read occurred.

Retain K=3 and the new 12h champion as measured design references. Do not call
the differing deviation values tuned optima or silently change the indicator
to introduce a new reversal rule. Strategy qualification remains a separate
step before floating feature selection.
