# ZigZag parameter forwarding, range and effect audit

Owner concern, 2026-10-28: a parameter adapter, overly coarse steps, insufficient
span or local optimum could explain unchanged DEAP champions. Audit completed
without changing the pinned indicator or the running optimization objective.

## Actual exploration

| Parameter | Observed values per horizon | Observed range | Smallest observed step |
|---|---:|---:|---:|
| Depth | 95 | 2..96 | 1 bar |
| Deviation | 1,010 | 1..119.9 | 0.1 pip |
| Backstep | 24 | 1..24 | 1 bar |

13,232 distinct decoded genomes were evaluated across the four horizons.
Every ledger row's genes match its result parameters and its authenticated
payload hash. Direct interception of 6,688 calls to the real engine confirmed
that Params receives all three requested values without substitution.
Changing depth or backstep by one changes actual pivot hashes in all four
finalist probes. Therefore neither integer gene is frozen by the adapter.
Some neighbors leave the registered horizon band and are correctly ineligible;
this is a model-domain constraint, not an adapter clamp.

## Deviation is read, but ineffective in the examined domain

For 1,276 within-horizon groups having identical depth/backstep and different
deviation values, none changed fitness, pivot count, complete support or median
pivot spacing. Real-engine probes at each finalist additionally compared full
candidate-map and pivot-series hashes, not only aggregate scores.

Scale probe in pips: 0, 1e-9, 1e-6, 0.001, 0.01, 0.1, 1, 5, 120, 1200, 1e6.
Every value produces identical candidate maps and pivots within each of the
four fixed depth/backstep cases on all TRAIN chunks. Internal threshold is
`deviation_pips * 0.0001`; it is not a percentage.

## Code-level explanation

Reviewed `heuristic-strategy/app/short_horizon/zigzag.py`:

Source SHA-256: `c06592f6478f3887213e985a80de5823dd953d3887ce25bec0272197860affda`.

- Lines 60-62 calculate the deviation threshold and trailing-window extrema.
- Lines 66-77 compare the current low with the window minimum, but emit a low
  candidate only when the current low equals that minimum. At such an accepted
  extreme the difference is zero, so a nonnegative deviation does not reject it.
- Lines 78-89 apply the symmetric condition to highs.
- The only remaining possible influence is cleanup of previous candidate maps.
  That path also left the complete maps identical in the real-data probes.
- Lines 93-134 select alternating pivots from those maps. This phase receives
  no deviation parameter and contains no explicit minimum peak-to-valley
  reversal test.

As a bounded additional check, all 729 length-six paths over a three-price
alphabet were tested with 15 legal depth/backstep pairs: 10,935 cases comparing
deviation 0 versus 1e6. Candidate maps never differed. A separate attempt to
find an out-of-contract backstep counterexample in this small family also
found none; that is not used as a positive control or claimed as a proof.

The source module's introduction says an opposite extreme follows a
deviation-sized reversal. That description is not supported as a controlling
minimum-reversal rule by the implementation examined here. This semantic
documentation discrepancy must not be repeated in reports.

Conclusion: a wider deviation range does not fix the measured inactivity;
this is not a missing optimizer-to-indicator parameter transfer. The evidence
is scoped to this pinned implementation, tested inputs and parameter domain,
not every indicator named ZigZag.

## Consequences

Keep the completed optimization as evidence for this implementation. The exact
2,004-pair integer grid at deviation=5 confirms all four winning scores, so
there is no better point on that finite two-gene grid left unexplored.
For subsequent work with this same definition, fixing deviation=5 as a declared
convention avoids spending search budget on observed-equivalent genotypes.
It must not be described as a learned optimal deviation.

If amplitude-threshold control is required, a genuinely active reversal rule
must be a separately specified and tested indicator variant, including exact
units and high/low same-bar ordering. No such semantic change was made here.

[Structured probes, ledger statistics and grid summary](../audits/evidence/zigzag_k3_p20_g300_20261028/RESULTS.json).
The external [MetaQuotes catalog entry](https://www.mql5.com/en/code/56) was
consulted as provenance context. Direct source download was unavailable (HTTP
403 / unsupported content type), so this audit does not claim byte-for-byte
parity with an independently downloaded official MQL5 source version.
