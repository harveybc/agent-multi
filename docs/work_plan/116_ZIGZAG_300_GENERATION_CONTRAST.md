# ZigZag 40 versus 300 generations

Owner request 2026-10-28: extend each horizon to 300 generations, population
at least 20, and compare with the completed baseline. Keep population 30 so
only generation budget changes. Keep seeds 7/19/43, horizons 3/6/12/24, TRAIN,
calendar, objective, cost assumptions, ranges and operator probabilities fixed.
The old run used 40 generations, not 30; compare against the actual 40.

## Tests and acceptance declared before execution

- Authenticate baseline specification, results, seed receipts and candidate
  ledger records before copying reusable evaluations to a fresh run root.
- Original source/data pins must still match; add extension tool/design pins.
- Preserve all baseline files; require byte-identical file digests after run.
- Run all 300 generations for every seed, with the same initial populations.
- Check all 40 recorded generations in every new chain exactly match the old
  chain (genes, fitness and eligibility), not merely its final score.
- Final per-seed eligible fitness must not decrease under unchanged elitism.
- Run independent scalar path reconstruction and interval non-overlap audit.
- Emit the paired per-seed and best-per-horizon comparisons automatically.

The cache is deterministic evaluated content, not a trained warm start. Reusing
it does not skip generations or alter random draws. Old ledgers are rehashed,
their result hashes checked, and adopted records get the new campaign identity.
No baseline seed/final receipt is adopted as a completed 300-generation unit.
An interrupted new seed restarts deterministically with its verified cache.

The sequential command performs search, audit and comparison without further
approval or manual intervention. Completed receipts resume under the same pins.
Results remain TRAIN hindsight opportunity fitness (inverse hours), not Sharpe
or account return. Neutral deviation and lower-band scale behavior identified
in plan 115 are preserved for this controlled contrast, not silently repaired.
SL/TP/volume fitting and phase 2 floating selection do not start in this run.
