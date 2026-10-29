# Four/five future-event extension

Owner request, 2026-10-28: test whether K=4 or K=5 reveals useful marginal
improvements or a recognizable pattern beyond K=2/3. This extends 118 without
changing its score, costs, horizon bands, pivot search or missing-data policy.

## Frozen comparison

- TRAIN 2020-2023 and admitted origin grid unchanged.
- K=4 and K=5, nominal horizons 3/6/12/24 h.
- DEAP population 20, generations 30, seeds 7/19/43, 24 new searches.
- Same depth/deviation/backstep bounds and genetic operators as 119.
- New identities and fresh candidate ledgers. No previously scored genome is
  adopted into these searches. Serial execution, followed by scalar audits.
- Previous K=2/3 champions can be compared only after receipt authentication;
  their results remain historical immutable artifacts.

## Analysis declared before results

Report the fixed-denominator objective, coverage, champion genes and seeds.
Compute paired scores on origins shared by all four champions within each
horizon. Also evaluate K=2/3/4/5 holding the K=3 champion genes fixed and use
their shared origins: this separates additional exits from reoptimization.
Count the ordinal of the chosen exit and the distribution of gains. Explicitly
identify whether gains come from a few origins or are broadly distributed.

On identical origins and genes, ideal best score cannot decrease when the exit
set grows: this is a mathematical property to verify, not evidence of learned
predictability. Global fitness can decrease because more origins lack all K
future events before the observed session chunk ends. Never hide this effect.

## Verification

Regression tests cover both K=4 and K=5, unchanged prefixes when later events
and later prices are perturbed, monotonic best score on a fixed complete
origin, scalar best-exit enumeration, and rejection of unregistered K values.
Each completed search arm must pass the post-run receipt and scalar audit.
