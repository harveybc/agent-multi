# Same-Market Compatibility Results

## Verified Completion

Design252 finished2026-11-10T08:48:39.945931Z (03:48 Bogota), in64.74 seconds.
Runtime `0e3502c1`;550 files authenticated. Collection identity:
`1e5fe1e467cb6442aacf343b64fc36176caeb4bee1ea1a5ff1fef9edca49aa26`.
Worker and collector completed successfully. Zero optimizer steps, source
weights unchanged, no input/state feedback or strategy change, no promotion.

Evidence: `/var/tmp/sac70-state-compat-20261110`.
Public [bank table](generalization/state_compatibility_20261110/COMPATIBILITY.csv),
[plot](generalization/state_compatibility_20261110/COMPATIBILITY.png),
[summary](generalization/state_compatibility_20261110/RESULT.json).
SOURCE_DONE authenticates the complete source collection, not just this subset.

## Coverage And Controls

-512 previously selected flat-entry origins,16 quarterly TRAIN episodes.
-1536 requested history banks at1/4/12 decision steps; one quarter-terminal
  exclusion leaves1535 banks over1507 distinct market rows.
-844 comparable banks:747 complete flat/long/short triplets and97 pairs.
-691 banks have only the flat history after native position closures. They
  are recorded, not included in the multi-state conflict denominator.
-7353 native suffix simulations, including48 identical-state control suffixes.
-16 negative controls, maximum return/cash discrepancy0.0. Every reference
  account trace hash and terminal cash matched the archived native reference.
-All seven model checkpoints passed exact repeated-input equality and exact
  full-TRAIN actor preservation. Native feature/mask/time equality passed for
  every compared history. No position or fill was assigned by hand.

## Value Conflict Versus Action Conflict

All844 comparable banks have nonzero action-gap fitting floors: a single Q
vector cannot reproduce all their state-conditional target vectors exactly,
even after removing state-common value offsets.

However, only190/844 banks (22.51%) have no shared optimal action. The other654
allow a common optimal action despite incompatible exact Q targets. Exact-value
fitting incompatibility must not be equated with inevitable action failure.

| Role | Steps after opening | Comparable banks | No common optimum | Fraction |
|---|---:|---:|---:|---:|
|Fit2020-2022|1|351|78|22.22%|
|Fit2020-2022|4|228|60|26.32%|
|Fit2020-2022|12|63|12|19.05%|
|Monitor2023|1|117|20|17.09%|
|Monitor2023|4|67|17|25.37%|
|Monitor2023|12|18|3|16.67%|

These are correlated constructed histories, not natural position frequencies
or independent samples. Later-step figures are conditional on survival and
cannot be compared as an unbiased effect of position age. Steps are decisions,
not necessarily elapsed clock hours.

### Concrete Witness

The bank at market row2495, one decision after its origin, has the largest
best-common-action mean regret in this panel. Shown explicitly as an illustrative
extreme, not a representative average. Discounted future rewards:

| Native account state | NOP | BUY | SELL | Best action |
|---|---:|---:|---:|---|
|Flat|0.005151|-0.434354|-0.489735|NOP: stay flat|
|Long|-0.489466|-0.489466|-0.006889|SELL: close long|
|Short|-0.394709|-0.004000|-0.394709|BUY: close short|

The observations are byte-identical. A deterministic shared three-signal
output cannot produce all three choices at that instant. The native controller
already interprets opposite signals as closes, but this alone does not resolve
the conflict. The best common choice in this example is NOP, with0.291095 mean
regret under equal weighting of the three counterfactual states.

## The Conflict Is Not The Whole Error

For each bank, minimize mean regret over common actions. Averaging those minima
gives0.023125 on fit banks and0.016972 on monitor banks. These are finite-panel,
equal-state-weight, fixed-continuation one-step bounds, not live profit bounds.

| Frozen decision source | Fit mean regret | Monitor mean regret |
|---|---:|---:|
|Best common-action diagnostic lower bound|0.023125|0.016972|
|Unchanged actor|0.215739|0.197915|
|Initial calibrated critic|0.201775|0.217526|
|Heads-only r19/r43|0.211375 /0.211632|0.202909 /0.204527|
|Last-core r19/r43|0.212555 /0.211655|0.202952 /0.201325|
|Full-representation r19/r43|0.200254 /0.197497|0.187892 /0.187568|

The actor's excess over the common-choice bound remains0.192614 on fit and
0.180943 on monitor. Thus hidden account state is a demonstrated limitation of
this exact value/choice interface, not a sufficient explanation of all error.
The ranking of diagnostic critics differs from249 because these are alternate
native histories with equal per-bank state weights, not the earlier natural
reference-state panel. This does not overwrite249 or justify model promotion.

## Next Controlled Design

Preserve the champion. Propose role-specific value outputs for flat entry,
long management and short management, with the native controller selecting
the applicable output outside the learned market encoder. This can keep the
same market feature inputs; it is nevertheless an explicit interface change
requiring its own contract, tests and paired contrast, not something silently
enabled by this audit. Keep account-state features out of the model input.

Role routing alone may not remove value conflicts between two positions of
the same direction with different entry, SL/TP, age or size. Do not promise it
as a complete repair before testing. Also quantify the large residual error
above the common-choice bound; do not attribute it all to the interface.

No2024 trading evaluation or TEST scoring occurred. The2023 monitor role was
used in source pretraining. Future trajectories here are retrospective labels
only. Alternative histories change account state while market paths are fixed;
the simulator does not model market impact. No new trading profitability result
is claimed.
