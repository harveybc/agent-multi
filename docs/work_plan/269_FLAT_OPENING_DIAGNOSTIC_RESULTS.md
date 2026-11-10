# 269. Flat-opening state and gradient results

Completed2026-11-10T20:44:02Z; design268.158 TRAIN timestamps:79 changed
anchors and79 nearest unchanged controls,2844 full native continuations.
All three shards and automatic collection completed;328 sealed files verified
before the additional gradient decomposition.21 focused tests pass, including
native account replay, autograd/finite-difference agreement, cloning isolation,
cache parity and unequal-size gradient-stratum reconstruction.

Runtime7ddda401, identity
`95b09ba9eded64e2d00c13bc7207a27a9ab337f95196fef8aaf64f280edf6a53`.
Two workers finished their53-anchor shards in160.65/170.91 seconds. The third
worker's12 completed anchors were transferred to the first available worker;
all receipts were verified and the remaining40 completed without repeating
the12 sealed cells. Its87.40-second final receipt covers the resumed invocation,
not the total initial-plus-resumed wall time. No dataset reduction occurred.

## Native state evidence

At every timestamp, force all three signals from the preserved greedy history
and the real stochastic calibration history, with identical continuation tapes.
Both complete native accounts reproduce exactly. All old267 flat-state
soft-gap witnesses reproduce within the declared1e-9 tolerance.

| Harmful TRAIN occurrences | Count |
|---|---:|
| All occurrences in267 reproduced here |60|
| Q disagrees with all three flat-state soft continuations |20|
| Q disagrees with all three calibration-state continuations |26|
| Q conflict in both histories |17|
| Q conflict only in flat history |3|
| Robust sign reversal between histories |9|

These are repeated seed/objective occurrences, not independent trades.
The9 sign-reversal occurrences are ONE timestamp/action pair repeated across
three objectives and three actor seeds:2020-06-12T01:00Z,NOP->SELL.
Flat-account gaps are[-0.10393,-0.35051,-0.72600]; calibration-account gaps
are[+0.10259,+0.10219,+0.10282]. The same market observation/action can have
opposite value in the two actual account contexts. This witnesses the interface
ambiguity, but does not show it causes all refinement failures. Neither
history's balance/position was fabricated. Differences include several state
fields, so this is not a controlled causal intervention on position alone.

## Unchanged controls and calendar legality

All90 calendar-forbidden flat anchors produce identical soft returns for all
three forced actions under each continuation. The31 eligible changed anchors
and37 eligible unchanged controls are reported separately; do not inflate
ranking quality by counting the prohibited-action ties as successful decisions.
Controls were matched on time/year/old action and flat position, not entry-gate
status. Consequently raw changed-versus-control rates are descriptive, not an
unconfounded causal comparison or a significance test.

| Critic | Eligible changed pairs | Q conflicts | Eligible control pairs | Q conflicts |
|---|---:|---:|---:|---:|
| SmoothL1 |62|22|74|23|
| Mean squared + variance |62|23|74|28|
| Mean squared +0.5 variance |62|23|74|26|

Each anchor contributes two alternatives to its original action. Three
stochastic draws are a diagnostic, not an exact expected return. Critic
ordering errors are also present outside the changed-signal sample.

## Actor gradient evidence

The analytic categorical SAC derivative matches autograd and finite differences.
The complete TRAIN mean-head gradient projects toward all60 harmful changed
boundaries, but ALSO toward all56 helpful occurrences in this flat-anchor
panel. It is not selectively harmful: a shared update moves both kinds.
The local per-observation gradient favors39/60 harmful changes;21 instead
move against that local direction while the aggregate gradient favors them.
The Q component alone favors53/60 harmful changes in the aggregate, and
entropy favors50/60. Removing entropy alone is therefore not established as
a repair. These counts are directional, not estimates of cash contributions.

This is the initial deterministic mean-head SAC derivative, NOT a replay of
256 stochastic AdamW steps. It omits optimizer preconditioning, later
gradients and posterior sampling. The mean-prior KL gradient is zero at the
initial mean. Do not claim this diagnostic reconstructs the entire optimizer.

## Full-support gradient decomposition

Follow-up defined after the panel completed, before any new fit. Replayed the
actual TRAIN calibration history and divided its24025 rows into:

| Actual calibration state | Rows | Harmful boundaries favored by its contribution |
|---|---:|---:|
| Flat, opening allowed |11540|57/60|
| Flat, opening forbidden |4828|60/60|
| Holding an order |7657|60/60|

Each stratum retains denominator24025, so their gradients reconstruct the
original full gradient. Subtracting the forbidden-flat contribution still
leaves all60 harmful boundaries favored. This is an algebraic diagnostic,
not a retraining ablation. No rows were removed, reweighted or fitted.

## Disposition and next correction gate

Preserve selected69, its actor/encoders, and the existing trading controller.
No new training, model promotion, state input or rolling normalization occurred.
Its reference TRAIN/VALIDATION returns remain+10.7921%/+14.5911%.

The evidence does not support a one-line fix by excluding calendar rows,
shrinking posterior uncertainty, removing entropy or matching only position.
Before another actor update, the next contrast should qualify action-value
ordering on operational flat-state NOP/BUY/SELL counterfactuals, including
unchanged controls and equivalent-action/calendar witnesses. Explicitly keep
opening and holding target semantics separate in the diagnostic; do not
silently add state to the network or reuse a mixed-role scalar as a proven
opening score. Test a critic correction without actor updates first, against
the preserved current critic. Only a verified ranking repair should feed
another bounded actor refinement and complete account evaluation.

This does not replace the source with a historical70-input result or restart
unchanged floating selection. Old239/245 contrasts remain relevant negative
controls: smaller updates or closer Q magnitudes did not guarantee rankings.

## Evidence

Full collection:
`~/.local/share/agent-multi/opening69-20261110/`.
Compact public [native summary](generalization/opening_diagnostic_20261110/SUMMARY.json)
and [gradient-stratum summary](generalization/opening_diagnostic_20261110/GRADIENT_STRATA.json).
The native summary digest is
`8a1ceac788dc2825237217b9775275b25cd8ab4bb83ade3f3f0130a5d90319b7`.
The per-anchor states, counterfactuals and gradients stay in the authenticated
local collection; this public packet is not a complete copy of those files.
