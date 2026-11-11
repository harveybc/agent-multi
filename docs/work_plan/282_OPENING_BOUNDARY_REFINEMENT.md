# Executable Opening Boundary Refinement

2026-11-11. Owner authorizes the next TRAIN-only diagnostic after281. Preserve
qualified69 reference and the seed43 tradeoff candidates; neither is overwritten.
Start from the same qualified control19 bundle, not a validation-selected winner.

## Frozen Design

Use the current greedy actor's actual native trajectory in FIT2020-2022 and
MONITOR2023. At flat states, probe each opening's first native step on a fork to
establish executability without changing the trajectory. Do not inspect profits
when choosing the cohort. Native fill eligibility can depend on the next bar;
it is a retrospective diagnostic restriction, not an observation or live filter.

For each year and BUY/SELL boundary, choose up to6 closest NOP-winner states
and6 closest direction-winner states by absolute logit difference. The competing
third class must not be the winner. Require24h separation within each stratum.
Add4 far-margin executable flat controls,2 restricted flat controls,4 long and4
short holding controls per year. Deduplicate timestamps and record all requested
strata; sparse strata remain sparse, never padded. At most152 anchors before
deduplication. Full trading accounts still cover all valid historical rows.

Execute three actions per anchor, then the frozen policy through the end of
that role's period. Real discounted reward and USD account continuation are
labels only. No ZigZag, account-state input, new feature, rolling scale or window
change. Period boundaries and holding equivalence use the281 gates.

Six paired arms: seeds19/43 x retained critic / fresh-head absolute calibration /
fresh-head pairwise-gap calibration. Calibration4096 updates, batch128, identical
initial heads and schedules for the two fitted objectives within seed. Pairwise
loss is the existing tested three-pair Huber plus common-level Huber. Only critic
heads train; actor representation stays frozen. Then only actor heads train with
the native categorical SAC objective and original alpha, learning rate and weight
decay. Fixed observations0/1/8, endpoint8. No stopping/checkpoint choice on the
monitor or validation. This isolates calibration loss on one boundary cohort,
not a new hyperparameter search or a full refit.

## Gradient Evidence

Before updates, take the exact first FIT minibatch. Differentiate its actor loss
into Q and entropy terms. For each cohort row and opening direction compute
`-dot(grad(logit_direction-logit_NOP), grad(loss_FIT))`, separately by term and
in total. This is the infinitesimal SGD direction, not a claim about AdamW.
Also save actual margin changes after1 and8 AdamW steps, and full native account
changes. Report which negative native action advantages receive positive margin
pressure, including holding/restricted controls. Monitor labels never enter a
gradient. Compare constant symbols and trace profitable/harmful changed actions.

## Requirements And Tests

| ID | Contract | Evidence |
|---|---|---|
|B1|Selection precedes reward labels|Outcome perturbation leaves chosen timestamps unchanged; nonfinite/foreign periods rejected|
|B2|True two-class boundaries|Third-class winner excluded; both sides selected when available; deterministic spacing and deduplication|
|B3|Native eligibility without parent mutation|Forked first steps, unchanged parent state, parity with executed continuation first effects|
|B4|Gradient decomposition|Sum of Q and entropy derivatives equals total; finite-difference SGD witness|
|B5|TRAIN-only same-state labels|Existing support, timestamps, terminal, holding equivalence and source gates|
|B6|Paired implementation|Common initial weights/schedules; frozen actor/encoder/alpha; actual nonzero updates and exact cold accounts|
|B7|Autonomous completion|Pinned finite queue: fit then independent all-endpoint attribution and controls; authenticated collection|

## Acceptance And Disposition

The experiment is complete only with six terminal arms, cold replay, component
weight checks and independent account reconciliation. A larger return alone is
not proof of mechanism repair; compare the gradient signs, executed changes and
constant-control regret. Report all arms and saved checkpoints. No automatic
promotion, live orders, TEST access or repeated feature selection is authorized
by this diagnostic. Any useful candidate requires separate qualification before
the next changed-policy floating pass. Historical STL/live-source work is separate.
