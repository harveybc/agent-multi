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

## Verified Launch

40 focused tests pass, including finite-difference gradient agreement and
positive/negative native-advantage classification. CLI import passes. Runtime
`c0ef78dd` was deployed immutably. Primary RTX5090 readiness and memory gates
passed; baseline full native accounts match the qualified source exactly.
Selected151 anchors produced453 real native continuations; forked first-step
executability matched subsequent continuation outcomes and parent-state gates
passed. The six-arm fit and independent report are queued sequentially with
automatic hashed collection and bounded execution. No live model was replaced.

Identity:
`60a36773859b58500336c63964799e9d7d45db90514761cd2432cded95411ace`.

## Completed Results

All six arms completed in442.39seconds; independent reporting took32.19seconds.
The collector authenticated489 worker files and40 report files. Both services
exited0. All cold accounts and frozen-component checks pass. The twelve endpoint
account differences reconcile exactly, with maximum attribution error0USD.
Within each seed the absolute/pairwise initial heads and minibatch schedule
hashes match. There are113 FIT anchors and38 MONITOR anchors; no exact logit
ties in the selected panel. Monitor/validation labels never enter an update.

Full-period native results at the frozen endpoint of8 additional actor updates:

| Critic | Seed | TRAIN return % | VAL return % | TRAIN trades | VAL trades | VAL Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Qualified reference |19|11.565540|14.591090|747|206|2.165596|
| Retained |19|11.040730|14.693262|744|206|2.179247|
| Retained |43|11.040730|14.693262|744|206|2.179247|
| Absolute calibrated |19|12.392145|14.693262|743|206|2.179247|
| Absolute calibrated |43|12.392145|14.693262|743|206|2.179247|
| Pairwise calibrated |19|12.392145|14.591090|743|206|2.165596|
| Pairwise calibrated |43|12.392145|14.693262|743|206|2.179247|

Both absolute-calibration seeds improve on the source in both periods, with
identical executed accounts, not identical weights. This is replication across
two initialization/batch seeds on the same history, not independent market data.
The pairwise loss does not improve trading over absolute calibration here.

For either absolute-calibrated endpoint:

| Metric | TRAIN | VALIDATION |
|---|---:|---:|
| Net USD |1239.214485|1469.326185|
| Sharpe |0.617273|2.179247|
| Maximum DD % |7.930291|4.044574|
| Orders/week |3.555024|3.814815|
| Win rate % |61.507402|58.252427|
| BUY count / wins |231 /154|24 /15|
| SELL count / wins |512 /303|182 /105|
| Early closes: helpful / harmful |29 /30|4 /2|
| TP / SL / forced closes |401 /222 /61|107 /67 /26|

The native costs, signal/command counts, holding durations and all controls are
in the summary. Early-close helpfulness is the fixed-position passive-protection
comparison, not the chronological account-effect decomposition.

## What Changed And What Did Not

Absolute/pairwise calibration adds82.660530USD in TRAIN versus the qualified
reference. Ten effective opening interventions explain it exactly: six helpful,
four harmful. The retained-critic controls lose52.480965USD in TRAIN, from seven
opening interventions (four helpful, three harmful). All validation gains are
the same10.217220USD, from avoiding the same harmful short-position early close
already identified in281: NOP lets forced-flat settle within that decision
interval. This gain also occurs with retained critics. It is not evidence of a
new calibration-specific early-close repair. Pairwise19 leaves VAL unchanged.

The gradient audit exposes a second limitation. In executable flat states there
are79 beneficial and86 harmful direction-versus-NOP comparisons in FIT, and24
beneficial/32 harmful in MONITOR. Absolute19 and both pairwise fits push every
beneficial comparison's margin down under the shared-head first-minibatch SGD
direction, and the actual8-step AdamW changes also reduce all those margins.
Absolute43 suppresses79/79 and24/24 under that SGD direction, and78/79 and23/24
under the actual AdamW endpoint. Thus fewer harmful openings can accompany a
broad reduction in willingness to open, including useful openings. These are
margin comparisons, not counts of executed orders lost.

This is not exclusively wrong Q signs: absolute19 has26 FIT and5 MONITOR flat
comparisons where both native return and predicted Q favor the opening over
NOP, yet the shared-head gradient reduces that opening margin. Pairwise43 has9
and5 respectively. The measurement localizes the issue to the complete actor
update, but does not yet separate local softmax three-action competition from
cross-sample gradient coupling or AdamW preconditioning. Entropy is not the
dominant gradient norm in these probes: the calibrated Q term is22-39x larger.
Gradient decomposition maximum numerical error is below9.5e-9.

Critic monitor regret improves from0.281122 to0.210960/0.221569 (absolute19/43)
and0.217214/0.182796 (pairwise19/43). However constant SELL's monitor regret is
0.180804, still lower than every learned critic here. Pairwise43's66% pairwise
accuracy therefore cannot alone certify useful learned discrimination. Constant
SELL is a diagnostic symbol, not a suggested trading strategy; effects depend
on the external position state.

## Disposition And Next Gate

Keep all six checkpoints plus both earlier references. No live model was
replaced, no feature changed, no TEST data read. The last cumulative floating
pass remains69 ->69 ->69 with28 groups. Do not repeat that unchanged pass.

Next: a read-only TRAIN decomposition of local-logit gradients, per-sample and
cross-sample shared-head contributions, and actual AdamW steps, including true
native-return advantages as an offline diagnostic control. This distinguishes
critic ranking from optimization interference without first changing the loss,
adding state inputs or launching a long refit. Qualify the saved absolute19
candidate (prespecified seed, not a best-seed search) before any next changed-
policy floating pass; preserve absolute43 as its paired replication. The
experiment reported here is complete; that follow-up has not been launched.

Evidence: [manifest](generalization/opening_boundary_20261111/MANIFEST.json),
[full results](generalization/opening_boundary_20261111/SUMMARY.json),
[gradient counts](generalization/opening_boundary_20261111/BOUNDARY_GRADIENTS.json),
[positive-Q pressure](generalization/opening_boundary_20261111/POSITIVE_Q_PRESSURE.json),
[curves](generalization/opening_boundary_20261111/RETURNS.png).
Report identity:
`0f4919b3457191d71eefb3d87f1cb0d23bd46afa153aa49f3180afdbd4bdc4c4`.
