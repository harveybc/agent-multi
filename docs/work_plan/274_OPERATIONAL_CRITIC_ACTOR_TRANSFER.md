# Operational critic to actor transfer

## Requirement and boundary

Continue271 independently of the STL coverage repair273. Test whether the
improved opening critic helps the actual selected69 policy under a bounded
actor intervention. This is not a new full training run or feature-selection
pass. Preserve all69 inputs, their encodings, the13xstride2 window, execution
contract and the champion. No additional state enters the model.

## Frozen design

Four sequential arms: seeds19/43 x old calibrated critic / full-representation
absolute-loss critic from271. Each starts from the same actor. The corrected
critic uses its corresponding seed; control uses the shared calibrated critic.
Use only271 FIT anchors with both opening actions executable, entirely2020-22;
2023 diagnostic anchors and2024 never enter actor gradients. This deliberately
bounded diagnostic tests the region where the new critic was qualified, not an
unqualified extension to every holding state. Evaluate full2020-23 TRAIN and
full2024 validation accounts, including effects on holding/closing decisions.

Only existing actor `.head.` parameters train; actor representation, both
critics, targets and alpha remain frozen. Use the exact categorical SAC actor
objective `mean(sum_a p(a|x)*(alpha*log(p(a|x))-min(Q1,Q2)(x,a))))`, with detached
Q and alpha, original learning rate/weight decay and batch128. No TD update or
target update is made. Common deterministic minibatch schedules within seed.
Fixed observations at0,1,8,32 updates; endpoint32, no checkpoint selection on
validation. Save every measured checkpoint, all native metrics and a return
plot. No automatic promotion or follow-on selection from a degrading endpoint.

## Tests before implementation

| Requirement | Unit/component check | Real integration evidence |
| --- | --- | --- |
| Source identity | Reject different source, support or control identity | Hash-check sealed critic and source bundles |
| TRAIN-only fit | Reject bad labels; exclude monitor and forbidden anchors | Record selected timestamps and common schedule hash |
| Actual SAC update | Explicit three-category expectation; detached Q/alpha | Nonzero actor-head tensor changes, finite gradients |
| Frozen modules | Reject any non-head change | Compare frozen parameters and source digest at every point |
| Exact initial policy | Restore identical inference trainability | All valid logits and reference accounts match parent |
| Export/reload | Checkpoint retains source inference flags | Cold-load first checkpoint and reproduce logits exactly |
| Repeated execution | Exclusive lock and sealed completed-arm skip | Persistent sequential worker plus collector |

The runner refuses on failed identity, parity, activity metadata or frozen-state
checks. It does not repair source data, read TEST, add rolling normalization or
change the simulator. Negative results remain measurements, not a reason to
promote or silently switch objectives.

## Executed result

Runtime97e3a3cb; experiment
`b7fe6b252a0fd0e4f183d6f4092535a6ae34a4d80b52c2f073812d4b729e78ff`.
Completed2026-11-11T07:56:48Z on the primary GPU in37.95 seconds, not a full
retraining run:45 qualified FIT anchors,32 updates per arm, full native TRAIN
and validation account evaluations. Four arms completed; collector authenticated
162 files. Nine focused tests passed. Original policy accounts match exactly,
actor heads really change, frozen parameters/buffers remain exact and every
arm's first exported checkpoint cold-reloads to exactly the same logits.

Both refinement seeds produce identical account metrics below; they are not
independent pretrained representations. They retain slightly different signal
changes in the corrected arm (57/58 TRAIN decisions at32updates), so identical
account metrics do not imply every model tensor or signal is identical.

| Critic | Actor updates | TRAIN return % | VAL return % | VAL Sharpe | VAL trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original actor | 0 | 10.792108 | 14.591090 | 2.165596 | 206 |
| Control | 32 | 12.194819 | 13.560523 | 2.036657 | 208 |
| Corrected | 32 | 12.946001 | 12.971293 | 1.949927 | 205 |

All measured points exceed2 orders/week in both periods. At1 control update,
TRAIN rises to11.565540% with unchanged14.591090% validation return; at8 control
updates TRAIN is11.545821% with unchanged validation. Corrected critic at1 update
gives11.143332% TRAIN; at8 updates10.510082%; both retain14.591090% validation.
These intermediate observations are archived, not retrospectively declared the
fixed-endpoint winner. No checkpoint was promoted or passed to live trading.

Conclusion: the improved internal Q-ranking metric did not transfer to better
validation trading at the declared32-update endpoint. The main next step is to
attribute the corrected/control action changes and check the saved1-update
candidate before any further updates or floating-selection pass. Do not repeat
an unchanged mask pass, run a long refinement schedule or claim STL is required
for this diagnosis. The STL coverage repair remains a separate open task.
