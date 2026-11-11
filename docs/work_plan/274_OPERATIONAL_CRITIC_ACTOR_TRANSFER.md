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
