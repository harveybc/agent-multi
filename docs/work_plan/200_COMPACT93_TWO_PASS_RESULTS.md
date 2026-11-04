# Compact actor: another deletion and reinsertion round

Finished2026-11-04 16:10:22 America/Bogota (21:10:22 UTC).
Runtime `8f923c05`; experiment105.89s, service109.43s, peak host RAM1.7GiB,
no swap. One sequential primary-GPU job, no training, no intermediate approval.

## Counts and decision

| Stage | Input features | Learned groups |
|---|---:|---:|
| Input to this round |94|30|
| After ordered removal |93|30|
| After ordered reinsertion |93|30|

Removed: **`RAW_ELAPSED_HOURS`**. Reinserted: none. The group `inv.raw` remains
because it still contains other inputs. The frozen deterministic branch response
from compaction198 remains intact, so fusion has30 learned slots plus one
constant-response slot. There are93 actual model inputs, not94 with a hidden mask.

89 atomic blocks were tested for removal in existing order. One improved the
score;88 worsened it against the accepted subset at their respective turn.
No eligible candidate tied the reference, so no equality deletion occurred.
The reverse pass tried the one removed feature and rejected its reinsertion.
All candidates met the activity requirement.90 unique subsets were evaluated;
reinserting the one feature returned to the already evaluated baseline and
therefore used its authenticated cache result.

The first four deletion candidates were evaluated before the accepted deletion,
and were not retested later in this single ordered round. This is exactly the
requested one ida/vuelta, not a proof that all remaining93 features are jointly
necessary or that selection has reached a global optimum.

## Metrics

Same TRAIN2020-2023 and VALIDATION2024, complete original support, fixed TRAIN
scaler,13 observations at stride2 and native risk/cost/calendar contract.
Criterion: VALIDATION net return per market-week, with >2 orders/week required
in both partitions. TRAIN is evaluated/reported but cannot offset a validation
regression. Removal accepts improvements or exact ties; reinsertion is strict.

| Metric | TRAIN before | TRAIN after | VALIDATION before | VALIDATION after |
|---|---:|---:|---:|---:|
| Net return % |210.1869|213.6512|10.4471|**11.0357**|
| Net profit USD |21,018.69|21,365.12|1,044.71|1,103.57|
| Sharpe |3.6164|3.6450|1.4496|**1.5106**|
| Maximum DD % |9.8647|9.8646|4.7276|4.7276|
| Trades |1,827|1,827|298|299|
| Win rate % |64.3678|64.4226|57.7181|57.8595|
| Orders/week |8.7416|8.7416|5.5185|5.5370|

VALIDATION improvement is0.58865 percentage points of cumulative net return.
TRAIN and VALIDATION returns cover different durations, so their raw percentage
difference is not an annualized overfitting estimate. The recorded same-unit
return/week gap increases slightly:0.812215 ->0.817889 percentage points/week;
both scores improve, but this round does not demonstrate a smaller gap.
VALIDATION is exposed development data used repeatedly for selection, not an
independent generalization test. TEST2025 remains unread.

| Final trade detail | TRAIN | VALIDATION |
|---|---:|---:|
| BUY trades / wins |929 /586|138 /75|
| SELL trades / wins |898 /591|161 /98|
| BUY net profit USD |9,554.48|130.08|
| SELL net profit USD |11,810.64|973.49|
| Early closes |340|64|
| Early closes long / short |175 /165|41 /23|
| SL / TP / forced closes |423 /1,011 /53|91 /123 /21|
| Commission USD |4,203.61|498.48|
| Spread USD |2,817.22|337.68|
| Slippage USD |2,161.77|292.27|
| Financing USD |383.17|38.89|

Final VALIDATION early closes:31 helpful and33 harmful relative to passive
protection of the same position, mean regret0.42719 times initial position risk.
TRAIN185 helpful and155 harmful, mean regret0.42575, one proposal preempted.
These are same-position counterfactuals, not account-continuation attribution.
The small selection improvement does not resolve that remaining policy issue.

## Preservation and verification

-30 focused tests for the adapter, compaction, scoring and subset contracts;
  seven additional tests of the pinned feature_selector v0.2.4 two-pass plugin.
- Prior compact control reproduces the saved probe and full native accounts.
- Original compact actor tensors unchanged after all masks; no optimizer step.
- Final93-input actor exported from the authenticated original108 actor,
  preserving previously and newly removed channel effects correctly. Original
  learned values remain; first convolutions are physically sliced.
-32,267 actor parameters, down from32,275 in the94-input compact actor;
  the same104 deterministic response-buffer values and31 fusion slots remain.
-30,234 valid-window logits are bitwise identical between the selected masked
  policy and final compact actor: max error0, zero changed actions.
- Full TRAIN and VALIDATION actions, trade books and equity traces are exact.
- Saved final bundle cold-loads and reproduces full emission exactly on the
  same GPU backend. No cross-device bitwise equality is asserted.
- All90 cache entries authenticated by campaign/subset identity and result
  digest. Independent offline replay through the actual selector reproduces
  the full selection result exactly.
- A completed-run restart reused the cache, replayed controls and final accounts,
  authenticated the artifact and produced the same result; no new training.

The standalone selector test invocation initially found an older ambient package;
rerunning with the pinned v0.2.4 path passed all seven tests. The GPU worker always
used that pinned path and authenticated the plugin witness; it was unaffected.

## Artifact and next use

The final bundle is `frozen_actor.compact.v1`, inference-only, with ordered93
input columns, original projected TRAIN constants, graph metadata, weights and
constant responses, plus probe and reload evidence. Load with
`app.frozen_actor_compaction.load_compact`; no retraining is required for inference.
Critics, optimizer and replay are not migrated. Preserve original decision-row
eligibility for paired comparisons; do not silently admit new rows after removal.

| Object | Identity / SHA256 |
|---|---|
| Campaign |`950512db554647ec71b728812805f43275250711ddef5913f5c7717679608f76`|
| RESULT.json |`08d2ab6ab32aeeeb3a762d3625ee34ac1a3958999152405dc38049a30d526877`|
| bundle/actor.pt |`571b87c6e71840a17ccf9345df2d1ec237c573f5d79a2b15ad175b21646453cd`|
| bundle/MANIFEST.json |`781fd25c518b8b51f89c2a83df04394a6f1613d2a2c6874dc687152f09c3fe4c`|

This authorized round is complete. Retain the93-input actor as the current
inference candidate. No additional selection round, training or DEAP was
automatically added to this one-round request.
