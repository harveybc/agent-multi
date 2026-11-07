# Selected76: Next Selection and Refit Cycle

Owner authorized2026-11-07: one ordered deletion pass followed by one reinsertion
pass over the completed76-input model, then fresh training on the resulting
physical channels. Primary RTX5090 after reboot; existing secondary replications
are not interrupted.

## Fixed Contract

Parent identity `8145612ef9f01c9de669fcb31189744cac20f538eafd2b6803ebba4a8a432f72`,
selected checkpoint epoch6, validation return4.7357%, Sharpe0.7692. Parent fit
completed40 epochs and restored its best weights with exact reload.

- Selection uses feature_selector `two_pass_greedy`, fixed weights and full native
  TRAIN/VALIDATION accounts. Remove on improvement or exact tie; reinsertion must
  improve. Keep indivisible encoding blocks together. No arbitrary target count.
- Preserve support, fixed TRAIN scaler,13 observations/stride2, costs and calendar.
  More than2 orders/week in both periods remains required. No TEST access.
- A checkpoint produced on another GPU may differ numerically. Permit probe
  logits only within atol1e-6/rtol1e-5, require identical argmax actions, and require
  exact full native account reproduction before and after the pass. Record the
  actual maximum difference. Do not claim bitwise logits when they differ.
- Refit removes discarded channels and empty groups physically. New weights,
  optimizers and replay; same TCN/attention and critic dropout as the parent.
- Four TRAIN years2020-2023 and VALIDATION2024. Maximum2000 complete epochs,
  patience20 after epoch20, validation return/week, min_delta1e-8. Save best from
  epoch1, inference bundles, epoch recovery and diagnostic plots.

## Automation and Evidence

`tools/run_selection_refit_cycle.py` chains authenticated selection, staged atomic
preparation, CUDA acceptance and the resumable structural trainer. A separate
collector copies and authenticates both final results and training bundles.
Configuration digest and process locks prevent changed contracts or duplicate
execution on restart. No later selection cycle is launched automatically.

| Requirement | Evidence |
|---|---|
| No partial selection promoted | closure digest and both-pass termination gate |
| No changed trading under GPU tolerance | finite/shape/tolerance/action tests plus full account parity |
| Reusable reduced-input model | selected recipe lineage, actual-shape CUDA canary, bundle reload |
| Recovery without partial preparation | temporary preparation directory followed by atomic rename |
| No manual transition between stages | one systemd chain, separate automatic collector |

Focused CPU verification:36 passed,6 CUDA skips, including automatic stage order
and no duplicate training after completion. Runtime evidence is produced by
the chain; a launch is not a completed selection or a trained model result.

## Launch Verification

Runtime `ff7a137e`; cycle started2026-11-07 at13:04UTC on RTX5090 after the reboot.
Selection identity `9be13b93b3d77efe644571313aa744e591bdc1ab8e6ccb47f216294540ada02b`.
Cross-device maximum probe error was1.1920928955078125e-6, within the combined
absolute/relative bound, with identical probe actions. Full native TRAIN and
VALIDATION control accounts matched the archived parent exactly. Both cycle and
collector services are enabled; primary user lingering is enabled. The obsolete
strict cross-device attempt was stopped and disabled, not reused as evidence.

## Completed Selection and Automatic Transition

Both passes completed at13:06:53UTC:76 initial columns,69 after deletion,70 after
reinsertion;28 groups retained,80 unique evaluations. All support and full-control
account checks passed again after masking.

| Frozen checkpoint treatment | TRAIN return % | VAL return % | VAL Sharpe | VAL orders | VAL win % | VAL max DD % |
|---|---:|---:|---:|---:|---:|---:|
| Parent76 inputs |10.8156|4.7357|0.7692|197|53.81|4.7188|
| Selected70 inputs |11.5145|13.8085|2.0971|206|57.77|3.6249|

These are interventions on frozen epoch6 weights, not fresh70-channel results.
The chain automatically prepared the physical70-channel recipe and entered the
training task at13:07UTC, without another operator command. Fresh-fit identity:
`eb4179dc4d1058a4243f41da671f46f49b3010bae3c8037f1ab94609e66d03bf`.
Real CUDA canary passed with128 transitions of shape13x70; first native SAC update
observed after4,096 warmup steps. Cycle service active with zero restarts, about
3.6GB aggregate RAM at launch. Training is running, not merely queued.
