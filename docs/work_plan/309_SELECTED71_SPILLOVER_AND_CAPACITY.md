# Selected71: spillover before core capacity

Continuation of307 within phase2, before another cumulative ida/vuelta. The
source299 checkpoint is unchanged. Retsu owns real-feed implementation under
`../handoffs/MUSASHI_TO_RETSU_SELECTED71_IMPLEMENTATION_2026_11_12.md`.

## Tests Before Diagnosis

| Requirement | Evidence to produce |
| --- | --- |
| No new training or validation fitting | CPU TRAIN-only forward passes; no optimizer steps |
| Correct source | Authenticate source299,307 receipts, recipe and support |
| Decompose actual saved correction | Final layer weight, bias, hidden response and interaction reconstruct logit delta |
| Frozen upstream | Source and candidate encoder state and activations match |
| No source mutation | Check file hashes and state at end |
| Not a capacity theorem | Effective rank describes sampled activations, not learnable information capacity |
| Negative controls | Reject nonfinite/mismatched arrays; known linear and nonlinear fixtures reproduce |

Report all TRAIN rows, greedy changes, margin movement, activation spectra and
the fixed target's nearest representation neighbours. Do not turn neighbour
labels into protected constraints or choose examples using validation. CPU
actions may differ near GPU boundaries: compare and disclose, never rewrite
the sealed GPU accounts. No profit claim comes from this forward-only study.

## Next Capacity Contrast

The current architecture is branch_width8/branch_blocks2, core_width16/
core_blocks2, head_width8, causal patch attention embedding8/heads2/feedforward16.
The proposed isolated contrast is core_width8 vs16 vs32, all with two core
blocks, unchanged attention, branches, windows, features, strategy, support,
optimizer and stopping contract. Parameter count and actual instantiated
shapes must be recorded. This is a plan, not a launched training queue.

Width changes also change the input dimension of the head. Therefore do not
claim exact inherited function unless a tested morphism proves it. A fair
fresh comparison includes the width16 fresh control; the preserved trained299
is a separate reference, not its substitute. Matched seeds and equal data/
update budgets are required. Freeze the rule before looking at validation.
Keep the proven head-reset findings; do not silently warm-start optimizer
state after structural changes. Live-incompatible repaired data require a
new shared data identity before promotion.

Finish the finite309 diagnosis first, then choose a bounded paired core trial
or head-local repair from TRAIN evidence. Keep running STL queues independent;
do not erase their outputs, promote STL31 as live-ready, or repeat unchanged
floating to occupy GPUs. The next return pass includes all phase2 removals
from the original120 universe, not only the last deletion pass. No phase1
discard is silently reintroduced.

## Executed Diagnosis

CPU run completed in16.23seconds on all24026 admissible TRAIN windows; no
validation inference, fitting or account replay. Identity
`fdbc292b8f1cf42db778bd0df148dda95d85d54ecad3adaaa2f98499538c4116`.
Source and candidate encoder activations are exactly equal. The only changed
parameters belong to the four head tensors. Six greedy changes match the
sealed307 GPU timestamps exactly. The original source is unchanged.

| Measurement | Result |
| --- | ---: |
| Actor parameters |30733 |
| Online twin critic parameters |61466 |
| Flattened core width |208 (13 x16) |
| Core centered effective rank |46.8643 |
| Core dimensions covering99% variance |169 |
| Head hidden width / effective rank |8 /3.9581 |
| Head dimensions covering99% variance |7 |
| RMS margin movement from hidden response |0.0011617504 |
| RMS margin movement from final weights |0.0004732380 |
| RMS margin movement from final biases |0.0002324586 |
| RMS margin movement from interaction |0.0000003784 |

These contributions are not independent variances and must not be added as
percentages of explained variance. At the fixed target, hidden response
contributes0.005394 to a total SELL-vs-NOP margin movement0.005578, about97%.
The final winning margin is only0.000005305: portability near the boundary
requires explicit qualification, not an arbitrary relaxed equality tolerance.
The decomposition reconstructs actual float32 forward differences with maximum
error2.0168e-6. Its algebraic identity is independently tested in float64.

Interpretation: this saved correction acts mainly through the first head
layer's nonlinear response, not merely a common final bias. This does NOT
establish insufficient core capacity or prove that widening the head helps.
The first proposed capacity comparison should isolate head_width8 vs16 on the
same frozen encoder, with a paired fresh-head control and fixed critic targets
appropriate to that experiment. Preserve299 as a separate reference. Only
after this contrast consider core8/16/32 with all other settings paired.
Neither capacity contrast has been launched; current GPUs continue STL.

Six new unit tests and six existing account/queue regressions pass. An initial
attempt completed inference but failed JSON serialization of a numpy scalar;
it produced no sealed result. The scalar was converted explicitly and the
full run repeated in a new root. Repository-local Ruff executable is absent;
no lint-clean claim is made.

Evidence: [RESULT.json](generalization/selected71_spillover_20261112/RESULT.json).
Large TRAIN arrays and full source receipts remain in the owner-local run
root `/var/tmp/sac71-head-spillover-20261112-r2`. No model was promoted and no
unchanged floating pass was started.
