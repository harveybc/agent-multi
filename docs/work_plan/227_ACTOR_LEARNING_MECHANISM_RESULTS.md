# Actor Learning Mechanism Contrast

Design:[226](226_ACTOR_GRADIENT_AND_OPTIMIZER_CONTRAST.md).
The frozen selected69 candidate is not modified. This investigates the actual
selected70 epoch0/epoch40 failure and its critic-return anchor. No data, input
encoding, strategy, position feedback, model capacity or paper deployment change.

## CPU Gradient Decomposition

Three checkpoints, eight128-row batches spanning TRAIN,1024 identical market
observations each. Actor value and entropy gradients sum to the native TorchRL
actor gradient (maximum discrepancy7.24e-7). Subtracting each row's common Q
offset preserves its value gradient (maximum discrepancy7.79e-7). All weights
and original trainability flags remain unchanged.

Numbers below are arithmetic means across the eight batches. The norm ratio
is the ratio of mean component norms, not the norm of an averaged gradient.
Q units are discounted reward units, not account return percentages.

| Snapshot | Q(BUY)-Q(NOP) | Q(SELL)-Q(NOP) | Value gradient norm | Entropy gradient norm | Ratio | Q prefers NOP |
|---|---:|---:|---:|---:|---:|---:|
| Preserved epoch0 |-0.011534|-0.005562|0.003867|0.004092|0.945|72.17%|
| Degraded epoch40 |-1.103106|-0.610468|0.048961|0.000626|78.267|90.14%|
| Critic anchor4096 |0.216528|0.099153|0.216993|0.000626|346.872|62.40%|

The anchor has exactly the degraded actor; only its critics differ. Hence its
entropy-gradient component is identical, while its value signal changes.
No actor update or new trading gain is inferred from the anchor.

For the degraded actor, the local mean NOP probability derivative under an
infinitesimal SGD step is +0.00139852 from the value term and -0.00002450 from
entropy. Signs differ across batches (value positive in5/8); this is not a
claim that every observation moves toward NOP. At epoch0 the means are
+0.00115180 and -0.00170119. These are derivatives, not measured AdamW updates,
not account returns and not a forecast of long-run training.

Interpretation: the critic's action differences, rather than only its common
negative Q offset, supply a much stronger preference against BUY/SELL after
refinement. The entropy term locally opposes NOP concentration on average;
it is not supported as the direct driver of this concentration. This narrows
the mechanism but does not by itself identify why the critic learned those
action differences or prove their correctness.

## Same Observation and Signal, Different Position

Authenticated original TRAIN trajectories under action-sampling seeds7/19/43,
aligned at the same timestamp and same categorical signal. The original GPU
probabilities are reproduced exactly from their sealed logits. Different
position directions are measured outside the network; no feedback is added.

Representative pair7/19 (the other two pairs are also saved):

| Snapshot | Position match | Rows | Different controller commands | Immediate reward RMSE | Soft return RMSE |
|---|---|---:|---:|---:|---:|
| Epoch0 | same |10524|0|0.017106|0.673184|
| Epoch0 | different |5529|512|0.155084|0.790173|
| Epoch40 | same |23508|0|0.002269|0.121649|
| Epoch40 | different |265|43|0.165755|0.303246|

For example, the same BUY symbol can open a long position or close an existing
short. This is the declared controller contract, not a newly discovered decoder
bug. Different account states can therefore create different value targets
for identical market observations/actions. Future sampled actions, balances,
position age and protection levels also differ; these observational pairs do
not isolate a causal position effect or provide an irreducible-error estimate.
The next causal test would hold those factors fixed where possible in the native
engine, not silently add state channels or declare the architecture impossible.

CPU probabilities differ from the archived GPU probabilities by at most6.59e-6
at epoch0 and3.21e-5 at epoch40; zero greedy action disagreements. Epoch40 does
not pass the1e-5 probability tolerance, and its report says so. Observational
pairing uses the exact archive probabilities, not those differing CPU values.

CPU REPORT SHA256:
`7839447d3366feb55da9971109c2f6d8132d4bb07ba70464ccea03e1d56bdcd0`.

## Optimizer Memory Contrast

Completed2026-11-09 at09:25:24 UTC (04:25:24 America/Bogota) on the physically
identified RTX5090. The sequential queue verified each finished unit before
continuing and authenticated pair equality before writing QUEUE_CLOSE.json.
Both arms use the same epoch40 weights,4096 updates, batch128 and full24025-row
TRAIN schedule. Only AdamW's stored moments/step counters differ. Reset clears
eight optimizer states, changes no parameter groups, LR or weight values.

| Updates | Retained RMSE | Reset RMSE | Retained bias | Reset bias |
|---:|---:|---:|---:|---:|
|0|25.525045|25.525045|-25.021730|-25.021730|
|1024|15.427805|19.863303|-14.637509|-19.253094|
|2048|9.448890|12.913522|-8.224373|-12.032337|
|4096|5.006720|6.062444|-2.474531|-4.157739|

At4096, alternate action draws19/43 give retained RMSE5.008122/5.001600,
reset6.062777/6.058929. These share TRAIN market dates and are not independent
model fits. The zero-value reference RMSE is0.295424/0.279264/0.293172 for
7/19/43. Neither neural critic beats it. A constant critic is not a useful
trading policy and does not establish correct action rankings.

Runtime seconds:262.53 retained,262.23 reset. The retained control reproduces
the earlier4096-update calibration values exactly. Actor and online encoders
remain unchanged at each recorded checkpoint. The frozen selected69 and70
checkpoint hashes were independently rechecked and remain unchanged.
The coordinator independently authenticated all12 terminal-listed unit files,
both DONE receipts, the pair contract and the aggregate comparison digest.

Conclusion: resetting AdamW is not the remedy under this matched budget and
actually worsens recovery. This does not prove optimizer settings never matter;
it rules against this particular proposed fix. The combined gradient evidence
supports investigating critic action-gap learning/target fit next, not widening
the representation or declaring the inputs uninformative.

Queue identity:
`bad5ff2ad55e9431f56c5198de45f33cd5e872bfe96e0c03aa73f2ae69a2be74`.
COMPARISON SHA256:
`f6ac890c17e3dbd7b319642e1c932d44c3fe4c053447c967272f59f59e6ace1a`.
Retained identity:
`3c2bc8cea10e8b0f10914748eee033a2de150b6598103459280cf388715f308b`.
Reset identity:
`a3d7960996b04d2d737c66f8ec8f4b95cb66e943673c8d7d05c4b445883708de`.

Next controlled distinction: retained versus freshly initialized critic head
with the encoder/actor fixed, common realized-return labels, and per-action
calibration as well as the constant reference. A native counterfactual position
test can separately isolate the ambiguity observed above. Neither is claimed
as already executed, and no long joint refinement starts from these diagnostics.

## Verification and Phase2 Integration

Focused tests:29 passed. Native gradient parity, immutable source checks and
exact archived probability matching additionally passed on real data. The
wrong-GPU regression rejects the process before loading data. Execution errors
and their corrections are recorded in226, not silently removed.

The reusable diagnostic is a checkpoint gate for phase2 structural export and
refinement: preserve source identity, inspect per-module weight/activation drift,
inspect actor gradients when refinement changes behavior, and replay full native
accounts before accepting an edit. It is not an additional expensive pass per
feature ablation. The simulator and representation stay unchanged in this run.

`export_components` now automatically includes a hashed WEIGHT_PROFILE.json
for actor/critic/target extractors, fusion, attention, core and heads. It records
finite values, norms and singular-value summaries at the structural/refinement
export boundaries. This metadata-only hook does not alter parameters and does
not reinterpret a tensor's flattened rank as model capacity. Historical bundles
are immutable; the hook applies to future exports, not retroactive rewrites.
