# Sequential SAC Regularization Ablations

Owner authorization2026-10-19: continue automatically after the active paired
early-stop campaign, testing regularizers separately before combinations.
Do not change the active scientific runtime. Poor generalization can reflect
overfitting AND regime/distribution shift; early stopping alone, dropout, L2
or BatchNorm are hypotheses, not guaranteed solutions.

## Frozen Order And Scope

| Order | Profile | Critic dropout | Critic Adam L2 | Critic BatchNorm |
| --- | --- | --- | --- | --- |
| 1 | dropout005 | 0.05 | 0 | false |
| 2 | l2_1e6 | 0 | 1e-6 | false |
| 3 | dropout005_l2_1e6 | 0.05 | 1e-6 | false |
| 4 | batchnorm | 0 | 0 | true |

These are bounded initial ablations, not optimized strengths or a DroQ/CrossQ
reproduction. Dropout follows each critic hidden activation; BatchNorm is
between each critic hidden Linear and ReLU, standalone. No output-layer noise,
no dropout in the actor, no BatchNorm/dropout combination. L2 uses the existing
Adam optimizer's coupled weight_decay on critic parameters only (including
biases); actor and entropy optimizers are unchanged. This distinguishes it
from AdamW. Preserve the two critic networks and target update algorithm.
The target critic remains in eval mode; both critic and policy inference must
be deterministic with dropout disabled. BatchNorm running statistics must be
saved/restored and synchronized to the target by native SB3 support.

All arms start fresh, seed7, the same15 selected inputs,13 observations at
stride2 observed open-market bars,32/32 hidden units, leverage50 and the same
business protections, sizing, costs and full TRAIN2020-23/VALIDATION2024.
No TEST, refit, DOIN, feature changes, new temporal SAC fits or donor weights.
Unchanged paired early stop: every5000 steps, patience10, min_delta1e-8,
maximum100000 transitions, negative mean TRAIN/VALIDATION daily log growth.
Retain separate full metrics and gap: a smaller gap or better mean with
losing VALIDATION is not profitability. All four arms run regardless of sign.

## Automation And Safety

Use the existing materializer/queue/publisher/collector. A pinned request_v5
adds exactly one validated regularization profile to request_v4. Finite
systemd workers are chained by authenticated predecessor closure (including
checkpoint reports), resource gates, and minute timers. No LLM polling or
owner approval between successful experiments. One heavy job only on5090;
5070Ti is not a second slot. Existing bounded attempts and typed failures
remain explicit, never fabricated completion. Publication/readback and local
closed metrics backup are required; a second physical collector runs on its
own timer. Working code/SQLite stay on native filesystem; the archive uses
the approved bulk volume. Configs/policy/optimizer/BN state and metrics are
retained. Independent typed failure may be investigated without editing
scientific identities; no silent parameter fallbacks are permitted.

## Tests Before Deployment

| ID | Behavioral and structural proof |
| --- | --- |
| RG1 | Invalid/extra/profile-drift settings rejected; baseline default unchanged |
| RG2 | Only critic modules changed; actor initialization identical at same seed |
| RG3 | Dropout stochastic in training, deterministic in eval; target remains eval |
| RG4 | L2 parameter groups exactly as declared; real updates finite |
| RG5 | Native save/load + best-checkpoint restore preserve weights, optimizers and BN buffers |
| RG6 | request_v5 changes only approved profile versus v4, other scientific settings identical |
| RG7 | Predecessor verifies paired counts and hashes; missing/corrupt monitor refuses |
| RG8 | Four sealed configs and ordered gate pins; idempotent preparation, no fits in prepare |
| RG9 | Generated timers/unit paths and GPU/env override; no parallel worker ownership |
| RG10 | Existing full engine subprocess monitoring, publication and collection regressions stay green |

Local CPU tests and a finite synthetic GPU smoke precede scientific admission.
No scientific campaign is started merely because its tests were written.
Update this document with exact evidence and deployed code pin before return.

## Verified Engineering And Runbook

182 focused CPU tests passed, including native SAC gradients/save/load for
all four profiles, five real-engine monitoring cases (baseline plus profiles),
unchanged parent RNG/replay, exact policy/optimizer/BatchNorm buffer restoration,
request_v5 parity, paired-predecessor corruption controls, immutable sequential
preparation, unit generation and existing publication/physical-copy coverage.
GPU admission is intentionally deferred to each worker's bounded ExecCondition
AFTER its predecessor closes and the GPU resource gate passes. A successful
GPU smoke is recorded separately as synthetic, never financial evidence.

```bash
python -m tools.selected_rl_regularization_sequence prepare \
  --base-request <private-request-v4.json> --root <private-sequence-root> \
  --predecessor-root <active-early-stop-root> --predecessor-sha256 <contract-sha256>
python -m tools.selected_rl_regularization_sequence render \
  --spec <private-sequence-root>/SEQUENCE.json --environment-file <private-env-file>
python -m tools.selected_rl_regularization_sequence status \
  --spec <private-sequence-root>/SEQUENCE.json
```

Install only generated units in the user's systemd directory, reload and
enable their timers. They carry resource/scientific closure gates and cannot
launch a second GPU-heavy worker while the preceding owner remains alive.
The publication timer owns delivery/readback; the existing collector CLI on
the second physical machine owns closed artifact snapshots. Fresh experiments
have independent OLAP experiment-set keys and explicit regularizer tags.

## Primary Sources

- [SB3 SAC policy extension points](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [Dropout Q-functions](https://arxiv.org/abs/2110.02034): motivation for critic regularization, not a claim to implement that full algorithm.
