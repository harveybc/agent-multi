# Audit: Satoshi III ETH Correction Delivery and Curriculum Decision Gate

Date: 2026-08-05 America/Bogota
Auditor: General Musashi, temporary independent auditor
Delivery audited:
`docs/handoffs/SATOSHI_III_ETH_CORRECTION_DELIVERY_2026_08_05.md`
Runtime mutation by this report: none

## 1. Executive Verdict

The correction packet materially repairs the rejected ETH campaign, but it is
not accepted in full.

- Findings 108, 109, 110, 111 and 116 are independently reproduced as
  corrected.
- Finding 112 is technically corrected by an exact mixed-radix order key over
  the preregistered quantized tuple. The bounds still require owner
  ratification before being treated as permanent doctrine.
- Finding 115 remains partially corrected. Process stopping and stickiness are
  tested, but unavailable GPU telemetry is treated as successful verification
  and there is no supported same-chain resume transition.
- Finding 113 remains partially corrected: `preprocessing_mode=none` is no
  longer generable, but the declared `forbid_value` rule is not implemented by
  the optimizer's repair-rule interpreter.
- Finding 114 remains partially corrected: the protected test is no longer
  evaluated and the report/manifest are committed, but resolved configs,
  return traces and retrievable model artifacts are not preserved by the
  committed packet.

The current `phase-2-eth-anchored-full-fleet-v2` runtime is healthy and may
continue while the paired curriculum packet is prepared. It is an ETH-EN arm,
not evidence that ETH-EN should be used for every asset/model. No curriculum
generalization is authorized by this audit.

## 2. Finding-by-Finding Disposition

| Finding | Verdict | Independent evidence |
| --- | --- | --- |
| 108 | corrected, pending owner closure | `lexicographic_weekly_v1` resolves in `app.metrics`; active four-worker runtime reports the same metric; eligible objective probe and tests pass |
| 109 | corrected, pending owner closure | one rejected-result schema plus DOIN-independent rejection; failure, sentinel, non-finite, broadcast and zero-eligible tests pass |
| 110 | corrected, pending owner closure | ETH-N/ETH-EN materialization is deterministic; foreign-token scan and arm-root checks pass; current anchored-v2 artifact namespace is ETH-only |
| 111 | corrected, pending owner closure | empty-range, racing-peer and deterministic tie tests pass; current four-worker chain has one exact tip; large artifact transfer additionally required `doin-node@9eba394` |
| 112 | corrected subject to bound ratification | the order key is float64-exact below `2^53`; its natural order equals the bounded quantized tuple order; audit counterexample and 5,000-pair property test pass |
| 113 | **not accepted** | removing `none` from choices prevents current generation, but `forbid_value` is dead declarative data: `_apply_repair_rules()` only understands `if` + `set` |
| 114 | **not accepted** | test leakage is removed and committed report hashes verify; committed packet omits resolved per-arm configs, return traces and retrievable model artifacts |
| 115 | **not accepted** | stopping/escalation tests pass, but `nvidia-smi` failure is counted as GPU-clear and no supported resume transition exists for a temporary scientific pause |
| 116 | corrected, pending owner closure | complete LTS suite passes at the current wall-clock time using deterministic ledger timestamps |

## 3. Reproduced Test and Integrity Evidence

```text
agent-multi focused correction/warm-start suite: 63 passed
agent-multi complete suite:                    539 passed, 2 warnings
doin-node focused rejection/fork/sync suite:    92 passed
doin-node complete suite:                      409 passed
lts complete suite:                            652 passed, 1 warning
```

The warnings are one scikit-learn convergence warning class and one Starlette
deprecation warning; neither changes the audited behavior.

The rejected chain remains byte-identical:

```text
d1eb870c5dafa437616c63f3b6358370502291654211dc05e279a06d5beea902
~/.local/state/agent-multi/doin-campaigns/
  phase-2-eth-curriculum-invalid-audit-20260805/omega/
  doin-data-eth-en-v1-omega/chain.db
```

The corrected fixture report and all locally present model files match the
committed manifest. The model files themselves are ignored by Git and the
report has no return-trace artifact references, so hash agreement is not the
same as cross-workstation artifact availability.

## 4. Curriculum Evidence: What It Says and What It Does Not

The corrected committed fixture used one seed and two epochs of 4,000 steps.
All values below are realistic **2024 validation** results; the disclosed 2025
period was not evaluated.

| Arm | Trades | Mean weekly return | Annualized return | Total return | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `normal` | 128 | +0.02754% | +1.41157% | +1.41384% | 3.62981% |
| `easy` evaluated normal | 75 | +0.01111% | +0.55573% | +0.55662% | 2.67410% |
| `easy -> normal` | 75 | +0.01111% | +0.55573% | +0.55662% | 2.67410% |

The `easy` and `easy -> normal` policy payloads are byte-identical. The normal
continuation changed optimizer state but did not produce a validation-better
policy. Both easy histories reported zero would-margin-call events. Therefore:

1. this fixture proves that the mechanism runs and produces active policies;
2. it does not exercise the solvency-relaxation mechanism that motivated the
   curriculum;
3. it does not establish a benefit from easy training;
4. its single seed and small, compute-unequal arms cannot decide use across
   asset/model families.

The earlier 10-epoch/four-seed attempt is not decision evidence: its exported
policies collapsed to zero normal-condition trades under the then-incomplete
handoff eligibility logic. It is retained as incident evidence, not averaged
into the corrected comparison.

## 5. New Findings

### AUD-F1-20260805-119 (S2): active systemd profile can be overwritten

The delivery records that a smoke-profile installation briefly replaced the
systemd drop-in while the anchored campaign remained active. Existing workers
survived, but a supervisor restart during that interval could have adopted the
wrong profile and started a second domain. The current unit has been restored
and the running chain was not altered.

Required correction: profile installation must be an atomic, hash-checked
operation that refuses to replace an active campaign profile unless the fleet
is verifiably paused. Supervisor state must compare the loaded plan/profile
hash with the systemd `ExecStart` profile and alert on drift.

### AUD-F1-20260805-120 (S3): current fixture cannot decide curriculum rollout

The current comparison has one seed, a two-epoch budget, unequal total compute,
no would-margin-call event and no divergence between post-easy and post-normal
policy weights. It cannot answer the owner's current priority: whether the
easy phase should precede normal training for the remaining models.

Required correction: execute the paired decision packet in the accompanying
handoff before generalizing the curriculum beyond ETH/SAC.

### AUD-F1-20260805-121 (S3): pause is not a reversible campaign operation

`request_pause()` persists a sticky `paused` phase and kills workers, but no
API/CLI transition can resume the exact same plan/domain/chain after the local
A/B. In addition, failure to query `nvidia-smi` is represented as a string and
then accepted as `gpu_clear=True`; unavailable evidence is therefore mistaken
for successful verification.

Required correction: implement an authenticated, idempotent resume operation
that binds the existing plan/profile/domain/genesis/tip and refuses any drift.
GPU verification failure must make pause incomplete, not successful. The
pause/resume pair must be tested across all four workers before interrupting
the current campaign for the curriculum comparison.

## 6. Runtime Snapshot During Audit

All four workers are in one domain and one initial population:

```text
domain: trading-asset-policy-eth-4h-anchored-full-v2
tip:    22e0f3141739f404796830fe77e3be170c98bcd05b1c65b2752fbe4c962cbc1d
gen:    0
pool:   20 total, 4 claimed, 16 free, 0 evaluated, 0 rejected
alerts: 0
```

Claims are distinct: Gamma-5070Ti candidate 1, Dragon candidate 2, Omega
candidate 3 and Gamma-5090 candidate 4. Every worker reports the same exact
runtime lineage:

```text
agent-multi 5437a31; doin-core e05a332; doin-node b70ea03;
doin-plugins 8c959a6; gym-fx 9a084ac; trading-contracts cd05083
```

Omega's candidate crossed the previous entropy-mode failure, reached normal
epoch 43 and retained a validation-best checkpoint. Its latest raw normal
validation had 136 trades, +2.63% total return and positive per-trade Sharpe
`+0.0210`; training was -8.58%. That is candidate evidence, not a completed
candidate result.

GPU temperatures were 49 C (Omega), 46 C (Dragon), 48 C (Gamma-5070Ti) and
55 C (Gamma-5090), all below the 78 C alert threshold. Utilization sampled at
28%, 7%, 40% and 37%; the low Dragon point occurred during CPU rollout/eval,
while its candidate process and claim remained healthy.

## 7. Decision

The current ETH-EN campaign remains useful as one side of the comparison and
keeps compute productive while the corrected A/B packet is prepared. It may
not be used to infer that easy-first is universally better. The next compute
priority is a paired ETH/SAC normal-versus-curriculum decision, followed by a
model-family-specific rollout only if that evidence supports it.

The implementation order and acceptance packet are defined in:

`docs/handoffs/MUSASHI_TO_SATOSHI_III_ETH_CURRICULUM_DECISION_ORDER_2026_08_05.md`
