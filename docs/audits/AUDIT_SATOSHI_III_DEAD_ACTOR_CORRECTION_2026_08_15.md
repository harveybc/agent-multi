# Audit: SAC Dead-Actor Root Cause and Observation-Contract Correction

Date: 2026-08-15 America/Bogota
Auditor: General Musashi
Subject branch: `satoshi/l2-dispatch-20260815`
Accepted subject commit: `8189d4f93dd7ac0b3cbd26d0cb2c2c714b60a2ad`
Finding: `AUD-P1LR-20260815-235`
Runtime scope: Paper/offline training only; sealed 2025 untouched

## 1. Verdict

The root cause is independently reproduced. The implementation correction is
accepted. The corrected observation survives a real easy-to-normal SAC
boundary in six independent seeds.

The old P1LR result cannot select the L1 recipe because every terminal policy
used the defective observation and the representative sealed terminal actor is
constant with a fully inactive first hidden layer. The historical enum remains
`INCONCLUSIVE`; it gains the qualifier
`INVALID_FOR_L1_RECIPE_SELECTION_OBSERVATION_CONTRACT_235`.

L2 must not run with the old frozen L1 recipe or its 2,724-input anchors. The
next decision experiment is the corrected-observation L1 factorial ordered in
`MUSASHI_TO_GENERAL_SATOSHI_III_DEAD_ACTOR_ACCEPTANCE_AND_CORRECTED_L1_ORDER_2026_08_15.md`.

Finding 235 is `independently_verified_correction_pending_corrected_L1`.

## 2. Same-weight sealed-artifact reproduction

Subject artifact family:

`p1_difficulty_lr_factorial_20260811_v1_decision/c0e53cf18b7d60dd/seed101/P1E_LR3E5/attempt-c47b5d7c2ac7f5c7-01`

The replay uses the same sealed weights and the same approved rows. The
counterfactual neutralizes only the 32 raw closes and 32 raw differences.

| Role / artifact | Original 2,724 inputs | Raw block neutralized |
| --- | --- | --- |
| fit / post-easy | 37/256 live; action std 0.0428 | 256/256 live; std 0.0745 |
| fit / terminal | 0/256 live; constant `-0.001271069` | 256/256 live; std 0.0802 |
| inner / post-easy | 21/256 live; action std 0.0170 | 256/256 live; std 0.0691 |
| inner / terminal | 0/256 live; constant `-0.001271069` | 256/256 live; std 0.0745 |

On fit, the raw block's mean absolute contribution to layer one is about 60.1
times the remaining observation; on inner validation it is about 98.5 times.
This is the decisive causal evidence because weights, observations and all
other inputs remain fixed.

## 3. Independent CPU mechanism fixture

Command:

```text
python tools/dead_actor_observation_fixture.py --updates 3000
```

Evidence:

`~/.local/share/agent-multi/dead_actor_observation_fixture_20260815/seed101_updates3000.json`

SHA-256:

`e369ee33b32976855a2d8bc195a42cb2706aae06078b42b69536df5c8ea0b81a`

| Observation | At initialization | After 3,000 updates |
| --- | --- | --- |
| raw price window, 2,724 dims | 145/256 live; one action; std 0 | 122/256 live; 23 newly inactive |
| feature-only, 2,660 dims | 256/256 live; 1,024 actions; std 0.07310 | 256/256 live; none newly inactive; std 0.07020 |

All 111 units inactive at raw-window initialization remained inactive. This
fixture uses real ETH observations and a real SAC actor, but the two arms have
different input shapes and therefore different initial tensors. It supports
the mechanism; it is not mislabeled as same-weight proof.

## 4. Code verification

Accepted behavior:

1. Observation-contract declaration is fail-closed.
2. `include_price_window=false`, 83 ordered features and expected dimension
   2,660 are bound before environment/model construction.
3. Fit and validation probes span their intervals rather than sampling only
   the opening rows.
4. Liveness and constant-action facts are recorded at checkpoints.
5. A bounded probe is not described as proof over unseen observations.
6. A 2,724-input anchor cannot warm-start or be contracted into the corrected
   actor.
7. L2 candidates carry their observation declaration explicitly.
8. Anchor-fallback and cross-arm treatment-realization gates execute.

Tests reproduced:

- actor/observation/L2 focused suite: `141 passed`;
- complete `agent-multi` suite: `1,528 passed`, two unrelated sklearn
  convergence warnings;
- sensitivity-gate suite after dynamic-host false-positive correction:
  `15 passed`;
- outgoing-commit sensitivity scan: clean.

## 5. Six-seed GPU boundary validation

Contractual seeds 101/202/303/404 ran on the four fleet GPUs. Seeds 505/606
were supplemental attempts to falsify the result while released GPUs would
otherwise be idle. Every run used:

- real ETH nested roles: 11,509 fit, 2,190 monitor, 2,190 inner and 2,196 outer
  scored rows;
- 256 causal context rows per validation role;
- one 20,000-step easy phase and three 20,000-step normal phases;
- a clean 2,660-input actor;
- no legacy anchor and no sealed-test access.

| Seed | Easy inner live / std | Easy probe trades | Normal live counts | Normal action-std range | Elapsed s |
| ---: | --- | ---: | --- | --- | ---: |
| 101 | 256/256 / 0.02971 | 2 | 256, 256, 256 | 0.01483–0.02676 | 1,042.3 |
| 202 | 256/256 / 0.02578 | 2 | 256, 256, 256 | 0.01359–0.01853 | 1,026.3 |
| 303 | 256/256 / 0.02220 | 2 | 256, 256, 256 | 0.01205–0.02121 | 559.1 |
| 404 | 256/256 / 0.02077 | 5 | 256, 256, 256 | 0.01536–0.02080 | 460.2 |
| 505 | 256/256 / 0.03117 | 2 | 256, 256, 256 | 0.01322–0.03318 | 555.9 |
| 606 | 256/256 / 0.02837 | 14 | 256, 256, 256 | 0.01466–0.02943 | 456.9 |

All 18 normal-epoch probes report 256/256 varying units and
`constant_policy=false`. All six easy handoffs are trained-epoch artifacts
classified `VIABLE`, not passthrough anchors or diagnostic fallbacks.

The four contractual terminal load proofs:

| Seed | Terminal SHA-256 | Loaded observation/action shapes |
| ---: | --- | --- |
| 101 | `3cc49a2b53ac30dcadba02db96dc128f7b61059dde0242448926c38ff35e26cd` | `(2660,) / (1,)` |
| 202 | `36dd4d85488710ec6beade49f135a2dafb738008981d07a163836e61a864ae08` | `(2660,) / (1,)` |
| 303 | `9c74007e7b79e524edaf70ecdf8a217ab524516062d94851687944e01287761f` | `(2660,) / (1,)` |
| 404 | `e3d30363e989a5d4dc7869477e20bea0f0b450aa65b653cecdae749531b9534d` | `(2660,) / (1,)` |

The consolidated contractual JSON/meta evidence is stored under:

`~/.local/share/agent-multi/observation_boundary_validation_collection_20260815/`

## 6. What this result does and does not establish

Established:

- the raw-window representation caused the observed actor pathology;
- the corrected representation reaches both phases;
- actors remain internally responsive through three normal epochs;
- the correction works across six initializations and four GPUs;
- the new terminal format is loadable and dimensionally correct.

Not established:

- easy is better than normal;
- `3e-5` is better than `1e-4`;
- a profitable or sufficiently active policy has been selected;
- L2 should start;
- the raw price path contains no useful information after a causal,
  scale-free transformation.

Several normal epochs had few or zero scored trades even while the actors
remained alive. That is not hidden and is why actor liveness is a diagnostic,
not a trading-performance objective. The corrected L1 factorial must now
measure difficulty and learning rate from valid policies and normal-realistic
evidence.

## 7. Required next action

General Satoshi III implements the new content-addressed P1LR v2 contract,
zero-update genesis artifacts and 16-cell corrected mechanics screen. The
screen dispatches without another ceremonial approval after tests pass. L2
remains parked until corrected L1 produces a defensible recipe.

Retsu remains read-only and receives the root-cause reproduction as his first
evidence drill.
