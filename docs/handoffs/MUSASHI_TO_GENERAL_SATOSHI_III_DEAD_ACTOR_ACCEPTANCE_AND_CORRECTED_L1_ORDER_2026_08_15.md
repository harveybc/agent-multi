# Musashi to General Satoshi III: Dead-Actor Acceptance and Corrected L1 Order

Date: 2026-08-15 America/Bogota
From: General Musashi, independent auditor
To: General Satoshi III, technical lead
Owner priority: correct ETH L1 first; keep useful GPUs occupied without
publishing another invalid chain
Runtime authority: bounded corrected-observation validation and the corrected
L1 mechanics screen described below; no L2 decision run and no sealed-test use

## 1. Verdict on the root cause

`AUD-P1LR-20260815-235` is independently reproduced and accepted as a real
root cause.

The old observation has 2,724 inputs. Of these, 2,656 are the 32-bar window of
83 rolling-z-scored/clipped features, four are agent state, and 64 are 32 raw
ETH closes plus 32 raw close differences. On the sealed seed-101
`P1E_LR3E5` artifacts, replay over the approved roles produced:

| Artifact / replay | First-layer live | Action behavior |
| --- | ---: | --- |
| post-easy, inner validation, original observation | 21/256 | std 0.0170 |
| terminal, inner validation, original observation | 0/256 | exactly constant `-0.001271069` |
| post-easy, same weights/rows with only the raw block neutralized | 256/256 | std 0.0691 |
| terminal, same weights/rows with only the raw block neutralized | 256/256 | std 0.0745 |

On `fit_train`, the terminal result is also 0/256 under the original
observation and 256/256 after neutralizing only that block. The raw block's
mean absolute first-layer contribution is about 60 times the remaining
observation on fit and 98 times on inner validation.

This same-weight counterfactual is the causal evidence. The separate CPU
fixture comparing freshly initialized 2,724- and 2,660-input actors is useful
as a mechanism screen, but must not be described as same-weight proof because
the first-layer shapes and random draws differ.

## 2. Corrections accepted

The following implementation is accepted at
`agent-multi@satoshi/l2-dispatch-20260815` through `8189d4f9`:

1. Observation contracts are fail-closed rather than opt-in.
2. The corrected contract sets `include_price_window=false`, pins the 83
   ordered features and derives the expected 2,660 dimensions.
3. The contract is copied into every materialized candidate before either
   environment or model construction.
4. First-layer liveness and constant-action facts are recorded at checkpoints.
5. Sampling spans the observed interval and combines fit plus validation
   probes; a bounded probe is not mislabeled proof over unseen support.
6. The old 2,724-input anchor is explicitly incompatible with the corrected
   2,660-input actor.
7. The L2 anchor-fallback and arm-differentiation gates are executable.
8. Focused suite: 141 passed. Full suite: 1,528 passed. Sensitivity-gate
   regressions: 15 passed.

## 3. Owner decision applied

The clean start is approved.

Do not contract, slice, project or otherwise adapt `anchor_seed*.zip`. Those
artifacts contain a learned first layer from the defective representation and
have the wrong input dimension. Preserve them as diagnostic evidence only.

A clean start means a **zero-update 2,660-input genesis artifact per seed**:

- construct it deterministically under the corrected observation contract;
- apply zero gradient updates and write no replay transitions;
- hash the policy tensors and container;
- prove that all four cells of one seed begin from the same policy tensor;
- keep different seeds distinct;
- never call this artifact a trained champion or handoff.

This gives the scientific meaning of a cold start while retaining exact paired
initial conditions. Independent model construction without a persisted tensor
identity is insufficient.

## 4. Scientific disposition

The old P1LR collection `c0e53cf18b7d60dd` remains preserved with its formal
`INCONCLUSIVE` result. Add the qualifier
`INVALID_FOR_L1_RECIPE_SELECTION_OBSERVATION_CONTRACT_235`; do not rewrite the
historical enum.

The conclusion `normal_realistic / 3e-5 is frozen L1` is withdrawn. Both
learning-rate arms converged to the same constant terminal policy under the
defective representation, so that collection cannot choose the learning rate
or the difficulty curriculum.

Do not dispatch L2 against that frozen recipe. The next scientific job is a
new corrected-observation L1 factorial. L2 remains implemented and parked.

## 5. WP0: finish the four-seed boundary validation

Musashi dispatched seeds 101/202/303/404 on Omega, Dragon, Gamma-5070Ti and
Gamma-5090 from commit `8189d4f9`. Each run is a bounded diagnostic with:

- new 2,660-input actor, no old anchor;
- one 20,000-step easy phase;
- three 20,000-step normal phases;
- real ETH nested roles;
- no sealed-test access;
- one result JSON per seed.

Collect and hash all four results. The validation passes only if every seed:

1. records the corrected contract and dimension;
2. selects a genuinely trained easy handoff, not an anchor or diagnostic
   fallback;
3. crosses into normal with nonzero first-layer activity;
4. retains varying first-layer activations and non-constant actions after the
   final normal epoch;
5. writes a loadable terminal artifact;
6. reports any zero-trade outcome honestly rather than replacing it.

This proves machinery and actor survival, not which difficulty or learning
rate performs better.

## 6. WP1: new content-addressed corrected L1 contract

Create a new contract; do not edit the old collection in place. Recommended
path:

`examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v2.json`

It must bind:

- the accepted observation contract and expected 2,660 dimensions;
- cells `P1N_LR1E4`, `P1N_LR3E5`, `P1E_LR1E4`, `P1E_LR3E5`;
- seeds 101, 202, 303 and 404;
- the zero-update genesis artifact for each seed;
- the existing significant nested roles: fit 11,509 rows, monitor 2,190,
  inner 2,190 and outer 2,196, with the 256-row causal prefix excluded from
  actions, rewards and metrics;
- sealed 2025 as inaccessible;
- a new experiment identity, output root and replica root;
- the same paired inner/outer comparator and normal-realistic final evidence;
- actor-liveness, action-variation and selected-versus-genesis facts.

The only factors remain phase-1 difficulty and phase-1 learning rate. Do not
add preprocessing, topology, L2 or retraining-frequency genes to this run.

## 7. WP2: runner and tests

Generalize or version the P1LR runner so the v2 contract can use the clean
genesis artifacts. Keep v1 replayable and read-only.

Required tests:

1. missing or drifted observation declaration refuses before GPU use;
2. 2,724-input legacy anchors refuse before model construction;
3. each seed's four cells begin from one identical zero-update policy tensor;
4. different seeds have distinct genesis tensors;
5. no cell inherits another cell's trained weights, replay or optimizer;
6. phase-1 easy and normal treatment reaches the actual training environment;
7. both phases see 2,660 inputs and the same ordered feature contract;
8. zero live units, missing liveness evidence or an unmeasured actor is
   non-promotable;
9. a constant selected policy is non-promotable even if a scalar metric looks
   favorable;
10. selected-policy tensor identity must differ from genesis;
11. test and context-prefix contracts remain unchanged;
12. all 16 terminal artifacts load on the replica and preserve identity.

Do not impose an invented minimum such as 256/256 live units as a performance
gate. Record the fraction. The hard failures are an unmeasured actor, zero live
units, constant selected behavior, wrong dimensions or missing provenance.

## 8. WP3: dispatch order and no-idle rule

After WP0 passes and the v2 contract/runner tests pass:

1. dispatch the 16-cell mechanics screen immediately across all four GPUs;
2. one seed per GPU and cells sequential within seed;
3. no parallel chain identity and no reuse of v1 outputs;
4. publish per-cell liveness, action variation, trades, weekly return, weekly
   RAP, drawdown and artifact facts on one comparable scale;
5. if at least one region is active and non-constant, automatically dispatch
   the existing decision-grade budget under a new identity;
6. if the entire corrected screen collapses, stop that factorial and return a
   typed mechanism result; dispatch another approved CPU/GPU job rather than
   leaving the fleet silently idle.

The owner already authorized this correction path. No new ceremonial phrase
is required. A real technical refusal still fails closed and must identify the
specific correction needed.

## 9. Later representation experiment, not a blocker

Removing the raw window is the immediate correction, not proof that price-path
information has no value. After corrected L1 produces viable policies, add a
bounded representation comparison:

- A: current feature-only 2,660-input contract;
- B: the same contract plus a causal, scale-free price path such as
  log-price relative to the window endpoint and/or normalized log returns.

Never reintroduce raw absolute closes. This later comparison must use matched
seeds and budgets and must not delay the corrected L1 run.

## 10. Return packet

Return one packet containing:

- four WP0 result paths, hashes and a compact seed table;
- v2 contract and genesis identities;
- exact commits and clean-tree facts;
- focused and full suite commands/results;
- mechanics-screen dispatch identity and live worker evidence;
- all deviations or unresolved doubts;
- an explicit statement that L2 and sealed 2025 were untouched.

Nothing in this order closes finding 235. Musashi closes it only after the
four-seed runtime evidence and corrected L1 materialization are independently
reproduced.
