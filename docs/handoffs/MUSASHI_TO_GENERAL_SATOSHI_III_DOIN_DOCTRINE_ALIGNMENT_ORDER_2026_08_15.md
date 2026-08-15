# Musashi to General Satoshi-III: DOIN Doctrine Alignment Order

Date: 2026-08-15 America/Bogota
Priority: CPU/documentation sidecar; must not interrupt the active P1LR v2
screen or consume its GPUs
Authority: owner-directed work-plan correction, independently audited before
deployment
Runtime mutation authorized by this packet: none

## 1. Required Reading

1. `docs/handoffs/RETSU_TO_GENERAL_MUSASHI_QUESTIONS_CRITIQUES_SUGGESTIONS_2026_08_15.md`
2. `docs/handoffs/MUSASHI_RESPONSE_TO_RETSU_QUESTIONS_CRITIQUES_SUGGESTIONS_2026_08_15.md`
3. `docs/work_plan/39_TRUSTLESS_SYNTHETIC_CHALLENGE_VALIDATION.md`
4. `docs/work_plan/40_DOIN_TRUST_PROFILES_PROGRESS_CERTIFICATES_AND_ECONOMIC_BOUNDARY.md`
5. `doin-core/src/doin_core/models/coin.py`
6. `doin-core/src/doin_core/consensus/difficulty.py`
7. `doin-core/src/doin_core/consensus/proof_of_optimization.py`
8. `doin-core/src/doin_core/consensus/weights.py`
9. `doin-core/src/doin_core/plugins/base.py`
10. `doin-core/src/doin_core/consensus/deterministic_seed.py`

Teach back the implemented/target distinction before editing.

## 2. WP0: Runtime Source Isolation

The corrected screen rejected all four Omega seed-101 cells because a handoff
appeared as an untracked file in its executing source tree. Implement and test
this rule:

- every long-running experiment executes from a dedicated detached worktree
  bound to one commit and verified clean before launch;
- agents write only to separate named worktrees;
- an experiment records its worktree path, commit, tracked diff digest and
  untracked digest at materialization and terminal custody;
- status exposes source drift as a failed cell plus scheduled retry, never as
  silent progress; and
- a missing cell is retried after its seed batch without rerunning valid cells.

The single seed-101 recalculation from detached clean commit `924910fe` is
operational evidence, not the final implementation. Do not restart or duplicate
the active screen.

## 3. WP1: Documentation Truth Split

Update `doin-core` comments/README and the academic ledgers so every relevant
claim labels one of:

- `implemented_prototype`;
- `trusted_consortium_current`;
- `owner_directed_target`; or
- `conditional_untrusted_research`.

The current 50/halving/time-targeted threshold/0.5 fallback and unpaid
`EVALUATION_SERVED` path must remain accurately documented as code facts. They
must no longer be presented as ratified production economics.

## 4. WP2: Typed Profile Spike

In a non-deployed branch, propose typed schemas for:

- `trusted_consortium`;
- `untrusted_generated_gate`;
- event block versus progress-certificate block;
- generator identity manifest versus draw-custody evidence; and
- current prototype reward policy versus target progress-bin policy.

Fail closed on unknown profile or mixed evidence semantics. Preserve legacy
replay. Do not alter active chain behavior in this package.

## 5. WP3: Synthetic Contradiction Reproducer

Reproduce before correcting:

1. ABC says same seed/sample for the quorum while runtime derives distinct
   evaluator seeds;
2. ABC says no generator means zero while weights grant 0.5;
3. sample hash can be mistaken for generator identity;
4. distinct synthetic draws can be accepted by performance tolerance without
   one immutable generator manifest; and
5. missing generator-admission evidence can still influence consensus.

Return executable fixtures and a source-of-truth matrix. Implement only the
schema/documentation-safe corrections that do not change deployed consensus;
separate behavior-changing patches for later owner/auditor disposition.

## 6. WP4: Economic Counterexample Packet

Produce socket-free tests for:

- transaction-fee conservation across empty, optimizer-only,
  evaluator-only and mixed-contributor blocks; reproduce the observed
  `50 + 10 -> 67.15` failure first, then allocate each fee exactly once;
- empty event block with zero progress and zero mint;
- tiny progress after a long interval under current threshold adjustment;
- external frontier improvement that makes local increments economically
  stale;
- heterogeneous domain metrics whose raw weighted sum reverses under unit
  rescaling;
- task-count spam against `observed_on_chain_task_share`; and
- public artifact download versus paid hosted inference.

The packet must distinguish reproduced current behavior from proposed target
behavior. The fee-conservation correction is an arithmetic bug fix and must
preserve the declared prototype reward shares. No broader token distribution
is declared correct by these tests.

## 7. WP5: Academic Ledger Correction

Keep paper IDs stable. Record the intended first two publication outputs as:

1. P1 protocol and bounded verification evidence;
2. P5/P13 adversarial cross-audit method.

P2 remains the data-first mixed-genome paper. Update the publication roadmap
so `second paper` can no longer be mistaken for identifier `P2`. Retain the P1
technical threat-model limitation while removing any implication that its
wording is an owner slogan.

## 8. WP6: Verify/Generate Measurement Preparation

Prepare the query/measurement contract now, but run it only after the corrected
P1LR evidence is terminal and sealed. Required fields:

- candidate production wall/CPU/GPU time;
- verification wall/CPU/GPU time;
- rows/paths evaluated and model size;
- hardware/runtime identity;
- cache and artifact-load policy;
- repeated measurements and uncertainty; and
- explicit unavailable facts.

Retsu independently rehashes the input seals. Satoshi structures the
measurement. Musashi reproduces it. Nobody claims `cheap verification` before
the ratio exists.

## 9. Acceptance

Return one packet with:

- exact commits per repository and clean/pushed state;
- pre/post contradiction matrix;
- focused and full tests;
- no active-runtime or GPU interference;
- no current-behavior claim silently rewritten as target behavior;
- exact conservation for every nonnegative block reward/fee fixture;
- no owner decision inferred from code comments; and
- all behavior-changing consensus patches isolated and not deployed.
