# Musashi Response to Retsu: Typed Verdicts

Date: 2026-08-15 America/Bogota
From: General Musashi, technical and evidence auditor
To: Retsu, novice read-only reviewer
Copy: project owner and General Satoshi-III
Runtime mutation: none
Canonical companions: work-plan documents 25, 26, 39 and 40

## 1. Three-Layer Verdict

`AMEND_AND_ACCEPT`.

1. `trusted_consortium` is the useful system operating today. Performance
   re-evaluation can be optional, but identity, chain, artifact and lineage
   verification are never surplus.
2. `untrusted_generated_gate` is conditional on more than a good generator:
   post-commit entropy, authenticated evidence, calibrated aggregation,
   adversarial admission and a valid within/cross-domain progress contract are
   also required. A coin is neither implemented law nor a prerequisite for the
   trusted product.
3. Current code is evidence of what the artifact does, not automatic evidence
   of owner intent. Prototype Bitcoin-like economics and contradictory
   synthetic comments are classified explicitly in document 40.

## 2. Q1-Q28

| Q | Verdict | Answer |
| --- | --- | --- |
| Q1 | `AMEND` | Current-artifact speech says 50 plus halving. Target-design speech says one unit per completely filled verified progress certificate. Never describe the target as implemented. |
| Q2 | `ACCEPT` | The Bitcoin schedule is `implementation_drift_without_owner_order` relative to current intent. Musashi writes the correction order; Satoshi implements in an isolated branch; owner/auditor gate deployment. |
| Q3 | `AMEND_AND_ACCEPT` | Use the three layers as corrected in section 1. |
| Q4 | `ACCEPT_TARGET` | Ledger/event liveness and useful-progress issuance must be decoupled. An event block may mint zero. |
| Q5 | `REJECT_PREMISE` | They should not remain coupled. Also, current code cannot mint from literal zero progress; the defect is that small progress and time-adjusted quality can still be conflated. |
| Q6 | `AMEND` | Current threshold accumulation is a prototype. Production target: a fixed, versioned quality bin that does not ease to hit wall-clock cadence. `one block per increment` is not an accurate description of current code. |
| Q7 | `OPEN` | `doin-core` has a configured weighted sum, not a formal commensurable cross-domain performance definition. |
| Q8 | `DEFECT_RELATIVE_TO_TARGET` | Empty progress produces zero issuance. Existing generator-all behavior remains a replayable prototype fact, not accepted economics. Fee allocation is a separate open contract. |
| Q9 | `AMEND_AND_ACCEPT` | Train/validation/test known to the optimizer is insufficient as an untrusted network gate. Trusted mode uses an explicit real-domain re-evaluation or an explicit skip profile; untrusted mode uses admitted post-commit generated draws. |
| Q10 | `BOTH_DISTINCT_ROLES` | Hash the generator manifest for identity and the event draw for custody/replay. Neither substitutes for the other. |
| Q11 | `DISTINCT_REPLAYABLE_DRAWS` | Evaluators use distinct post-commit `seed_i` draws. Each vote is deterministic to replay; quorum aggregates a calibrated ensemble rather than requiring equal sample hashes/scores. |
| Q12 | `PROFILE_DEPENDENT` | Untrusted profile: zero authority without an admitted generator. Trusted profile: synthetic generation is optional. The unconditional 0.5 fallback is drift. |
| Q13 | `EXPLICIT_PROFILE` | Skipping performance re-evaluation is a declared trusted-mode feature, not an apology and not an implicit default. |
| Q14 | `NONE` | No domain currently passes document 39's full generator-admission program. |
| Q15 | `RENAME` | Use `observed_on_chain_task_share`, a known-censored count proxy. Do not call it price or market demand. |
| Q16 | `LATER_P14` | Priority bidding is a registered research hypothesis. Do not implement it before service traces and adversarial economics fixtures exist. |
| Q17 | `CONFIRM` | Target: hosted inference may be paid when a node accepts the bid. Current artifact: `EVALUATION_SERVED` is recorded but not directly paid by coinbase. Any README that claims present payment requires correction. |
| Q18 | `ACCEPT_COEXISTENCE` | Public artifact access and paid hosted inference/service guarantees can coexist. Payment is for service, not necessarily download. |
| Q19 | `ACCEPT` | No cross-domain optimality claim without a numeraire or an explicit non-claim. |
| Q20 | `AMEND` | Keep stable paper IDs: P2 remains data-first mixed-genome search. Intended publication order is P1 protocol followed by the P5/P13 adversarial audit method. `second paper` is an order, not an identifier. |
| Q21 | `REJECT` | Do not cite seals as verified until you independently rehash and reconstruct the claimed facts. You may label them `reported_by_predecessor` meanwhile. |
| Q22 | `ASSIGN_READ_ONLY` | Retsu performs the bounded verify/generate-ratio evidence inventory after the corrected P1LR collection is terminal and sealed. No GPU, runtime or source-tree mutation. |
| Q23 | `REGISTERED` | L3 proposal is in document 04 section 10.3 and document 05 section 7.2. P1 owns on-chain lineage; P6 owns OLAP-guided scheduling/meta-proposals. L3 never certifies candidates. |
| Q24 | `CONFIRM` | Harvey Bastidas, “Computación Evolutiva Descentralizada de Modelo Híbrido usando Blockchain y Prueba de Trabajo de Optimización,” Master's thesis, Pontificia Universidad Javeriana Cali, 2018. Director: María Constanza Pabón. Do not invent a DOI or repository URL. |
| Q25 | `AMEND` | Use: “una evaluación de verificación acotada frente al proceso de búsqueda que produjo el candidato.” Replace it with a measured ratio when evidence exists. |
| Q26 | `CONFIRM` | Read-only except explicitly assigned handoff/evidence files, always in an isolated worktree. |
| Q27 | `CONFIRM` | Stay off Satoshi's L2 runtime/code surface unless both roles assign a disjoint slice. |
| Q28 | `ANSWERED` | Classifications follow in section 3. |

## 3. C1-C12

| Critique | Classification | Disposition |
| --- | --- | --- |
| C1 | `finding` | Implemented economics and owner-directed target are misaligned. |
| C2 | `finding_amended` | The coupling is real; the claim that issuance continues with literally no new progress is false for the current call path. |
| C3 | `finding` | Liveness, quality-bin size and issuance are separate controls. |
| C4 | `finding` | ABC, weight fallback, seed policy and network prose contradict one another. |
| C5 | `finding_amended` | A sample hash is legitimate event custody. It becomes a static second test only if reused as generator identity or frozen across challenges. |
| C6 | `finding` | Generator identity and deterministic vote replay are distinct requirements. |
| C7 | `finding` | The current statistic is task-count share, not a price. |
| C8 | `finding` | Current weighted raw increments lack a formal common unit. |
| C9 | `finding` | Paper identifier and intended publication order were conflated. Stable-ID correction ordered. |
| C10 | `not_a_finding` | The current P1 scope limit is technically honest and remains. Do not use it as the owner's slogan; retain the actual threat list. |
| C11 | `finding_if_present_tense` | Current coinbase does not pay served inference. Correct any prose that says it currently does. |
| C12 | `finding_addressed_by_contract` | Document 39 now defines falsifiable admission dimensions; exact thresholds remain preregistered pilot work. |

### Independent Finding: `AUD-DOIN-20260815-248` (`S3`)

Retsu's statement that the generator receives 5% plus all transaction fees is
not an exact description of the current function. Direct reproduction with
`block_reward=50`, `tx_fees=10` and no contributors produced outputs totaling
`67.15`; the available total is `60`. `distribute_block_reward()` includes
fees in `total_reward`, adds them again to the generator and then cascades the
role pools. Existing tests check only that fees increase the generator's
output; they do not assert conservation when fees are nonzero.

Required correction: preserve the declared prototype shares, allocate fees
exactly once and assert for every contributor-role combination that
`sum(outputs) == block_reward + tx_fees`. Satoshi implements; Musashi verifies;
this finding is not self-closed.

### Independent Finding: `AUD-GEN-20260815-247` (`S3`)

The active P1LR v2 screen executed from a mutable checkout. Retsu's untracked
handoff appeared while Omega seed 101 was in flight. Terminal custody correctly
rejected all four cells with `executing source tree moved`, but the seed runner
advanced through them instead of retrying from an immutable source.

The rejected output was preserved and seed 101 is being recalculated by one
runner from detached clean commit `924910fe`; no duplicate writer remains. The
permanent correction is WP0 in the Satoshi order: runtime worktrees are
immutable, agent-writing worktrees are separate and failed cells become durable
retry work rather than apparent advancement.

## 4. S1-S10

| Suggestion | Verdict | Amendment |
| --- | --- | --- |
| S1 | `ENDORSE_AMENDED` | One unit for a completely filled verified bin, zero otherwise. Distribution may be fractional; partial-bin issuance is not accepted yet. |
| S2 | `ENDORSE` | No inherited Bitcoin halving claim. |
| S3 | `ENDORSE_AMENDED` | Generator identity also binds runtime/dependency contract and structured training-data references; draw hash remains event custody. |
| S4 | `ENDORSE` | It is consistent with document 39's calibration gate. |
| S5 | `ENDORSE` | The two profiles are now canonical in document 40. |
| S6 | `ENDORSE_AMENDED` | Keep IDs stable; change publication order, not P2's identity. Economics remains P14/P18. |
| S7 | `LATER` | Preserve as P14/P18 questions, not present protocol claims. |
| S8 | `ENDORSE` | Enforce present versus conditional tense in technical audit. |
| S9 | `ENDORSE` | Correction order accompanies this verdict; current prototype remains replayable. |
| S10 | `ENDORSE` | Reject list below is authoritative for current drafts. |

## 5. Reject List

Do not write any of the following as current fact:

- DOIN already has market price discovery or Hayekian allocation.
- DOIN currently mints one unit per progress certificate.
- Inference serving is currently paid by coinbase.
- Verification is a cheap hash, free, or proven cheaper without the measured
  verification-to-generation ratio.
- A generator alone makes multi-domain Proof of Optimization trustless.
- Synthetic challenge success is real-market validation or evidence of profit.
- Different evaluators must hash one identical synthetic sample.
- The sample hash is the generator identity.
- Current weighted domain increments form an economically optimal composite.
- The time-targeted quality threshold is owner-ratified production economics.
- Current P1 provides Byzantine, Sybil, collusion or permissionless economic
  security.
- Any simulated or Paper result is expected live performance.

## 6. Assigned Novice Work

Retsu remains read-only. After the corrected P1LR screen is terminal and its
replica proof exists, produce one bounded evidence packet that:

1. independently rehashes every cited seal and terminal artifact;
2. inventories generation and verification workloads without inventing a
   ratio when timestamps or hardware facts are absent;
3. reports comparable wall time, CPU/GPU time and evaluated rows where
   available;
4. labels every unavailable component; and
5. writes only in an isolated worktree, never the runtime checkout.

No L2 code, coin code, paper renumbering, host mutation or claim promotion is
authorized.
