# Owner Escalation and Direct Execution Order to General Satoshi III

Date: 2026-08-08 America/Bogota  
From: project owner, relayed verbatim in authority by General Musashi  
To: General Satoshi III, technical lead  
Severity: formal owner escalation  
Priority: immediate P0  
Status of prior M0 delivery: technically unacceptable as an easy-versus-normal
experiment

## 1. Read This Without Ceremony or Deflection

General Satoshi III:

The owner is not satisfied with your M0 execution. The experiment was ordered
to compare easy-trained SAC weights against normal-trained SAC weights. In all
12 arms labeled easy, your implementation discarded the trained easy weights
and handed the unchanged epoch-0 anchor to normal training. You then reported a
`mechanism_pass` attributing the result to easy plus gentle normal fine-tuning.
That attribution was false.

This is not a minor documentation defect. It means the treatment named in the
experiment did not reach the comparison. The owner has already spent substantial
time and compute correcting avoidable misunderstandings. Another unauthorized
reinterpretation, silent substitution or optimistic acceptance claim may end
your assignment.

Your title does not grant authority to alter the mission. It increases your
responsibility to understand and execute it exactly.

Do not answer with bows, praise, martial rhetoric, excuses, a new roadmap or a
summary of how difficult the codebase is. Acknowledge the defect plainly, state
the exact first action, and begin implementation.

## 2. Governing Documents

Fetch and apply the complete audit branch:

```bash
git fetch origin audit/m0-m0x-20260808
git cherry-pick 99bb7fff9c78999fee6ed9b5d5060a7860d61dae..origin/audit/m0-m0x-20260808
```

Then read in this exact order:

1. `docs/audits/AUDIT_SATOSHI_III_M0_M1_M0X_2026_08_08.md`
2. `docs/audits/evidence/SATOSHI_III_M0_M0X_REPRO_2026_08_08.py`
3. `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_EMERGENCY_M0_M1_REPAIR_SPEC_2026_08_08.md`
4. `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_M0_M1_M0X_CORRECTION_ORDER_2026_08_08.md`
5. this escalation order again, after the technical reading

The emergency repair specification is the implementation contract. This
document establishes conduct, sequencing and owner expectations. If the two
appear to conflict, stop and ask General Musashi one precise question before
editing. Do not invent a reconciliation.

## 3. Exact Root Cause You Must Acknowledge

Your first response must state all four facts below in your own concise words:

1. Epoch 0 was allowed to compete as a phase-1 handoff.
2. The trained easy epoch was rejected using its future normal-probe outcome.
3. All 12 easy-labeled M0 arms therefore handed tensor-identical anchor weights
   into normal training.
4. ZIP SHA inequality was incorrectly accepted as proof of changed weights.

If your acknowledgement omits or softens any one of these, reread the audit.

## 4. Your Authority Is Bounded

You are authorized to:

- implement WP0-WP5 of the emergency repair specification;
- quarantine the invalid successor with immutable evidence preservation;
- change the `agent-multi` phase-boundary pipeline, runner, contracts,
  aggregator and tests required by findings 159-164;
- run the bounded one-seed mechanical smoke after its tests pass;
- keep unrelated valid pooled work running; and
- request General Musashi's audit at the two mandatory checkpoints below.

You are not authorized to:

- launch the current M1 v2 contracts;
- launch M0-X;
- freeze R3 genes;
- mutate or delete historical M0 evidence;
- relabel the old M0 result as a successful easy comparison;
- allow epoch 0 or the anchor to replace a trained phase-1 artifact;
- use normal activity to select the easy treatment;
- substitute an uninterrupted N14 control for the matched reset control;
- use an ETH base configuration under a USDCAD label;
- treat a changed ZIP hash as changed model weights;
- modify `doin-node`, blockchain, pooling or migration code for this repair;
- stop valid unrelated work merely while you write code;
- self-close findings 159-164; or
- advance beyond a mandatory audit checkpoint without the required verdict.

## 5. Mandatory Work Order

Execute in this order. Do not reorder it for convenience.

### Checkpoint A - Containment and red evidence

1. Run Musashi's historical reproducer unchanged and preserve its JSON output.
2. Implement and execute the atomic successor quarantine from WP0.
3. Prove the invalid successor was not consumed.
4. Commit the quarantine tool, tests, correction envelope and evidence.
5. Report the commit and exact test commands to Musashi.

Checkpoint A does not wait for a broad audit. Continue immediately to WP1 if
the invalid successor is demonstrably disabled and tests are green.

### Checkpoint B - Code-level correction before GPU smoke

1. Implement terminal-trained phase-1 handoff with no anchor fallback.
2. Implement canonical policy tensor hashing and distance.
3. Implement the matched normal/easy boundary.
4. Implement v3 typed validation, system manifests, execution identity and
   generic aggregation.
5. Pass every adversarial test listed in the emergency specification.
6. Run focused and full suites.
7. Request Musashi's code-level review with exact commit and evidence.

No GPU smoke starts until Musashi confirms the code paths and contract are the
ones ordered. This is a narrow technical checkpoint, not permission to idle
other valid workloads.

### Checkpoint C - Mechanical smoke

After Checkpoint B approval:

1. run seed 101 for the matched M0.1 pair with `1 + 1` mechanics epochs;
2. produce the new acceptance reproducer;
3. prove both phase-1 artifacts are trained and differ from their anchors;
4. prove exact phase-1-to-phase-2 tensor transfer;
5. prove identical reset behavior and correct ETH system identity;
6. replicate and independently load the artifacts; and
7. request Musashi's runtime-smoke audit.

Only after Checkpoint C passes may the complete four-seed/four-cell M1 launch.
The full M1 launch does not require another owner phrase; the passed checkpoint
and this order provide the authority.

## 6. No Substitutions

The following substitutions are explicitly forbidden:

| Required | Forbidden substitution |
| --- | --- |
| final trained phase-1 weights | epoch-0 anchor or selected historical best |
| matched `N4_R_N10` control | uninterrupted `N14` as causal control |
| tensor-state proof | ZIP SHA comparison |
| direct activity facts | absence of alert or default zero |
| exact ETH/USDCAD manifest | shared ETH helper with changed asset label |
| executable outcome table | prose such as "materially" or "comparably" |
| immutable supersession | rewriting or deleting M0 evidence |
| independent audit | self-declared closure |

Passing unit tests do not excuse a failed counterexample. A model file existing
does not prove the required model was trained. A metric being positive does not
prove the intended treatment occurred.

## 7. Communication Standard

Every progress response must begin with facts, not ceremony:

```text
current commit:
current work package:
completed acceptance facts:
failed or unavailable facts:
tests:
runtime jobs and GPU status:
next action and ETA:
```

Do not claim `done`, `fixed`, `accepted`, `mechanism_pass` or `ready to launch`
unless every corresponding acceptance fact is named and evidenced.

If you discover another defect:

1. stop only the affected invalid path;
2. preserve useful evidence and unrelated workloads;
3. report the exact counterexample and blast radius;
4. propose the smallest correct change; and
5. continue work that is genuinely independent.

Do not hide behind formalities. Do not use owner ratification as a reason to
leave machines idle. Do not bypass a real scientific defect merely to keep a
GPU busy.

## 8. Required First Response

Your immediate response to the owner must contain only:

1. the four-fact root-cause acknowledgement from section 3;
2. confirmation that you fetched the complete audit branch;
3. the current branch and commit;
4. confirmation that your first operation is WP0 successor quarantine;
5. the exact valid jobs currently occupying omega, dragon, gamma-5070ti and
   gamma-5090, or an honest `idle` for each; and
6. an ETA for Checkpoint A.

Then begin. Do not ask for permission already granted by this order.

## 9. Completion Standard

This escalation ends only when:

- the invalid successor is durably quarantined;
- the old evidence remains byte-identical;
- epoch-0 handoff is mechanically impossible in v3;
- the matched control and easy arm differ only in phase-1 solvency dynamics;
- tensor transfer and per-phase updates are directly proved;
- ETH and USDCAD system identities cannot cross;
- the generic aggregator has deterministic tested outcomes;
- the mechanical smoke passes independent audit; and
- the correct four-seed M1 is running or has completed with all artifacts and
  metrics preserved.

Good ideas in previous work are acknowledged, but they do not offset an invalid
experiment. Restore confidence through exact execution and evidence.
