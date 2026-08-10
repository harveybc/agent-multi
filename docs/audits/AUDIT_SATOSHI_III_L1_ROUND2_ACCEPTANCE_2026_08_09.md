# Independent Audit: L1 Round-2 Acceptance

Date: 2026-08-09 America/Bogota  
Auditor: General Musashi (Codex), independent auditor during the role swap  
Subject: `agent-multi@edcdb044` (report-only tip)  
Executable fleet revision: `agent-multi@4ca7361c`  
Subject branch: `satoshi/m0-aggregation-hardening`  
Runtime mutation by this audit: none

## 1. Verdict

**DO NOT ACCEPT THE ROUND-2 PACKET YET. DO NOT START DECISION IDENTITY
`dce2903ce0d25ca5`.**

The correction is close and most of it is sound. The report-only delta from
`4ca7361c` to `edcdb044` contains only the return packet. Omega, Dragon and
Gamma are clean at the same full executable commit `4ca7361c`; all three use
`gym-fx@efa49160`. The smoke produced 16/16 immutable mechanics records, every
record is `mechanics_smoke` and `decision_eligible=false`, the 96 focused tests
and complete 885-test suite pass, and the restart evidence contains twenty
new failed-attempt directories plus the planted partial while preserving a
single successful attempt under its separate identity.

Acceptance is blocked by five independently reproduced defects. Most
importantly, aggregation writes its result inside the sealed input tree. The
published source/replica digest is `bdb644e6...`, but the source tree now
rehashes to `5916ef72...` after aggregation. A separate aggregator CLI also
bypasses the collection/replica gate. Gamma's two assigned GPU UUIDs are only
checked for visibility; neither deployed environment binds
`CUDA_VISIBLE_DEVICES`, so the 5070 Ti and 5090 assignment is not enforced.
The exact normal contract omits the required explicit financing treatment,
and phase-1 realized epochs count the epoch-0 baseline as training.

These are correction-sized defects, not reasons to redesign the experiment.
Apply the accompanying bounded order, rerun one corrected mechanics smoke,
and return the evidence. No new owner phrase is required.

## 2. Findings 188-195

| Finding | Independent disposition | Evidence |
| --- | --- | --- |
| 188 | verified corrected, pending closure | real CLI exit tests pass; `SEED_FAILED` restarted repeatedly into preserved attempts and later completed under a corrected disposable identity |
| 189 | verified corrected, pending closure | resolver confines record paths to the sealed root; source-deletion acceptance tests pass |
| 190 | partially verified | replica is mandatory and the pre-aggregation whole-tree digests matched; findings 196-197 still defeat the end-to-end authority claim |
| 191 | verified corrected, pending closure | runner binds the manifest agent and explicitly validates the curriculum wrapper |
| 192 | verified corrected, pending closure | protected entries are applied and plugin failure cannot submit an unprotected fallback |
| 193 | verified for spread/slippage/min-equity; broader exact-system gate remains open | positive spread, explicit slippage and min-equity are applied; finding 199 covers omitted financing treatment |
| 194 | verified corrected, pending closure | manifest v2 was generated from clean `80390bf1`; materialization refuses dirty provenance |
| 195 | verified for N/E mode label; truthful epoch count remains open | normal and easy labels now match execution; finding 200 covers the baseline-count error |

## 3. New Findings

### AUD-F1-20260809-196 (S2): aggregation mutates the sealed input tree

`collect_l1_factorial.main()` passes the sealed parent to
`write_aggregation()`, which writes
`sealed/<experiment>/aggregation/l1_factorial_aggregation.json`. The tree was
sealed and replicated at digest `bdb644e6...`; after the write, the exact
source tree rehashes to `5916ef72...`, while Dragon's replica remains
`bdb644e6...`.

Impact: the published seal is no longer immutable and the asserted
source/replica equality is false after the very operation it authorizes.

### AUD-F1-20260809-197 (S2): direct aggregation bypasses seal and replica authority

The public `aggregate_l1_factorial.py` CLI accepts only experiment/output-root
arguments and calls `aggregate()` directly. It does not require or validate a
collection manifest, a sealed-tree digest, or the mandatory replica proof.

Impact: the same decision artifact can be produced through an unverified path,
making the mandatory replica gate procedural rather than enforced.

### AUD-F1-20260809-198 (S2): assigned Gamma GPU UUIDs are not execution bindings

The launcher verifies that the assigned UUID appears in `nvidia-smi -L`, but
does not require or set `CUDA_VISIBLE_DEVICES`. Both deployed Gamma files are
exactly `L1_EXTRA_ARGS=--smoke`; neither binds its assigned GPU. The model
config uses CUDA/auto selection, so both processes may select the first visible
device even though the contract names different UUIDs.

Impact: four-worker concurrency and maximum-capacity claims are unproved, and
the decision run can overload one Gamma GPU while leaving the other idle.

### AUD-F1-20260809-199 (S3): normal financing treatment remains implicit

The correction order required commission, spread, slippage, financing,
margin/min-equity, leverage and protection to be explicit. Manifest v2 and
`validate_normal_contract()` contain no financing binding or validation. The
current Backtrader path does not apply overnight financing, so the honest
binding is an explicit disabled/unsupported treatment, not silence.

Impact: the exact normal-system record remains incomplete and future parity
analysis cannot distinguish deliberate zero financing from omission.

### AUD-F1-20260809-200 (S3): realized phase-1 epochs include baseline telemetry

Every smoke arm requested and trained one phase-1 epoch, with 500 gradient
updates. The record reports `phase1_realized_epochs=2` because
`easy_epochs_run=len(history)` counts epoch 0 baseline telemetry plus epoch 1
training. The same off-by-one will propagate into decision records.

Impact: compute accounting and early-stopping analysis overstate trained
phase-1 epochs and cannot be trusted without reconstructing history rows.

## 4. Reproduction

Evidence:

- `docs/audits/evidence/repro_runs/MUSASHI_L1_ROUND2_ACCEPTANCE_REPRO_2026_08_09.py`
- `docs/audits/evidence/repro_runs/MUSASHI_L1_ROUND2_ACCEPTANCE_REPRO_2026_08_09.json`

Independent results:

- exact executable identity: `4ca7361c` on Omega, Dragon and Gamma, all clean;
- dependency identity: `gym-fx@efa49160` on all three hosts;
- prior round-1 reproducer: five corrected counterexamples no longer
  reproduce; the immutable absolute-path survivor is handled by the resolver;
- round-2 reproducer: findings 196-200 reproduce; the restart-attempt
  counterexample does **not** reproduce;
- focused suite: **96 passed**;
- full suite: **885 passed**, two declared sklearn convergence warnings;
- current L1 workers: clean terminal, no residual factorial process;
- GPU sample: Omega 42 C, Dragon 31 C, Gamma 25/41 C; no training load at the
  acceptance gate.

## 5. Decision

The owner should **not ratify acceptance yet**. Satoshi may implement the
bounded correction order immediately under standing authorization. The next
acceptance packet must carry a new manifest, smoke identity and predicted
decision identity; `dce2903ce0d25ca5` is retired before launch because these
decision-bearing corrections change the code/system contract.

