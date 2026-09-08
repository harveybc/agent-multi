# Musashi to General Satoshi: T2 C57-C65 and Model M3

Date: 2026-09-07

Authority: owner requested all executable lanes remain moving. This order
authorizes bounded CPU work only. It grants no venue, live, key, B4 mutation,
sealed-2025, checkpoint promotion or additional GPU authority.

Priority: P0 T2 runtime correction, then P1 M3 CPU calibration. B4 v7 remains
under its existing service and must not be modified, restarted or used as a
source checkout.

## 1. Inputs

P0 starts from the pushed candidate tip
`4e5f19dccd24d6fb2cd7046ad173070bfbf98617` and consumes:

- `docs/audits/MUSASHI_AUDIT_T2_C48_C56_2026_09_07.md`;
- the C48-C56 PRE/POST already committed at that tip; and
- the seven exact counterexamples in sections C57-C63 below.

P1 starts in a separate clean worktree from
`origin/musashi/model-capacity-m0-m2-20260907` at
`5c3c1dc700f411b263766d2028c752dedfc7607e` and consumes:

- `docs/work_plan/44_DATA_CENTRIC_SIGNAL_MODEL_INFORMATION_AND_CAPACITY.md`;
- `docs/research/model_capacity/M3_M6_CONFIRMATORY_DESIGN_DRAFT_2026_09_07.md`;
- `docs/audits/evidence/MODEL_CAPACITY_M0_M2_PILOT_V3_SUMMARY_2026_09_07.json`.

Do not merge the two worktrees or identities.

## 2. P0: T2 Corrections

### C57: wall authority that cannot be renewed by restart

Freeze the PRE in the audit as an executable regression. Replace the lossy
five-second accounting with a descriptor-bound, strict, self-integral monotonic
protocol. Requirements:

- no `exists()`/`read_text()` check-then-reopen path;
- exact owner, mode and regular-file checks from the consumed descriptor;
- malformed, duplicated, reordered, truncated interior or transplanted
  records fail closed;
- every interval that can execute is charged before or durably at its
  boundary, so repeated crashes cannot recover elapsed time;
- bind a boot identity and monotonic-clock facts; a reboot or clock authority
  ambiguity stops for review rather than renewing the budget; and
- model both physical outcomes of the final fsync.

Use an existing reviewed append-only durability primitive where possible. Do
not invent another overwrite/restore protocol.

Acceptance includes repeated sub-cadence crashes, torn final writes, an
interior-line mutation, ledger replacement between read and append, reboot
identity change and two real processes.

### C58: complete lock protocol

Replace bare release existence with an immutable release intent and a separate
completion witness bound to the exact session, session digest, UUID, holder PID
and generation. Reclaim must re-read and revalidate both descriptor-first
under the exclusive election.

An absent, empty, malformed, permissive, symlinked, stale, transplanted or
fsync-uncertain completion never frees the lock. Validate strict schemas and
self-digests for SESSION, TAKEOVER, release intent and completion. Test both
physical outcomes of the last fsync and two real contenders.

### C59: authority-bound claims and terminals

Every claim and every terminal must bind at least:

- sealed design physical and self digests;
- design-review and execution-record digests;
- executor code identity and pinned commit/tree;
- exact unit binding and unit id;
- attempt id, mode and campaign generation; and
- the digest of the claim that the terminal closes.

Deep verification under current physical authority is mandatory before a
terminal can remove a unit from future work. A stale or transplanted terminal
must become typed `UNCERTAIN`, never `TERMINAL_FAILED`.

### C60: disposition is external authority, not a candidate command

The disposition CLI must first pass the same current gates as execution and
fully verify the uncertain claim. It additionally consumes a separate external
Musashi disposition record from the fixed private reviewer root, pinning the
claim digest, unit, attempt, current execution record, decision and reason.

Candidate-authored text, a repository JSON or a self-digest grants nothing.
The record is needed only if a real uncertain attempt occurs; do not author or
simulate a productive one in this correction.

### C61: root and intermediate-directory custody

Open the results root from a fixed trusted parent, component by component with
`O_NOFOLLOW`. Require exact uid/0700 for the root and control directories and
uid/0600 for control/evidence objects. A symlink root or intermediate
component, permissive preexisting directory, foreign owner, non-normal path or
path replacement fails before writes. Apply the same rule to heartbeat and
wall evidence; no fixed shared `.tmp` path.

### C62: one effective wall and memory limit

The fit supervisor must receive the currently remaining global wall and use
`min(remaining_global, per_fit_limit)`. It may not start when no positive
budget remains. Reserve/charge the non-interruptible interval durably before
dispatch so a worker or parent crash cannot make it free.

After an `OK` payload, verify the child exited; otherwise kill and reap before
returning. EOF, send failure and crash are typed and reaped. Demonstrate no
orphan process. Reconcile parent plus child memory conservatively; document
what the configured RSS bound actually covers instead of claiming a stronger
physical guarantee.

### C63: final physical adjudication controls process success

Any persistence or verification failure after a claim halts the campaign
immediately as typed uncertainty. Do not catch it, increment a counter and
continue. Before release and successful exit, re-adjudicate every sealed unit:

- each completed record is deeply reverified against current physical inputs;
- each failed terminal is deeply reverified against current authority;
- no `UNCERTAIN` state exists; and
- counts equal the sealed population exactly.

The exact PRE (`rc=0` with one uncertain claim) must exit nonzero and preserve
the named uncertainty.

### C64: independent verification and mutation

Extend the independent POST and focal battery with one isolated kill for every
requirement above. At minimum run mutations that:

- reintroduce bare release existence;
- omit current-authority binding from terminals;
- restore the five-second crash credit;
- let the supervisor ignore remaining wall;
- follow the results-root symlink; and
- permit success with final uncertainty.

Publish pass, fail and skip counts separately from the final code tip. Run the
full suite and name inherited failures exactly.

### C65: stop before scoring

Return with no productive execution record authored, no sealed unit scored and
no scientific ledger. Required disposition:
`T2_C57_C65_READY_FOR_FINAL_MUSASHI_RUNTIME_RECORD`.

Musashi will independently rerun the PRE/POST, review the final checkout,
install the external execution record and launch the bounded CPU service. This
step is not delegated back to the owner.

## 3. P1: Model-Information M3 CPU Calibration

After P0 is pushed, continue in the separate M3 worktree. This is authorized
to keep the CPU scientific lane moving while B4 occupies the GPU.

### M3.0: pre-result seal

Turn section C1 of the M3-M6 draft into an executable immutable design before
computing any separability outcome. Verify the exact finite-N Cover formula
and state whether the classifier is homogeneous or affine; data generation,
formula and feasibility solver must use the same convention. Include the
primary citation and a machine-checked formula test for small enumerated cases.

Freeze K={32,64,128}, N/K={1.25,1.5,1.75,2,2.25,2.5,2.75}, at least 200
independent labeling tasks per cell, a precision-driven extension rule,
simultaneous intervals, seeds, solver tolerances and all resource limits.

### M3.1: solver and controls

Primary outcome is deterministic linear feasibility, not perceptron training.
Cross-check a frozen subset with an independently formulated solver. Include:

- random Gaussian points with a numerical general-position diagnostic;
- independent random binary labels;
- an easy separable positive control;
- a deliberately inconsistent negative control; and
- permutation and sign-symmetry checks.

Solver disagreement, numerical ambiguity or resource exhaustion is typed and
remains in the denominator. It is never silently converted to nonseparable.

### M3.2: bounded execution and verdict

Run CPU-only with CUDA hidden, one logical worker unless the sealed resource
contract explicitly licenses more, maximum six wall-hours and observable
heartbeat/stop-file. Reconstruct every aggregate from immutable per-task
records in a fresh process.

Allowed verdicts are `COVER_CALIBRATION_CONFIRMED_WITHIN_DECLARED_PRECISION`,
`NOT_CONFIRMED` or `INCONCLUSIVE`. This stage does not estimate MLP
intelligence, residual capacity or exact Kolmogorov complexity and grants no
DOIN feature.

Return the design, results, independent verifier, mutations and exact counts.
Do not start the residual-capacity intervention M4 until Musashi reviews M3.

## 4. Reporting and Runtime Status

While working, report B4 service status and current cell/epoch from read-only
telemetry only. Do not restart it. Report T2 as blocked by C57-C65, not by the
owner. Report M3 as CPU-only and separate from both campaigns.

The return packet must name all remaining blockers and assign each to one of:
Satoshi, Musashi, owner/operator or external evidence. A blocker that Satoshi
or Musashi can resolve must not be assigned to the owner.
