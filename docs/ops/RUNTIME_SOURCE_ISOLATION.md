# Runtime Source Isolation (WP0, order 2026-08-15 §2)

Status: ENFORCED (runner launch path) + CONVENTION (agent write
discipline). Owner: General Satoshi III. Mechanism:
`tools/runtime_worktree.py`.

## The incident this corrects (2026-08-15)

The corrected P1LR v2 screen rejected all four omega seed-101 cells
because a third agent wrote an untracked handoff file
(`docs/handoffs/RETSU_TO_...md`) into the CANONICAL checkout while the
experiment executed from it. The source-identity guard
(`pipeline_plugins._system_config.assert_source_identity_unmoved`,
`dirty_untracked_digest None -> 'aadbeca8…'`) correctly refused — but:

1. the experiment should never have been executing from a shared
   writable checkout;
2. the failure surfaced as an untyped `SEED_FAILED` killing all four
   cells; and
3. recovery was manual (a pinned worktree was created and the seed
   relaunched by hand).

## The five rules

### 1. Experiments execute ONLY from pinned detached worktrees (ENFORCED)

Every long-running experiment executes from a DEDICATED DETACHED
worktree bound to one commit and VERIFIED CLEAN before launch, under
the declared runtime root:

    ~/Documents/GitHub/.runtime/          (override: P1LR_RUNTIME_ROOT
                                           or AGENT_MULTI_RUNTIME_ROOT)

Create/verify it with the ONE shared mechanism (operators and units):

    python tools/runtime_worktree.py ensure <commit> [--label p1lr-v2]
    # -> ~/Documents/GitHub/.runtime/agent-multi[-<label>]-<commit12>

and point the systemd unit's `WorkingDirectory` at it (the
`examples/systemd/pin_p1lr_decision_runtime.sh` drop-in pattern).

The runner launch path enforces the rule: `tools/
p1_difficulty_lr_factorial.py --seed N` refuses TYPED, before any
GPU/model work, when the executing tree is

* not a detached worktree under the runtime root
  -> `REFUSED_SOURCE_NOT_ISOLATED` (exit class 4), or
* dirty / carrying untracked files at launch
  -> `REFUSED_SOURCE_DIRTY` (exit class 4).

`--no-isolation-check` exists for socket-free tests ONLY; a fleet
launch MUST enforce. Exit class 4 is `RestartPreventExitStatus`: a
configuration refusal is recorded FAILED and never blindly restarted.

### 2. Agents write only to separate NAMED worktrees (CONVENTION)

No agent ever writes into the canonical checkout
(`~/Documents/GitHub/agent-multi`) while anything might execute from
it — and per rule 1, nothing may execute from it either. Agent work
happens in named worktrees:

    ~/Documents/GitHub/.worktrees/agent-multi-<agent>-<topic>-<date>
    (e.g. .worktrees/agent-multi-retsu-doctrine-20260815)

on a branch named `<agent>/<topic>-<date>`. Handoffs, findings and
docs are committed there and land in canonical only through the
owner/auditor merge path. Another agent CANNOT be technically stopped
from violating this — which is exactly why rule 1 makes every
experiment IMMUNE to such writes: a pinned detached worktree does not
see them, and a launch from a contaminated tree refuses typed.

### 3. Records bind source custody at materialization AND terminal custody (ENFORCED)

Every cell record carries `source_isolation`
(`agent_multi.p1lr_source_isolation.v1`): per code root (agent-multi,
gym-fx) the worktree path, commit, tracked-diff digest and untracked
digest — captured AT MATERIALIZATION and again AT TERMINAL CUSTODY —
plus the launch block with the explicit `verified_clean_at_launch`
fact (or the typed reason enforcement was off). The pre-existing
sealed `dirty_untracked_digest` stays byte-frozen (v1 identities fold
it in); the split digests are additive. Read-path validators never
demand the new block of pre-WP0 records.

### 4. Source drift is a FAILED CELL plus scheduled retry — never silent progress (ENFORCED)

`assert_source_identity_unmoved` now raises the typed
`SourceDriftError` (`failure_class: source_drift`), re-proven at
materialization (before any GPU work in the cell) and at terminal
custody. `tools/multifront_status.py` (`classify_p1lr_failure`) and
`tools/p1lr_idle_guard.py` classify it distinctly:

* `failure_class: source_drift`, `retry_eligible: true`, with the
  scheduled retry named (systemd `Restart=on-failure` retries exit 1;
  the idle guard's §7.8 bounded restart covers a dead unit);
* `REFUSED_SOURCE_*` launch refusals classify
  `source_isolation_refused`, `retry_eligible: false` — fix the
  launch root first (remediation names the exact commands);
* the status block names failed cells in `state_basis` and
  `failures[]` — a drifted cell can never render as quiet inactivity.

### 5. Missing cells retry AFTER the seed batch without rerunning valid cells (ENFORCED)

A seed rerun (`--seed N`, optionally `--retry-missing` to emit the
typed plan `agent_multi.p1lr_seed_retry_plan.v1` first) reuses every
COMPLETE cell record byte-identically (`ALREADY_COMPLETE` — the
pipeline never re-executes) and runs ONLY the missing/failed cells,
in contract cell order. An existing-but-invalid record REFUSES rather
than being overwritten; recover it explicitly.

## Recovery runbook (source drift)

1. `python tools/runtime_worktree.py ensure <commit> --label <exp>`;
2. point the unit at the pinned worktree (drop-in pattern above) or
   `cd` into it for a direct launch;
3. `python tools/p1_difficulty_lr_factorial.py --seed N
   --retry-missing` — complete cells reuse, failed cells rerun;
4. never delete or edit the pinned worktree while the experiment is
   live; never retarget it to another commit (typed refusal
   `REFUSED_WORKTREE_COMMIT_MISMATCH`).
