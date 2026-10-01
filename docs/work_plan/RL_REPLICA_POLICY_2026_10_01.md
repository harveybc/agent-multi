# Executable RL replica admission

## Requirements and test design

Scope: lane G matrix, queue and direct cell runner; no remote process operations,
training, changes to historical evidence or automatic retries.

| Requirement | Acceptance evidence |
| --- | --- |
| P1 one screening seed per arm | persisted exclusive claim, fresh instance refuses repeat |
| P2 second/third require prior justification | exact-config authorization, nonempty reason/owner, persisted before claim |
| P3 fourth refused | matrix rejection and runner refusal even with a reason |
| P4 preserve evidence | existing result remains byte-identical; partial attempt held |
| P5 successor fail-closed | shell queue S0/202 does not reach admission/training; direct runner refuses before ML imports |

Architecture: one standard-library policy shared by matrix, shell queue and
direct runner; campaign-root state locked for read/modify/write. Shell checks
are advisory; only the runner claims immediately before training. The default
lane seed plan is 101 (screen), 202 and 303 (held unless authorized); 404 refused.
Historical 16-cell matrices remain evidence, not launch authorization.

Discovery, requirements and unit/integration test design completed before
implementation. Scientific model/data validity remains outside this admission
change. Deployment to remote running checkouts is not part of the tests.

## Operator contract

- All runners for the same campaign must use the same persistent output/state
  root (`--replica-root`, or `RL_REPLICA_ROOT`, or the output parent by default).
  Never create another root to evade the count. Multi-host operators must
  reconcile/import historical evidence before dispatch; this local ledger
  does not query remote hosts or the warehouse.
- `REPLICA_POLICY.json` stores seed plan, immutable authorizations and claimed
  arm/seed slots. `flock` serializes claims; atomic replacement persists them
  before training. Failed claims remain held. No automatic retry/restart is
  authorized by this change; prior partial evidence needs a separate reviewed
  recovery workflow. The count is conservative across config revisions within
  the same campaign root.
- Allowed reason codes: `published_protocol`, `final_contrast`,
  `stochastic_variability`, `nondeterminism_diagnosis`. A reason and its owner
  must be nonempty; declaration time is assigned by the tool, not the caller.
  Authorizations bind the actual canonical config, not its self-reported hash.
  Later config edits do not inherit authorization. Existing claims/checkpoints
  cannot be retroactively authorized. Files are operator-controlled, not signed
  third-party evidence; scientific adequacy of a reason remains review work.
- Historical results (including fourth seeds) are preserved, never rerun or
  rewritten. Preservation is not retrospective validation of their contents.
  Pilot and smoke roots must remain explicitly separate evidence classes.
- The historical draft/frozen matrices remain untouched. New default matrices
  have four cells. An explicit three-seed plan marks extra rows held; four-seed
  materialization is refused. The executable lane policy supports only the
  frozen seed identities 101, 202 and 303; other seeds fail closed.

Read/check without training (may create the policy lock directory):

```bash
python -m rl_temporal.replica_policy check \
  --root /path/to/campaign --out /path/to/campaign/RL-S0_seed202 \
  --cell /path/to/RL-S0_seed202.json
```

Only after an actual predeclared design justification, persist it explicitly:

```bash
python -m rl_temporal.replica_policy authorize \
  --root /path/to/campaign --out /path/to/campaign/RL-S0_seed202 \
  --cell /path/to/RL-S0_seed202.json --reason-code final_contrast \
  --reason '<predeclared scientific justification>' --declared-by '<reviewer>'
```

No authorization is shipped or created by this commit. A held queue row is
skipped, not automatically retried if authority changes later. Exit 3 means
held/rejected/preserved in the check CLI; exit 2 means a policy error. The
direct runner exits 0 when preserving a result without any ML imports.

## Deployment boundary

No remote files, services or processes were touched, including the live
RL-D1_seed202. The sandbox handoff test proves that the **updated** queue will
not launch S0/202 without persisted justification; the direct runner repeats
the gate before importing SB3. An already-running remote shell and its old
checkout do not magically acquire this commit. The operator must adopt the
updated queue/runner/policy for its successor before claiming that the remote
handoff is protected. Until deployment is confirmed, live handoff protection
is PENDING, not complete. Do not interrupt the current cell to apply it.

## Verification

RED: the new policy tests first failed collection because
`rl_temporal.replica_policy` did not exist. GREEN: CPU-only admission tests,
shell handoff sandbox, actual arm-config construction, no model training.
The suite-wide checkout preservation fixture remained enabled. Exact final
test totals are recorded in the adjacent method state.
