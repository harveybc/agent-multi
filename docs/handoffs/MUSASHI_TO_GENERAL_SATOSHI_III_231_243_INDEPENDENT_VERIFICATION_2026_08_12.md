# Musashi to General Satoshi III: Independent Verification 231-245

Date: 2026-08-12 America/Bogota
From: General Musashi, temporary independent auditor
To: General Satoshi III, technical lead
Runtime rule: verification runs in parallel; do not stop compute or brokers

## Objective

Independently verify the correction packet in:

- `docs/audits/AUDIT_SATOSHI_RETURN_231_233_AND_RUNTIME_CORRECTIONS_2026_08_12.md`
- `docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md` section 1ak
- `docs/work_plan/38_NEAT_LESSONS_L1_L2_CURRICULUM_AND_FEATURE_SELECTION.md`

Do not close findings you implemented or reported. Return reproduced facts,
counterexamples and exact commands/commits; the owner disposes closures.

## Non-interference rule

The active decision identity is `8cc6ca5e45e4f993`. Four
`p1lr-decision@` units must continue uninterrupted. Verification must be
read-only against runtime state and use temporary local fixtures for attacks.
Never stop, restart, re-materialize or warm-start the decision run. Never
submit or cancel a broker order. Do not clear the IBKR `hold`; that remains an
owner-signed operation after direct flat reconciliation.

## Required verification

1. Re-run the unchanged 231-233 reproducer and report each boolean.
2. Attack 234 with active final/best checkpoints that are byte-identical and
   prove custody succeeds only when both paths/hashes/load facts are valid.
3. Reproduce the old 2,191-versus-2,190 terminal transition, then prove exact
   row equality through bridge and nested selectors for active and inactive
   policies.
4. Prime the nested-role cache, mutate a role CSV without changing its
   manifest, and prove re-verification rejects it.
5. Materialize all 16 decision cells and prove one execution-profile hash,
   `buffer_size=40000` and two complete pass-equivalents on every seed.
6. Load a saved SAC with a larger archived buffer through both warm-start
   paths. Prove source transfer capacity 1, target capacity preserved, empty
   replay, fresh optimizer state, identical transferred policy tensors and no
   full archived replay allocation.
7. Re-run the LTS tests for 238-241 and verify current Alpaca, IBKR and MT5
   heartbeat hashes join their manifests. Treat current protected positions as
   read-only evidence.
8. Install from `requirements-ci.txt` in an isolated environment and run the
   exact six Tier-A test files. Confirm pytest 9.0.3 and no Dependabot alert
   remains after GitHub processes the pushed default branch.
9. Observe decision identity `8cc6ca5e45e4f993` through at least two fresh
   heartbeat intervals. Record four workers, zero restarts, cgroup memory,
   GPU temperature/utilization and the first landed record when it exists.
10. Verify the runtime-pin drop-in on all three hosts points to clean detached
    `agent-multi@182bac7e` worktrees and that preflight from each pinned
    worktree still derives `8cc6ca5e45e4f993` while the canonical checkout may
    advance independently. Do not restart a worker merely to prove the pin.
11. Inject sequential-reader latency with a heartbeat written after collection
    starts. Prove displayed age is nonnegative and genuine clock lead is
    reported separately.

## Return packet

Return one append-only Markdown packet with:

- exact repository heads and clean/dirty states;
- per-finding verdict (`reproduced`, `verified_corrected`, or `still_open`);
- test counts and evidence hashes;
- live worker table and broker table;
- every doubt or newly discovered defect; and
- no claim of closure.

Continue approved CPU work and social collection while verification runs.
No formal review gate is permission to leave compatible compute idle.
