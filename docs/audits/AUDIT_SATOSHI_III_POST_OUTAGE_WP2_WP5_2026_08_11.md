# Audit: Satoshi III Post-Outage WP2-WP5

Date: 2026-08-11 America/Bogota  
Auditor: General Musashi, independent verifier  
Subject branch: `satoshi/post-outage-209-223`  
Subject tip: `8ab0d6d6b106d702cee851e15c64b079d9d40d6b`  
Verdict: **corrections required before the mechanics screen starts**

## 1. Executive Verdict

Satoshi delivered substantial, testable work. The terminal-custody correction,
GPU readiness classifier, phase-1 handoff telemetry, v2 migration preflight and
the 2x2 factor declarations are real. The branch is clean and pushed, 150
focused tests pass, and the complete `agent-multi` suite passes 1,120/1,120.

The screen must not run at this tip. Its materializer binds the nested split
contract SHA in metadata but does not pass the nested split contract to the
executing pipeline. It therefore executes the legacy split path and uses the
wrong evidence roles. An independent real-artifact replay observed 42 rows for
the train tail and 2,196 rows from 2024 under the label `inner_validation`; the
approved nested contract requires a 2,190-row 2022 train monitor and a
2,190-row 2023 inner validation. The same config also retains
`lexicographic_weekly_v1` instead of the required paired metric.

The pause prevented invalid compute. It is not permission for continued idle:
the accompanying correction order authorizes immediate corrected preflight,
deployment and 16-cell screen without another owner phrase.

## 2. Evidence Reproduced

- Subject branch and origin agree exactly at `8ab0d6d6`; worktree clean.
- `git diff --check 83888d0c..8ab0d6d6`: clean.
- Focused suite: 150 passed.
- Full `agent-multi` suite: 1,120 passed, two non-failing sklearn convergence
  warnings.
- Compilation and JSON parsing: pass.
- Ladder terminal verifier: `TERMINAL_CUSTODY_VERIFIED`; all four terminal
  paths and hashes match; fresh tree digest remains
  `cdb6ef9947887992fc0a133a8c66adb76d64a4484cccb5cfc9f63fbea1c2ed8e`.
- Same-artifact replay, D2 post-easy artifact `2620b722...`: raw action exactly
  `0.014354467391967773`, one float32-unique value, zero `abs(a)>=0.1`, typed
  `THRESHOLD_EXPOSES_PREEXISTING_COLLAPSE` on both replayed legacy roles.
- Same-artifact replay, anchor `cb27375c...`: `NO_THRESHOLD_COLLAPSE` on both
  replayed legacy roles.
- Four physical GPUs are visible; temperatures sampled at 42 C, 28 C, 24 C and
  40 C. No P1LR worker or service is deployed or running.

Canonical independent reproducer:

- `evidence/repro_runs/MUSASHI_POST_OUTAGE_224_230_REPRO_2026_08_11.py`
- `evidence/repro_runs/MUSASHI_POST_OUTAGE_224_230_REPRO_2026_08_11.json`

## 3. Prior-Finding Dispositions

| Finding | Independent disposition |
| --- | --- |
| 217 | Remains open. Checker behavior is corrected, but remote default master still has the two broken links; the branch-only README change is not closure. |
| 220 | Remains open. A successor screen is implemented but neither validly materialized nor dispatched. |
| 221 | Implementation verified at unit level; integrated acceptance is blocked by finding 224 because the wrong evidence roles reach the pipeline. |
| 222 | Wording corrected and replay mechanism implemented; replay result is diagnostic only until rerun on the exact approved roles. |
| 223 | Independently verified corrected, pending owner closure. The adversarial collector tests and read-only four-terminal proof both pass. |

## 4. New Findings

### AUD-F1-20260811-224 (S2): factorial and replay claim nested evidence while executing legacy roles

`materialize_cell_config()` stores the nested contract SHA in `_identity`, but
the executable config has no `nested_split_contract`. The first real config
instead carries the legacy ranges and `selection_metric=lexicographic_weekly_v1`.
The replay labels legacy `val` as `inner_validation`, although it is the 2024
outer-validation interval. This would use outer truth during checkpoint
selection and invalidate the intended L1 attribution.

Required correction: execution must consume the typed nested contract directly:
fit 11,509 rows, monitor 2,190, inner 2,190, outer 2,196, sealed 2025 absent;
paired metric only; exact role hashes and context counts recorded in every
cell. Generate a new experiment identity. No existing output may be relabeled.

### AUD-F1-20260811-225 (S2): screen can declare eligibility with zero replica proofs

`screen_verdict()` sets `replica_terminal_loads` to an explanatory string and
does not evaluate it. The independent 16-record counterexample supplied no
replica evidence and still returned `SCREEN_VIABLE_REGION`, exit 0. It also
does not require the per-checkpoint summary at aggregation time.

Required correction: a typed P1 collection/replica proof is a mandatory input.
Require exactly 16 successful terminal loads bound to experiment, seed, cell,
relative path and SHA, plus per-checkpoint and split-manifest bindings. Missing,
duplicate, foreign, altered or unloaded artifacts refuse the verdict.

### AUD-F1-20260811-226 (S3): the conditional decision run is declared but not executable

The CLI supports `--seed` for the one-pass mechanics screen and
`--screen-verdict`; it has no decision-run mode, materializer or aggregator.
Every materialized cell is hard-coded to one phase-1 and one phase-2 epoch.
Consequently a viable screen cannot trigger the already-authorized decision
run with 2,000-checkpoint ceiling, patience 60/floor 40, best restoration,
outer truth and paired main/interaction effects.

Required correction: implement the decision path before calling WP4 complete.
Decision cells start from the exact per-seed anchors, never screen terminals.

### AUD-F2-20260811-227 (S3): unsigned/expired resume files can deny the owner operation

The owner signature protected IBKR correctly; no unauthorized resume occurred.
However, `ibkr_resume_after_reconciliation.py` counts every top-level JSON
before signature/expiry classification. A second unsigned/expired payload made
the valid owner operation refuse with `2 capability file(s)`. An interactive
TTY and a public phrase do not prove that the caller is the owner; the
passphrase-protected signature is the actual boundary.

Required correction in `lts`: select an explicitly named signed capability or
classify signatures and validity before ambiguity checks; ignore only invalid,
unsigned and expired files while still refusing two valid signed capabilities.
Tests must use isolated stores. Documentation must stop claiming TTY alone
excludes an agent.

### AUD-GEN-20260811-228 (S3): consolidated status contradicts the cleared IBKR hold

Direct durable state after the signed owner operation is `halt=none`, zero
positions and zero orders. The current consolidated queue still reports
`operational_but_held` and asks the owner to clear the hold because it gives a
stale prior decision (`halted:hold`) precedence over current durable state.

Required correction: current service state and fresh broker reconciliation are
authoritative. Preserve the old rejection as history and report IBKR as
write-enabled, flat and waiting for the next H4 decision.

### AUD-GEN-20260811-229 (S3): completed code was neither deployed nor dispatched

Omega, Dragon and Gamma remain on `bd65787c`; no `p1lr-screen@` unit exists on
any host, no P1LR record exists, and 0/4 workers run. The subject branch is only
on origin. This is an operational completion defect, not a scientific result.

Required correction: after findings 224-225 pass the deterministic preflight,
deploy one exact corrected commit to all hosts, install the readiness timer and
P1LR units, and start seeds 101/202/303/404 immediately. Audit proceeds in
parallel and may stop a run only on concrete identity/safety failure.

### AUD-GEN-20260811-230 (S4): Gamma historical replica stopped incomplete

No rsync process is active. Gamma contains 219 GB while Dragon currently has
173 GB at the intended path. No deletion occurred and 42 GB remains free on
Gamma.

Required correction: resume the bandwidth-limited, restartable replica, then
prove source/destination inventory and content digests. No source deletion is
authorized.

Post-finding runtime action: the auditor resumed the copy at 19:35 COT as the
transient user unit `gamma-history-replica.service`, with rsync `--partial` and
`--bwlimit=25000`. It is active with zero restarts. Finding 230 remains open
until the completed source/destination digest proof lands.

## 5. Four-Front Runtime Status

- **Front 1:** old L1 factorial terminal 16/16; new P1LR screen 0/16 and not
  deployed. GPUs healthy but idle for training. This audit supplies the exact
  correction path and standing dispatch authority.
- **Front 2:** Alpaca Paper write-enabled with one exposure; MT5 Demo connected,
  write-enabled and one exposure; IBKR Paper connected, flat, `halt=none`, zero
  orders/positions. The consolidated IBKR queue label is wrong per finding 228.
- **Front 3:** 9,513 posts collected, 716 enriched, 254 eligible backlog, zero
  drafts; publishing remains human-gated.
- **Front 4:** subject branch clean/pushed; 1,120 tests pass. Findings 224-230
  are open; correction 230 is executing. No S0/S1 event observed.

## 6. Acceptance Boundary

The corrected mechanics screen may launch without a new owner phrase only
after a no-training preflight proves the exact nested role counts/hashes,
paired selection, sealed-test absence, 16 distinct cell identities, four GPU
bindings and an executable replica gate. The long decision run is conditional
on a real viable screen plus a complete 16-terminal replica proof. No Paper or
Demo risk control is relaxed.
