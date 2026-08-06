# Audit: Corrections 143-150

Date: 2026-08-06 America/Bogota  
Auditor: General Musashi, independent verifier  
Technical-lead packet: `docs/handoffs/SATOSHI_III_143_150_CORRECTION_PACKET_2026_08_06.md`  
Audited head: `agent-multi@154f5784`  
Runtime mutation: none; no broker call, venue restart, campaign pause, RT1 run or smoke launch

## 1. Verdict

Corrections 143 and 146 are independently verified and may move to
`independently_verified_pending_owner_closure`. Corrections 144, 145 and
147-150 remain open. RT1-A remains `MATERIALIZED_NOT_EXECUTED` and the pending
smoke remains blocked.

The supplied typed probe is a valid replacement for the retired exception-is-a-
pass harness, and SQLite now commits interval and authoritative restart state in
one transaction. The remaining acceptance claims fail under the executable
counterexamples in:

`docs/audits/evidence/SATOSHI_III_143_150_ACCEPTANCE_REPRO_2026_08_06.py`

## 2. Findings

### AUD-F1-20260806-151 - S3 - Replica authority is caller-supplied text

The validator rejects `replica_authority == LOCAL_HOST`, but does not observe a
second host or storage authority. A primary and replica in the same local temp
directory pass when the record says `replica_authority="dragon"`. The publisher
also copies to a local path and obtains the authority from an environment
variable. Artifact loading and terminal cross-binding are corrected; physical
independence is not.

Evidence: independent `144_replica_authority_spoof`; decision runner lines
165-177 and 205-225. Finding 144 remains open through 151.

### AUD-F1-20260806-152 - S2 - The handover is an arithmetic assertion, not a correct close

`gym-fx` exposes `info.position` as direction `{-1,0,1}`. The runner treats that
direction as the number of units and computes `abs(position) * price *
commission`. With configured `position_size=0.01`, price 2,000 and commission
0.0002, the runner charges 0.4 instead of 0.004: a 100x error. It charges no
configured spread or slippage, submits no simulator close, reconciles no broker
state and then hardcodes `flat_after_handover=true`.

The forced-hold warm-up and exact `h` sample count are corrected, but the
post-close balance, handover facts and performance metrics are not valid.

Evidence: independent `145_handover_cost_and_proof`; runner lines 164-193 and
577-640; `gym-fx/app/bt_bridge.py` publishes position direction rather than
position size. Finding 145 remains open through 152.

### AUD-F1-20260806-153 - S2 - Consecutive origins do not inherit the adapted model

The authoritative pointer is loaded once before the loop and never updated in
memory after a new origin commits. In a real two-origin CPU run, origin 1's
before-model differed from origin 0's after-model by a maximum policy-weight
delta of `0.00958772`. A restart happens to reload the last SQLite pointer, so
the defect hides in the crash/replay fixture: uninterrupted execution resets
each later origin to the original anchor/fresh initialization.

This defeats incremental adaptation, paired cadence comparison and the business
requirement that the successor starts from the current model/account state.

Evidence: independent `147_in_process_model_chain`; runner lines 441-509 and
649-689. Finding 147 remains open through 153.

### AUD-F1-20260806-154 - S3 - Restarted summaries discard prior latency observations

`latencies` is process-local and committed origins are skipped without loading
their persisted durations. After two origins and a restart for origin 3, the
database p95 was `3.81919 s`, while the summary reported only the new sample,
`3.58468 s`. With 20 persisted updates this can falsely satisfy the owner-ratified
p95 deadline when old durations lie between two-thirds and one cadence.

Evidence: independent `148_restart_latency_window`; runner lines 476-503 and
690-741. Finding 148 remains open through 154.

### AUD-F1-20260806-155 - S3 - Untracked source is still reported as a clean tree

The identity calls `git status --untracked-files=no` and hashes only `git diff
HEAD`. An executable untracked Python fixture was visible to Git as `??`, while
`source_tree_digest()` returned `clean=true` and no diff hash. The tracked-diff
correction is useful, but it does not bind the complete source used by Python.

Evidence: independent `149_untracked_source_identity`; runner lines 326-352.
Finding 149 remains open through 155.

### AUD-F1-20260806-156 - S3 - Same-second cached state with the same PID proves rejoin

Resume timestamps have second precision and `_observed_after()` accepts
equality. The independent fixture supplied a worker observation equal to
`accepted_at`, retained the same PID/start-tick generation, and obtained
`rejoin_proven=true`. The bounded deadline is corrected, but the required fresh
process generation and strictly post-acceptance poll evidence are absent.

Evidence: independent `150_same_second_and_pid_generation`; supervisor lines
167-180, 3548-3563 and 3679-3694. Finding 150 remains open through 156.

### AUD-F1-20260806-157 - S2 - Every RT1-A arm omits its final interval

The origin range stops at `block_start + block_bars - cadence_bars` with an
exclusive Python endpoint. In a 28-day block it evaluates 83/84 eight-hour
intervals, 55/56 twelve-hour intervals, 27/28 daily intervals and only 3/4
weekly intervals. The cadence-dependent omission biases the comparison most
strongly against the weekly arm and violates the materialized 28-day contract.

Evidence: independent `157_block_coverage`; runner lines 470-474.

### AUD-F1-20260806-158 - S2 - Any compatible SAC file can claim mature-anchor status

`--anchor-model` proves only that a ZIP loads and records its hash/path. It does
not require or bind a champion-origin manifest, selection evidence, resolved
genome, training/data contract or eligibility decision. A compatible fresh-init
mechanics checkpoint therefore satisfies the same performance-run anchor path.
Clean-tree plus nonempty path is the entire promotion gate.

Evidence: independent `147_anchor_provenance`; runner lines 355-382, 410-418
and 521-529. Finding 147 remains open through 158.

## 3. Accepted Corrections

| Finding | Disposition | Independent basis |
| --- | --- | --- |
| 143 | independently verified pending owner closure | typed outcomes distinguish harness/fixture errors from exact refusal and postcondition passes; deliberate stale-symbol and malformed-fixture cases expose the retired harness |
| 146 | independently verified pending owner closure | interval and `rt_state_v2` commit together; restart reads SQLite, verifies model bytes and replays a post-artifact crash exactly once |

Partial work preserved:

- 144 now performs real SAC loads and exact terminal artifact cross-binding;
- 145 now forces hold during warm-up and scores exactly `h` samples;
- 147 refuses an absent anchor unless the run is explicitly mechanics-only;
- 148 has explicit schema fields and a guard that reads them;
- 149 binds tracked modifications;
- 150 persists a bounded timeout and returns to stable paused state on expiry.

Those improvements remain; they are insufficient for closure for the reasons
above.

## 4. Verification

```text
independent acceptance reproducer: completed; network_used=false
supplied typed probe: 10/10 typed outcomes
focused correction suites: 77 passed
agent-multi full suite: 629 passed, 2 warnings
lts full suite: 661 passed, 1 warning
```

Passing suites do not override the executable counterexamples.

## 5. Runtime Disposition

- RT1-A: remain materialized, do not execute.
- Smoke 123/124/127: do not launch.
- Venues: do not restart or mutate for this correction cycle.
- Active `full-v2`: preserve unchanged until the owner explicitly decides the
  prepared pause/archive operation. Its zero-activity candidates remain
  decision-ineligible.
