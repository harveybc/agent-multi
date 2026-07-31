# 04. Open Findings Register

Version: 1.2.0
Date: 2026-07-30
Owner: Satoshi (state updates); closure of S0-S2 requires an independent
verifier (normally Musashi) per `../README.md`.

This file is the cross-session source of truth for finding state. Full finding
stanzas live in their originating report; this register carries identity,
state and the next required action only.

## 1. Open

| ID | Sev | State | Title | Source report | Next action | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| AUD-F1-20260730-005 | S3 | open (re-sampled 23:51 COT: **3 h 25 min**) | Equal-height chain fork: dragon on tip `603dfe1a…`, other three on `4b4f06a1…` at height 9; finalized anchor identical on all four, so no parallel lineage and no corruption. **Key mitigating evidence:** chain height has not advanced past 9 during the whole window (finalized height stuck at 2, 7 unfinalized blocks), so finalization has had no opportunity to resolve it — persistence is convergence latency pending a new block, not a demonstrated fork-choice failure | `../AUDIT_FULL_CROSS_FRONT_2026_07_30.md` section 3.2 | Re-sample at the generation-2→3 boundary (gen 2 at 17/20, ~2-4 h out) when new blocks seal; run read-only AT-F1-011. **No chain mutation on current evidence.** Escalates to S2 if the split survives that boundary, or Dragon holds unique accepted transactions, or finalized anchors diverge | Musashi (decision) + Satoshi (test) |

| AUD-F3-20260731-006 | S3 | open | Prompt-injection screening is five English-only regexes and is the **sole** barrier, because flagged content is withheld rather than sanitized. Spanish phrasing, paraphrase, code fences, homoglyphs and base64 all pass unflagged. Bounded by compensating controls: packet declares content untrusted, publishing disabled, no tool access, human review required | `../AUDIT_FULL_CROSS_FRONT_2026_07_30.md` section 5.3 | Add Spanish patterns and multilingual malicious-post fixtures; consider quarantine over silent drop so evasions stay auditable | Musashi |
| AUD-F3-20260731-007 | S3 | open | Document 23 section 4 paid-token caps, 80 % circuit breaker and 100 % hard disable, and section 7 model-call cost facts, have no located implementation; social OLAP has no model-call table. The collector itself is deterministic, so the gap is on the Hermes consumer side | `../AUDIT_FULL_CROSS_FRONT_2026_07_30.md` section 5.3 | Record model-call facts and enforce caps before cadence or packet size increases | Musashi |
| AUD-F3-20260731-008 | S4 | open | `digest_packet` applies `LIMIT` in SQL then drops injection-flagged rows in Python, so hostile posts consume top-N slots and can crowd legitimate findings out of the review packet | `../AUDIT_FULL_CROSS_FRONT_2026_07_30.md` section 5.3 | Filter in SQL or over-fetch and trim after filtering | Musashi |

| AUD-GEN-20260731-009 | **S2** | open | **No CI in any of the ten repositories.** Document 09 section 12 declares "CI covers unit, property, integration and contract tests" and section 2 requires a CI future-row mutation test; neither exists. All verification is manual/local, and public Tier A repos have no automated secret or dependency scanning | `../AUDIT_QUALITY_SECURITY_TESTING_2026_07_31.md` section 6 | Start with one minimal workflow per Tier A repo running the existing suite; add the leakage mutation gate to `agent-multi` | Musashi |
| AUD-GEN-20260731-010 | S3 | open | Property/metamorphic layer declared in document 09 section 1.2 (ten named invariants) is effectively unimplemented — 0-1 files per repo. These catch silent accounting/netting/permutation/staleness defects that unit tests pass through | `../AUDIT_QUALITY_SECURITY_TESTING_2026_07_31.md` section 6 | Implement the ten invariants in their owning repos (`gym-fx`, `lts`, `agent-multi`) | Musashi |
| AUD-GEN-20260731-011 | S3 | open | System/acceptance level exists operationally but not as runnable suites; only `prediction_provider` has an acceptance/production taxonomy. Acceptance regression is detected by operating the system rather than testing it | `../AUDIT_QUALITY_SECURITY_TESTING_2026_07_31.md` section 6 | Adopt the `prediction_provider` taxonomy; convert channel-switch/rollback and one bounded replay first | Musashi |
| AUD-GEN-20260731-012 | S3 (Tier A) / S4 | open | Dependency reproducibility rests on one conda environment hash; no per-repo pinning or SBOM. Document 24 section 3.6 requires package/binary provenance. Supply-chain exposure for public Tier A repos | `../AUDIT_QUALITY_SECURITY_TESTING_2026_07_31.md` section 6 | Lock-file + SBOM for Tier A; keep the env hash as the fleet control | Musashi |
| AUD-F3-20260731-013 | S4 | open | Social relevance scoring is coarse term-matching, saturating at 0.71, so ranking within the largest band is near-arbitrary while the digest sorts by it | `../AUDIT_QUALITY_SECURITY_TESTING_2026_07_31.md` section 6 | Length-normalise, weight distinctive terms, add recency; or use the score as filter not sort key | Musashi |

All four bootstrap/status findings were independently verified closed by
Satoshi on 2026-07-30 22:44 COT; see `../AUDIT_DELTA_2026_07_30_02.md`
section 1. The `clientId 7` sub-hypothesis in AUD-F2-20260730-004 is withdrawn
as not reproduced.

### 1a. Technical-lead triage, 2026-07-31

These state changes preserve the original report history and supersede the
initial breadth-first classifications above:

| ID | Current state | Triage |
| --- | --- | --- |
| AUD-F3-20260731-006 | implemented_pending_independent_verification | Multilingual/encoded screening and fixtures implemented |
| AUD-F3-20260731-007 | implemented_pending_independent_verification | Conservative token reservations, hard caps and model-call facts implemented |
| AUD-F3-20260731-008 | implemented_pending_independent_verification | SQL filters quarantined rows before `LIMIT` |
| AUD-F3-20260731-013 | implemented_pending_independent_verification | Weighted, length-normalized, recency-aware full-corpus scoring implemented |
| AUD-GEN-20260731-009 | open, re-triaged S3 | No CI confirmed; S2 impact not demonstrated |
| AUD-GEN-20260731-010 | open, narrowed S3 | Invariant set incomplete, not absent; inventory exact missing cases |
| AUD-GEN-20260731-011 | rejected_as_written | Runnable acceptance/system/multi-node suites exist; open only specific missing scenarios |
| AUD-GEN-20260731-012 | open, re-triaged S4 | Canonical 143-pin fleet lock exists; per-repo locks/SBOM remain release hardening |

Evidence and rationale:
`../CODEX_AUDIT_TRIAGE_2026_07_31.md`. Independent re-verification task:
`../../handoffs/SATOSHI_POST_FIX_VERIFICATION_TASK_2026_07_31.md`.

## 2. Closed / Resolved

| ID | Sev | State | Closure | Verifier | Date | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| AUD-F2-20260730-004 | S2 | verified_closed | User accepted the TWS Paper disclaimer; watchdog now requires a recent authenticated reconciled session and overlapping preflights fail closed | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `lts@12d389d`; `205 passed`; successful systemd observer/watchdog run |
| AUD-GEN-20260730-001 | S3 | verified_closed | Document 13 phase summary, ledger and immediate tasks now record the deployed four-worker v2 campaign | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |
| AUD-F1-20260730-002 | S4 | verified_closed | Document 13 records measured fleet throughput, the 10-14 day full-budget range and an end-of-stage-1 duration/evidence decision point | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |
| AUD-GEN-20260730-003 | S4 | verified_closed | Musashi recovery prompt v1.1.0 includes docs 08/11/14/16 and refreshes the runtime warning | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |

## 2b. Observations Pending Verification (not yet findings)

| Ref | Observed | Verify in |
| --- | --- | --- |
| OBS-20260730-A | Job-0 record `started_at` is 2026-07-29T07:16:30Z (02:16 COT) but the Omega node process start implies ~18:18 COT the same day, a ~16 h gap with `restart_count=0`. Likely the job record marks plan materialization while workers launched after the deployment sequence, but unverified. | `worker_events` table, next delta session |
| OBS-20260730-D | Gamma is the resource-constrained host: 6.12 GB of 15.34 GB RAM available, ~2.4 GB swap in use, 50.66 GB free disk (12 %), campaign-cgroup `sock_throttled` = 6,228 while omega/dragon report 0. No OOM kills; both GPUs healthy. | Trend check each delta session; escalate if `sock_throttled` grows or swap use trends up |
Resolved observation: `OBS-20260730-B` was the symptom that became
`AUD-F2-20260730-004`; the finding is now verified closed.

Resolved observation: `OBS-20260730-C` is covered by the live tier-0 collector
at `agent-multi@12d394ff`. Its verified systemd packet recorded all three
hosts, four GPUs, utilization/temperature/memory, RAM, swap, disk and campaign
cgroup OOM counters.

## 3. Verified Non-Findings (do not reopen without new evidence)

From `../AUDIT_BOOTSTRAP_2026_07_30.md` (2026-07-30):

1. Four-worker fleet lineage fully consistent (one plan hash, generation,
   population fingerprint, chain height, finalized anchor, component
   revisions; distinct claims; zero alerts).
2. Deployed `agent-multi@6a7bf5a` is code-identical to HEAD `21bcc427`
   (docs-only delta).
3. All 11 repos clean and synced; no user changes at risk.
4. Job-1 `planned_candidates=0` is the fail-closed materializer design, not a
   missing budget.
5. Gen-0 vs current-generation population fingerprints differing is
   per-generation fingerprint behavior, not divergence.
6. `predictor` recent commits are Apr-May 2026 historical work, consistent
   with reference-only role.

## 4. Register Rules

- Add a row when a report opens a finding; move to section 2 with verifier,
  date and closure evidence reference when closed.
- Never edit severity/history in place; append a state-change note if a
  severity is re-triaged, with reason.
- Each delta session (AT-GEN-010) re-verifies that open rows still reproduce
  and that closed rows have closure evidence recorded by a non-reporter.
