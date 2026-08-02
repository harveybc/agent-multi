# 04. Open Findings Register

Version: 1.3.0
Date: 2026-07-31
Owner: Satoshi (state updates); closure of S0-S2 requires an independent
verifier (normally Musashi) per `../README.md`.

This file is the cross-session source of truth for finding state. Full finding
stanzas live in their originating report; this register carries identity,
state and the next required action only.

## 1. Open

| ID | Sev | State | Title | Source report | Next action | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| AUD-GEN-20260731-025 | S3 provisional | open, self-reported | Satoshi omitted `doin-plugins` from D2 after the governing finding named it, an enumeration-drift defect acknowledged by the auditor | `../AUDIT_GS_COUNTER_RESPONSE_AND_AT_F1_001_2026_07_31.md` section 3 | Harvey adjudicates severity; retain in P13 incident corpus | Harvey + Satoshi |
| AUD-F1-20260731-026 | S2 provisional | open | AT-F1-001 reconstructed the configured full-period L2 correctly but certified it as weekly-fraction; train tail covers 3 weeks, validation 53, and the mean-weekly reconstruction changes the champion score from positive to negative | `../CODEX_CROSS_REVIEW_OF_SATOSHI_AT_F1_001_2026_07_31.md` | Harvey selects the authoritative objective; no mid-chain mutation; then add a regression and correct the audit disposition | Harvey (decision) + Musashi (implementation) + Satoshi (verify) |
| AUD-GEN-20260731-027 | S3 provisional | open | Satoshi report timestamp 04:40 COT predates audited commits created at 04:58 and 05:16; local file birth is 13:31 | `../CODEX_CROSS_REVIEW_OF_SATOSHI_AT_F1_001_2026_07_31.md` | Publish a non-destructive provenance addendum with separate evidence and write times | Satoshi |
| AUD-GEN-20260731-028 | S3 provisional | open | Satoshi declared AT-F1-001 reported and opened finding 025 but wrote only the report, leaving backlog, findings register and recovery state contradictory to his mandatory handoff lifecycle | `../CODEX_CROSS_REVIEW_OF_SATOSHI_AT_F1_001_2026_07_31.md` | Reconcile recovery state and publish the bounded correction report; backlog/register were repaired by cross-review | Satoshi |
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

### 1d. Executable response, 2026-07-31 ~03:35 (Musashi)

Full disposition:
`../CODEX_DISPOSITION_OF_SATOSHI_INNOVATION_AUDIT_2026_07_31.md`.

- **020 remains open S4.** Read-only evidence found seven peer-tip adoptions
  and 7-second median announcement-to-convergence latency where pairing was
  possible. Recurrence is supported; finalized-anchor divergence and a safety
  failure are not.
- **021 remains open S4.** Three complete 20-candidate generations measured
  8.42% aggregate tail-barrier idle; generation 2 measured 12.05%. This
  confirms material straggler waiting but does not cross the finding's
  aggregate S3 threshold. Evidence:
  `../evidence/SWARM_EFFICIENCY_MEASUREMENT_2026_07_31.json`.
- **009 remains open S3.** The first repository-local Tier A workflow now
  exists and GitHub Actions run `30617139414` passed on commit `af343923`.
  Other Tier A repositories remain without their bounded gates.
- **010 remains open S3.** Future mutation is in the gate; unavailable-market
  and stale/invalid-signal router guards now have executable tests. End-to-end
  fill/ledger fixtures and the remaining mapped invariants are still missing.
- P7/P9 narrowed, P11 held, P14 deferred, P16 first priority with prior-art
  state corrected to `unverified`, and P19 admitted. P15 is retained as a
  separately queryable child of P6 until an objective-plane experiment
  justifies a complete merge.

### 1c. Governance-response audit, 2026-07-31 ~02:45 (Satoshi)

Full evidence: `../AUDIT_MUSASHI_GOVERNANCE_INNOVATION_RESPONSE_2026_07_31.md`.

Closure recommendations for Harvey or an independent verifier (neither
reporter nor implicated party may verify):
**005** → verified_closed (height-9 competition resolved by finalization
advance 2→3 with unanimous anchor; no safety defect; no mutation ever
performed); **014** → verified_closed (Arendt removed, no-weight statement
verified); **015** → verified_closed with hash-pin strengthening;
**016** → verified_closed via the verified authorized-claim table;
**017** → verified_closed (validator ran: 5 packages valid; test passed).

New open findings:

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260731-020 | S4 | open | Recurring equal-height minority tips repeatedly involving Dragon (heights 9 and 10 observed; earlier competitions at 6 and 9 per Musashi's logs). Convergence works; cause and latency unmeasured. Named test: per-peer minority-tip census + announcement-to-adoption latency by route | Musashi (test) + Satoshi (verify) |
| AUD-F1-20260731-021 | S4 (S3 if measured >10 %) | open | Generation-barrier straggler idle: fast workers wait at each generation tail (observed ≥1.3 h × up to 3 GPUs in gen 2); estimated 6–14 % fleet capacity, measurable from existing ETA log pairs. Origin: project owner's direct operational observation. First sub-experiment of registry line P6 | Musashi (measure) + Satoshi (design review) |

Registry decisions proposed: P15 merged into P6; P7/P9/P11 narrowed; P14
deferred (no real traces); P16 first collision priority. Prior-art first pass:
5 primary sources opened and verified (Hyperband, IPFS, BOCPD, Huang et al.
2024, plus PBFT carried); Semantic Scholar returned HTTP 429 — recorded, not
papered over. Process observation (S4): findings 009/010 were answered with
inventories where the artifacts (one CI workflow; nine fixtures) were cheaper.

### 1b. Post-fix verification and academic audit, 2026-07-31 (Satoshi)

Post-fix verification (`../AUDIT_POST_FIX_VERIFICATION_2026_07_31.md`):
006/007/008/013 → **verified_closed** (code + 16-test authorized run);
009 → open S3 accepted; 010 → open S3 narrowed (inventory pending, AT-QUAL-024);
011 → **rejected_as_written, withdrawn** (all six cited suites verified to
exist; residue is specific unautomated scenarios, S4); 012 → open S4 accepted.
Fork check: `deferred_no_new_boundary` (gen 2 at 19/20, finalized still 2).

New findings from the academic audit
(`../AUDIT_ACADEMIC_PUBLICATION_PROGRAM_2026_07_31.md`):

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-GEN-20260731-014 | S3 | open | Unregistered participant "Arendt" appears as independent corroborator in the closure chain; no versioned role definition anywhere in the authority model | Musashi + Harvey |
| AUD-GEN-20260731-015 | S3 | open | Dual-role conflict: auditor is now academic lead and P5 is a paper about the auditor's own system; proposed controls: enumeration-rule incident corpus, Musashi raw-timestamp verification, external review before P5 preprint, in-paper conflict disclosure | Harvey (accept) + Musashi (doc amendment) |
| AUD-ACAD-20260731-016 | S3 | open | P1 vocabulary implies Byzantine/adversarial guarantees while evidence is cooperative crash-fault with signed identities; scope threat model before drafting | Satoshi (framing) + Musashi (code verification) |
| AUD-ACAD-20260731-017 | S4 | open | No `papers/<id>/` scaffold or `claims.csv` exists; claim-to-evidence contract unenforceable once drafting starts | Musashi |

Paper states: P1 evidence_incomplete, P2 evidence_incomplete, P3 outline,
P4 evidence_incomplete, P5 evidence_incomplete. None evidence_ready.

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
| AUD-F2-20260801-029 | S3 | **verified_closed** | Fleet-safe read-only `/v1/status` route on Dragon (redacts fingerprints/balances/tickets/HMAC); Omega consolidated watchdog treats Dragon remote status as authoritative via 0600 env config; false `mt5_bridge_missing` alarm eliminated. Implementer: Musashi. Independently verified by Satoshi: `lts@a5fe0d97` matches, all 4 claimed file SHA-256 match byte-exact, live packet 18:17Z shows heartbeat age 8.6 s / connected / read_only=true / demo / 0 positions / 0 orders / 6 symbols with no MT5 event active, `lts tests/unit: 101 passed` reproduced. Port-62024 SSH clarification accepted; watchdog evidence correctly no longer depends on interactive access. Rider bonus verified same pass: `agent-multi@06de651f` conflict-proving selection-metric test (hash match, suite 35 passed) mechanically guards job-1's weekly-robust fitness | Satoshi (reporter; implementation by the other party — S3 role separation satisfied) | 2026-08-01 | `../CODEX_RESPONSE_TO_AUDIT_DELTA_2026_08_01.md`; live watchdog packet; independent test reruns |
| AUD-F2-20260730-004 | S2 | verified_closed | User accepted the TWS Paper disclaimer; watchdog now requires a recent authenticated reconciled session and overlapping preflights fail closed | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `lts@12d389d`; `205 passed`; successful systemd observer/watchdog run |
| AUD-GEN-20260730-001 | S3 | verified_closed | Document 13 phase summary, ledger and immediate tasks now record the deployed four-worker v2 campaign | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |
| AUD-F1-20260730-002 | S4 | verified_closed | Document 13 records measured fleet throughput, the 10-14 day full-budget range and an end-of-stage-1 duration/evidence decision point | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |
| AUD-GEN-20260730-003 | S4 | verified_closed | Musashi recovery prompt v1.1.0 includes docs 08/11/14/16 and refreshes the runtime warning | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |

### 1i. Temporary-role auditor session, 2026-08-01 ~21:25 (Musashi)

Role state: `ROLE_SWAP_ACTIVE`. Runtime was observed through GET-only campaign
APIs and deterministic watchdog packets; no campaign, broker, Hermes or
production-code mutation occurred.

- `AT-ACADEMIC-031` is **reported**: 16 seeded P1/P5 sources checked, 15
  verified and IACR ePrint 2017/203 rejected as superseded by 2018/559. P1
  and P5 claim matrices were seeded; all efficacy and novelty claims remain
  `planned` or `evidence_incomplete`.
- The owner's six approved improvements were converted into acceptance
  criteria and backlog tasks `AT-GEN-033` through `AT-ACADEMIC-038`.
  Implementation remains Satoshi's responsibility; Musashi cannot close the
  acceptance contract he authored.
- No new finding ID was opened. A malformed pre-existing P4 ledger row was
  structurally repaired while validating the shared CSV; its content and
  evidence state were not promoted.
- `AUD-F4-20260801-034` remains **open**. Both temporary-role participants are
  parties to it, so neither may close it; owner or post-handback independent
  disposition remains required.

Runtime snapshot at 2026-08-01 21:28 America/Bogota: all four workers were
online and `running` in the same job/domain, generation 7, with identical
finalized height 8/hash `4b4f06a1...`, matching component versions and
`join_ready=true`. Campaign progress was 143/480 candidates (29.79% of the
full planned budget; early stopping may reduce it), measured throughput
1.9496 candidates/hour, and full-budget ETA 622266 seconds (7.20 days).
Front-2 watchdog had no active event keys; Alpaca, IBKR and MT5 read-only
probes were available; shadow submitted zero orders.

### 1h. Independent verification of the social-accounting corrections, 2026-08-01 ~19:50 (Satoshi)

Per the owner's direct instruction, the correction contract at `lts@f6d8b21`
and `agent-multi@c24b6ce8` was independently reproduced against current
commits:

- **030 → verified_closed**: doc 28 platform table corrected; owner order
  recorded; registry-exclusion test
  (`test_current_registry_excludes_mql5_demo_and_keeps_protected_ctrader_api`)
  reproduced passing.
- **031 → verified_closed**: hand-refired original attack — full withdrawal
  now crystallizes exactly 10.00 at the configured rate; partial withdrawal
  charges precisely the withdrawn share (5.00); retained-share profit charged
  once later; dip-and-recover to HWM charges nothing (B2b). Conservation test
  in suite.
- **032 → verified_closed**: `max_overshoot_ratio` rejection verified via the
  suite's overshoot test reproduced locally (auditor's hand harness was
  outpaced by the contract's new required fields — a strictness win recorded
  in Musashi's favor).
- **033 → verified_closed**: parametrized step-alignment rejection reproduced.
- Additional demanded verifications: **event-chain tamper detection
  hand-proven** (valid → tampered copy → invalid) on a COPY of the DB;
  v1→v2 migration covered by `test_olap_migrates_v1_tables_without_losing_rows`
  (reproduced); idempotency-duplicate rejection reproduced; suites reproduced:
  focused 17, complete lts **234 passed** (exact match), agent-multi tests/unit
  **404 passed** (claimed 405 — one-test collection delta at newer docs HEAD;
  immaterial, noted for precision).
- Verifier: Satoshi (reporter); implementer: Musashi — role separation per
  established precedent and explicit owner instruction.

**New finding:**

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F4-20260801-034 | S4 | open | `social_trading_cli.py --database` creates the schema at the argument path but persists run/event rows to the default state DB — scenario isolation impossible; discovered when the auditor's scratch DB stayed empty while the default DB grew | Musashi |

**Auditor disclosure:** as a consequence of 034, the auditor's verification
runs (including run `social-c5c91977ff8b4e8d`) were unintentionally persisted
to `~/.local/state/lts/social-trading-lab.sqlite` (9 runs total present).
Lab-only, zero orders, no broker surface — but the writes were not intended
and are disclosed per the auditor's own standard. Tampering was performed
ONLY on scratch copies.

### 1g. Social-trading reality-loop audit, 2026-08-01 (Satoshi)

Full evidence: `../AUDIT_SOCIAL_TRADING_REALITY_LOOP_2026_08_01.md`.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F4-20260801-030 | S3 | implemented_pending_independent_verification | Current docs and registry agree: cTrader/eToro controls first, MQL5 Signals live-only. Owner verified cTrader Copy and submitted Open API app | Satoshi verification |
| AUD-F4-20260801-031 | S3 | implemented_pending_independent_verification | Withdrawal-time proportional fee crystallization, gross/net flow facts, money conservation regression | Satoshi verification |
| AUD-F4-20260801-032 | S4 | implemented_pending_independent_verification | Bounded overshoot plus notional, margin and reserve rejection | Satoshi verification |
| AUD-F4-20260801-033 | S4 | implemented_pending_independent_verification | Minimum and maximum volume step-alignment validation | Satoshi verification |

Implementation evidence: `lts@f6d8b21` and
`../../handoffs/MUSASHI_RESPONSE_AUDIT_F4_SOCIAL_2026_08_01.md`. The
implementation owner does not close these findings.

Verified non-findings (do not reopen without new evidence): no HWM double
charge on recovery; manager fee-immunity guards hold; deposit/withdraw HWM
defaults correct; registry matches official sources on all five checked
claims (MQL5 demo ban, cTrader SL/TP non-copy + equity-to-equity formula +
demo-investor rules + fee caps); lab has zero broker/secret/network surface
(stdlib-only, proven); `orders_submitted=0` deterministic, hashes stable;
111 lts tests pass at `db80d97`; no platform→alpha or social→DOIN path found;
Hermes cannot subscribe/publish/allocate/order. Recommended platform reorder
(cTrader-investor first; MQL5 only with a live-capital decision; Darwinex
Zero as sole virtual provider track) awaits Harvey. Legal questions flagged
for a qualified professional — not answered technically.

### 1f. OWNER DECISION — Objective contract: Alternative A (2026-08-01)

Harvey ratified **Alternative A** from the AT-F1-012 decision packet
(`../AUDIT_OBJECTIVE_CONTRACT_AND_CURRICULUM_2026_07_31.md` §4),
`owner-ratified`, relayed verbatim: "go with A".

Effects, binding by owner authority:

1. Job 0 runs to completion unchanged; its champion is **initialization
   evidence under the declared full-period proxy objective** — never a
   performance claim. No mid-chain mutation (both Generals' standing position,
   now owner-confirmed).
2. **Job 1 is the authoritative selection** under `robust_weekly_rap_fitness`
   (weekly units, immutable scenario suite).
3. Rider (i) — the conflict-proving selection-metric test — was already
   implemented (`agent-multi@06de651f`) and independently verified. Satisfied.
4. Rider (ii) — relabeling the job-0 champion "alpha handoff under
   full-period proxy objective" in all artifacts/docs — Musashi action at or
   before archive time.
5. Archive-time verification — elite warm-start set must contain the weekly
   top-2 candidates against the FINAL chain — event-driven audit task
   (AT-F1-013, scheduled on job-0 completion; not pre-claimed).

Consequent finding states:

- **AUD-F1-20260731-026 (S2) → verified_closed** by owner decision: the
  full-period/weekly conflict is resolved by decree — job-0's objective is
  officially the documented proxy; the weekly contract governs the
  authoritative stage, which is mechanically test-guarded. Verifier: Harvey
  (owner), on the evidence packet both Generals produced and cross-verified.
- **AT-F1-001 → closed**: the safety contract (floors, brackets, firewall,
  artifact integrity) was verified and stands; the unit-contract question that
  kept it open is resolved by the owner decision. Alternative C is formally
  retired (falsified by measurement); Alternative B expires unexercised.

### 1e. Cost-realism check prompted by the owner, 2026-08-01 (Satoshi)

Owner concern: cost/spread/fee assumptions may be defaults, invalidating all
downstream results. Checked three layers deep:

- **Verified non-finding:** job-0 costs are NOT zero/defaults. The deployed
  config selects `/environment/execution_difficulty = "easy_floor"`, resolved
  from `examples/config/execution_curriculum/project3_execution_cost_curriculum_v1.json`:
  commission 0.00005/side, full spread 0.0001, slippage 0.25 bps/side,
  positive as contracted. The base `/environment/commission = 0.0` keys are
  legacy fields overridden by the named profile (nested-truth check applied).
- **OBS-20260801-E (S4):** doc 20 states easy_floor slippage `0.000075`; the
  authoritative curriculum file implements 0.25 bps/side (0.000025). Same
  sign/magnitude, different number — doc precision fix needed.
- **OBS-20260801-F (S3-candidate):** the MT5 observation symbol set (ETH,
  SOL, BTC, ADA, DOGE, EURJPY) **does not include USDCAD** — the asset the
  entire fleet is optimizing has no live cost-observation stream on the active
  venue. Proposed: add USDCAD (and the doc-21 FX universe) to the EA watch
  list. Cheap, directly serves cost realism.
- **Structural gap (feeds AT-F2-014 proposal):** all curriculum scenario
  numbers are hand-declared constants; observed venue facts (Alpaca spreads,
  MT5 snapshots, IBKR sessions) exist but nothing feeds them back into
  scenario/nominal/stress profiles. The observed→scenario calibration loop is
  designed (P4 item 3, doc 21 §4.4) but unimplemented. Scenario changes must
  mint new domain hashes at job boundaries only — never mid-chain.

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
