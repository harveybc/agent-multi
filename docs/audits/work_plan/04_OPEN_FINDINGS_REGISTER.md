# 04. Open Findings Register

Version: 1.7.0
Date: 2026-08-03
Owner: Musashi during `ROLE_SWAP_ACTIVE`; closure of S0-S2 requires an
independent verifier per `../README.md` and dual-party findings go to Harvey.

This file is the cross-session source of truth for finding state. Full finding
stanzas live in their originating report; this register carries identity,
state and the next required action only.

## 1. Open

| ID | Sev | State | Title | Source report | Next action | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| AUD-F1-20260731-026 | S2 provisional | open | AT-F1-001 reconstructed the configured full-period L2 correctly but certified it as weekly-fraction; train tail covers 3 weeks, validation 53, and the mean-weekly reconstruction changes the champion score from positive to negative | `../CODEX_CROSS_REVIEW_OF_SATOSHI_AT_F1_001_2026_07_31.md` | Harvey selects the authoritative objective; no mid-chain mutation; then add a regression and correct the audit disposition | Harvey (decision) + Musashi (implementation) + Satoshi (verify) |
| AUD-GEN-20260731-027 | S3 provisional | open | Satoshi report timestamp 04:40 COT predates audited commits created at 04:58 and 05:16; local file birth is 13:31 | `../CODEX_CROSS_REVIEW_OF_SATOSHI_AT_F1_001_2026_07_31.md` | Publish a non-destructive provenance addendum with separate evidence and write times | Satoshi |
| AUD-GEN-20260731-028 | S3 provisional | open | Satoshi declared AT-F1-001 reported and opened finding 025 but wrote only the report, leaving backlog, findings register and recovery state contradictory to his mandatory handoff lifecycle | `../CODEX_CROSS_REVIEW_OF_SATOSHI_AT_F1_001_2026_07_31.md` | Reconcile recovery state and publish the bounded correction report; backlog/register were repaired by cross-review | Satoshi |

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
| AUD-F2-20260801-039 | S2 | **verified_closed** | Versioned risk-increasing order contract makes naked entries and stop-entry/protective-stop ambiguity inexpressible; four exact adversarial tests independently pass | Musashi | 2026-08-02 | `trading-contracts@e068bb5`; `../AUDIT_SATOSHI_II_L0_CORRECTION_PACKET_2026_08_02.md` |
| AUD-F2-20260802-043 | S2 | **verified_closed** | Long/short SL and TP are anchored to persisted decision reference before reservation | Musashi | 2026-08-02 | `lts@9fe9b64`; exact wrong-side reproductions |
| AUD-F2-20260802-045 | S2 | **verified_closed** | `BEGIN IMMEDIATE` serializes budget check/write across connections; 20 repeated races admitted one of two 1% intents against a 1% cap | Musashi | 2026-08-02 | `lts@9fe9b64`; correction audit |
| AUD-F2-20260802-046 | S2 | **verified_closed** | Post-validation/serialization failure rolls back reservation and persists replayable rejection | Musashi | 2026-08-02 | `lts@9fe9b64`; correction audit |
| AUD-F2-20260801-040 | S3 | **verified_closed** | Separate risk/gross/margin/day-loss dimensions and atomic reservation accounting independently pass directed, concurrent and generated-event checks | Musashi | 2026-08-02 | `trading-contracts@e068bb5`; `lts@6af0300`; `../AUDIT_SATOSHI_II_L0_INTEGRATED_PACKET_2026_08_02.md` |
| AUD-F2-20260802-044 | S2 | **verified_closed** | Persisted exposure lifecycle remains in aggregate risk and position totals independently of terminal order state | Musashi | 2026-08-02 | `lts@6af0300`; 6,000 generated events |
| AUD-F2-20260802-049 | S2 | **verified_closed** | Immutable originals conserve day risk and one logical position across cumulative partial fills | Musashi | 2026-08-02 | `lts@6af0300`; exact reproduction plus 25/50/75/100% audit |
| AUD-F2-20260802-050 | S2 | **verified_closed** | Signed multi-asset exposure identity and short flatten direction persist from immutable decisions | Musashi | 2026-08-02 | `lts@6af0300`; directed and generated-event audit |
| AUD-F2-20260802-051 | S2 | **verified_closed** | Cancel and flatten intents carry exact target order-intent identity | Musashi | 2026-08-02 | `trading-contracts@cd05083`; `lts@6af0300` |
| AUD-F2-20260802-052 | S2 | **verified_closed** | Venue, account fingerprint and environment capability substitution now rejects | Musashi | 2026-08-02 | `lts@6af0300`; exact cross-venue/account reproductions |
| AUD-GEN-20260801-035 | S3 | **verified_closed (owner)** | Direct per-venue order/position counts replace inference from alert absence | Harvey (owner), after Musashi verification | 2026-08-02 | `../../handoffs/OWNER_CLOSURE_DISPOSITION_REQUEST_2026_08_02.md`; `agent-multi@cfae3335` |
| AUD-GEN-20260801-036 | S3 | **verified_closed (owner)** | Contradictory queue states reject and failed jobs remain excluded | Harvey (owner), after Musashi verification | 2026-08-02 | same owner disposition; `agent-multi@b0196a73` |
| AUD-GEN-20260801-037 | S4 | **verified_closed (owner)** | Wrong-type payloads degrade to explicit unavailability; 23 focused, 427 full and 1,500-shape stress passed | Harvey (owner), after Musashi verification | 2026-08-02 | same owner disposition; `agent-multi@c1860130` |
| AUD-GEN-20260801-038 | S4 | **verified_closed (owner)** | Append-only chronology corrected to authoritative 19m40s Git interval | Harvey (owner), after Musashi verification | 2026-08-02 | same owner disposition; `agent-multi@8b660d27` |
| AUD-F4-20260801-034 | S4 | **verified_closed (owner)** | Social CLI database override now preserves scenario isolation | Harvey (owner), dual-party disposition | 2026-08-02 | same owner disposition; `lts@11d8958` |
| AUD-F1-20260730-005 | S3 | **verified_closed (owner)** | Equal-height competition resolved by finalization with no safety defect; recurrence remains under open finding 020 | Harvey (owner) | 2026-08-02 | same owner disposition; finalization evidence in prior reports |
| AUD-GEN-20260731-025 | S3 owner-adjudicated | **verified_closed (owner)** | Enumeration-drift incident retained in P13 corpus; no further correction required | Harvey (owner) | 2026-08-02 | same owner disposition |
| AUD-F2-20260801-029 | S3 | **verified_closed** | Fleet-safe read-only `/v1/status` route on Dragon (redacts fingerprints/balances/tickets/HMAC); Omega consolidated watchdog treats Dragon remote status as authoritative via 0600 env config; false `mt5_bridge_missing` alarm eliminated. Implementer: Musashi. Independently verified by Satoshi: `lts@a5fe0d97` matches, all 4 claimed file SHA-256 match byte-exact, live packet 18:17Z shows heartbeat age 8.6 s / connected / read_only=true / demo / 0 positions / 0 orders / 6 symbols with no MT5 event active, `lts tests/unit: 101 passed` reproduced. Port-22022 SSH clarification accepted; watchdog evidence correctly no longer depends on interactive access. Rider bonus verified same pass: `agent-multi@06de651f` conflict-proving selection-metric test (hash match, suite 35 passed) mechanically guards job-1's weekly-robust fitness | Satoshi (reporter; implementation by the other party — S3 role separation satisfied) | 2026-08-01 | `../CODEX_RESPONSE_TO_AUDIT_DELTA_2026_08_01.md`; live watchdog packet; independent test reruns |
| AUD-F2-20260730-004 | S2 | verified_closed | User accepted the TWS Paper disclaimer; watchdog now requires a recent authenticated reconciled session and overlapping preflights fail closed | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `lts@12d389d`; `205 passed`; successful systemd observer/watchdog run |
| AUD-GEN-20260730-001 | S3 | verified_closed | Document 13 phase summary, ledger and immediate tasks now record the deployed four-worker v2 campaign | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |
| AUD-F1-20260730-002 | S4 | verified_closed | Document 13 records measured fleet throughput, the 10-14 day full-budget range and an end-of-stage-1 duration/evidence decision point | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |
| AUD-GEN-20260730-003 | S4 | verified_closed | Musashi recovery prompt v1.1.0 includes docs 08/11/14/16 and refreshes the runtime warning | Musashi | 2026-07-30 | `../CODEX_AUDIT_FINDING_CLOSURE_2026_07_30.md`; `agent-multi@2617f4cc` |

### 1k. Counterpart-session continuity event, 2026-08-01 ~22:35 (Musashi)

Owner reported that the prior Satoshi conversation was deleted after a
security problem. No versioned evidence establishes the root cause, so no new
security finding is asserted from the report alone. Repository continuity was
reconstructed instead of relying on the lost chat.

- The predecessor completed corrections 035-037 at `b0196a73`, wrote the
  bounded response at `49dcb20d`, added continuous demo-trading doctrine in
  work-plan document 29 at `92e9c756`, and issued its audit addendum at
  `fa5342a0` before loss.
- A successor cold-start specification was created at
  `../../handoffs/GENERAL_SATOSHI_II_NOVICE_TECHNICAL_LEAD_PROMPT_2026_08_01.md`.
  It preserves the temporary technical-lead role without attributing memories
  to a new conversation and requires staged reconstruction before activation.
- Prompt version 1.1.0 adds the owner-directed `Bella Flor Safety Code`:
  defensive authorized cybersecurity only; dangerous cyber, biological,
  chemical, weapons and physical-harm work is outside mission scope; safe
  portions of mixed requests continue rather than causing global inactivity.
- One predecessor dirty change remains preserved and unstaged:
  `AUDIT_SIX_IMPROVEMENTS_FIRST_PACKET_2026_08_01.md` contains only a malformed
  date-space edit in the observed diff. It was not overwritten or included in
  this continuity work.
- Fresh read-only status at 22:31 America/Bogota: all four DOIN workers were
  online in the same job/domain/generation 7 and finalized anchor; 150/480
  full-budget candidates complete; venue direct counts were zero orders and
  zero positions; no consolidated unavailable fields.

Role state remains `ROLE_SWAP_ACTIVE`, with counterpart activation pending
delivery and completion of `NOVICE_BOOTSTRAP`. Findings 035-037 remain
implemented pending Musashi verification; doctrine audit `AT-F2-039` remains
proposed and unexecuted.

### 1j. Six-improvements first packet, 2026-08-01 ~21:45 (Musashi)

Full evidence: `../AUDIT_SIX_IMPROVEMENTS_FIRST_PACKET_2026_08_01.md`.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-GEN-20260801-035 | S3 | open | Empty alert set can falsely report zero orders despite positive direct venue counts | Satoshi |
| AUD-GEN-20260801-036 | S3 | open | Queue validator accepts contradictory semantics and maps failed jobs to materialized | Satoshi |
| AUD-GEN-20260801-037 | S4 | open | Network provenance and partial/wrong-type payload honesty are incomplete | Satoshi |

Verified non-findings: the live packet materially agreed with independent
source reads; live queue states were coherent; 9 focused and 414 full tests
passed; all checks were read-only or temporary-fixture based. Criteria 1 and
4 are `reported_changes_required`, not accepted. Criteria 2/3/5/6 retain their
declared partial states and were not prematurely audited.

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

### 1h. Satoshi II cold start and continuous-demo doctrine, 2026-08-01 (Musashi)

Full evidence:

- `../AUDIT_SATOSHI_II_COLD_START_AND_STATUS_FIXES_2026_08_01.md`
- `../AUDIT_CONTINUOUS_DEMO_TRADING_DOCTRINE_2026_08_01.md`

State updates for prior findings:

| ID | State update | Evidence |
| --- | --- | --- |
| AUD-GEN-20260801-035 | correction independently reproduced; owner/post-handback closure recommended | direct per-venue reconstruction plus 18 focused tests |
| AUD-GEN-20260801-036 | correction independently reproduced; owner/post-handback closure recommended | original counterexamples reject; failed/completed jobs excluded |
| AUD-GEN-20260801-037 | remains open; correction incomplete | truthy wrong-type status, non-numeric direct count and wrong-type plan-job still crash |

New findings:

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-GEN-20260801-038 | S4 | open | Satoshi II report write time and claimed commit window conflict with authoritative Git commit time; append-only chronology correction required | Satoshi II |
| AUD-F2-20260801-039 | S2 | open, blocks L0 acceptance | `OrderIntent.v1` accepts risk-increasing entries without SL/TP and overloads `stop_price` between entry trigger and protection | Satoshi II |
| AUD-F2-20260801-040 | S3 | open, blocks L0 acceptance | `rel_volume` conflates notional, margin and loss-at-stop; no atomic aggregate loss reservation contract | Satoshi II |
| AUD-F2-20260801-041 | S3 | open, blocks L0 acceptance | Execution lifecycle cannot unambiguously represent partial fills, unknown acknowledgements, cancellation and per-leg bracket protection | Satoshi II |
| AUD-F2-20260801-042 | S3 | open, blocks L0 acceptance | Telegram hold/kill path is not yet required to bypass Hermes/LLM inference through a deterministic authenticated command surface | Satoshi II |

`AT-GEN-043` passed with the chronology correction above. `AT-F2-039` is
`reported_changes_required`; no broker write path was enabled. The assigned
technical packet is the adversarial L0 contract-first fixture set, after the
bounded finding-037 correction.

State update, 2026-08-02 independent verification:

- `AUD-GEN-20260801-037`: correction independently verified with 23 focused
  tests, 427 full unit tests, the submitted wrong-shape packet and an
  additional 1,500-case deterministic JSON-shape stress run with zero
  exceptions. Owner/post-handback closure recommended; Musashi does not
  self-close the finding he authored.
- `AUD-GEN-20260801-038`: append-only chronology correction independently
  verified against Git. Owner/post-handback closure recommended.
- Live-demo vertical observed state: not implemented or running at
  `agent-multi@55d72575`, `trading-contracts@534b034`, `lts@11d8958`,
  `prediction_provider@ac4d9e2`. The interface map appeared as concurrent
  Satoshi II work, but it must be included in the implementation packet and
  is not an acceptable standalone return.
- Canonical evidence:
  `../AUDIT_SATOSHI_II_037_038_AND_LIVE_DEMO_STATUS_2026_08_02.md`.

### 1i. Satoshi II L0 implementation delta, 2026-08-02 (Musashi)

Canonical evidence:

- `../AUDIT_SATOSHI_II_L0_IMPLEMENTATION_DELTA_2026_08_02.md`

Positive evidence: versioned v2 contracts and the zero-network LTS library
are substantive and pushed; 84 trading-contract tests and 145 LTS unit tests
independently pass.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F2-20260802-043 | S2 | open, blocks L0/L1 | Market order accepts wrong-side SL/TP because protection is not anchored to decision reference price | Satoshi II |
| AUD-F2-20260802-044 | S2 | open, blocks L0/L1 | Filled open exposure disappears from gross/margin/position totals and bypasses max positions | Satoshi II |
| AUD-F2-20260802-045 | S2 | open, blocks L0/L1 | Check-then-reserve spans transactions; concurrent intents exceeded a 1% cap with 2% persisted risk | Satoshi II |
| AUD-F2-20260802-046 | S2 | open, blocks L0/L1 | Post-reservation validation failure leaks active capacity with no recorded decision | Satoshi II |
| AUD-F2-20260802-047 | S3 | open, blocks L1 | Accepted flatten/cancel and emergency-flatten paths emit no risk-reducing intent | Satoshi II |
| AUD-F2-20260802-048 | S3 | open, blocks L0 acceptance | L0 is a tested library but has no runner, config, deployment unit or continuously advancing ledger | Satoshi II |
| AUD-F2-20260802-049 | S2 | open, blocks L0/L1 | Partial fill undercounts daily risk, double-counts one logical position and permits aggregate risk above cap | Satoshi II |
| AUD-F2-20260802-050 | S2 | open, blocks L0/L1 | Exposure loses sign, asset/instrument and provenance; short flatten increases risk and non-FX fill becomes USD.CAD | Satoshi II |
| AUD-F2-20260802-051 | S2 | open, blocks L0/L1 | Cancel intent uses placeholders and cannot identify the original or broker order to cancel | Satoshi II |
| AUD-F2-20260802-052 | S2 | open, blocks L0/L1 | LTS accepts capability evidence from a different venue/account | Satoshi II |

Correction state update, 2026-08-02:

- 043, 045 and 046 are independently reproduced and `verified_closed` by
  Musashi at `lts@9fe9b64`.
- 044 and 047 remain `in_progress`; their first counterexamples are fixed but
  049–051 expose unsafe partial/short/cancel behavior.
- 039 is independently verified closed. 040–042 remain open pending the
  connected runtime contract.
- Canonical evidence:
  `../AUDIT_SATOSHI_II_L0_CORRECTION_PACKET_2026_08_02.md`.

### 1l. Satoshi II integrated L0 runtime audit, 2026-08-02 (Musashi)

Canonical evidence:

- `../AUDIT_SATOSHI_II_L0_INTEGRATED_PACKET_2026_08_02.md`

Independent suites pass 95 contracts, 16 provider-mechanics, 60 focused L0,
295 full LTS and 429 agent-multi unit tests. An additional 6,000 generated
events and cumulative 25/50/75/100% fills preserve the submitted invariants.
Findings 040, 044 and 049-052 are verified closed.

The deployed process is genuinely alive and zero-network, but its first active
reservation permanently consumes the gross cap; it then replays a rejection
instead of exercising lifecycle mechanics. `AT-F2-040` remains
`reported_changes_required`.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F2-20260802-053 | S2 | open, blocks L0/L1 | Concurrent identical intents race outside the atomic unit; one crashes on the decision primary key instead of replaying | Satoshi II |
| AUD-F2-20260802-054 | S2 | open, blocks L0/L1 | Lifecycle previous-state validation occurs before the transaction; concurrent reports produced `requested -> filled -> accepted` | Satoshi II |
| AUD-F2-20260802-055 | S2 | open, blocks L1 | Owner kill is persisted accepted before effects; a crash makes same-nonce retry reject without resuming flatten/cancel | Satoshi II |
| AUD-F2-20260802-056 | S3 | open, blocks L0 | Continuous L0 saturates after one pending reservation and does not continuously evaluate persisted invariants or alert on its degraded state | Satoshi II |
| AUD-F2-20260802-057 | S2 | open, blocks L1/L2 | No route binding connects policy asset, quote symbol and venue instrument; BTC intent against ETH quote/instrument was accepted | Satoshi II |
| AUD-F2-20260802-058 | S3 | open, blocks L1/L2 | Future-dated quotes have negative age and are accepted as fresh; quote geometry lacks an explicit validation gate | Satoshi II |

Findings 041, 042, 047 and 048 remain open under the connected failures 054,
055 and 056. No L1 broker write is authorized.

### 1m. Solvency-relaxation curriculum collision test, 2026-08-02 (Musashi)

Canonical evidence:

- `../AUDIT_SOLVENCY_RELAXATION_CURRICULUM_2026_08_02.md`

The owner's historical NEAT observation is accepted as a credible research
prior and registered as P20. It does not alter job 0 or job 1: 12,966 visible
current-campaign evaluation summaries remain far above the inherited 1%
equity floor, while stochastic training termination causes are unobservable.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260802-059 | S4 | open, blocks P20 measurement only | `gym-fx` collapses insolvency and natural data exhaustion into one boolean; training/candidate/OLAP evidence cannot measure termination cause | Lieutenant Satoshi II |

The approved diagnostic design compares realistic hard reset, penalized
liquidation-plus-recapitalization with chronological continuation, and a
deterministic randomized-start control. Every arm is selected under realistic
solvency. No GPU run or campaign mutation is authorized by this registration.

### 1n. L0 acceptance and MT5 AT-F2-006, 2026-08-02 (Musashi)

Canonical evidence:

- `../AUDIT_L0_ACCEPTANCE_AND_MT5_AT_F2_006_2026_08_02.md`

State changes independently verified against Satoshi II's implementation:

- 053-058 -> `verified_closed`;
- connected findings 041, 042, 047 and 048 -> `verified_closed`;
- `AT-F2-040` -> `verified_passed`;
- the post-packet settled-exposure retry defect fixed at `lts@77bf02e` is
  classified S4 and `verified_closed` in the same independent pass.

Evidence includes 68 focused and 303 full LTS tests, 95 contract tests, 16
provider-mechanics tests, 431 agent-multi tests, direct read-only ledger
reconstruction, valid lifecycle hash-chain verification and a controlled
service restart with no duplicate effects or network submissions.

New MT5 write-readiness findings:

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F2-20260802-060 | S2 | open, blocks MT5 L1 only | Reviewed and deployed MT5 EA is read-only, has no command path, and running EX5/source provenance is not attested | Lieutenant Satoshi II |
| AUD-F2-20260802-061 | S2 | open, blocks MT5 L1 only | Deployed MT5 account-fingerprint allowlist is empty and therefore fail-open | Lieutenant Satoshi II |
| AUD-F2-20260802-062 | S3 | open, blocks MT5 L1 only | MT5 bridge reads unbounded request bodies before authentication | Lieutenant Satoshi II |

These findings do not block the IBKR Paper L1 path. The protection gate is
amended for owner ratification: the first minimum-size bracket canary is the
native-protection verification instrument, with complete parent/SL/TP group
transmission and deterministic flatten/hold on any ambiguous evidence.

### 1o. IBKR Paper L1 adapter audit, 2026-08-03 (Musashi)

Canonical evidence:

- `../AUDIT_SATOSHI_II_IBKR_L1_ADAPTER_2026_08_03.md`
- `../evidence/IBKR_L1_ADAPTER_REPRO_2026_08_03.py`

The bracket translator and read-only preflight are retained. IBKR Paper L1 is
not activated: the audited submission path performs no broker call and the
adapter is not integrated into the accepted L0 runtime.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F2-20260803-063 | S2 | open, blocks IBKR L1 | `submit_bracket()` reports submitted after counters without calling the broker | Satoshi III successor technical lead |
| AUD-F2-20260803-064 | S2 | open, blocks IBKR L1 | Repository phrase plus arbitrary local token is self-authorization; no durable owner-issued capability or atomic token ledger exists | Satoshi III successor technical lead |
| AUD-F2-20260803-065 | S2 | open, blocks IBKR L1 | Cancelled/rejected/wrong-type/wrong-price protection can verify as protected; cancel/flatten/hold is only a string | Satoshi III successor technical lead |
| AUD-F2-20260803-066 | S2 | open, blocks IBKR L1 | L1 adapter is absent from the accepted L0 risk, outbox, lifecycle, runner and OLAP path | Satoshi III successor technical lead |
| AUD-F2-20260803-067 | S3 | open, blocks IBKR L1 | L1 profile admits arbitrary venue/host and non-positive limits; declared quantity/distance/spread fields are unused | Satoshi III successor technical lead |
| AUD-F2-20260803-068 | S4 | open, does not independently block L1 | Account evidence compares single-hash identity with double-hashed account-set identity and raises a false discrepancy | Satoshi III successor technical lead |

### 1p. IBKR Paper L1 Milestones A-E independent audit, 2026-08-03

Canonical evidence:

- `../AUDIT_SATOSHI_III_IBKR_L1_MILESTONES_A_E_2026_08_03.md`
- `../evidence/IBKR_L1_MILESTONES_A_E_REPRO_2026_08_03.py`

Corrections 063-068 reproduced: 164 focused and 467 full LTS tests pass.
They are implemented and independently verified against their original defect
statements, pending the role-swap closure protocol. This does not authorize
IBKR L1 because the integrated path exposed the following new findings:

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F2-20260803-069 | S2 | independently_verified_pending_owner_closure | Protection can disappear after initial acknowledgement; parent fill is accepted using synthetic planned SL/TP legs without current broker verification | owner closure |
| AUD-F2-20260803-070 | S2 | independently_verified_pending_owner_closure | Flatten trusts outbox delta/current account and can increase or reverse direct broker exposure before reconciliation | owner closure |
| AUD-F2-20260803-071 | S2 | independently_verified_pending_owner_closure | Partial fills create broker exposure invisible to L0 because fill sync requires terminal `Filled` and assumes full requested quantity | owner closure |
| AUD-F2-20260803-072 | S2 | independently_verified_pending_owner_closure | Restart rebuilds a mutable bracket and drops capability `contract_con_id`, weakening acknowledgement identity | owner closure |
| AUD-F2-20260803-073 | S3 | independently_verified_pending_owner_closure | Crash after capability/effect commit but before first call remains `journaled_pending` forever and stalls the canary | owner closure |
| AUD-F2-20260803-074 | S3 | independently_verified_pending_owner_closure | L1 flatten appends lifecycle directly because accepted L0 protection semantics are not intent-class-aware | owner closure |

### 1q. Selected-model Paper continuity audit, 2026-08-03

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F2-20260803-075 | S3 | implemented_pending_independent_verification | After TWS reconnect, a filled parent disappeared from open-order facts although its completed-order and execution facts remained available; exact bracket verification therefore invoked fail-closed recovery. `lts@cffdc13` joins completed orders to executions strictly by permanent order id and reconstructs only directly evidenced completed parents. | Satoshi III verifies; owner closes |
| AUD-F2-20260803-076 | S4 | implemented_pending_independent_verification | Temporary TWS quote unavailability (`10197`, competing market-data session) made the continuous Paper runner exit and enter a systemd restart loop. `lts@8b67235` records `waiting_for_quote`, submits zero orders, keeps monitoring and retries without process churn. | Satoshi III verifies; owner closes |
| AUD-F2-20260803-077 | S3 | implemented_pending_independent_verification | A cumulative fill within floating-point tolerance of the requested quantity could be classified terminal while retaining a non-exact `filled_units`, violating the execution contract and stalling reconciliation. `lts@8b67235` canonicalizes the terminal amount and rejects negative cumulative facts. | Satoshi III verifies; owner closes |
| AUD-F2-20260803-078 | S4 | implemented_pending_independent_verification | The venue-neutral L0 serializer labeled Alpaca evidence as `ibkr_paper.bracket.v1`. `lts@5cc606f` binds future protected-intent evidence to the actual venue; historical append-only payloads remain unchanged. | Satoshi III verifies; owner closes |
| AUD-F2-20260803-079 | S3 | implemented_pending_independent_verification | `LtsMt5ModelBridge.mq5` failed MetaEditor compilation at line 669 because `BarJson()` omitted the string-concatenation operator before the closing JSON brace. `lts@ebdfec5` repairs the syntax and adds a source regression assertion; the corrected ISO is mounted in the VM. | Satoshi III verifies MetaEditor zero-error compile; owner closes |
| AUD-F2-20260803-080 | S2 | implemented_pending_independent_verification | The MT5 execution EA could not poll signed GET commands because empty-body SHA-256 handling failed and response-header lookup was case-sensitive. `lts@74ec402` adds the canonical empty hash, crypto self-test and case-insensitive header parsing. | Satoshi III reproduces and verifies; owner closes |
| AUD-F2-20260803-081 | S3 | implemented_pending_independent_verification | The execution bridge inherited the read-only operational-status label, causing fleet consumers to misclassify the writable EA. `lts@74ec402` reports the v2 execution bridge, `read_only=false` and `execution_enabled=true`. | Satoshi III verifies direct status; owner closes |
| AUD-F2-20260803-082 | S2 | implemented_pending_independent_verification | The EA required `OrderCheck` to return the `OrderSend` success retcode `10009`; valid checks return boolean success with retcode `0`, so every protected command was refused before broker submission. `lts@5aeea9c` separates check success from strict send-result validation. | Satoshi III reproduces source/runtime correction; owner closes |
| AUD-F2-20260803-083 | S3 | implemented_pending_independent_verification | The consolidated watchdog treated every MT5 position as unexpected under an obsolete read-only assumption. `lts@44bb639` now requires ticket-, symbol-, side-, volume-, SL- and TP-level reconciliation to a successful model command; altered or foreign exposure remains critical. | Satoshi III verifies tests and live status; owner closes |
| AUD-F2-20260803-084 | S3 | implemented_pending_independent_verification | An Alpaca effect could reach `terminal_flat` without applying accepted/filled/closed lifecycle facts to L0, leaving an active reservation after the broker was flat and rejecting the next valid signal as `max_concurrent_positions`. `lts@bc974d5` reconstructs immutable effect identity, reconciles cumulative broker fills and terminal state idempotently, and retries only the signal blocked by that stale terminal reservation. | Independent auditor reproduces exact fixture and runtime repair; owner closes |
| AUD-F2-20260803-085 | S3 | implemented_pending_independent_verification | The consolidated watchdog treated authorized Alpaca/IBKR exposure as unexpected, and the read-only IBKR observer saw only its own client's orders rather than both protection children. `lts@6daf85e` requests all open API-client orders and requires fresh account-, environment-, instrument- and model-bound writable heartbeats plus direct count/protection reconciliation. | Independent auditor verifies 544-test suite and zero-alert live packet; owner closes |

### 1r. Succession/regime audit and P0/K0 disposition, 2026-08-03 (Musashi)

Canonical evidence:

- `../AUDIT_CHAMPION_SUCCESSION_AND_REGIME_RESEARCH_2026_08_03.md`
- `../../work_plan/32_CHAMPION_SUCCESSION_AND_REGIME_RESEARCH.md` v1.1.0

P0 state changes are appended without rewriting section 1q:

- 075-078 and 080-085 -> `independently_verified_pending_owner_closure`;
- 079 -> `implemented_pending_vm_compile_verification`; direct Dragon status
  confirms the corrected v2 bridge is connected and write-enabled, but the
  exact MetaEditor zero-error/zero-warning output remains independently
  unavailable;
- Dragon's MT5 bridge and model runner are `systemd --user` services. A
  system-scope lookup is not downtime evidence.

K0 is accepted as `proceed_with_revisions`; K1 may start, K2 remains gated.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260803-086 | S3 | corrected_pending_independent_verification | Sharpe-specific deflation was applied by analogy to RAP without a defined estimator | Satoshi III verifies |
| AUD-F1-20260803-087 | S3 | corrected_pending_independent_verification | Regime headroom could use hindsight labels and did not prove an actionable causal router | Satoshi III verifies |
| AUD-F2-20260803-088 | S3 | corrected_pending_independent_verification | Seven-day tenure conflated operational compatibility with statistical superiority | Satoshi III verifies |
| AUD-F1-20260803-089 | S3 | corrected_pending_independent_verification | Parallel R1 DOIN domains violated the canonical one-swarm/one-chain sequence | Satoshi III verifies |
| AUD-GEN-20260803-090 | S4 | corrected_pending_independent_verification | Regime references were imprecise and the AIMS ensemble-HMM paper year was wrong | Satoshi III verifies |

### 1s. TWS restart and post-recovery continuity, 2026-08-04 (Musashi)

Direct runtime evidence after the owner restored TWS Paper:

- port 7497 accepted an IBKR API connection and the model runner resumed;
- direct broker facts reported zero positions and zero open orders;
- the previously ambiguous L1 effect reconciled to `terminal_flat`;
- the L0 ledger still contained `halt=hold`, so the fresh model decision was
  correctly rejected as `halted:hold`;
- the consolidated watchdog subsequently reported no active incident.

The fail-closed recovery therefore protected the Paper account. Two continuity
defects remain. They are independent of broker profitability and must not be
hidden by the successful flat reconciliation.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F2-20260804-091 | S3 | open, blocks unattended IBKR continuity | A TWS connection refusal during runner construction exits the process; systemd restarted it roughly every 20 seconds for about 11 hours, leaving a stale heartbeat instead of an advancing degraded heartbeat and accumulating more than 2,000 restarts. | Satoshi III implements; Musashi verifies |
| AUD-F2-20260804-092 | S2 | open, blocks new IBKR Paper risk after recovery | The reported owner path for clearing the post-recovery hold does not exist: `DemoExecutionConfig` permits only risk-reducing verbs, explicitly rejects `resume`, `apply_owner_command()` never clears `halt`, and the deployed profile has no command phrase. The account is proven flat but the runner remains indefinitely blocked. | Satoshi III implements; Musashi verifies; owner authorizes use |

Finding 092 does not authorize a generic risk-enabling command. Its correction
must implement a narrowly scoped, replay-resistant `resume_after_reconciliation`
transition that requires fresh direct evidence for the exact Paper account:
zero positions, zero open orders, no unknown effect, a terminal recovery state,
no active P0/P1 cause and an explicit owner-issued capability. It must update
the ledger atomically, preserve an audit trail and fail closed on every missing
fact. Manual SQLite edits are prohibited.

### 1t. Live alerting and business-evidence delivery audit, 2026-08-04

Canonical evidence:

- `../AUDIT_SATOSHI_III_LIVE_ALERTING_AND_BUSINESS_EVIDENCE_2026_08_04.md`
- `../evidence/SATOSHI_III_LIVE_ALERTING_REPRO_2026_08_04.py`

Finding 091 is corrected and independently verified pending owner closure.
Finding 092 remains open: the replacement resume path is not accepted because
093 and 094 defeat its new-hold and owner-authorization contracts. The owner
must not run the current resume CLI.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F2-20260804-093 | S2 | open, blocks IBKR Paper resume | Resume checks safety state before its transaction; a later `kill` is overwritten by `halt=none` | Satoshi III implements; Musashi verifies |
| AUD-F2-20260804-094 | S2 | open, blocks IBKR Paper resume | Public phrase, unsigned JSON and PTY checks do not authenticate the human owner | Satoshi III implements; Musashi verifies; owner provisions human boundary |
| AUD-GEN-20260804-095 | S2 | open, blocks alert acceptance | Secret-shaped values in ordinary/nested JSON bypass regex redaction and can reach SQLite/Telegram | Satoshi III implements; Musashi verifies |
| AUD-GEN-20260804-096 | S2 | open, blocks alert acceptance | Forwarding keys are not host/source-bound and arbitrary non-empty recovery evidence resolves incidents | Satoshi III implements; Musashi verifies |
| AUD-GEN-20260804-097 | S2 | open, blocks alert failover acceptance | Worker treats SSH ingestion as Telegram delivery, leaving owner router/transport as an unobserved single point of failure | Satoshi III implements; Musashi verifies |
| AUD-F2-20260804-098 | S3 | open, blocks business-evidence acceptance | Consolidated status reports read-only/zero-order facts contradicted by current writable heartbeats and Paper history | Satoshi III implements; Musashi verifies |
| AUD-F2-20260804-099 | S3 | open, blocks IBKR accounting acceptance | Broker effect is terminal-flat but its L0 exposure remains open | Satoshi III implements; Musashi verifies |
| AUD-F2-20260804-100 | S3 | open, blocks unattended reconciliation | TWS cache remained contradictory after 1100/1102 without an explicit authoritative-source convergence contract | Satoshi III implements; Musashi verifies |
| AUD-F2-20260804-101 | S3 | open, blocks Alpaca continuity acceptance | Daily order-budget exhaustion is an exception/restart condition rather than a durable decision outcome | Satoshi III implements; Musashi verifies |
| AUD-F2-20260804-102 | S2 | independently_verified_pending_owner_closure | One Alpaca bar produced four protected Paper round trips through fresh retry identities; fixes at `a9b9d41`/`9a8d568` prevent recurrence | owner closes |
| AUD-F2-20260804-103 | S3 | open, blocks retry acceptance | IBKR construction backoff retries every exception, including fatal config/account/security/programming defects | Satoshi III implements; Musashi verifies |
| AUD-F2-20260804-104 | S3 | open, blocks continuous MT5 Demo trading | MT5 stop closed the broker position, but L0 retained an active reservation and only a `requested` lifecycle; all later due bars reject `max_concurrent_positions` | Satoshi III implements; Musashi verifies |

The delivery scores 56.7/100 under the preregistered rubric and is not
accepted. Complete suites pass, Project 3 is terminalized, DOIN is untouched,
and MT5 produced one real protected Demo round trip. A later direct-account
snapshot corrected the initial interpretation: the broker is flat and finding
104 tracks the stale L0 capacity. Those strengths are retained while the
correction/business-evidence order executes.

### 1u. Live alerting overnight delta, 2026-08-04 (Musashi)

Canonical evidence:

- `../AUDIT_SATOSHI_III_LIVE_ALERTING_OVERNIGHT_DELTA_2026_08_04.md`
- `../evidence/SATOSHI_III_OVERNIGHT_DELTA_REPRO_2026_08_04.py`

The addendum through `agent-multi@5e811a64` is not the ordered correction
packet. The canonical reproducer still breaks 093, 095 and 096; 094 is also
unchanged. Direct runtime evidence independently verifies correction 099:
`lts@83cc286` closed the prior ghost exposure through the accepted lifecycle
API while preserving the new acknowledged, protected IBKR Paper position.
Finding 099 is now `independently_verified_pending_owner_closure`; finding 100
remains open because direct-flat authority still depends on the TWS client
cache.

Current direct venue facts:

- IBKR Paper: one selected-model USD.CAD short, 25,000 units, with matching
  native TP and SL children; fresh monitoring heartbeat;
- Alpaca Paper: broker flat, but one orphan active reservation blocks the next
  eligible bar;
- MT5 Demo: broker flat and bridge healthy, but reservation 104 remains active
  with only a `requested` lifecycle event.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F2-20260804-105 | S3 | open, blocks continuous Alpaca Paper trading | Broker is flat and defect-era effects are terminal, but an orphan active L0 reservation without active exposure/effect still consumes the single-position capacity | Satoshi III implements; Musashi verifies |
| AUD-F2-20260804-106 | S3 | open, blocks TWS clock-alert acceptance | A fresh heartbeat with both clock fields absent is accepted as clock-healthy without validating the runner state; malformed `decided` evidence can suppress/recover the alert | Satoshi III implements; Musashi verifies |
| AUD-F2-20260804-107 | S3 | corrected_pending_independent_verification | Retired MT5 read-only unit remained enabled and restart-looped against the execution bridge port; `lts@29d6f6c`/`76b2afc` retire and mask it | Satoshi III verifies; owner closes |

### 1v. ETH champion/stage-curriculum audit, 2026-08-05 (Musashi)

Canonical evidence:

- `../AUDIT_SATOSHI_III_ETH_CHAMPION_STAGE_CURRICULUM_2026_08_05.md`
- `../evidence/SATOSHI_III_ETH_CURRICULUM_REPRO_2026_08_05.py`
- `../../handoffs/MUSASHI_TO_SATOSHI_III_ETH_CURRICULUM_CORRECTION_ORDER_2026_08_05.md`

The delivery is rejected. The ETH campaign was stopped and disabled on all
three hosts after direct evidence showed an impossible objective contract,
failed candidates accepted as champions, cross-asset artifact paths and three
same-height chain tips. The invalid run is preserved as evidence and is not
resumable scientific work.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260805-108 | S1 | open, blocks any ETH campaign restart | Runtime trains/checkpoints with `risk_adjusted_return` while the outer `lexicographic_weekly_v1` objective is unimplemented and raises after training | Satoshi III implements; Musashi verifies |
| AUD-F1-20260805-109 | S1 | open, blocks any optimization domain | Plugin evaluation failures return finite worst fitness without rejection and can become champions/accepted blocks | Satoshi III implements; Musashi verifies |
| AUD-F1-20260805-110 | S1 | open, blocks ETH artifact materialization | ETH optimizer outputs still target historical USDCAD paths and conflict with the campaign handoff namespace | Satoshi III implements; Musashi verifies |
| AUD-F1-20260805-111 | S2 | open, blocks distributed smoke/full campaign | Four workers persisted on three equal-height tips; fork resolution raised `IndexError` during a peer rollback/fetch race | Satoshi III implements; Musashi verifies |
| AUD-F1-20260805-112 | S2 | open, blocks candidate/champion selection | Weighted transport scalar reverses the declared authoritative lexicographic order | Satoshi III implements; Musashi verifies |
| AUD-F1-20260805-113 | S3 | open, blocks GPU smoke acceptance | Genome emits forbidden `preprocessing_mode=none` candidates with no repair rule | Satoshi III implements; Musashi verifies |
| AUD-F1-20260805-114 | S3 | open, blocks curriculum evidence acceptance | Three-arm mechanism fixture exposes protected-test outcomes and its claimed result packet is not preserved/addressable | Satoshi III implements; Musashi verifies |
| AUD-F1-20260805-115 | S3 | open, blocks unattended campaign operations | Supervisor service stop leaves compute workers alive; direct TERM/INT did not interrupt active candidates | Satoshi III implements; Musashi verifies |
| AUD-F2-20260805-116 | S3 | open, blocks claimed full-suite acceptance | LTS rolling-evidence test uses wall-clock lifecycle creation against a fixed `as_of` and fails after that timestamp | Satoshi III implements; Musashi verifies |
| AUD-F1-20260805-117 | S1 | corrected_pending_independent_verification; blocks ETH curriculum restart | The easy phase changed solvency only, accepted zero activity, selected early-stop checkpoints by flat economic equity and exported a post-easy artifact even when no action reached the strategy | Musashi implemented emergency correction; independent technical lead verifies |
| AUD-F1-20260805-118 | S2 | corrected_pending_independent_verification; blocks any zero-deadband curriculum | `gym-fx` interpreted configured `continuous_action_threshold=0.0` as `0.33`, preventing the intended no-deadband easy action mapping | Musashi implemented emergency correction; independent technical lead verifies |

Findings 117-118 were reproduced from seed 2703 before correction. The
replacement seed-2711 smoke produced 883 easy-training trades, 895 submitted
protected entries, zero protected-entry rejections and 122 realistic-validation
trades. Suites: `gym-fx` 84 passed; `agent-multi` 526 passed. This evidence
permits deployment for a corrected smoke but does not close either finding and
does not authorize resuming the rejected chain from findings 108-116.

### 1w. ETH correction delivery and curriculum decision audit, 2026-08-05

Canonical evidence:

- `../AUDIT_SATOSHI_III_ETH_CORRECTION_DELIVERY_2026_08_05.md`
- `../../handoffs/MUSASHI_TO_SATOSHI_III_ETH_CURRICULUM_DECISION_ORDER_2026_08_05.md`

State transitions are append-only; section 1v remains the original history.

| Finding | Appended state | Basis / remaining work |
| --- | --- | --- |
| 108 | independently_verified_pending_owner_closure | objective resolves and current runtime uses `lexicographic_weekly_v1` |
| 109 | independently_verified_pending_owner_closure | plugin and DOIN independently reject failures/sentinels/non-finite results |
| 110 | independently_verified_pending_owner_closure | deterministic ETH-only arm roots and current ETH-only runtime namespace |
| 111 | independently_verified_pending_owner_closure | fork tests plus one exact live four-worker tip; large block sync extended at `doin-node@9eba394` |
| 112 | corrected_pending_owner_bound_ratification | exact bounded mixed-radix tuple order verified; bounds remain an owner decision |
| 113 | partially_corrected_open | `none` removed from choices, but declared `forbid_value` rule is unsupported by the runtime interpreter |
| 114 | partially_corrected_open | test disabled and report hash committed; resolved configs, return traces and retrievable artifacts remain absent |
| 115 | partially_corrected_open | stop/escalation works; unavailable GPU evidence passes and no same-chain resume operation exists |
| 116 | independently_verified_pending_owner_closure | current complete LTS suite passes with deterministic timestamps |

New findings:

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260805-119 | S2 | open, blocks unattended profile switching | An active campaign's systemd profile can be overwritten; a restart in the observed interval could launch/adopt the wrong domain | Satoshi III implements; Musashi verifies |
| AUD-F1-20260805-120 | S3 | open, blocks curriculum generalization | One-seed/two-epoch unequal-compute fixture with zero margin events cannot decide normal vs easy-normal rollout | Satoshi III executes paired packet; Musashi verifies; owner decides rollout |
| AUD-F1-20260805-121 | S3 | open, blocks temporary A/B interruption | Operator pause is sticky with no supported same-chain resume, and unavailable GPU telemetry is accepted as clear | Satoshi III implements; Musashi verifies |

### 1x. ETH curriculum decision preflight audit, 2026-08-06

Canonical evidence:

- `../AUDIT_SATOSHI_III_ETH_DECISION_PREFLIGHT_2026_08_06.md`
- `../../handoffs/MUSASHI_TO_SATOSHI_III_ETH_DECISION_PREFLIGHT_CORRECTION_ORDER_2026_08_06.md`

The N14/EN4_10/E4 four-seed design is accepted, but execution is withheld.
`full-v2` remains running and untouched. Prior findings 113/114/115/119/121
remain partially corrected and open.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260806-122 | S2 | open, blocks temporary A/B interruption | Resume authenticates no operator, verifies only plan/profile drift and reports success before proving rejoin to the bound chain/population | Satoshi III implements; Musashi verifies |
| AUD-F1-20260806-123 | S2 | open, blocks profile switching/restart | Profile drift emits an alert without blocking worker launch; installer fails open when active status is unavailable or pause is unverified | Satoshi III implements; Musashi verifies |
| AUD-F1-20260806-124 | S3 | open, blocks verified pause | Nonzero `nvidia-smi` with empty stdout is accepted as GPU-clear | Satoshi III implements; Musashi verifies |
| AUD-F1-20260806-125 | S3 | open, blocks curriculum decision evidence | Packet lacks terminal-weight evaluation and cannot distinguish zero margin events from absent telemetry | Satoshi III implements; Musashi verifies |
| AUD-F1-20260806-126 | S3 | open, blocks four-GPU decision execution | Base contract is unpinned; runner/aggregator lack tests, completeness gates, idempotent orchestration and implemented replica verification | Satoshi III implements; Musashi verifies |
| AUD-F1-20260806-127 | S2 | open, blocks efficient full campaign restart | Activity-ineligible epochs never consume patience; four no-trade candidates can run all 2,000 epochs while the log mislabels the state as step warm-up | Satoshi III implements; Musashi verifies |

### 1y. J0/J4 and retraining-frequency audit, 2026-08-06

Canonical evidence:

- `../AUDIT_SATOSHI_III_J0_J4_AND_RETRAINING_FREQUENCY_2026_08_06.md`
- `../evidence/SATOSHI_III_J0_J4_RETRAINING_REPRO_2026_08_06.py`
- `../../handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_J0_J4_RETRAINING_CORRECTION_ORDER_2026_08_06.md`

Corrections 123, 124 and 127 pass focused mechanical tests but still require a
bounded runtime smoke. Corrections 113, 122, 125 and 126 are not accepted. The
active `full-v2` domain remains untouched while the correction order executes.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260806-128 | S2 | open, blocks any campaign rejoin | Empty bound and observed chain identity can return `rejoin_proven=true` and resume successfully | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-129 | S2 | open, blocks N/EN/E decision | Real pipeline reloads best weights without preserving terminal policy; runner can claim both were evaluated while terminal evidence is absent | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-130 | S2 | open, blocks four-seed orchestration | Existing arm record is reused without binding data, code, base contract, resolved config, budget, anchor or artifacts | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-131 | S2 | open, blocks curriculum promotion | Malformed, lineage-incompatible packets with no finite decision metrics can aggregate as promotion-eligible | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-132 | S3 | open, blocks genome-repair acceptance | Repair accepts nonexistent genes and deterministically selects the first allowed categorical value, creating an ordering prior | General Satoshi III implements; General Musashi verifies |
| AUD-F2-20260806-133 | S3 | open, blocks exact-controller inventory | J4 is Omega-local, classifies by model-id substring and hard-codes SAC authority false instead of joining exact manifest hashes | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-134 | S2 | open, blocks adaptation-contract freeze | Static annual validation and unevidenced 32/256 windows do not establish the next-interval/week retraining business contract | General Satoshi III implements RT0-RT2; General Musashi verifies; owner selects cadence |

### 1z. Corrections 128-134 and RT1 ruling, 2026-08-06

Canonical evidence:

- `../AUDIT_SATOSHI_III_128_134_CORRECTIONS_AND_RT1_RULING_2026_08_06.md`
- `../evidence/SATOSHI_III_128_134_CORRECTION_REPRO_2026_08_06.py`
- `../../handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_128_142_CORRECTION_ORDER_2026_08_06.md`

The independent reproducer reports every new counterexample reproduced with
zero network and no runtime mutation. Corrections 128-133 are partial; finding
134's descriptive manifest is retained, but its RT0/RT1 evidence is rejected.
The active `full-v2` campaign remains running and untouched.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260806-135 | S2 | open, blocks campaign rejoin | Rejoin ignores bound component/domain revisions and proves no ancestry from the pre-pause tips | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-136 | S2 | open, blocks N/EN/E decision | Exact-id arm reuse accepts incomplete evidence and terminal weights are not retrievable/replicated from the arm artifact manifest | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-137 | S2 | open, blocks curriculum promotion | Duplicate physical seed packets are silently overwritten; empty common identity and per-arm lineage drift can promote | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-138 | S3 | open, blocks genome repair acceptance | Repair validation accepts an absent typed schema and a forbidden value outside the declared categorical domain | General Satoshi III implements; General Musashi verifies |
| AUD-F2-20260806-139 | S2 | open, blocks exact live-controller authority | J4 grants SAC authority without freshness, model/config/input parity, inference eligibility or observation parity | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-140 | S2 | open, blocks RT0/RT1 | Rolling-origin scoring includes 256 warm-up bars and resets account equity/effects at every origin | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-141 | S2 | open, blocks restart-safe adaptation | RT identity omits decision inputs and mutable checkpoint/OLAP writes can double-adapt after a crash; supplied evidence is stale under unchanged runner version | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-142 | S3 | open, blocks exact data contract | The executable decision runner retains contradictory `train_years=4`/`test_years=1` beside explicit dates | General Satoshi III implements; General Musashi verifies |

### 1aa. Corrections 135-142 acceptance audit, 2026-08-06

Canonical evidence:

- `../AUDIT_SATOSHI_III_135_142_ACCEPTANCE_2026_08_06.md`
- `../evidence/SATOSHI_III_135_142_ACCEPTANCE_REPRO_2026_08_06.py`
- `../../handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_135_150_VERDICT_AND_CORRECTION_ORDER_2026_08_06.md`

Append-only dispositions: 135, 137, 138, 139 and 142 are independently
verified for their named corrections and await owner closure. Findings 136,
140 and 141 remain open. RT1-A is materialized but execution remains forbidden.

| Prior finding | Appended state | Basis / remaining work |
| --- | --- | --- |
| 135 | independently_verified_pending_owner_closure | component/domain drift and exact bound-tip ancestry pass; new 150 tracks freshness/deadline |
| 136 | open | 144 proves promotion still accepts nonexistent/unloadable artifacts |
| 137 | independently_verified_pending_owner_closure | duplicate physical packets, malformed identity and per-arm lineage drift rejected |
| 138 | independently_verified_pending_owner_closure | typed schema and executable provenance-bearing repair verified |
| 139 | independently_verified_pending_owner_closure | exact authority join verified mechanically; deployment remains one-seat conditional |
| 140 | open | 145 proves active warm-up trading, h+1 scoring and disappearing exposure |
| 141 | open | 146-149 defeat crash, subject, handover-guard and source-identity contracts |
| 142 | independently_verified_pending_owner_closure | contradictory year fields absent from executable configs |

New findings:

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260806-143 | S3 | open, blocks acceptance-evidence trust | Generic exception handling marks stale APIs and malformed fixtures as corrected | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-144 | S2 | open, blocks N/EN/E promotion | Runner trusts asserted load proof and mismatched terminal references; aggregator promotes nonexistent artifacts | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-145 | S2 | open, blocks RT0/RT1 | Warm-up actively trades, three-bar cadence scores four facts and open exposure disappears at origin boundaries | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-146 | S2 | open, blocks restart-safe adaptation | Crash after SQLite commit and before pointer replace skips an origin while retaining stale model/account state | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-147 | S2 | open, blocks RT1 execution | Runner initializes fresh SAC instead of adapting a hash-bound mature ETH champion and is sequenced before R3 | General Satoshi III implements after R3; General Musashi verifies |
| AUD-F1-20260806-148 | S3 | open, blocks deadline acceptance | Deadline rule names zero unreconciled handovers but records/checks no handover fact | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-149 | S3 | open, blocks exact experiment identity | Git HEAD identity ignores uncommitted source state | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-150 | S3 | open, blocks unattended rejoin | Rejoin ancestry has no post-resume freshness requirement or bounded pending timeout | General Satoshi III implements; General Musashi verifies |

Runtime observation, not a new chain finding: active `full-v2` is one coherent
chain and all four GPUs are working, but the runtime predates activity patience
and all four current candidates remain zero-trade after 1,002-1,353 epochs.
Continuing or pausing that preserved run is an owner decision.

### 1ab. Corrections 143-150 acceptance audit, 2026-08-06

Canonical evidence:

- `../AUDIT_SATOSHI_III_143_150_CORRECTIONS_2026_08_06.md`
- `../evidence/SATOSHI_III_143_150_ACCEPTANCE_REPRO_2026_08_06.py`
- `../../handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_143_158_CORRECTION_ORDER_2026_08_06.md`

Append-only dispositions: 143 and 146 are independently verified pending owner
closure. Corrections 144, 145 and 147-150 remain open. RT1-A and smoke remain
blocked.

| Prior finding | Appended state | Basis / remaining work |
| --- | --- | --- |
| 143 | independently_verified_pending_owner_closure | typed probe outcomes and deliberate stale-harness fixtures reproduced |
| 144 | open | real load/cross-binding fixed; 151 proves replica authority remains self-asserted |
| 145 | open | warm-up/exact-h fixed; 152 disproves close quantity, costs and flat proof |
| 146 | independently_verified_pending_owner_closure | atomic SQLite row/state and post-artifact replay reproduced |
| 147 | open | 153 resets uninterrupted succession; 158 accepts an unproven anchor |
| 148 | open | 154 drops persisted latency history after restart; handover facts inherit 152 |
| 149 | open | 155 proves untracked executable source is reported clean |
| 150 | open | timeout fixed; 156 accepts same-second evidence with unchanged PID generation |

New findings:

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260806-151 | S3 | open, blocks artifact promotion | Replica authority is a caller-supplied string over a local copy | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-152 | S2 | open, blocks RT0/RT1 metrics | Handover treats direction as units, omits costs and hardcodes flat without execution | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-153 | S2 | open, blocks adaptation | Consecutive origins in one process do not inherit the preceding adapted model | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-154 | S3 | open, blocks cadence deadline | Restarted summary computes p95 from only the new process session | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-155 | S3 | open, blocks exact experiment identity | Untracked executable source is excluded from cleanliness/digest | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-156 | S3 | open, blocks unattended rejoin | Same-second observation with unchanged PID generation proves rejoin | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-157 | S2 | open, blocks RT1-A comparison | Every declared RT1-A cadence omits its final complete block interval | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260806-158 | S2 | open, blocks performance RT | Bare compatible SAC ZIP can claim mature champion-anchor status | General Satoshi III implements after R3; General Musashi verifies |

### 1ac. M0/M1/M0-X independent audit, 2026-08-08

Canonical evidence:

- `../AUDIT_SATOSHI_III_M0_M1_M0X_2026_08_08.md`
- `../evidence/SATOSHI_III_M0_M0X_REPRO_2026_08_08.py`
- `../../handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_M0_M1_M0X_CORRECTION_ORDER_2026_08_08.md`
- `../../handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_EMERGENCY_M0_M1_REPAIR_SPEC_2026_08_08.md`

M0 raw artifacts and metrics are preserved, but `mechanism_pass` is withdrawn
as easy-curriculum evidence: all 12 declared easy arms handed unchanged epoch-0
anchor weights into normal training. Current M1/M0-X contracts are blocked;
unrelated valid fleet work remains authorized and must not be idled.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260808-159 | S2 | open, blocks M0 successor/M1/R3 | All 12 easy arms hand unchanged epoch-0 anchors into normal training while aggregation attributes survival to easy | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-160 | S2 | open, blocks model-change evidence | ZIP hash inequality is accepted as proof of changed policy weights | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-161 | S2 | open, blocks M1 causal interpretation | N14 versus E4_N10 confounds easy dynamics with replay/optimizer/model reconstruction at the phase boundary | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-162 | S2 | open, blocks M0-X | Generic runner labels USDCAD while unconditionally materializing ETH data/base/observation config | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-163 | S2 | open, blocks M1/M0-X launch | V2 has no executable aggregator/exact decision semantics and execution/output identities collide across variants | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-164 | S3 | open, blocks durable acceptance | V2 schema does not enforce factorial/system lineage and reported replicas omit model artifacts | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-165 | S2 | open, blocks M0-X inference | Proposed USDCAD system has only 1,604 4h training rows and no exact multi-year date/regime contract | General Satoshi III materializes sufficient data/anchor after M1; General Musashi verifies |

### 1ad. Satoshi III WP0 quarantine acceptance audit, 2026-08-08

Canonical evidence:

- `../AUDIT_SATOSHI_III_WP0_QUARANTINE_2026_08_08.md`
- `../evidence/SATOSHI_III_WP0_QUARANTINE_REPRO_2026_08_08.py`
- `../../handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_WP0_QUARANTINE_CORRECTION_ORDER_2026_08_08.md`

The real invalid M0 successor is launch-ineligible, byte-preserved under its
content-addressed retired path, and its five envelope hashes independently
match. That accepts only finding 159's runtime-containment subcriterion. WP0
remains unaccepted because its negative-consumer and recovery proofs fail
adversarial cases. Unrelated valid fleet work remains authorized.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260808-166 | S3 | open, blocks WP0 acceptance | Consumer inspection reports whole roots scanned while ignoring SQLite, JSONL, CSV and logs | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-167 | S2 | open, blocks WP0 acceptance | Schema-only idempotency returns success for launch-eligible/corrupt supersession and missing envelope | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-168 | S3 | open, blocks WP0 acceptance | Missing aggregation/table/manifest bindings are written as null while evidence is certified immutable | General Satoshi III implements; General Musashi verifies |
| AUD-GEN-20260808-169 | S3 | open, blocks clean suite/WP0 acceptance | New mutating quarantine executable is absent from the reviewed engineering-surface declaration | General Satoshi III implements; General Musashi verifies |

### 1ae. L1/L2 curriculum, stopping and feature-selection audit, 2026-08-08

Canonical evidence:

- `../AUDIT_L1_L2_CURRICULUM_FEATURE_SELECTION_AND_STOPPING_2026_08_08.md`
- `../../work_plan/38_NEAT_LESSONS_L1_L2_CURRICULUM_AND_FEATURE_SELECTION.md`
- `../../handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_L1_L2_CURRICULUM_FEATURE_SELECTION_EXECUTION_ORDER_2026_08_08.md`

The historical fixed-epoch ETH packets remain preserved as diagnostics. They
cannot decide the owner's train-plus-validation stopping, L1/L2 curriculum or
online feature-selection questions. The new nested program is standing
owner-authorized; review waiting does not authorize idle compute.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260808-170 | S2 | open, blocks decision-bearing L1 comparison | Fixed 14-epoch N/EN allocation disables the required L1 stopping experiment | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-171 | S2 | open, blocks new-domain L1 selection | Lexicographic checkpoint selection explicitly uses validation alone | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-172 | S2 | open, blocks decision-bearing L2 | Optimizer fitness/patience consumes one selected split rather than nested paired evidence | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-173 | S2 | open, blocks full-year evidence claim | Seven-day train monitor is shorter than scaling context and split warm-up shortens score years | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-174 | S3 | open, blocks online feature-selection claim | Current ETH path fixes 83 inputs and implements neither inherited sparse masks nor an L1 learned gate | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-175 | S3 | open, blocks staged-maturation claim | L2 stages freeze genes but use global mutation/crossover and no stage-local diversity contract | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260808-176 | S2 | open, blocks curriculum attribution | Current matrix cannot separate L1 curriculum, L2 curriculum and their interaction | General Satoshi III implements; General Musashi verifies |
| AUD-GEN-20260808-177 | S3 | open, operational governance | Repeated owner-phrase gates remain in already-approved research flow and can idle the fleet | General Satoshi III removes redundant gates; General Musashi verifies runtime queue behavior |

### 1af. L1 round-1 and round-2 acceptance audits, 2026-08-09

Canonical evidence:

- `../AUDIT_SATOSHI_III_L1_CORRECTION_RETURN_2026_08_09.md`
- `../AUDIT_SATOSHI_III_L1_ROUND2_ACCEPTANCE_2026_08_09.md`
- `../evidence/repro_runs/MUSASHI_L1_CORRECTION_RETURN_REPRO_2026_08_09.py`
- `../evidence/repro_runs/MUSASHI_L1_ROUND2_ACCEPTANCE_REPRO_2026_08_09.py`
- `../../handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_L1_ROUND2_CORRECTION_ORDER_2026_08_09.md`

Findings 188-195 have independent correction dispositions in the round-2
audit. The decision run remains blocked by findings 196-200. This is a bounded
evidence/runtime correction, not a redesign and not a new owner gate.

| ID | Sev | State | Title | Owner |
| --- | --- | --- | --- | --- |
| AUD-F1-20260809-188 | S2 | verified corrected, pending closure | Failed seed had been mapped to a successful systemd exit | General Satoshi III corrected; General Musashi verified |
| AUD-F1-20260809-189 | S2 | verified corrected, pending closure | Sealed remote records retained unusable absolute paths | General Satoshi III corrected; General Musashi verified |
| AUD-F1-20260809-190 | S2 | partially verified; 196-197 block closure | Replica was optional and its whole-tree digest unverified | General Satoshi III implements remainder; General Musashi verifies |
| AUD-F1-20260809-191 | S2 | verified corrected, pending closure | Exact manifest did not govern actual plugins | General Satoshi III corrected; General Musashi verified |
| AUD-F1-20260809-192 | S2 | verified corrected, pending closure | Normal execution could fall back to unprotected orders | General Satoshi III corrected; General Musashi verified |
| AUD-F1-20260809-193 | S2 | verified for original defect; 199 remains | Normal-realistic profile lacked spread/slippage and explicit min-equity | General Satoshi III corrected; General Musashi verified |
| AUD-F1-20260809-194 | S3 | verified corrected, pending closure | Frozen manifest came from a dirty obsolete tree | General Satoshi III corrected; General Musashi verified |
| AUD-F1-20260809-195 | S3 | verified for mode label; 200 remains | Phase-1 evidence mislabeled normal arms as easy | General Satoshi III corrected; General Musashi verified |
| AUD-F1-20260809-196 | S2 | open, blocks decision launch | Aggregation mutates the sealed input and breaks source/replica digest equality | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260809-197 | S2 | open, blocks decision evidence | Direct aggregator CLI bypasses collection and replica authority | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260809-198 | S2 | open, blocks four-GPU launch | Assigned Gamma GPU UUIDs are visible but not bound to execution | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260809-199 | S3 | open, blocks exact normal contract | Financing treatment remains implicit | General Satoshi III implements; General Musashi verifies |
| AUD-F1-20260809-200 | S3 | open, blocks truthful compute evidence | Phase-1 realized epochs count baseline telemetry as training | General Satoshi III implements; General Musashi verifies |

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
