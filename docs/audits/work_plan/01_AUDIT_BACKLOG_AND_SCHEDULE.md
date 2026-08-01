# 01. Audit Backlog and Schedule

Version: 1.4.0
Date: 2026-07-31
Owner: Satoshi
Reviewer: Musashi

## 1. Task Identity and States

Task IDs are stable: `AT-<F1|F2|F3|GEN>-NNN`. They are audit *verification*
tasks, distinct from finding IDs (`AUD-...`, tracked in file 04) and from
Musashi's implementation ledger (work-plan document 13).

```text
proposed -> scheduled -> in_progress -> reported -> closed
                              |             |
                              +-> blocked   +-> deferred
```

`reported` means the audit report exists; `closed` means Musashi/user triaged
every resulting finding. A `blocked` task records the smallest missing
authorization or evidence.

## 2. Cadence Mapping

From document 24 section 7, mapped to this backlog:

| Cadence slot | Mechanism | Backlog coverage |
| --- | --- | --- |
| 5 min / hourly | Deterministic watchdogs and Hermes (not Satoshi) | Inputs to the snapshot (file 03) |
| 24 h delta | Satoshi delta session | AT-GEN-010 plus the highest scheduled task |
| 72 h front rotation | Satoshi focused session | Next unaudited front task below |
| Weekly full | Satoshi cross-front session | AT-GEN-009 plus full-front sweep |
| Monthly | Recovery/supply-chain session | AT-GEN-008 |
| Event | Campaign transition, incident, broker activation, contract change | AT-F1-003, AT-F2-006, or ad hoc |
| Paper outline/evidence freeze/pre-submission | Academic audit | AT-ACADEMIC-030 or its dated successor |
| Weekly/quarterly innovation cycle | Academic research | AT-ACADEMIC-032 permanent queue |

## 3. Risk-Ranked Backlog

Ordered by current risk-weighted value. "Hermes pre-collect" marks the share
of the work that must arrive pre-collected per files 02/03 rather than being
explored by Satoshi at full token cost.

| ID | Front | State | Task | Trigger | Hermes pre-collect |
| --- | --- | --- | --- | --- | --- |
| AT-F1-011 | F1 | **reported (owner disposition pending)** | Equal-height fork classification for AUD-F1-20260730-005: the height-9 competition resolved after finalization advanced; finalized anchors and claims/population remained coherent. Satoshi recommended closing the original finding and tracking recurrence under finding 020 | Harvey or independent disposition of finding 005 | classification evidence is in `AUDIT_MUSASHI_GOVERNANCE_INNOVATION_RESPONSE_2026_07_31.md` |
| AT-F1-001 | F1 | **reported (finding open)** | Protected-entry v2 eligibility/bracket contract verification (details 4.1); cross-review rejected the PASS because full-period RAP was certified as weekly RAP | Harvey metric-contract decision, then correction addendum | block-8 split metrics and artifact hash independently reproduced |
| AT-F1-012 | F1 | **reported** | Ranking sensitivity measured (champion flips full↔weekly; 5/5 sign flips; 0 eligibility flips; n=5 with 13 exclusions documented) and curriculum inheritance traced (job 1 recalculates robust weekly fitness; dual selection-key risk flagged for pre-launch test). Decision packet A/B/C delivered; recommendation A with two riders. Awaiting Harvey's objective decision | `../AUDIT_OBJECTIVE_CONTRACT_AND_CURRICULUM_2026_07_31.md` + `../evidence/AT_F1_012_OBJECTIVE_RANKING_2026_07_31.csv` | complete |
| AT-F2-002 | F2 | scheduled | Broker-boundary fail-closed and secret-redaction audit (details 4.2) | next 72 h | OLAP schema dumps, adapter config hashes |
| AT-F1-003 | F1 | scheduled (event) | Champion archive and job-0 to job-1 transition verification (details 4.3) | job-0 convergence event | supervisor history JSON, artifact hashes |
| AT-GEN-010 | GEN | scheduled | Doc-drift delta sweep: doc 13 refresh vs runtime; verify open findings still reproduce | every 24 h delta session | git log delta, snapshot diff |
| AT-F1-004 | F1 | proposed | Reconstruct weekly/annual return and RAP for the current champion from atomic OLAP facts; confirm no unit mixing on dashboard | after first stable champion | OLAP extract |
| AT-F1-005 | F1 | proposed | Dataset/manifest SHA verification across all four workers vs versioned manifest | with AT-F1-001 or remote access | per-host hash listing |
| AT-F2-006 | F2 | proposed | MT5 EA source security review: HMAC, nonce persistence, timestamp window, demo-only, firewall allowlist | before EA attach/canaries | EA source snapshot |
| AT-F1-007 | F1 | proposed | L1 contract spot-check: patience 60/floor 40/best-checkpoint restore actually wired; `epoch_timesteps` derivation recorded | 72 h rotation | resolved config extract |
| AT-F3-008 | F3 | proposed | Social S0/S1 pre-activation review: allowlist, injection fixtures, budget caps, publishing disabled | before S1 activation | spec/config diff |
| AT-GEN-008 | GEN | proposed | Continuity evidence: evidence-pool snapshot hashes exist and verify; backup retention matches doc 13 section 6 | monthly slot | hash listing, `quick_check` results |
| AT-GEN-009 | GEN | proposed | Weekly recovery-prompt reviews (both Musashi's and Satoshi's) against architecture changes | weekly slot | git delta of docs |
| AT-F2-009 | F2 | proposed (machine telemetry materialized) | Dragon/Gamma linger + SSH-bridge dependency check; MT5 VM state; Alpaca scheduling mechanism | first session requiring process-level remote evidence | deterministic machine packet plus bounded process verification |
| AT-ACADEMIC-030 | GEN | reported | Audit the five-paper publication program, seed a related-work ledger and classify evidence readiness | complete; awaiting cross-review disposition | paper artifact manifests and publication-plan diff |
| AT-ACADEMIC-031 | GEN | scheduled | Verify the seeded P5/P1 primary sources and seed their claim/evidence matrices | after S0/S1 and AT-F1-011 | primary-source ledger delta |
| AT-ACADEMIC-032 | GEN | scheduled permanent | Audit Musashi's governance response, P1 scope and P6+ continuous-research program; then maintain future-work, collision-test, replication and retirement queue | dedicated 2026-07-31 handoff, then weekly/quarterly cadence | dated innovation audit, prior-art delta, future-work files |

## 3b. Continuous Deepening Program (added 2026-07-31)

Per user direction, all three fronts receive continuous and increasingly deep
coverage. Repository quality bars follow **exposure tiering**, not uniform
ceremony (framework in `../AUDIT_QUALITY_SECURITY_TESTING_2026_07_31.md`
section 2):

- **Tier A (public/adversarial):** `doin-core`, `doin-node`, `doin-plugins`,
  `lts` — CI mandatory, adversarial/property tests, dependency provenance,
  threat model, security review before public/multi-user exposure.
- **Tier B (trust-critical internal):** `agent-multi`, `prediction_provider`,
  `financial-data` — CI, leakage/cutoff, invariants, boundary integration.
- **Tier C (libraries):** `trading-contracts`, `gym-fx`, `heuristic-strategy` —
  CI, unit plus property/contract fixtures, compatibility tests.

| ID | Front | Depth | Task | Prerequisite |
| --- | --- | --- | --- | --- |
| `AT-SEC-020` | F1 | deep | `doin-core` crypto/trust primitives: identity, signing, hashing, commit-reveal, replay, quorum assumptions | none |
| `AT-SEC-021` | F1 | deep | `doin-node` untrusted-peer input: validation, bounds, dedup identity, fork handling, resource exhaustion | none |
| `AT-SEC-022` | F2 | deep | `lts` credential/redaction/order-authority review; prove no path bypasses risk and reconciliation | none |
| `AT-SEC-025` | F3 | deep | Moltbook adversarial fixtures: multilingual injection, citation forgery, crowd-out | after 006/008 fixes |
| `AT-F3-013` | F3 | medium | Hermes-side model call: provider, budget, and whether the packet is the only prompt input | none |
| `AT-QUAL-023` | all | medium | Verify CI adoption and the leakage mutation gate | AUD-GEN-...-009 fix |
| `AT-QUAL-024` | all | medium | Verify the ten document 09 invariants | AUD-GEN-...-010 fix |
| `AT-QUAL-026` | all | light | Dependency/SBOM verification for Tier A | AUD-GEN-...-012 fix |

Rotation honouring the 72-hour front-coverage rule: F1 deep security
(`AT-SEC-020`, `AT-SEC-021`) → F2 (`AT-SEC-022`) → F3 (`AT-SEC-025`), with
quality-verification tasks interleaved as their prerequisites land. This
complements, and does not replace, the operational tasks in section 3.

## 4. Detailed Specs for Scheduled Tasks

### 4.1 AT-F1-001: Protected-entry v2 contract verification

Objective: independently verify the exact contract whose v1 failure caused the
2026-07-29 incident, while the live campaign depends on it.

Checks:

1. Locate and read the wiring (not just docs) for: train-tail >= 1 trade,
   annual validation >= 12 completed trades, `-1e9` ineligibility fitness,
   action-collapse guard, bracket-only entries with fail-closed plugin errors.
2. Rerun the focused suites cited in doc 20 (`agent-multi` safety/campaign
   tests, `gym-fx` full suite) and compare pass counts; bounded, no training.
3. Reconstruct the current champion L2 fitness from its stored metric vector
   using `L2 = mean(RAP_tt, RAP_val) - beta*|RAP_val - RAP_tt|` and confirm
   weekly-fraction units end to end.
4. Confirm protected-test paths remain closed (`selection_uses_test` absent,
   `evaluate_test_split=false`, firewall test present and passing).

Bounded commands (read-only plus existing unit tests): `rg` for the sentinel
and floor constants; `python -m pytest -q <focused paths>`; supervisor/chain
API GET for the champion metric vector. Token class: medium. Output: findings
plus verified non-findings in `AUDIT_V2_CONTRACT_<date>.md`.

### 4.2 AT-F2-002: Broker-boundary fail-closed audit

Objective: prove the venue boundary fails closed and leaks no identity before
canaries are enabled.

Checks:

1. `lts` Practice client: hard-coded `api-fxpractice.oanda.com` only; OANDA
   Global Markets credentials rejected.
2. Alpaca/IBKR observers: read-only mode enforced in code, not config comments.
3. MT5 bridge (read phase): EA refuses non-demo, command endpoint returns no
   command, HMAC/nonce/timestamp checks fail closed.
4. Read-only SQLite over the paper OLAP stores: schema and sampled rows contain
   fingerprints only - no tokens, no raw account IDs (match patterns without
   printing candidate secret values).
5. Canary preconditions: `orders.enabled` absent/false, confirmation-phrase
   path intact.

Token class: medium. Output: `AUDIT_BROKER_BOUNDARY_<date>.md`.

### 4.3 AT-F1-003: Champion archive and transition verification (event)

Objective: at job-0 convergence, verify completion before successor startup.

Checks: identical archived artifact on every host (bytes, SHA-256), decoded
genome and metric vector present, five diverse elites archived, independent
hash-load of the `.zip`, curriculum config generated on every host from local
handoff artifacts with identical hashes, no v1 (`usdcad-...-shared-v1`)
lineage referenced anywhere in the v2 successor, verified stop of all four
workers before job-1 start.

Token class: medium-low if the supervisor history JSON is pre-collected.
Output: `AUDIT_CHAMPION_ARCHIVE_<date>.md`.

### 4.4 AT-F1-011: Equal-height fork classification (read-only)

Objective: classify AUD-F1-20260730-005 without touching the chain, and give
Musashi the evidence he stated was missing before any repair decision.

Checks:

1. Retrieve block 9 from Dragon (tip `603dfe1a…`) and from a majority node
   (tip `4b4f06a1…`); compare generator identity, transaction IDs and types.
2. Determine whether either branch carries transactions absent from the other,
   specifically `optimae_accepted` or `candidate_evaluated`. A Dragon-only
   accepted transaction escalates the finding to S2.
3. Read the `ForkChoiceRule` deterministic tie-break in `doin-core` and
   evaluate which of the two tip hashes it should select at equal height.
4. Re-sample tips after finalization advances past height 9. A fork that
   dissolves at finalization is benign convergence latency, and the finding
   closes as `false_positive` or `accepted_risk` with a monitor.

Explicit non-goals: no chain repair, no worker restart, no reorganization, no
recommendation to mutate state. Output is classification evidence only.

Token class: low-medium. Output: `AUDIT_FORK_CLASSIFICATION_<date>.md`.

### 4.5 AT-F1-012: Objective-contract and curriculum audit (read-only)

Objective: quantify whether the full-period L2 used by the active job selects
different candidates from a mean-weekly-normalized L2, then trace whether the
queued curriculum successor corrects or inherits that selection pressure.

Checks:

1. Reconstruct both objective scalars for every accepted candidate with enough
   atomic train-tail and validation evidence; never impute missing rows.
2. Compare ranks with Spearman/Kendall, top-1/top-5 overlap, sign changes,
   eligibility changes and current-champion disposition.
3. Trace job-1 handoff and materialization through executable code and config,
   including genomes, weights, elites, fitness, weekly folds and difficulty.
4. Give Harvey a falsifiable decision matrix for unchanged completion,
   boundary-only replacement or retaining the current objective.
5. Classify the academic consequence without asserting novelty.

Explicit non-goals: no swarm mutation, restart, replacement campaign,
fitness-code edit or queued-config edit. Invocation 07 must complete first.

Token class: medium. Output:
`AUDIT_OBJECTIVE_CONTRACT_AND_CURRICULUM_2026_07_31.md` plus a row-level CSV.

## 5. Scheduling Notes

- Do not run more than one heavy task per session; report and stop.
- If an `S0`/`S1` appears, it preempts this schedule.
- If nothing changed since the last snapshot and no slot is due, the correct
  action is a one-paragraph "no change, next trigger" note - not a re-audit.
