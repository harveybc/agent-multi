# 01. Audit Backlog and Schedule

Version: 1.0.0
Date: 2026-07-30
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

## 3. Risk-Ranked Backlog

Ordered by current risk-weighted value. "Hermes pre-collect" marks the share
of the work that must arrive pre-collected per files 02/03 rather than being
explored by Satoshi at full token cost.

| ID | Front | State | Task | Trigger | Hermes pre-collect |
| --- | --- | --- | --- | --- | --- |
| AT-F1-011 | F1 | **scheduled (next)** | Equal-height fork classification for AUD-F1-20260730-005: compare block 9 across branches for unique accepted transactions, read the `ForkChoiceRule` tie-break, re-sample after finalization advances. Read-only; proposes no chain mutation (details 4.4) | now, while the fork is observable | snapshot tip/anchor fields (already collected) |
| AT-F1-001 | F1 | scheduled | Protected-entry v2 eligibility/bracket contract verification (details 4.1) | next 24-48 h | test-suite outputs, champion metric JSON |
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

## 5. Scheduling Notes

- Do not run more than one heavy task per session; report and stop.
- If an `S0`/`S1` appears, it preempts this schedule.
- If nothing changed since the last snapshot and no slot is due, the correct
  action is a one-paragraph "no change, next trigger" note - not a re-audit.
