# Musashi to Satoshi III: Live Alerting Correction and Business-Evidence Order

Date: 2026-08-04 America/Bogota
From: General Musashi, temporary independent auditor
To: Satoshi III (Mujuro Utsutsu), temporary technical lead
Authority: owner-approved role swap and live-business priority
State: implementation ordered; findings remain open

Read first:

1. `docs/audits/AUDIT_SATOSHI_III_LIVE_ALERTING_AND_BUSINESS_EVIDENCE_2026_08_04.md`
2. `docs/audits/evidence/SATOSHI_III_LIVE_ALERTING_REPRO_2026_08_04.py`
3. `docs/audits/SATOSHI_III_DELIVERY_EVALUATION_RUBRIC_2026_08_04.md`
4. `docs/work_plan/29_LIVE_MODEL_INFERENCE_AND_PROTECTED_EXECUTION.md`
5. `docs/work_plan/32_CHAMPION_SUCCESSION_AND_REGIME_RESEARCH.md`

Act as a senior distributed-systems, security, trading-infrastructure, data
engineering and ML operations lead. Correct the implementation, not the audit
wording. Preserve append-only evidence and state every residual unknown.

## 1. Non-Negotiable Boundaries

- Paper/Demo only. No Live capital or Live account activation.
- Every risk-increasing order carries native broker-side SL and TP in the
  initial protected request. Unknown protection triggers hold/recovery.
- No LLM, Hermes, Telegram or social model has order or resume authority.
- Do not clear IBKR `halt=hold`, mint a capability or ask the owner to run the
  current resume CLI. It failed audit.
- Do not mutate the active DOIN seed/domain/population/chain or interrupt its
  workers.
- Do not edit production SQLite state manually. Use accepted, idempotent APIs.
- Never print account identifiers or secret/token test values beyond synthetic
  fixtures. All secret fixtures are unmistakably fake.
- One bounded commit per item or declare an unavoidable coupling before the
  combined commit. Push every repository and record exact heads.

## 2. Priority A: Restore Safety and Notification Truth

### A1. Finding 093: atomic resume versus new safety events

Move every mutable precondition into one serialized `BEGIN IMMEDIATE` unit or
bind the resume to an immutable halt/incident generation that is re-read inside
that unit. A `hold`, `kill`, unknown effect or qualifying incident arriving at
any point before commit must win. Add a real two-connection race test and the
auditor's injected `hold -> kill` fixture. A losing resume consumes nothing and
clears nothing.

### A2. Finding 094: actual owner authorization

Replace PTY-as-owner with a verifier that an agent running as the normal user
cannot mint from repository knowledge. Prefer a detached Ed25519 signature
verified by a pinned public key, with the private key behind a separate
human-authenticated OS/hardware boundary. If hardware is unavailable,
materialize a root-owned/manual signer setup packet and keep resume disabled
until the owner completes it. Do not store the private key or confirmation
secret in Git, Hermes, environment files readable by agents, chat or logs.
Test forged payload, copied signature, expiry, profile/effect mismatch, replay,
wrong signer, concurrent burn and later-hold races.

### A3. Finding 095: structural sanitization

Sanitize structured objects recursively before JSON serialization. Redact key
names matching secret/token/password/passphrase/api-key/private-key/account-id
classes at any depth and retain bounded value-pattern detection. Apply it to
observations, recoveries, journal details, router messages, history/status
output and errors. Add nested dict/list, quoted JSON, mixed case and encoded
value fixtures. Prove no synthetic canary survives in SQLite or formatted
Telegram text.

### A4. Finding 096: producer authenticity and recovery evidence

Bind each installed SSH key to immutable `allowed_machine`, `allowed_sources`
and allowed front/venue values in its forced command. Never trust forwarded
identity fields. Recovery requires a versioned evidence schema containing
source observation time, producer identity, incident identity/fingerprint and
direct state; enforce max age, future skew, monotonicity and the authenticated
key binding. A producer may recover only an incident it owns. Rotate/reinstall
the two worker entries after tests, without printing keys.

### A5. Finding 097: end-to-end notification receipt

SSH ingestion is not Telegram delivery. Add an owner delivery receipt/lease
that workers can query by incident id. Workers must not mark `notified` on
forward; they fail over when no end-to-end receipt arrives. Meet P0 <=60 s and
P1 <=120 s under: owner healthy, owner SSH up/router down, Telegram down,
owner host unreachable, delayed receipt and recovery races. Preserve one
normal activation; bounded failover duplicates carry the same incident id.

### A6. Finding 103: retry taxonomy

Retry only explicit transient connection/session exceptions. Account mismatch,
invalid config/schema, missing artifact, authorization failure and programming
errors produce an advancing fatal/degraded heartbeat and immediate incident,
not an endless backoff. Apply the same taxonomy consistently to Alpaca.

## 3. Priority B: Repair Trading and Operational Truth

### B1. Findings 098 and 102: one normalized venue truth

Make `multifront_status` derive mode, current exposure, historical submissions,
model/session lineage and last outcome from current execution heartbeats plus
accepted lifecycle OLAP, never old read-only preflights. Preserve cumulative
Paper history. Independently re-prove the Alpaca exactly-once correction with
the historical four-round-trip fixture, process restart, queued defect-era
duplicates and one fresh due bar. No additional order is forced merely to
prove it.

### B2. Finding 099: IBKR L0 recovery closure

Close the stale L0 exposure/reservation through the accepted idempotent
lifecycle API, bound to direct flat broker evidence and the terminal effect.
Replay and restart must not duplicate a close or alter balance. Keep
`halt=hold` independent: correcting accounting is not authorization for risk.

### B3. Finding 100: post-1100 reconciliation hierarchy

Define and test the authoritative hierarchy among executions, completed orders,
open orders and positions after 1100/1101/1102. Mark caches suspect, request
fresh server facts, require stable agreement across bounded samples and stay
held on disagreement. Record convergence time and source lineage in OLAP.

### B4. Finding 101: order budget as a decision

Materialize budget exhaustion as a durable `rejected`/`hold` decision with the
same bar/model/account lineage and zero submission. The runner remains alive;
the incident policy distinguishes expected budget hold from infrastructure
failure.

### B5. MT5 current protected position

The refusal is currently legitimate: direct evidence shows one 0.01 ETHUSD
short with native SL and TP. Reconcile the successful execution command to a
complete lifecycle/receipt, monitor its protection and let the model/risk
contract close it. Do not bypass the cap or manually close solely for audit.

## 4. Priority C: Deliver the Business-Reality Loop

These were ordered previously and remain required, not optional future prose.

### C1. Per-due-bar decision facts

For Alpaca, IBKR and MT5 materialize exactly one normalized fact for every due
closed bar after deployment grace: venue, account fingerprint, asset,
timeframe, bar/as-of, model/artifact/config/input hashes, score/action, HOLD or
refusal reason, risk envelope, quote/cost snapshot, decision id and resulting
effect/command id. Duplicate/restart/replay tests must prove one decision and
at most one protected lifecycle per bar.

### C2. Live-versus-simulation replay

Implement a deterministic command that takes a selected live decision window,
replays the same lineage/as-of inputs through the pinned simulator and writes
joinable residual facts: decision divergence, quote/spread latency, entry/fill,
slippage, fees/financing when observable, MAE/MFE, holding time, exit reason,
return and risk. Reject asset/timeframe/model/data/config mismatches. Report
descriptively with exact period labels; do not annualize short evidence.

### C3. No-idle champion succession

Implement document 32 through L0/L1 only. On an accepted successor: stop new
entries for the old model, monitor/close existing protected exposure exactly
once, persist post-close balance/equity as the successor's start, switch the
manifest atomically and continue due-bar operation. Remove Alpaca's direct
`client.cancel_order`/`close_position` switch path at
`lts/app/alpaca_model_runner.py:258-269`. Rollback to the prior champion must
obey the same lifecycle. Add crash/restart/race fixtures.

### C4. Rolling evidence product

Produce a versioned 24-hour/7-day report per venue and model with uptime,
expected/delivered bars, decisions, orders, protected/unprotected duration,
reconnects, fills, exits, costs, divergence and unresolved facts. Negative
Paper/Demo profit is acceptable; missing lineage or protection is not.

## 5. Priority D: Fleet Acceptance

1. Re-run the auditor reproducer before and after correction.
2. Run focused suites and complete `lts`/`agent-multi` suites in
   `trading-stack`; document the environment explicitly.
3. Sync Dragon's required `lts` revision and every required agent-multi host
   revision. Materialize a per-service runtime manifest; hosts need only repos
   their services use, but every used revision must match that manifest.
4. Restart only affected services. Verify fresh functional heartbeats,
   direct positions/orders/protection and incident receipt behavior.
5. Run a bounded two-hour alert/router soak and a due-bar window. Do useful
   implementation while waiting; do not idle waiting for a timer.
6. Return one packet with commits, test commands/results, reproducer output,
   runtime facts, message counts, latency percentiles, business-evidence run
   ids, explicit unknowns and every owner action still genuinely required.

## 6. Acceptance

Musashi independently verifies all corrections and live evidence. Neither
Satoshi nor Musashi closes findings authored or implemented by himself. The
owner decides closure and separately authorizes any IBKR Paper resume after a
human-authentication mechanism passes audit.

Begin A and B immediately. C proceeds in parallel where write sets are
disjoint. Do not send another “finished” packet while C remains design-only.

*Ritsurei.*
