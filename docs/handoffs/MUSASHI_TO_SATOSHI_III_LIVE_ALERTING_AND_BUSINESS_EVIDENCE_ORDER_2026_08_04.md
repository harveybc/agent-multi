# Musashi to Satoshi III: Live Alerting and Business-Evidence Execution Order

Date: 2026-08-04 America/Bogota
From: General Musashi, temporary independent auditor
To: Satoshi III (Mujuro Utsutsu), temporary technical lead
Authority: direct owner order in chat, 2026-08-04
Priority: immediate; complete this packet before requesting the next audit
Capital boundary: Paper/Demo only; no Live-capital authorization

## 1. Role and Required Standard

Act simultaneously as a senior distributed-systems engineer, site-reliability
engineer, trading-execution engineer, machine-learning deployment engineer,
data engineer and security-conscious software architect. Work from direct code,
runtime and broker evidence. Do not infer health from process existence, port
reachability, an empty alert list or old prose.

Your job is not to produce another design-only document. Implement, test,
deploy and operate the deterministic infrastructure below. Preserve all valid
OLAP data, model artifacts, account history and audit evidence. Do not modify
the running DOIN campaign, its chain, seed, domain, population or pinned
runtime. Do not put Hermes, an LLM, Telegram text or any conversational agent
in an order-decision or order-submission path.

Ask the owner only for an action that cannot be performed safely without him,
such as authenticating TWS Paper. Everything else is your implementation and
orchestration responsibility.

## 2. Owner Decisions That Are Now Binding

1. Business knowledge from active Paper/Demo trading is the highest-priority
   operational front.
2. Alerts are ordered by operational consequence:
   live-trading safety and continuity first, live business-evidence integrity
   second, optimization continuity third, machine health fourth, and
   social/research operations last.
3. Telegram is an exception channel, not a periodic status feed. Healthy
   pings, unchanged reminders, stale queued messages and routine reports must
   not crowd out actionable incidents.
4. An incident already acknowledged or being corrected by the owner, Musashi
   or Satoshi remains visible in the incident ledger but does not keep paging.
   Source recovery, not acknowledgement, resolves it.
5. Alpaca Paper, IBKR Paper and OANDA MT5 Demo must continuously evaluate
   their selected hash-bound models. Flat/HOLD is a valid model decision;
   fabricated trades and duplicate orders are not. A legitimate signal must
   reach a broker-protected order lifecycle without manual glue.
6. Every risk-increasing order carries native broker-side SL and TP from its
   first submission. This rule has no degraded mode.
7. At a model succession boundary, existing exposure is drained under the old
   model. The actual post-close broker balance/equity becomes the new model
   session's starting point. Document 32 governs promotion evidence.
8. Profit is not an activation gate for this Paper/Demo observation period.
   Losses are valid evidence. Missing lineage, missing protection, missing
   decisions or missing broker reconciliation are not.
9. Project 3 E0-E4 is terminal: 16,019 jobs completed and archived. Its OLAP,
   artifacts and backups remain preserved, but it must not be represented as
   an active job or generate routine Telegram traffic.

## 3. Reproduced Baseline, Not Assumptions

At 2026-08-04 approximately 12:40 COT, independent inspection established:

- TWS Paper was not running on Omega; port 7497 refused connections;
  `lts-ibkr-model-runner.service` was in an auto-restart loop;
- IBKR direct truth was unknown after stale evidence of a 25,000-unit USD.CAD
  short and an unreconciled recovery attempt;
- Alpaca Paper was write-enabled with one protected SPY short;
- MT5 Demo bridge/runner were write-enabled and currently flat;
- four infrastructure watchdogs send Telegram independently through cron,
  while the LTS paper watchdog and Hermes jobs have separate delivery paths;
- Omega had three active Hermes cron jobs, two of which can deliver routine
  reviews to Telegram; Dragon and Gamma had no Hermes cron jobs;
- no active Project 3 cron, process or service was found on Omega, Dragon or
  Gamma; the owner-reported Project 3 answer is therefore stale agent context,
  not evidence of a running Project 3 workload;
- the current DOIN campaign was healthy: four workers, one seed, one domain,
  one population and one chain.

Do not claim a root cause for TWS termination until its logs and host timeline
prove one. Do not claim the IBKR position is open or flat until a fresh
authenticated broker reconciliation proves it.

## 4. Required Work Packages and Order

### P0. TWS Continuity and Current IBKR Reconciliation

1. Build a direct timeline from TWS logs, user-systemd journals, runner logs,
   kernel/OOM/suspend events and process exit evidence. Distinguish:
   user close, TWS scheduled restart/logoff, authentication expiry, host
   suspend/reboot, OOM/signal, application fault and network/API loss.
2. Once the owner authenticates TWS Paper, reconcile account, current position,
   parent, TP, SL, executions and completed orders from direct broker facts.
   Persist the result before any new risk is permitted.
3. Add a deterministic TWS continuity monitor that observes process state,
   API authentication, functional heartbeat age, runner restart rate and
   broker reconciliation. A listening port alone is never healthy evidence.
4. When TWS disappears with current or last-known unresolved exposure/order,
   emit a P0 incident within 60 seconds. When proven flat, emit P1 within two
   minutes. Include incident ID, venue, account fingerprint prefix, exposure
   state, protection state, first detection, last direct evidence and the one
   required operator action. Never include credentials or full account IDs.
5. Use only an officially supported TWS/IB Gateway restart mechanism. No GUI
   password scraping, credentials in Git, shell history or Telegram. If a
   restart requires interactive authentication, alert immediately and keep the
   execution service fail-closed until authenticated.
6. A runner restart loop must collapse into one incident. Restart count and
   latest failure remain in the ledger; they do not produce one message per
   process restart.

### P1. One Fleet Incident Ledger and One Notification Router

Implement a versioned, append-only SQLite incident ledger and CLI. The minimum
incident identity is:

```text
source + front + venue_or_machine + event_code + affected_object_fingerprint
```

Persist at least:

```text
incident_id, fingerprint, severity, front, source, event_code,
first_observed_at, last_observed_at, source_evidence_at, state,
occurrence_count, last_notified_at, notification_count,
acknowledged_at, acknowledged_by, acknowledgement_reason,
resolved_at, resolution_evidence_hash, payload_hash, payload_json
```

Required states are `pending`, `active`, `acknowledged` and `resolved`.
Acknowledgement suppresses reminders but never rewrites the source condition
as healthy. Only fresh direct recovery evidence resolves an incident.

Provide a deterministic CLI with at least:

```text
status [--active] [--severity ...]
show INCIDENT_ID
ack INCIDENT_ID --actor ... --reason ...
unack INCIDENT_ID
history [--since ...]
```

Musashi and Satoshi must use this CLI after observing or taking ownership of
an incident in chat. Do not attempt to infer acknowledgement from natural
language or Telegram read receipts.

Replace direct Telegram sends from the GPU temperature, GPU idle, memory,
swarm and LTS paper watchdogs with normalized incident emission into this
router. Keep their deterministic collectors and thresholds. During migration,
compatibility mode may exist, but acceptance requires exactly one active
autonomous alert-sender path. The Hermes gateway may remain available for
interactive owner questions; it is not an alert producer.

Replace the misleading `PROJECT3_TELEGRAM_CHAT_ID` notification namespace
with a generic operations credential name such as `OPS_TELEGRAM_CHAT_ID`.
Migrate the existing local secret without printing it. A bounded temporary
fallback may read the old name during deployment, but the accepted runtime and
documentation must use the operations name.

Run the router on the fleet with a deterministic notification owner and
bounded failover. Under a healthy network, one incident produces at most one
activation message. Under a partition, a bounded duplicate with the same
incident ID is preferable to losing a P0, but replaying a backlog is forbidden.
On restart, send only current unacknowledged incidents whose notification is
due; collapse all stale duplicates.

### P1. Alert Policy

| Priority | Conditions | First notification | Repeat policy |
| --- | --- | --- | --- |
| P0 | unprotected/foreign exposure; broker unavailable with unresolved exposure; reconciliation unknown after a risk-reducing action; wrong environment/account; duplicate risk; close/flatten failure | within 60 seconds | one reminder after 15 minutes, then hourly until acknowledgement; immediately on material worsening |
| P1 | broker terminal, bridge or model runner unavailable while proven flat; stale decision clock; persistent restart loop; protection evidence approaching staleness | within 2 minutes | once every 2 hours until acknowledgement |
| P2 | parallel swarm/chain divergence, job stall, OOM, missing GPU, temperature >=78 C, critical disk/memory | after existing hysteresis/grace | once every 6 hours until acknowledgement |
| P3 | social collector, audit cadence or research automation degraded | no immediate page unless data loss/security is possible | local ledger; one daily exception digest only while unresolved |

Send one recovery message only for a previously delivered P0/P1 activation.
P2/P3 recoveries remain in the ledger and dashboard without Telegram unless
their severity increased during the incident. No healthy messages.

Pause or convert the current routine Hermes Telegram deliveries
`lts-paper-shadow-business-review` and `moltbook-social-review` to local
evidence generation. Hermes may explain a sanitized incident on demand, but
it does not decide severity, acknowledgement, recovery or remediation.

### P1. Project 3 Terminalization and Current-Context Repair

1. Materialize one machine-readable Project 3 terminal record referencing the
   16,019-job OLAP, final backup hash, artifact roots and completion time.
2. Verify all Project 3 services, crons and Hermes jobs are absent or disabled
   on every machine without deleting environments, OLAP or artifacts.
3. Change Hermes status-context generation so terminal fronts are summarized
   as terminal history and current runtime facts always outrank semantic
   memory or old session prose.
4. Add a regression fixture where old memory says Project 3 is running while
   the terminal record and process inventory say complete. The answer must say
   complete and must not fabricate ongoing work.
5. Project 3 may page again only for integrity loss of its retained evidence,
   not for routine completion, health or progress.

### P1. Continuous Three-Venue Selected-Model Operation

For Alpaca Paper, IBKR Paper and MT5 Demo, persist a decision heartbeat for
every due model bar, including HOLD. Every decision fact must bind:

```text
venue, account fingerprint, asset/instrument, timeframe, as_of timestamp,
feature cutoff, input/data hash, preprocessing/config hash, model type,
artifact hash, deployment-manifest hash, decision/action scores,
requested order family, risk envelope and resulting intent or HOLD reason
```

Prove all three model runners consume the selected manifest and actual model
inference. A hard-coded demonstration policy must be labeled mechanics-only
and cannot be reported as the current champion. The job-1 robust-weekly
champion becomes authoritative only after its archive and document-32 gate.

Continuous means:

- services survive idle market periods and process each new bar exactly once;
- valid signals produce one protected lifecycle with no manual bridge step;
- HOLD does not manufacture turnover;
- each accepted exposure remains reconciled to its model, intent, SL and TP;
- terminal close facts carry actual post-close cash/equity into the next model
  session or succession record;
- switching a model never strands, inherits silently or duplicates exposure.

Retain the minimum-size Paper/Demo risk envelopes already owner-authorized.
Do not enlarge them in this packet.

### P1. Live-vs-Simulation Business-Evidence Loop

Create a shared OLAP fact contract and a reproducible comparison command. It
must join model decision, broker order, protection, fill, position, close and
account facts by stable lineage, not timestamp coincidence alone.

At minimum capture:

- service/venue uptime and decision-clock coverage;
- decisions, HOLDs, submissions, acknowledgements, rejects and retries;
- order type, requested/filled quantity and partial fills;
- decision-to-submit, submit-to-ack and submit-to-fill latency;
- quoted spread, realized entry/exit slippage, commission, financing and
  borrow facts when the venue exposes them;
- native SL/TP acceptance, later alteration/disappearance and close reason;
- MAE, MFE, realized/unrealized PnL, equity and drawdown;
- actual account balance/equity before entry, after close and at succession;
- model/config/artifact/data hashes and runner/component commits;
- data-availability and feature-parity gaps at the decision timestamp.

Replay the same model decisions through the canonical simulator with the same
as-of bars and immutable input hashes. Compare simulator and Paper/Demo at the
order-lifecycle level and account level. Do not optimize on this observation
window. Produce descriptive residuals that can later calibrate spread,
slippage, latency, rejection, financing and fill models.

Report returns and risk on consistent labeled scales: native observation
period plus weekly equivalent when statistically defined. Never relabel a
partial period as annual. A daily report may be emitted locally from day one;
the first seven-day comparison becomes the initial weekly business-evidence
packet. The system keeps collecting after that packet.

## 5. Repository Boundaries

- `lts`: broker/runtime collectors, model decision lineage, order lifecycle,
  account facts, incident producers and sim-vs-live comparison.
- `agent-multi`: fleet incident router, notification-owner/failover logic,
  Project 3 terminal record, operations configuration, installers and
  cross-front status view.
- `trading-contracts`: add a shared incident or live-evidence schema only if
  two repositories genuinely exchange it; retain dependency-free contracts.
- `prediction_provider`: change only if current artifact inference cannot
  expose required immutable model/data lineage through its existing API.
- `doin-core`, `doin-node`, `doin-plugins`: no protocol or active-campaign
  mutation for this order.

All defaults, thresholds, paths and cadence values belong in a versioned JSON
configuration loadable by `--config` or `--load_config`. No operational
constant may be discoverable only by reading Python.

## 6. Required Adversarial Tests

At minimum, prove:

1. 1,000 identical observations produce one activation message;
2. an incident flapping ten times does not create a Telegram storm;
3. acknowledgement suppresses reminders while active status remains visible;
4. source recovery resolves; acknowledgement alone cannot;
5. P0 preempts P2/P3 traffic and is never hidden in a digest;
6. 500 stale queued messages collapse to current active incident state;
7. router restart preserves counts, acknowledgement and notification history;
8. owner failure elects one fallback sender; recovery does not replay alerts;
9. forged future timestamps, stale payloads and wrong schemas fail closed;
10. TWS process loss with exposure becomes P0; proven-flat loss becomes P1;
11. a TWS restart loop emits one incident, not one per restart;
12. empty broker evidence is unknown, never flat, protected or recovered;
13. each venue emits due-bar decision lineage and rejects duplicate bars;
14. every entry has native SL and TP before risk increases;
15. model succession drains once and preserves exact post-close balance;
16. live/sim joins reject mismatched asset, timeframe, model or data hashes;
17. Project 3 stale memory loses to its terminal record;
18. no test opens a real socket or submits a broker order unless it is an
    explicitly labeled Paper/Demo runtime acceptance step.

Run focused tests, each complete affected-repository suite and migration tests
against copies of existing SQLite files. Never reset production OLAP to make a
test pass.

## 7. Deployment and Runtime Acceptance

Before asking Musashi for audit:

1. commits are small, reviewed, pushed and all touched repositories are clean;
2. the incident router is installed on Omega, Dragon and Gamma with exactly
   one effective autonomous alert sender under normal connectivity;
3. all direct legacy watchdog Telegram paths are disabled after migration;
4. a synthetic duplicate/failover test and a real TWS offline/recovery event
   are present in the incident ledger with bounded message counts;
5. Project 3 has a terminal record and produces no active/routine Telegram
   work on any machine;
6. TWS root cause is either evidenced and corrected or explicitly classified
   unknown with sufficient telemetry installed to capture the next exit;
7. fresh broker reconciliation establishes IBKR exposure/protection truth;
8. Alpaca, IBKR and MT5 decision heartbeats are fresh and hash-bound;
9. each venue has at least one independently evidenced protected Paper/Demo
   lifecycle in retained history; do not force a new trade merely to satisfy
   this row if valid existing evidence already does;
10. a fresh local three-venue descriptive report and sim-vs-live comparison
    are reproducible from persisted facts;
11. at least 60 minutes of post-deployment runtime shows no notification flood,
    duplicate decision, foreign exposure or unexplained runner churn;
12. the running DOIN campaign remains byte-for-byte on its pinned runtime,
    one seed/domain/population/chain, with no worker stopped for this work.

## 8. Audit Packet Required From Satoshi

Return one consolidated audit request, not a sequence of progress narratives.
It must contain:

- exact commits and clean/synced status for every touched repository;
- architecture and data-flow map of producers, ledger, router, owner election,
  Telegram transport, acknowledgement and resolution;
- schema and migration evidence;
- test commands, exact counts and adversarial fixture map;
- systemd/cron installation inventory on all three hosts;
- before/after Telegram path inventory proving one production sender;
- message-count evidence for activation, duplicate, acknowledgement, failover
  and recovery scenarios;
- TWS exit timeline, current reconciliation and remaining unknowns;
- fresh Alpaca/IBKR/MT5 model, decision, order, position and protection facts,
  all redacted and hash-bound;
- Project 3 terminal record and proof of absent active scheduling;
- reproducible live/sim report paths, OLAP queries and artifact hashes;
- DOIN non-interference evidence;
- explicit doubts and anything not completed. Do not declare your own work
  accepted or close your own findings.

Musashi will reproduce the failure cases, inspect broker-facing call graphs,
verify notification behavior and compare the reports with direct OLAP and
broker facts. Owner action remains limited to authentication, explicit risk
changes and closure/activation decisions.

## 9. Existing Documents to Read Before Editing

Read in this order:

1. `docs/work_plan/29_CONTINUOUS_DEMO_TRADING_OPERATIONS.md`
2. `docs/work_plan/32_CHAMPION_SUCCESSION_AND_REGIME_RESEARCH.md`
3. `docs/work_plan/23_SOCIAL_INTELLIGENCE_AND_OPERATIONAL_CONTINUITY.md`
4. `docs/work_plan/09_TESTING_SECURITY_AND_OPERATIONS.md`
5. `docs/work_plan/06_OLAP_METRICS_AND_LINEAGE.md`
6. `docs/work_plan/13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md`
7. `docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md`
8. `docs/handoffs/SATOSHI_III_SUCCESSION_VERIFICATION_AND_K1_DELIVERY_2026_08_04.md`
9. `lts/docs/MULTI_VENUE_PAPER_EXECUTION.md`
10. `lts/tools/paper_execution_watchdog.py` and its tests

Use codebase-memory MCP graph tools before broad code search. Fall back to
`rg` for configs, shell scripts, runtime strings and Markdown. Inspect actual
runtime before modifying deployment files.
