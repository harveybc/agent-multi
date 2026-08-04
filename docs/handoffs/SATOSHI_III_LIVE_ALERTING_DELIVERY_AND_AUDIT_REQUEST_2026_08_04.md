# Delivery and Audit Request: Live Alerting, IBKR Recovery, Business-Evidence Order

Date: 2026-08-04 America/Bogota (evidence sampled through 18:55 UTC)
From: Satoshi III (Mujuro Utsutsu), temporary technical lead
Order: `MUSASHI_TO_SATOSHI_III_LIVE_ALERTING_AND_BUSINESS_EVIDENCE_ORDER_2026_08_04.md`
including the §9 urgent addendum.
This packet closes nothing. It reports implemented, deployed and operating
work, new findings, and explicitly incomplete items.

## 1. Commits and Repository State

Both repositories clean and pushed at packet time.

`agent-multi` (from `95a2816b`, base also contains your `5bc1c0dc`/`ac05354e`):

| Commit | Content |
| --- | --- |
| `8d53eb8c` | fleet incident ledger + CLI + notification router + 28 adversarial tests |
| `82eacdae` | 4 infra watchdogs migrated to ledger emission (zero direct sends left) |
| `5bf15b6e` | Project 3 terminal record + integrity verifier + Hermes context repair |
| `c91d2741` | forced-command forwarding shim + dedicated forwarding-key config |
| `58099aa6` | resolution sync to owner for every severity (found live, see §7) |
| `1eae0609` | shell-quoted SSH forwards so JSON survives transit (found live) |

`lts` (from `6daf85e`, branch `main`):

| Commit | Content |
| --- | --- |
| `045d70a` | deterministic TWS continuity monitor + 11 adversarial tests |
| `4cfac0b` | LTS paper watchdog migrated to ledger; P0 severity contract; summaries local-only |
| `775646d` | owner-gated `resume_after_reconciliation` (§9 addendum) + 18 adversarial tests |
| `b2c696e` | AUD-F2-20260804-091 fix: connect refusal → advancing degraded heartbeat loop |
| `a9b9d41` | Alpaca exactly-once: reconciliation repairs can no longer mint retry identities |
| `0cce126` | Alpaca runner 091 parity (degraded heartbeat instead of crash-loop) |
| `9a8d568` | consumer-side exactly-once: defect-era queued duplicates superseded via ledger |

Test suites at head: **lts 583 passed**, **agent-multi 483 passed**
(`conda run -n trading-stack python -m pytest -q tests` / `tests/unit`).

## 2. Architecture and Data Flow

```text
producers (deterministic collectors)                     ack/resolution
  gpu_temperature / gpu_idle / memory / swarm watchdogs     ack: CLI only
  lts paper watchdog (P0 severity contract)                 (incident_ledger.py
  tws_continuity_monitor (60 s systemd timer, omega)         ack/unack; never
  project3_terminal_verify (integrity only)                  inferred from text)
        │ observe / recover (fail-closed inputs, redaction)  resolution: ONLY
        ▼                                                    fresh source
  per-host SQLite incident ledger (append-only journal +     recovery evidence
  materialized state; pending/active/acknowledged/resolved)
        │
  incident_router.py (cron: omega 1 min; dragon/gamma 2 min)
        ├─ owner (omega): due-ness recomputed from ledger state (no queue);
        │    policy P0 60s/15m/hourly · P1 2h · P2 6h · P3 daily digest;
        │    single Telegram transport (OPS_TELEGRAM_CHAT_ID, bounded
        │    fallback to the legacy name); redaction pass on every message
        └─ non-owner: SSH forward through a dedicated ed25519 key whose
             authorized_keys forced command permits ONLY
             incident_ledger.py observe|recover (shim, allow-listed
             options); resolution sync for every severity; bounded
             per-severity failover duplicates carry the same incident id
```

Schema: `agent_multi.incident_ledger.v1` — `incidents` (all fields your
order requires, plus `delivered_severity`), append-only `incident_events`
journal, `ledger_meta`. Identity = sha256(source|front|venue_or_machine|
event_code|affected_object)[:16]; at most one non-resolved incident per
fingerprint (partial unique index); flap-reopen window collapses storms.

## 3. Adversarial Fixture Map (order §6)

| # | Proof | Test |
| --- | --- | --- |
| 1 | 1,000 identical observations → one activation | `test_incident_router.py::test_thousand_observations_one_activation_message` |
| 2 | 10 flaps ≠ storm | `…::test_flapping_ten_times_is_not_a_storm` + ledger reopen tests |
| 3 | ack suppresses, stays visible | `…::test_ack_suppresses_reminders_status_stays_visible` |
| 4 | recovery resolves; ack cannot | ledger `test_ack_suppresses_but_never_resolves` |
| 5 | P0 preempts, never in digest | `…::test_p0_preempts_and_never_hides_in_digest` |
| 6 | 500 stale → current state | `…::test_stale_backlog_collapses_to_current_state` (queue-less by construction) |
| 7 | restart preserves history | router + ledger restart tests |
| 8 | failover bounded, no replay | `…::test_nonowner_forwards_and_fails_over_bounded` |
| 9 | forged/stale/wrong-schema fail closed | ledger refusal tests (future skew, stale evidence, naive ts, bad severity, alien schema) |
| 10 | TWS loss w/ exposure P0, flat P1 | `test_tws_continuity_monitor.py` (incl. acknowledged-position P0) |
| 11 | restart loop = one incident | monitor test (delta 2195 in one payload) + `test_ibkr_model_runner_backoff.py` |
| 12 | empty evidence = unknown, never flat | monitor `test_empty_ledger_evidence_is_unknown_never_flat` |
| 13 | due-bar lineage, duplicate bars rejected | **partial**: `test_l0_retry_gate_is_exactly_once`, consumer supersede test; full per-venue due-bar contract not delivered (§10) |
| 14 | native SL+TP before risk | pre-existing suite coverage retained (543-line watchdog + venue tests); no new test this window |
| 15 | succession drains once, preserves balance | **not delivered** — Priority B schema code still awaits your concurrence (§10) |
| 16 | live/sim joins reject mismatches | **not delivered** (§10) |
| 17 | Project 3 stale memory loses | `test_project3_terminal.py::test_stale_memory_loses_to_terminal_record` |
| 18 | no real sockets in tests | all new tests use tmp ledgers/fakes; the only broker-facing steps were labeled runtime acceptance |

Resume addendum: 18 further adversarial tests in `test_ibkr_l1_resume.py`
(stale/empty/wrong-account evidence, open orders, positions, nonterminal
effects, active/unknown incident, expired/overlong/foreign capability,
replayed nonce, concurrent burn, crash-at-boundary rollback, idempotent
retry, spent-capability-vs-new-hold).

## 4. TWS Exit Timeline and Reconciliation (P0)

All times America/Bogota (UTC-5), from TWS launcher logs, user journal,
runner journal and the durable ledger:

1. **23:17:55 (Aug 3)** Error 1100 connectivity lost ×4 → 1102 restored.
   In the blip the protective orders vanished from direct evidence
   (parent gone; TP/SL later `Cancelled`).
2. **23:17:56-58** fail-closed response: `protection_health_failure` →
   `recovery_hold` (halt=hold) → cancels 687/688 → flatten order 692
   (BUY 25,000 MKT) → broker `Filled` → `recovery_unreconciled`.
3. **23:19-23:44** TWS-local API state was poisoned: snapshots kept
   listing the Cancelled STP 688 as open and reported `remaining_units
   +25000` for 27 minutes. The ledger never trusted it — state stayed
   `effect_unknown`.
4. **23:45:02** TWS performed an **orderly self-shutdown** (JTS
   Stopper/ShutdownTask threads; scope closed after 9 h 44 m). Launcher
   log states **“Daily auto-restart is not enabled.”** Root cause class:
   **scheduled daily auto-logoff/exit** — not crash, not OOM, not
   suspend, not network.
5. **23:45:51 → 12:29 (Aug 4)** runner crash-looped (NRestarts 2295)
   with a frozen heartbeat — your AUD-F2-20260804-091, now corrected.
6. **12:49-12:50** owner relit TWS; the runner resumed from the
   immutable effect contract; **17:49:58 UTC** `recovery_reconciled_flat
   {remaining_units: 0.0}` → `terminal_flat` persisted. Fresh read-only
   observer: `open_positions 0, open_orders 0` (fingerprint `9f9f5111…`).

Position truth: IBKR's server was flat from the 692 fill onward; the
+25000 readings were TWS-cache artifacts (nothing could trade while TWS
was down, and the account was flat when it returned). Remaining unknown:
why the 1100 blip destroyed the protective orders — I did not find a
server-side reason in available logs.

Corrective owner action (once): enable **daily Auto restart** in TWS
Lock-and-Exit settings so the scheduled logoff becomes a restart.

## 5. §9 Addendum Delivery State

- `resume_after_reconciliation` implemented (`lts@775646d`): all ten
  properties, per §3 of this packet's commit list; TTY-only mint
  (`tools/mint_resume_capability.py`, confirmation phrase) and TTY-only
  owner CLI (`tools/ibkr_resume_after_reconciliation.py`) which gathers
  fresh read-only broker evidence on a dedicated client id and refuses
  on any active/unknown P0-P1 venue incident.
- **Runtime acceptance is pending the owner**: mint one resume
  capability bound to `l1e-f4993c2dda8cdc2a` and run the CLI. I cannot
  and will not execute this transition myself. Until then the seat is
  safe-but-held: decisions correctly reject `halted:hold`.
- 091 corrected (`b2c696e`): construction refusal now writes advancing
  `degraded_error` heartbeats (phase=connect) with exponential backoff
  capped at `connect_backoff_max_seconds` (config 300 s), zero
  submissions, automatic reconnect+reconcile; mid-loop failures re-enter
  the same loop in-process. Runner redeployed (NRestarts back to 0,
  state `decided`).

## 6. NEW FINDINGS reported (I do not close these)

1. **Alpaca duplicate lifecycles (exactly-once violation; serious).**
   One SPY signal (bar 2026-08-03) produced **four** bracket lifecycles
   on 2026-08-04 (effects at 00:14, 13:33, 14:38, 16:21 UTC; 10
   execution receipts): `reconcile_terminal_effects()` returned repair
   items after every completed round trip and the retry path hashed the
   repair ids into the idempotency identity, minting a fresh key each
   time. The daily Paper order-budget guard blocked the fifth attempt
   and the runner crash-looped (116 restarts). Fixed at `a9b9d41`
   (deterministic `:l0-retry-1` identity, only for bars with zero
   effects) plus `9a8d568` (consumer supersedes satisfied retry-class
   decisions — this drained the defect-era queued duplicate through a
   code path, no manual SQLite). Runner redeployed; live state now
   `replayed_signal` with zero new submissions. The four Paper
   round-trip costs remain in the account as evidence.
2. **IBKR L0 exposure row not closed by recovery — FIXED post-packet.**
   Effect `l1e-f4993c2dda8cdc2a` went `terminal_flat` with the broker
   flat, but exposure `exp-oi2-rsv-f4993c2dda8cdc2a` (USD.CAD −25,000)
   stayed `open`, consuming `max_concurrent_positions` — it would have
   blocked the seat's first post-resume entry. `lts@83cc286` adds
   `reconcile_terminal_exposures` (the `bc974d5` analog): closure +
   reservation release only under terminal effect AND direct-evidence
   flat route position; 6 adversarial tests; verified live — the row
   closed at 2026-08-04T22:06:22Z on the runner's first tick after
   redeploy. Please verify the repair alongside §6.1.
3. **TWS API cache poisoning after 1100/1102** (evidence note, §4 item
   3): reconciliation should prefer executions/server-side queries over
   the open-order/position cache after a connectivity blip.
4. **Order-budget exhaustion surfaces as an exception**, not a decision
   outcome (mitigated by 0cce126's degraded-continuity, root cause
   remains in `alpaca_l1.submit`).
5. Two router defects found and fixed during live fleet acceptance
   (`58099aa6`, `1eae0609`) — see §7; they were caught precisely because
   acceptance was run against the real fleet.

## 7. Deployment and Message-Count Evidence

Installed and operating:

| Host | Router cron | Watchdog crons | Other |
| --- | --- | --- | --- |
| omega | `*/1` | 4 (migrated code) | continuity-monitor timer 60 s; Hermes gateway restarted; both LTS runners restarted on fixed code |
| dragon | `*/2` | 6 (migrated code) | forwarding key installed; MT5 units untouched |
| gamma | `*/2` | 6 (migrated code) | forwarding key installed |

`OPS_TELEGRAM_CHAT_ID` migrated from the legacy name in the local env on
all three hosts without printing it; the router prefers the new name with
the legacy names as bounded fallback. Hermes jobs
`lts-paper-shadow-business-review` and `moltbook-social-review` now
deliver `local` (jobs.json backed up first); the Hermes gateway remains
interactive-only. Legacy sends: **zero** `send_telegram(` call sites
remain in any watchdog (grep evidence); the LTS watchdog's periodic
summary is journal-only.

Acceptance runs recorded in the production ledger:

- **Synthetic P1** `INC-20260804184058-1a008620`: 3 duplicate
  observations → occurrence 3, ONE activation message; second router
  pass sent nothing; source recovery → ONE recovery message. Total 2
  messages.
- **Fleet-forward P2** `…-d04e6564` (dragon): forwarded through the
  forced-command key into the omega ledger; omega cron router sent ONE
  activation; dragon-side recovery synced resolution with ZERO Telegram
  (P2 policy). `status --active` on omega: **no incidents**.
- Real TWS offline/recovery: the outage predates the ledger; its durable
  evidence lives in the L1 broker-facts journal (§4). The continuity
  monitor now emits `tws_unavailable` recoveries every healthy minute.

Soak: router+monitor operating since ~18:37-18:41 UTC with no flood, no
duplicate decision, no foreign exposure, no unexplained churn at packet
time; the 60-minute mark completes ~19:41 UTC — please sample
`~/.local/state/agent-multi/incident-router/cron.log` and
`incident_ledger.py history` when reproducing.

DOIN non-interference: `doin-campaign-supervisor` active since
2026-07-29, NRestarts 0; one running DOIN unit each on dragon/gamma; no
worker stopped, no campaign file touched.

## 8. Three-Venue Runtime Facts (redacted, hash-bound, 18:46 UTC)

- **IBKR Paper**: heartbeat `decided`, artifact `dc95edcb…`, model
  `usdcad-4h-linear-live-v1`, fingerprint `c0ff137a…`; decisions
  rejected `halted:hold` awaiting the owner's resume; effects all
  `terminal_flat`.
- **Alpaca Paper**: heartbeat `replayed_signal`, artifact `b0ab77e0…`;
  stable after redeploy; account flat (0/0), protected SPY lifecycles
  retained in history (they satisfy acceptance row 9 — no trade was
  forced).
- **MT5 Demo (dragon)**: heartbeat `l0_refused`, artifact `539f9460…`,
  fresh; reason `max_concurrent_positions` (replayed). Doubt listed in
  §10 — I did not verify this window whether that cap reflects a real
  open ticket or held lifecycle capacity.

## 9. Project 3

Terminal record `records/project3_terminal_record.json` (16,019 jobs
verified by direct OLAP count; final backup
`project3-evidence-…-ae7caa31`, sha256 `73c46d56…`, 1,015,513,088 bytes);
`project3_terminal_verify.py` proves OLAP count, hash agreement, snapshot
rehash and scheduling silence (verified clean on omega; dragon/gamma have
zero project3 units/crons/Hermes jobs). Hermes context: generated
runtime-facts block with explicit precedence now heads
`~/.hermes/memories/MEMORY.md`; regression fixture proves stale "running"
prose loses. Paging is integrity-loss-only.

## 10. Explicitly Incomplete / Doubts

1. **Live-vs-sim business-evidence loop: not built.** The shared OLAP
   fact contract, replay-through-simulator command and descriptive
   residuals remain undelivered.
2. **Full per-bar decision-fact contract (order P1) partial**: the
   heartbeats and `live_model_inferences` bind venue, fingerprint,
   asset, timeframe, bar, input/artifact/config hashes and scores, but
   one normalized per-due-bar decision fact (incl. HOLD reason and risk
   envelope) with a duplicate-bar test per venue is not yet materialized.
3. Adversarial rows 14-16 have no NEW tests this window (14 relies on
   retained suites; 15 awaits your Priority B concurrence; 16 depends on
   item 1).
4. Resume runtime acceptance pending the owner (§5); the real
   `halt→none` transition has therefore never run in production.
5. MT5 `max_concurrent_positions` refusal unverified (§8).
6. Findings §6.2-6.4 are reported, not fixed.
7. The daily P3 digest and gamma-originated live traffic have not yet
   been observed in production (nothing qualifying occurred).
8. K2 (GBrain lockfile/postinstall packet) remains blocked as ordered.

## 11. Disclosure

During the pre-order evidence gathering I printed one raw Paper account
identifier into the owner chat (from a journal fact substring). The
doctrine is identifiers stay out of chat regardless of environment; the
ledger, router and monitor built today all redact that pattern
structurally. Reported once for the record; not repeated here.

## 12. Requested Actions

- **Owner (Master):** enable TWS daily Auto restart; when you wish the
  IBKR seat live again: `python tools/mint_resume_capability.py
  --profile examples/configs/ibkr_usdcad_model_profile_v1.json
  --resume-of-effect-id l1e-f4993c2dda8cdc2a` then
  `python tools/ibkr_resume_after_reconciliation.py --config
  examples/configs/ibkr_usdcad_model_runner_v1.json` (both interactive,
  on Omega). Closures for eligible findings remain yours.
- **General Musashi:** verify this packet; reproduce the failure cases;
  disposition findings §6.1-6.5; concur (or not) with the incomplete
  scope in §10 so the live-vs-sim loop and decision-fact contract can be
  the next bounded window.

*Ritsurei.* — Satoshi III (Mujuro Utsutsu)
