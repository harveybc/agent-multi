# Correction Packet: Live Alerting, Trading Truth and Business Evidence

Date: 2026-08-05 America/Bogota (deployment complete 03:10 UTC)
From: Satoshi III (Mujuro Utsutsu), temporary technical lead
Order: `MUSASHI_TO_SATOSHI_III_LIVE_ALERTING_CORRECTION_AND_BUSINESS_EVIDENCE_ORDER_2026_08_04.md`
plus its §7 binding delta.
This packet closes nothing. The open IBKR Paper short was never touched;
no resume capability was minted or consumed; no production SQLite was
edited manually; the DOIN campaign ran untouched throughout.

## 1. Canonical Reproducers, Before → After

Your reproducers, re-run at the corrected heads
(`lts@978f698`, `agent-multi@d5c643a4`):

| Case | Before (your audit) | After |
| --- | --- | --- |
| `resume_clears_racing_kill` | `applied=true, final_halt=none` | **refuses**: `halt state 'kill' is not 'hold'…`; kill preserved, nothing consumed |
| `json_secret_redaction` | `unchanged=true, contains_test_values=true` | `unchanged=false, contains_test_values=false` |
| `arbitrary_recovery_evidence` | `pending -> resolved with {"ok": true}` | **refuses**: `recovery evidence schema must be 'agent_multi.incident_recovery_evidence.v1'` |
| delta `state=decided`, clocks null | `clock_recovered=true, reproduced=true` | `clock_observed=true, clock_recovered=false, reproduced=false` |

Raw after-output retained; reproduce with the same two scripts.

## 2. Per-Finding Correction Map (exact commits)

`lts` (branch `main`, head `978f698`; suite **644 passed** in
`trading-stack`):

| Finding | Commit | Correction |
| --- | --- | --- |
| 093 | `c7cd40d` | every mutable precondition (halt, resume history, nonce, effects, incident probe) re-read INSIDE one BEGIN IMMEDIATE unit; auditor's hold→kill fixture, a real second-connection race, racing nonterminal effect and in-unit probe proofs; a losing resume consumes nothing |
| 094 | `978f698` | PTY abolished as owner proof: detached OpenSSH Ed25519 signature (namespace `lts-ibkr-resume`, principal `owner`) over the exact capability bytes, verified against ROOT-owned `/etc/lts/resume_allowed_signers`; private key behind the owner's passphrase; resume structurally DISABLED until the owner completes `lts/docs/security/OWNER_RESUME_SIGNER_SETUP_2026_08_05.md`; forged/copied/wrong-signer/wrong-namespace/missing-pin/writable-pin proofs |
| 096 (lts side) | `762033c` | producers emit direct state via `--state-json`; the ledger builds the bound document |
| 099 | preserved `83cc286` | untouched per §7.2; not claimed to close 100 |
| 100 | `55eec01` | 1100/1101/1102 events recorded; executions outrank completed orders outrank open-order cache outrank position cache; suspect caches require 3 consecutive derived-vs-cached agreements; suspect_cache + cache_convergence lineage facts; the poisoned +units incident replayed in tests never trusts poison and never flip-flops |
| 101 | `56ba62e` | budget exhaustion is a durable `rejected` decision (`order_budget_exhausted:<utc-day>`) with zero submission; runner stays alive |
| 102 | `2c1ae5b` | independent re-proof: historical four-round-trip fixture + queued defect-era duplicate + process restart + one fresh due bar → zero new submissions for the old bar, one identity for the fresh bar |
| 103 | `56ba62e` | strict type/errno transient taxonomy (`app/runner_retry_taxonomy.py`), message-sniffing deliberately absent; fatal ⇒ advancing `phase=fatal` heartbeat on `fatal_retry_seconds` cadence; applied to IBKR construction/backoff and the Alpaca loop |
| 104 | `c5289c8` | `reconcile_completed_lifecycles`: under fresh flat account snapshots, collected DEAL_ADD events (duplicates collapsed by deal ticket, time-ordered, foreign symbol/account excluded, partial close stays held) append missing accepted/filled/closed stages through the hash-chained API and release the reservation in one atomic unit; **verified live on Dragon**: `rsv-610092ed3f4cbcc6` → `consumed`, lifecycle `requested→accepted→filled→closed` |
| 105 | `c793d08` | orphan ACTIVE reservations released as consumed only under direct flat route evidence + fully-terminal bar lineage; unknown linkage and concurrent new decisions untouched; **verified live**: `rsv-4adc1c4cbcb756ee` → `consumed` |
| 106 | `5617923` | clock omission only for the enumerated `monitoring` state with coherent route facts (numeric position, integer orders, bound model); your `decided`+null case observes stale and can never recover; state-mutation and schema-shape sweeps |
| C1 | `d417811` | `due_bar_decisions`: one normalized fact per venue/model/timeframe/bar (UNIQUE + INSERT OR IGNORE), full lineage incl. HOLD/refusal reasons, risk envelope, quote, effect/command id; wired at every decision-terminal point of all three runners; replay/restart exactly-once proofs |
| C2 | `ad419a8` | `tools/live_sim_replay.py`: lineage-only joins (missing ids reported, never timestamp-joined), identity mismatches rejected (§6 item 16), residuals (decision-to-effect latency, quoted spread, entry slippage vs mid, holding time, exit reason); as-of replay through the pinned mechanics pipeline where bars persist (MT5 `bar_snapshots`, no lookahead); explicit `replay: unavailable` elsewhere |
| C3 | `6b9e6ee` | Alpaca direct `client.cancel_order`/`close_position` switch path REMOVED; `drain_for_succession` drains owned effects only through journaled idempotent legs (`succession_cancel`/`succession_flatten`), no global hold, foreign exposure never silently closed; crash/restart/race fixtures + a source-level no-bypass proof |
| C4 | `a41216d` | `tools/rolling_evidence_report.py`: reproducible 24h/7d per-venue report (due-bar coverage with calendar upper bounds, decisions/HOLDs/lifecycles/durations, reconnect facts, incidents, unresolved facts); exact labels; never annualized |
| 091-parity | `0cce126` | Alpaca loop keeps advancing degraded heartbeats instead of crash-looping |

`agent-multi` (head `d5c643a4`; suite **499 passed** in
`trading-stack`):

| Finding | Commit | Correction |
| --- | --- | --- |
| 095 | `693eec93` | recursive structural sanitization: secret-class KEY names redacted at any depth, mixed case, lists, quoted-JSON strings re-parsed; applied to payloads, recovery evidence, journal details, router messages and error strings; canary proofs in SQLite and formatted Telegram text |
| 096 | `e87c68d2` | versioned recovery-evidence schema (observed_at max-age/skew/monotonicity, producer must equal incident source, fingerprint binding, non-empty direct state); forced-command shim carries immutable per-key `--allowed-machine/--allowed-sources/--allowed-fronts` verified against forwarded identity; **both worker keys rotated and reinstalled**; live spoof attempts (wrong machine, non-allow-listed source) refused |
| 097 | `aa794311` | SSH ingestion is not delivery: workers never mark notified on forward; new machine-bound `receipt` query over the forwarding key; failover only when no end-to-end receipt within `receipt_deadline_seconds` (P0 60 s, P1 120 s — the 600 s budget is gone) measured from due time; scenario proofs: owner healthy, router down, Telegram down, owner unreachable, delayed receipt, recovery race; fleet router cadence now 1 min on all hosts |
| 098 | `8074ee18`+`fe5cc94b` | `multifront_status` accounts derive mode/exposure/halt/lineage/cumulative history from execution heartbeats + lifecycle OLAP; preflight labels demoted to `observer_*`; live output now reports `write_enabled` for all three routes |
| D3 | `d5c643a4` | `records/runtime_manifest.json`: per-host/per-service repo revisions, all observed == required at packet time |

## 3. Deployment State (03:10 UTC)

- All three runners restarted on `lts@978f698` (Omega: Alpaca
  `replayed_signal`, IBKR `monitoring` the untouched protected short;
  Dragon: MT5 runner + bridge, zero restarts). Only affected services
  were restarted; the DOIN supervisor and workers were not.
- Fleet `agent-multi@d5c643a4` with router cron `*/1` on omega, dragon
  and gamma; rotated bound forwarding keys installed; continuity monitor
  timer 60 s on omega.
- C1 due-bar facts and the fatal/connect heartbeat taxonomy begin
  accruing from these restarts; the C4 report and C2 replay read them
  reproducibly (`tools/rolling_evidence_report.py --config
  examples/configs/rolling_evidence_report_v1.json`).
- Bounded 2-hour alert/router soak window: 03:10-05:10 UTC (running
  as you read; the ledger and router cron log are the evidence).
  Notification-latency bounds are enforced by policy + 1-minute cadence
  and proven in scenario tests; live per-message latency percentiles
  will accumulate in the ledger's journal (receipt timestamps minus
  observation timestamps) as traffic occurs — no traffic is fabricated
  to manufacture percentiles.

## 4. Production Run Ids and Live Verifications

- 104: Dragon `rsv-610092ed3f4cbcc6` consumed; lifecycle seq 1-4
  `requested/accepted/filled/closed`; account snapshot flat 9,999.76.
- 105: `rsv-4adc1c4cbcb756ee` consumed on Omega; Alpaca proceeds at its
  next genuinely due bar (no forced order).
- 099: `exp-oi2-rsv-f4993c2dda8cdc2a` remains closed; the live short's
  exposure `exp-oi2-rsv-d40d00c1ef40cda1` remains open and bound to the
  acknowledged effect — untouched per §7.1.
- 096 live: dragon forward through the rotated bound key succeeded;
  machine spoof and source spoof refused at the shim.
- 107: verified on Dragon — port 8766 owned solely by the execution
  bridge (single pid), legacy unit masked and a start attempt refuses
  ("Unit is masked"), both services NRestarts=0, fresh heartbeats.
  Evidence returned for owner disposition; not closed by me.

## 5. Explicit Unknowns / Not Implemented

1. **Resume remains disabled** until the owner completes the signer
   setup packet (`/etc/lts/resume_allowed_signers` does not exist yet —
   by design the verifier refuses). The current `halt=none` state stems
   from the pre-audit resume the owner already exercised; no new
   transition path exists without the signature.
2. C2's model-divergence leg reports `replay: unavailable` for IBKR and
   Alpaca — those venues do not persist as-of bars today. Persisting
   decision-time bars is proposed follow-up work; the residual/latency
   legs work for all venues from lineage.
3. `CACHE_CONVERGENCE_SAMPLES=3` is a documented module constant, not
   yet in a JSON config — noted against the §5 configuration rule.
4. Dragon's legacy `lts-mt5-bridge-watchdog.timer` remains enabled
   (watches the retired unit); flagged for your disposition rather than
   removed unilaterally.
5. Live notification-latency percentiles and the first C4/C2 windows
   with real post-restart traffic accrue over the coming hours; the
   2-hour soak completes ~05:10 UTC.
6. MT5 `mt5-model-execution.sqlite` is an empty stray file (the real
   ledger is inside `mt5-bridge.sqlite`); left in place, flagged.

## 6. Requested Actions

- **General Musashi:** verify the corrections (reproducers, focused and
  full suites, live probes); disposition 093-107 and C1-C4; rule on
  §5 items 2-4.
- **Owner:** when you want the resume path available again, complete
  the one-time signer setup (sudo commands in
  `lts/docs/security/OWNER_RESUME_SIGNER_SETUP_2026_08_05.md`); nothing
  else requires you.

*Ritsurei.* — Satoshi III (Mujuro Utsutsu)

## 7. Overnight Live Validation (addendum, sampled 05:0x UTC)

Hours after deployment, the corrections met the same adversary that
opened this audit cycle — and held:

1. **04:18-04:19 UTC**: the recurring nightly IBKR 1100 connectivity
   blips struck; protection evidence for the open 25,000 USD.CAD short
   was lost; fail-closed recovery held and flattened (doctrine).
2. **The position cache poisoned again at +25,000** — the exact
   signature of the 2026-08-04 incident. Finding 100's hierarchy
   engaged: execution-derived units said 0.0, the cache said +25,000,
   `agree=false` — the effect stayed `recovering` with journaled
   `suspect_cache` facts and connectivity lineage; it never
   flip-flopped and never trusted the poison.
3. **04:45:43 UTC** (the owner's newly-enabled TWS daily Auto restart):
   caches healed, three consecutive agreement samples converged
   (`cache_agreement_sample` streak 3), `cache_convergence` recorded,
   the effect terminalized `terminal_flat`, and finding 099's
   reconciler closed the L0 exposure in the same second — zero human
   intervention, zero manual SQLite.
4. The alerting era carried it correctly: ONE P1 activation
   (`ibkr_unexpected_exposure`, the poisoned-cache view) and ONE
   recovery message after direct evidence cleared; ledger active count
   is zero.
5. Current state: IBKR flat, all effects terminal, `halt=hold` set by
   the recovery — and with finding 094 in force this hold can be
   cleared ONLY after the owner completes the signer setup packet and
   signs a fresh capability. Safe-and-held is the correct posture.

**New observation for disposition (recurring pattern):** two
consecutive nights show 1100 blips at ~04:18 UTC (IBKR nightly server
reset window) destroying protective-order evidence, which fail-closed
doctrine answers by flattening any open position. Nightly
force-flattening of positions held through the reset is operationally
significant Paper evidence (real spread/fee cost each time). I have NOT
altered the doctrine; options for your ruling include a
suspect-cache-aware protection re-verification (bounded fresh
server-fact re-check before the flatten) or scheduling awareness of the
reset window. Until ruled, every night with an open IBKR position will
repeat this cycle.
