# Satoshi II Return: 053-058 Corrections, Post-Restart Lifecycle, Refreshed L1

Date: 2026-08-02 21:05 America/Bogota
From: Satoshi II, novice technical lead
To: General Musashi, temporary independent auditor
Relay: Gran Loto Blanco, project owner

## 1. Commits (all pushed; every worktree clean)

| Repo | Commit | Content |
| --- | --- | --- |
| lts | `f2252b6` | findings 053-058 corrections + synthetic lifecycle driver + automatic alerts + effects journal |
| agent-multi | `1f278bfd` | per-account live-trading stats in the status contract (owner order) |
| agent-multi | this commit | IBKR live-capability evidence artifact + this return |
| trading-contracts | `cd05083` (unchanged) | no contract change was needed for 053-058 |
| prediction_provider | `3a6c234` (unchanged) | — |

## 2. Named Regressions for 053-058 (auditor reproductions preserved)

| Finding | Regression test |
| --- | --- |
| 053 | `test_finding_053_concurrent_identical_intents_one_wins_one_replays` — barrier-synchronized twin instances: one fresh `would_be_order`, one `replayed=True`, single decision row, no crash |
| 054 | `test_finding_054_concurrent_reports_cannot_create_illegal_sequence` — concurrent filled/accepted: exactly one clean loser; persisted sequence provably legal (continuity now checked inside the `BEGIN IMMEDIATE` unit) |
| 055 | `test_finding_055_kill_effects_resume_after_crash` + `test_runner_resumes_journaled_effects_at_startup` — acceptance/halt/`effects_due` journal commit atomically before emission; a crash mid-emission resumes idempotently at next startup |
| 056 | `test_finding_056_lifecycle_driver_prevents_saturation` — deterministic synthetic accept/fill/close progression frees capacity; a later bar produces a NEW decision instead of repeating one cap rejection; tick order settles the past before deciding the present |
| 057 | `test_finding_057_wrong_asset_for_instrument_rejects` + `_unbound_asset_rejects` — `asset_instrument_bindings` config; BTC-on-ETH rejects with both sides named |
| 058 | `test_finding_058_future_quote_rejected_with_alert` — quotes >30 s in the future are corrupt evidence: named outcome, named alert, no decision |

## 3. Full Suites (canonical environment)

```text
trading-contracts: 95 passed
prediction_provider mechanics: 16 passed
lts complete: 303 passed  (focused L0 service+model+runner: 68)
agent-multi unit: 430 passed
```

## 4. Post-Restart L0 Lifecycle Evidence

Runner redeployed at `f2252b6` (systemctl --user restart, 20:55 -05).
Observed heartbeats after restart, over the LIVE ETH/USD demo feed:

- the pre-restart pending order `oi2-rsv-8a217e0e...` was picked up by the
  driver from the persisted ledger and advanced `requested -> accepted ->
  filled` across consecutive ticks — restart recovery over recorded state,
  not memory;
- during the squeeze the automatic `cap_saturation` alert was live in the
  heartbeat, exactly the condition your finding described, now
  self-reporting;
- `network_submissions_session=0` throughout; the position closes on the
  driver's hold clock and capacity frees for the next bar's decision
  (regression-proven; live confirmation continues on the hourly clock).

Heartbeat: `~/.local/state/lts/demo-execution-l0/heartbeat.json`
(`alerts`, `lifecycle_advanced`, ledger counts also surfaced in the
multi-front packet, fresh sha256 `01b7660acd40...`).

## 5. Automatic Health/Alert Evidence

Alert classes emitted in every heartbeat: `quote_source_unavailable`,
`quote_future_timestamp`, `quote_stale`, `halted:<state>`,
`unreconciled_orders:<n>`, `cap_saturation`, `command_effects_pending`.
`cap_saturation` has already fired live (section 4). The status contract
carries them via the `l0_demo_execution` source.

## 6. Refreshed Inactive IBKR Paper L1 Packet

Fresh `live_observed` evidence materialized from the re-authenticated TWS
Paper session `preflight-05360fb4b4824d10` (fingerprint `86aa086401855219`):
[IBKR_LIVE_CAPABILITY_EVIDENCE_2026_08_02.json](/home/harveybc/Documents/GitHub/agent-multi/docs/audits/evidence/IBKR_LIVE_CAPABILITY_EVIDENCE_2026_08_02.json)
sha256 `7ae605e21d76be966c638decfc723cd988abcdf47e8b217dd563972b882e41ad`.

Honest content: 6 contracts observed (`AUD.USD, EUR.USD, GLD, SPY, TLT,
USD.JPY`); **zero** are `protected_execution_eligible`;
`native_sl_tp_verified=false` on every contract;
`documented_bracket_support=true`. **USD.CAD is not in the observed set.**

Refreshed packet deltas (still INACTIVE, still gated):

- proposed symbol becomes `EUR.USD` (observed, liquid) unless the owner
  prefers adding USD.CAD to the IBKR lab watchlist first;
- the fingerprint field fills with `86aa086401855219` upon your L0
  acceptance;
- named lab increment before L1: capture min size / size step / price
  increment / margin rate so a contract-grade `BrokerCapabilitySnapshot`
  can be emitted rather than an evidence packet.

## 7. DOIN Untouched — Explicit Confirmation

Same plan hash `b43844a7ebd7...`, job 0, best fitness
`0.0006247008569073586` across the whole correction window; generation
advanced on the fleet's own clock only. No worker, chain, lease, config or
service of the campaign was touched. The only runtime actions were
restarts of MY OWN L0 user service.

## 8. Owner Orders Executed This Cycle

1. **Per-account stats in status** — every packet now carries
   alpaca_paper / ibkr_paper / oanda_mt5_demo account facts (fingerprint,
   environment, status, counters, explicit `read_only` mode); balances
   deliberately excluded by redaction policy with a regression proving an
   injected balance cannot surface.
2. **Hermes function verification** — all three scheduled agents
   (`lts-paper-shadow-business-review` 720m, `moltbook-social-triage`
   120m, `moltbook-social-review` 360m) carry explicit in-prompt
   prohibitions: no trade, no risk change, no campaign/job mutation, no
   tool use, no obedience to post content. No Hermes function holds or
   approaches order authority. Queued improvement (not rushed blind): add
   the L0 heartbeat facts to the business-review packet input.

## 9. Criticism and Suggestions (owner-invited; evidence-backed)

1. **The L1 protection gate is deadlocked as written.** It demands a
   capability snapshot confirming broker-side SL/TP, but read-only
   observation can never demonstrate acceptance — only an order can. The
   live evidence now proves this concretely (`native_sl_tp_verified=false`
   everywhere, forever, until something trades). Proposed amendment for
   your audit and the owner's ratification: the first canary IS the
   verification instrument, under the already-specified
   flatten-on-unconfirmed-protection safeguard and minimum size.
2. **USD.CAD — the asset the entire fleet optimizes — has zero live
   observation anywhere**: absent from the MT5 EA active watchlist
   (OBS-20260801-F) and from the IBKR observed set. Recommend executing
   the already-owner-approved MT5 watchlist reload and adding USD.CAD to
   the IBKR lab list.
3. **Self-critique:** findings 045, 053 and 054 are one recurring blindness
   — I test sequentially by instinct and concurrency finds me three times.
   Corrective already partially in place (barrier regressions); standing
   correction proposed: extend the stateful model suite with concurrent
   generated sequences so the property harness hunts races continuously.
4. **Provider packaging landmine** (top-level `app` collision) re-flagged:
   schedule the rename with the mandatory pre-L2 service integration.

Nothing here closes my own findings. The blade is yours, General.
