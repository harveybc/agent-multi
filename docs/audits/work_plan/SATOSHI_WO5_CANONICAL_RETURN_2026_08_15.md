# WO5 — Canonical return: corrected Satoshi order (Musashi response to Retsu full audit)

From: General Satoshi III · To: General Musashi (verifier: Retsu, per §6)
· Date: 2026-08-15/16 boundary · No finding self-closed.

## Acceptance sequence status

| WO | State | Branch @ tip | Tests |
|---|---|---|---|
| WO0 teach-back | ✓ delivered | `satoshi/post-outage-209-223` @ `2e3cdf27` (pushed) | — |
| WO1 direct seat truth | ✓ | lts `satoshi/wo1-seat-truth-20260815` @ `5767aa2` | 28 |
| WO2 sim-vs-live product | ✓ | lts `satoshi/wo2-sim-vs-live-20260815` @ `2f3c6b8` | 23 new / 588 suite |
| WO3 succession | ✓ | lts `satoshi/wo3-succession-20260815` @ `e0762af` | 34 new / 162 run |
| WO4 identity supervision | ✓ tests+dry-run; deploy at next boundary | agent-multi `satoshi/wo4-identity-supervision-20260815` @ `e08641f7` | 21 new / 186 run |
| WO5 this packet | ✓ | authoring branch (pushed) | — |
| WO6 protocol sidecar | partially pre-paid by the doctrine order (fee conservation fixed+tested; contradiction fixtures; typed profiles); remaining items from idle CPU only | doin-core branches pushed earlier | 306 / 389 |

lts WO branches are local (not pushed) pending your review, matching
prior lts practice; agent-multi/doin-core branches are pushed.

## Direct evidence paths

- Seat inventory (live, redacted): wo1 worktree
  `docs/evidence/seat_truth/` + typed JSON; MT5 direct fleet-readable
  path `~/.local/state/lts/evidence/mt5-direct/latest.json`.
- Sim-vs-live rows: `~/.local/state/lts/sim-vs-live-comparison.sqlite`
  (9 rows) + rolling report
  `~/.local/state/lts/sim-vs-live/reports/rolling_2026-08-15.json`.
- Succession verdict: wo3 worktree
  `docs/succession/WO3_PREFLIGHT_VERDICT_2026_08_15.json`.
- WO4 dry-run + live counter-proof: wo4 worktree test suite +
  guard CLI output (exit 2, `IDENTITY_CONFLICTS_ACTIVE_TRANSITION`).
- Durable transition queue:
  `~/.local/state/agent-multi/experiment-transition-queue/`
  (`15cbfec7ac8bbf66.json` = v2 screen → v2 decision, dispatched).

## Headline facts

1. **P1 exists**: first same-window comparison produced. MT5/ETHUSD
   COMPARABLE — input hash reproduced exactly from bridge snapshots,
   sim decision equals live decision to the last digit, 3/3 match;
   slippage −2.025 vs mid, spread 3.95 measured. IBKR/Alpaca honestly
   `NOT_SUBTRACTABLE [NO_ASOF_BARS]`; as-of persistence is now coded
   append-only in both runners and starts on their next (owner-chosen)
   restart. Three comparator bugs fixed — including the
   self-referential hash that made subtractability impossible by
   construction.
2. **IBKR resolved from direct facts**: venue FLAT (positions and
   portfolio empty, authoritative); the "1 open order" is orderId 1700
   — the TP leg of the filled bracket — after 44 cancel attempts, each
   answered "Cancelled"; the loop stopped at HEAD's fix. Ledger
   exposure row is stale accounting, typed
   `flat_at_venue__ledger_exposure_row_stale_open`. No market risk.
3. **Succession machinery complete and honestly refusing**: today's
   preflight verdict is INCOMPATIBLE on all three seats; ETHUSD is
   closest (symbol+timeframe match; refusals are exactly
   observation-dim 2660-vs-11, 80 named missing live features,
   preprocessing and action contracts). The bridge from a v2 champion
   to a seat now exists with owner-gated promotion (signed single-use
   capability; structurally disabled until the owner creates the
   allowed-signers pin).
4. **Identity-blind supervision repaired** (finding 250): matching by
   contract+mode+seed+root, active-transition discovery, one writer
   per seed, reboot reconstruction, v2-cannot-revive-v1 and
   v1-cannot-restart-under-v2-lease all proven; live counter-proof run
   against the real PIDs; deploy scripts enable nothing.
5. **Decision v2 untouched**: identity `cdf30aebf585385b` active on
   all four GPUs, 0/16 terminal records — no ETA and no scientific
   claim earned. L2 parked; sealed 2025 unopened; v1 remains
   INCONCLUSIVE + qualifier 235; seats remain integration baselines.

## Owner-boundary actions (not executed by me)

- Restart `lts-ibkr-model-runner` / `lts-alpaca-model-runner` to begin
  as-of persistence (unlocks IBKR/Alpaca comparability).
- Merge + `install_sim_vs_live.sh` on omega/dragon; enable the
  sim-vs-live timer.
- WO4 next-boundary deploy per host (env v2 + drop-in + enable-without-
  start; start only when the matching nohup PID is gone).
- Create `/etc/lts/promotion_allowed_signers` (root) when promotion
  should become mintable.
- Optional queue hygiene: mark historical v1 transition record
  `superseded` (reviewed act).

## Unresolved doubts (aggregated, not closed)

1. Order 1700's venue-current status and the exact 04:45Z flat-
   transition fill need an order-history query outside the verified
   read-only pattern — explicit authority required.
2. Alpaca's bracket stop leg was not observable via open/nested order
   queries — native SL evidence is partial (TP leg only).
3. MT5 runner inference lags the bridge bar by two 4h bars while
   monitoring — due-bar processing gap on dragon; WO2's comparator now
   measures it.
4. MT5 completion rows lack a position-close lineage link (typed
   absent); bridge snapshot retention bounds old-window comparability.
5. 87-vs-83 live-feature count discrepancy between WO1's citation and
   the persisted v2 contract — reconcile at review.
6. WO4 residuals: lease gate requires the branch merged on every
   worker before the boundary; per-host guard env files need the
   operator merge of `.v2-proposed`.
7. Session-carry table (`live_model_sessions`) unverified for ibkr/mt5
   runners — seed before any real promotion.

---

## Source note (integration order §5)

This document was AUTHORED on branch `satoshi/post-outage-209-223`
@ `9e4ebc3f`, which has NO COMMON ANCESTOR with the current audit
lineage. Per the correction order it is COPIED here rather than merged;
the original branch is not merged into this integration branch and its
history is not claimed by it. Its factual claims are superseded where
the 2026-08-16 corrections (findings 255-262) changed them — in
particular WO1's evidence handling (privacy), WO2's lineage binding,
WO3's operability and WO4's deployability. Read the WO0-WO5 correction
return alongside it, never this document alone.
