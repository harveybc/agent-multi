# General Satoshi Audit Invocation 05

Date: 2026-08-01
From: General Musashi, technical lead
To: General Satoshi, independent auditor and academic lead
Authority: project owner Harvey

General Satoshi, please perform a strict independent delta audit. Hard
criticism and concrete improvement proposals are welcome. Reproduce material
facts instead of accepting this invocation, prose status or another agent's
conclusion. Do not mutate the active DOIN job, broker accounts, credentials,
system services or live state.

## Required Reading

Read in this order:

1. `docs/work_plan/README.md`
2. `docs/work_plan/27_REALTIME_FEATURE_AND_ASSET_PARITY.md`
3. `examples/config/live_parity/project3_realtime_feature_asset_contract_v1.json`
4. `app/live_parity.py`
5. `tools/project3_live_parity_audit.py`
6. `tests/unit/test_live_parity.py`
7. `docs/work_plan/20_PROTECTED_EXECUTION_ACTIVITY_GATE_INCIDENT_2026_07_29.md`
8. `docs/work_plan/22_MULTI_VENUE_PAPER_EXECUTION_AND_SOCIAL_TRADING.md`
9. `docs/handoffs/MUSASHI_DECISION_RELAY_2026_08_01.md`
10. `docs/audits/evidence/MUSASHI_MULTI_FRONT_STATUS_2026_08_01.md`
11. `docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md`
12. `docs/handoffs/CLAUDE_AUDIT_RECOVERY_PROMPT_2026_07_30.md`

Cross-repository material:

- `/home/harveybc/Documents/GitHub/lts/mt5/MQL5/Experts/LtsMt5ReadOnlyBridge.mq5`
- `/home/harveybc/Documents/GitHub/lts/docs/MULTI_VENUE_PAPER_EXECUTION.md`
- `/home/harveybc/Documents/GitHub/lts/tests/unit/test_mt5_bridge_lab.py`
- `/home/harveybc/Documents/GitHub/financial-data/work_plan/selection_artifacts_2026_07_10/doin_exploration_shortlist.csv`
- `/home/harveybc/Documents/GitHub/financial-data/_metadata/subscriptions.json`
- `/home/harveybc/Documents/GitHub/financial-data/_metadata/stage15_cryptoquant_value_probe.json`

## Audit Questions

### A. Real-time feature contract

Try to falsify the architecture rather than summarize it.

1. Does it correctly separate decision-data source, normalized facts, shared
   feature computation, model inference and execution venue?
2. Can any historical-only, revised, stale or unavailable source pass the
   live-inference gate accidentally?
3. Are availability masks and fallbacks constrained enough to prevent a model
   from learning a source that silently disappears live?
4. Is the `event_time`/`available_at`/vintage contract sufficient for macro,
   calendar, on-chain and filing inputs?
5. Does the contract consider all plausible high-value source families without
   pretending unintegrated or cancelled providers are operational?
6. Is computing complex transforms in one shared implementation the correct
   parity boundary? Identify any transforms whose batch/stream semantics still
   need an explicit tolerance or state contract.

### B. Asset and venue universe

1. Challenge the core set, diversification controls and research watchlist
   against the OLAP shortlist and current venue evidence.
2. Verify that weak standalone controls are retained only for defensible
   diversification, risk or parity reasons.
3. Audit every alias where research spot data may drive CFD, cash-FX, ETF or
   broker execution. Focus on basis, units, trading hours, financing and
   weekend behavior.
4. Determine whether any promising cell is incorrectly omitted or any listed
   cell lacks enough evidence even for its stated set.
5. Verify that broker availability is treated as a constraint rather than an
   alpha ranking.

### C. Active DOIN campaign

Use read-only supervisor endpoints on port 8795 and node endpoints where
needed.

1. Reproduce same plan, seed, domain, population fingerprint, dataset, worker
   versions, candidate claims and finalized anchor across four workers.
2. Investigate the persistent equal-height competing tips. Identify the exact
   existing resolver/retry path and whether convergence is guaranteed on a new
   block, timer or peer event. Do not propose a replacement consensus design
   unless a test proves the existing contract insufficient.
3. Check for duplicate candidate evaluation, independent populations or
   progress loss.
4. Recalculate throughput and ETA. Treat queued job ETA zero as unavailable,
   not immediate completion.
5. Verify owner-ratified Alternative A: job 0 is an initialization proxy; job 1
   is authoritative under robust weekly RAP. At archive, rider (ii) must label
   job 0 honestly and AT-F1-013 must test final-chain weekly top-2 inclusion in
   the elite warm start.
6. Verify the cost distinction: job 0 uses static slippage `0.000075`; job 1
   easy-floor begins at `0.000025` per side. Report any code path that mixes
   these contracts.

### D. Live and operational fronts

1. Reproduce Alpaca, IBKR, MT5 and shadow functional freshness from current
   facts, not process existence.
2. Verify the expanded MT5 default watchlist is read-only and does not imply
   runtime observation until the EA is reloaded.
3. Confirm no model is permitted to trade before closed bars, feature parity,
   protected canary, SL/TP and reconciliation pass.
4. Review Gamma's 88% root usage and the unverified QEMU guest-agent state as
   operational risks, assigning proportionate severity.
5. Recheck social collector freshness, Flash budget arithmetic, approval-gated
   publishing and the prohibition on direct social model inputs.

## Required Output

Produce a dated audit report under `docs/audits/` and update the findings
register/recovery state only where evidence warrants it. For every finding:

- stable ID, front, severity and status;
- observed fact versus inference versus proposal;
- exact reproduction command and evidence path;
- impact on safety, profit/risk, reproducibility or operational continuity;
- smallest correction and a test that would prevent recurrence.

Do not close a finding you introduced and materially corrected. Do not present
a proposal as owner authority. Preserve the owner's priority order: live
testing, optimization, academic work, social work. End with the three highest
expected-value improvements and explicitly state what evidence would falsify
each recommendation.
