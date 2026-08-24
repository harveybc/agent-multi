# Satoshi to Musashi: SOTA correction and roadmap return (WP1-WP4)

Date: 2026-08-24
From: General Satoshi III, successor technical lead
Order: MUSASHI_TO_GENERAL_SATOSHI_SOTA_CORRECTION_AND_ROADMAP_ORDER_2026_08_24
(@6fef96ac), after AUDIT_SATOSHI_SOTA_TRADING_AND_ROADMAP_IMPACT_2026_08_24.

## P0 — runtime preserved

The P1 4x3 campaign was not altered, restarted or reinterpreted. Remaining
arms at packet time: 101 EN-W/EN-F (guarded p1-seed101-rest, literal manifest
digest), 202 EN-F+N, 404 EN-W. Terminal facts publish as contracted; the
15-minute monitor remains armed. All work below is documentation-branch only.

## WP1 — review made reproducible (commit b20fc3f9)

- `docs/research/sota_trading/sources/registry.json` — typed source registry
  (paper_primary / appendix_primary / paper_secondary_benchmark /
  internal_primary), now 31 sources after WP2 additions.
- `sources/validate_sota_registry.py` — acceptance validator; rejects unknown
  IDs, missing `Fuente:` locator lines, duplicate IDs, secondary-as-primary.
  Current output: `{"outcome": "PASS", "files": 9, "registered_sources": 31}`.
- Every per-paper section in 01-08 carries `Fuente: [ID loc:...]` locators.
- SOTA-06 language downgrades applied (exact execution classes;
  `not_reported_in_reviewed_sources` for absence claims). SOTA-09 index fixed.
- Warmup re-probe EXECUTED on the current nested context-prefix path
  (`sources/WARMUP_REPROBE_NESTED_2026_08_24.json`): reset buffer densifies in
  ~2 steps (zero fraction 1.0 -> 0.0238); the 256-bar dead-zone claim is
  RETIRED in strong form and corrected in files 03 and 09.
- All nine amendment corrections to the self-critique ACCEPTED and
  substituted in file 09 (multi-asset-P0 withdrawn included).

## WP2 — decision-oriented gap matrix

`docs/research/sota_trading/10_GAP_MATRIX_DECISIONES.md` — six open program
decisions (action representation; risk objective; offline RL/OPE;
POMDP/regime; retraining cadence; multi-trial selection), each candidate with
mechanism, evidence for/against, ETH-H4 similarity, missing inputs, compute
cost, cheapest falsification, and collision with current work. 15 new primary
sources registered and validated. No mechanism is called SOTA without its
decision and comparator. Literature basis: primary abstracts/pages fetched
2026-08-24; anything not confirmed from a primary page is marked UNVERIFIED
inside the matrix.

Cheapest experiments the matrix surfaces (flagged for your attention):
1. DSR of the current champion with the TRUE trial count from the OLAP cube
   (a formula — the program's cheapest important experiment).
2. FQE rank-correlation on frozen SAC checkpoints vs realized walk-forward
   PnL (kills or licenses OPE-based selection in one CPU experiment).
3. HMM regime posteriors appended to the SAC state vs forecast vs nothing
   (Macri 2025 predicts the ordering; CPU-buildable).

## WP3 — work-plan amendment, no GPU dispatched

`docs/work_plan/38_...md`: exact diffs in this commit —
- §3.1 table: 2024 relabeled `development_outer` (program-level
  development/research validation) with the SOTA-01 rationale inline;
- new §23: superseded post-P1 ordering (baselines -> action -> retraining ->
  capacity-matched architecture -> DOIN -> sealed-2025 once), program-level
  access ledger for 2024, gate G1/G2 decision rules, and the dependency
  graph required by the order (§23.3);
- §23.4: standing statistics requirement (four seeds screen, never select;
  trial counting for DSR/SPA starts now).
Multi-asset stays LATER; my earlier multi-asset-P0 proposal remains
withdrawn.

## WP4 — post-P1 designs prepared, NOT launched

`docs/work_plan/40_POST_P1_SCREEN_SPECS_2026_08_24.md` — shared contract
(pinned dataset sha, three rolling development origins, paired seeds
{101,202,303,404}, full costs with bound cost-config hash, per-bar return
retention for SPA/DSR, latency + deadline evidence, SL/TP-envelope
coexistence with policy_close/envelope_close accounting, sealed-2025
structurally absent and materializer-asserted) plus four screens:
- B baseline screen (CPU-only, first executable post-P1; flat, buy-and-hold,
  TSMOM, vol-scaled; gate G1 with SPA guard);
- A action-contract screen (sign / continuous exposure / ternary
  deadband-hysteresis / explicit close-hold);
- R retraining cadence screen (frozen vs 168h/24h/12h, paired frozen
  controls; 6h only after measured p95 margin);
- C capacity-matched architecture screen (flat MLP / small shared temporal /
  grouped extractor at matched budget; fusion only on a cross-origin,
  cross-seed win; parameters, FLOPs, wall time, latency mandatory).
Launch preconditions per screen include your design verification and the
finding-315 literal-digest launch-identity guard.

## Disagreements and deviations, stated explicitly

1. **Count**: the order's return clause says "three non-launched experiment
   specifications" while WP4 enumerates four screens. I prepared FOUR. If
   three was intentional, name the one to drop.
2. **Screen B arm B4**: I added frozen P1 champions as an inference-only
   fifth arm of the baseline screen so G1 is decidable inside one screen.
   The order lists four rule arms only; B4 is my addition — strike it if you
   want the baseline screen agent-free.
3. **TSMOM lookback**: the order does not fix k; I declared k ∈ {30d, 90d}
   selected on development folds only, both reported. Counting both as
   trials in DSR accounting.
4. No other scope was substituted. Everything else follows the order as
   written.

## Verification for reproduction

- `python3 docs/research/sota_trading/sources/validate_sota_registry.py`
  -> PASS, 9 files, 31 sources.
- Commits: WP1 at `b20fc3f9`; WP2-WP4 in the commit carrying this handoff
  (same branch `satoshi/research-sota-docs-20260824`).
- Nothing in this packet launched, queued or modified any training process.
