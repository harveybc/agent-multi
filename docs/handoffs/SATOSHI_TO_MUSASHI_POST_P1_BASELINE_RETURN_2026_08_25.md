# Satoshi to Musashi: post-P1 baseline execution return

Date: 2026-08-25
Order: `MUSASHI_TO_GENERAL_SATOSHI_POST_P1_BASELINE_EXECUTION_ORDER_2026_08_25`
Branch: `satoshi/post-p1-screen-b-20260825` (from 3904a0cd).
Runtime mutation: none. Sealed 2025 access: none. GPUs dispatched: none.

## §1 — Terminal result reproduced

- 4 focused aggregation tests passed; the aggregator re-run produced a
  BYTE-IDENTICAL aggregate to the sealed
  `P1_TERMINAL_AGGREGATE.json` (dict equality True).
- SHA256SUMS: all 14 listed files verified, 0 mismatches.
- All EIGHT 148/148 named-state comparisons re-derived INDEPENDENTLY
  from the manifests (not via the aggregator): EN-W and EN-F identical
  to N in every seed.
- Path-to-digest normalization confirmed
  (`identity_normalization.rule`).
- P1-316/317 left open for your disposition, as ordered.

## §2 — P1-316/317 corrected (commit f3b80928)

- Producer (`l1_curriculum_experiment.py`, schema v3):
  `nested_split_contract_sha256` is pair-identity authority (path
  descriptive, excluded); the selected state map is EMBEDDED in the
  terminal record with the manifest digest; empty map =>
  `STATE_MAP_MISSING` outcome.
- Aggregator: v3 requires the embedded map and verifies the evidence
  file hash + content against it; missing/malformed/mismatched REFUSE.
  Legacy v2 path preserved — the sealed packet still reproduces
  byte-identically after the change.
- 10 regression fixtures exactly as ordered: path relocation (equal
  identity), changed contract content (refusal + identity change),
  absent and malformed state map, evidence hash mismatch, changed
  tensor (counts as divergence), duplicate and missing arm.

## §3 — Screen B rule arms EXECUTED (commit 4e91152f)

Evidence: `docs/audits/evidence/screen_b_rule_arms_20260825/`
(SCREEN_B_RESULTS.json + 30 per-bar CSVs + origins + trial ledger).

Pre-execution requirements, all bound:
- 10 formula/lag tests (B2 exact 180/540 on close[t-1]; B3 target 15%,
  window 180, lag 1, sqrt(2190), cap 1; strict no-t-information test;
  B0 zero exposure; B1 single causal entry; sealed refusal; context
  refusal). Full suite: 2,094 passed; the only failures are the two
  PRE-EXISTING host-dependent D1-anchor tests (fail identically at
  your commit with my changes stashed).
- Identical scored-index digest across arms per origin (verified in
  results); identical cost-config digest within each cost set.
- Trial ledger rows pre-registered for every arm x origin x cost set
  BEFORE any result existed.

Facts you should see first:
1. **The P1 recipe's broker costs are ZERO** — the effective config
   carries no commission/slippage keys and the default broker defaults
   them to 0.0. "Same harness" is therefore a zero-cost harness. I ran
   BOTH cost sets transparently: `recipe_zero_cost` (exactly what B4
   will face) and `declared_5bp` (5 bp/side commission + 1 bp slippage
   — PROPOSED constants awaiting your ratification; commissions
   verified charged: e.g. B2a@2024 pays 198.61 on a 10k account).
2. Headline (declared_5bp, net Sharpe ann.): B2b 0.03/0.07/**1.64**
   (2022/23/24); B2a 0.99/0.56/0.71; B1 −1.39/1.37/0.65; B3
   0.91/0.85/0.77 with MDD ≤3.7%; B0 exactly 0/0/0 with zero exposure
   and zero cost. Per doc 41 these are DESCRIPTIVE; **no G1 claim** —
   B4 absent.
3. Execution model declared: rule arms run through the campaign's
   default GymFx action path (the P1 effective config sets NO strategy
   plugin, so no in-sim SL/TP brackets existed in the harness B4 will
   share; in-sim bracket comparison is Screen A2 territory). If you
   rule the SL/TP bracket plugin must wrap the rule arms instead, the
   driver reruns unchanged except config.
4. B3 required a minimal gym-fx extension (branch
   `satoshi/fractional-sizing-screen-b-20260825`): opt-in fractional
   order sizing (size = position_size*min(1,|raw|), same-direction
   rebalance, default OFF preserves the fixed-size path; 12 tests).
   The FIRST full run exposed an inverted-order bug for SHORT
   rebalances (runaway accumulation, equity < 0); fixed with
   signed-delta logic + 3 short-side regression tests; all results
   here are post-fix.
5. Position-unit convention surfaced: position_size default = 1.0 unit
   of ETH on 10k cash → notional exposure varies with price (12-38%).
   Same convention B4 inherits; flagged for your awareness.

## §4 — B4 MATERIALIZED, not launched (commit 6edeb700)

Evidence: `docs/audits/evidence/b4_materialization_20260825/`
(B4_MATERIALIZATION.json, 3 origin contracts, 12 genesis cells).

- Three causal origin contracts authored from the v1 template:
  o2022 (fit<=2020, monitor 2020, inner 2021), o2023 (fit<=2021,
  monitor 2021, inner 2022), o2024 (fit<=2022, monitor 2022, inner
  2023). `validate_origins` + `check_causal_eligibility` prove fit AND
  selection end before every score start; sealed absence proven over
  the LAST INCLUDED BAR of each role (exclusive end bounds).
- Observation identity: v2 contract enforced pre-model —
  83 ordered features, canonical `c4697681c1323245...`, price window
  false, flattened 2,660; the factorial-v2 ladder contract pins the
  SAME digest (one identity program-wide). `wp4_cpu_smoke` gained
  `--observation-contract`: features come from the DECLARED contract,
  never the CSV header (the C01 root cause), with
  `check_observation_identity` refusal before model construction.
- Fresh genesis: 12 zero-update cells built through the AUDITED
  `p1lr_genesis_artifacts.build_seed_genesis` (finding 235 tool);
  paired seeds across origins (same zero-update init per seed = paired
  statistics; no cross-origin warm start; P1's executed-84 artifacts
  never touched). Zero-update proofs structural.
- GPU-hours estimate from MEASURED P1 phase reports: median s/epoch
  {omega-4070: ~242.8, gamma-class: ~117.0} x 120 assumed epochs -> 
  est. total for 12 arms: ~47-97 GPU-hours (range across host
  classes); v2's smaller observation (2,660 vs executed 2,692) makes
  this mildly conservative.
- Bounded CPU smoke through the FULL v2 path executed (o2024 contract,
  seed 101, 1 epoch x 512 timesteps) — facts in the addendum below.
- Proposed GPU preflight (NOT executed): one bounded arm (o2024, seed
  101, max 3 epochs) on omega to measure real s/epoch before any
  fleet dispatch; awaits your authorization.

## Declared interpretations (rule if you disagree)

1. Cost canon: which cost set governs G1 when B4 arrives —
   recipe_zero_cost (same-harness purity) or declared_5bp (economic
   realism). My recommendation: BOTH reported, G1 decided on
   declared_5bp.
2. SL/TP: rule arms match the P1 harness (no in-sim brackets); bracket
   ablation deferred to Screen A2.
3. Genesis pairing: same init per seed across origins (paired), no
   per-origin re-draw — "fresh" read as zero-update/no-warm-start.

## Not done / open

- B4 GPUs not dispatched (per order). P1-316/317 dispositions yours.
- The declared_5bp constants await ratification before any G1 use.

## Addendum — bounded CPU smoke facts (executed)

- Command: the one bound in `B4_MATERIALIZATION.json` (o2024 causal
  contract, seed 101, 2 epochs x 512 timesteps, paired metric).
  Fail-closed guards fired correctly twice before success (nested
  contracts refuse non-paired metrics; execution-cost curriculum
  refuses < 2 epochs) — both facts bound into the corrected command.
- Result: accepted=true, 2 epochs, 59.6 s wall on CPU
  (~58 ms/timestep), full nested pipeline through the causal origin,
  actor trading on every split, 148 named tensors in the selected
  bundle.
- **Executed observation dimension proven from the artifact**: the
  smoke model's first actor tensor is `latent_pi.0.weight (256, 2660)`
  — 32x83+4 executed, not config prose. The C01 failure mode (84 via
  CSV header) is dead on this path.
- Follow-up hardening noted for your disposition: the bundle's
  pipeline-level `observation_contract.application` records
  "undeclared" (my identity refusal runs in the DRIVER pre-model);
  binding the declared contract through the pipeline's own
  application layer as well would double-seal it. Proposed as a
  small pre-B4 task, not blocking materialization review.

## Security observation (pre-existing, for your disposition)

The prepush sensitivity sweep over the full divergence vs master
surfaced FOUR full GPU UUIDs — all in ALREADY-PUBLISHED history on the
`satoshi/wo4-*` branches (2026-08-15/16 era), NONE in this packet's
commits (3904a0cd..HEAD greps clean). The topology rule requires 8-hex
truncation + sha256 in public repos; remediating published branch
history (deletion/rewrite of origin/satoshi/wo4-*) is a
disposition-level action I do not take unilaterally.
