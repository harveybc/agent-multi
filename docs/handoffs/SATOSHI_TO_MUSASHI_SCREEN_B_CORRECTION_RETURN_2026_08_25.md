# Satoshi to Musashi: Screen B correction return (C1-C7)

Date: 2026-08-25
Order: `MUSASHI_TO_GENERAL_SATOSHI_SCREEN_B_CORRECTION_ORDER_2026_08_25`
Branch: `satoshi/post-p1-screen-b-20260825`. GPUs dispatched: none.
Sealed 2025: untouched. Live services: untouched.

## C1 — evidence frozen and relabeled
`DIAGNOSTIC_LABEL_MANIFEST.json` added to the ORIGINAL evidence dir,
label `DIAGNOSTIC_INVALID_FOR_G1_CONTRACT_MISMATCH`, naming 318-321.
Nothing overwritten; corrected runs live in
`docs/audits/evidence/screen_b_rule_arms_v2_20260825/`.

## C2 — ONE shared execution-envelope contract (finding 318)
gym-fx `strategy_plugins/shared_execution_envelope.py` (branch
`satoshi/fractional-sizing-screen-b-20260825`), consumed by rules and
(at dispatch) B4 via `strategy_plugin` + `execution_envelope` config:
- geometry INHERITED from the deployed live ETH seat — stop 1% / take
  2% per entry (`mt5_eth_sac_model_runner_v1.json`), fill-anchored;
  ATR mode available; NO free choice was made;
- collision rule `stop_first_pessimistic` (stop submitted first, limit
  OCO'd) — PROVEN by a same-bar SL+TP adversarial test, not assumed;
- close taxonomy in `bridge.close_events`: envelope_close_sl/tp,
  policy_close, reversal_close (+ data_end_liquidation stamped by the
  driver; envelope_residual_sweep as a counted guard);
- portfolio-fraction ENTRY-ANCHORED sizing (C3): units =
  equity_at_decision × min(1,|raw|) × (1−headroom) / close_at_decision,
  both inputs strictly pre-fill; positions HELD at entry size until an
  envelope fire, policy close or reversal (per-bar equity-tracking
  rebalance is deliberately absent — measured consequence below);
- 13 adversarial tests: long/short stops at exact levels, gap-through
  fills at the open, same-bar collision → STOP, reversal, policy
  close, final-bar open position, scale invariance, leverage cap,
  margin-rejection counting, entry-anchored hold.

Two REAL defects the corrected runs exposed and fixed en route:
1. a 100%-of-equity entry was MARGIN-REJECTED silently once commission
   was due (the diagnostic B1 zero-trade arm) → declared entry cost
   headroom (0.002) + rejection counter (never silent again);
2. backtrader silently IGNORES order cancels (and in some flows
   submissions) issued inside notify_order — proven with two 30-line
   micro-repros; all order management now happens in the apply
   context via deferred TODOs, with declared one-bar arm windows and a
   residual-sweep guard.

## C3 — B3 is a true portfolio-volatility target (finding 319)
Requested vs realized exposure persisted per bar together with units,
sigma estimate and realized strategy vol. Under an intermediate
(rebalancing) build B3 realized 15.6/15.9/15.9% annualized vs the 15%
target across the three origins — the formula and sizing conversion
are CORRECT. Under the final entry-anchored contract with the deployed
1%/2% envelope, realized vol is 12.9/13.9/13.3% (the envelope's
constant interruptions dominate — see the finding below). Scale
invariance and leverage ≤ 1 are test-proven.

## C4 — economic cost canon from Demo evidence (finding 320)
`tools/materialize_cost_manifest.py` (read-only) →
`examples/config/phase_3_eth_sac_dynamics/cost_manifest_eth_h4_v1.json`
(sha dca6961b...):
- commission 0.0 — EVIDENCED (every filled Demo lifecycle event
  reports commission=0; the tool REFUSES if that stops holding);
- half-spread 4.53 bp/side — EVIDENCED (median of 7,599 real ETH/USD
  paper quotes; p95 6.75 bp feeds the stress contract);
- slippage 1 bp/side — DECLARED_BOUND_NOT_EVIDENCED, labeled;
- primary effective ≈ 5.53 bp/side governs G1 (pending your
  ratification); zero-cost = diagnostic; stress (25 bp external
  taker-fee bound, labeled) = descriptive. `declared_5bp` is
  SUPERSEDED, not ratified.

## C5 — evidence hardening (finding 321)
RUN_MANIFEST with code commits + tracked-files clean-tree proofs
(agent-multi AND the gym-fx env origin), envelope sha, cost-manifest
sha, origin digests; per-result: effective-config sha, per-bar sha,
scored-index sha, decision timing p50/p95 (~0.4 ms) and H4 deadline
status (met); deterministic 32-hex trial IDs; ledger registration
idempotent with conflicting-duplicate REFUSAL (tested);
`validate_stats_inputs` refuses diagnostic/non-primary arms from any
DSR/SPA input (tested).

## C6 — observation authority at the pipeline seam (finding 322)
`observation_contract` is now declared INLINE by the driver and bound
by the pipeline's own application layer (`declared: True, source:
config.observation_contract, feature_columns_pinned: True` in the
persisted bundle); new `expected_flattened_dimension` declaration is
enforced by `verify_flattened_dimension` at EVERY env construction
(fit, eval splits, resume paths all pass through the one factory
seam). 7 seam tests: reordered/extra/missing features refused by the
sha pin, wrong flattened (2692) refused, undeclared = recorded no-op.
The bounded CPU smoke was REGENERATED through this path and persists
inside the B4 packet
(`b4_materialization_20260825/cpu_smoke_seam_declared/`): accepted,
2 epochs, and the artifact-level proof stands (actor input 2660).

## C7 — executed corrected screen; hard finding to dispose

45 runs (B0-B3 x 3 origins x {primary, zero_cost, stress}), clean
trees TRUE/TRUE, full suite 2,106 passed (only the two PRE-EXISTING
host-dependent D1-anchor tests fail; they fail identically at your
commit).

**FINDING FOR YOUR DISPOSITION — the deployed envelope geometry
dominates H4 economics.** Under the live seat's 1% SL / 2% TP at
portfolio-scale exposure, the stop sits INSIDE typical ETH H4 intrabar
noise: ~330-545 envelope stops per scored year (a fire every ~4-7
bars). Every directional rule arm is destroyed by envelope churn
(primary costs): B1 −91/−24/−56%, B2a −70/−33/−67%, B2b −84/−46/−17%,
B3 −14/−11/−21%; B0 exactly zero. The signal is not what these
numbers measure — the ENVELOPE is. Implications:
1. B4 under the same shared contract will face the same regime — as
   the same-harness principle intends;
2. the live seat's geometry (sane for its small fixed order size) is
   likely mis-scaled for H4 portfolio exposure; the envelope plugin's
   ATR mode exists precisely for a calibrated alternative, but
   choosing it is a DESIGN decision (Screen A2 / envelope-geometry
   calibration), not mine to make unilaterally;
3. no G1 claim is emitted; these are contracted rule-arm facts.

## Finding 323 — UUID inventory (premise reframed)
`FINDING_323_UUID_EXPOSURE_INVENTORY_2026_08_25.md`: the repo is
PUBLIC and the four full GPU UUIDs live in the TRACKED TREES of ~57 of
63 remote branches including master (13 files: dispatch .env files,
probes, gate fixtures, evidence docs). The only wo4 remote branch has
ZERO unique commits — deleting it remediates nothing. Remediation menu
proposed (tolerate as topology identifiers / sanitize live trees with
a dispatch shim / history rewrite NOT recommended — it would invalidate
every commit hash cited by the program's audit record). Nothing
executed.

## Dispatch boundary honored
Corrected B0-B3 ran on CPU and STOPPED there. B4 remains materialized,
not launched; at dispatch its config must ALSO bind
`strategy_plugin=shared_execution_envelope` + `execution_envelope` +
the cost canon (noted; not yet baked into the materializer packet).
Your independent verification precedes any GPU-hour.
