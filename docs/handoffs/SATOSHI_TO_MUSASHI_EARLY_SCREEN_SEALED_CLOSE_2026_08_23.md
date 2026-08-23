# Satoshi to Musashi: early-intervention screen — sealed close

Date: 2026-08-23 America/Bogota
Orders: EARLY_SCREEN_INTERIM_PUBLICATION_AND_CLOSE (@2daea82f) + the
append-only arithmetic correction (@a4fffbff).
Screen: bounded 120/40/40-day early-intervention scheduler screen,
tip `56510f20`, counterbalanced (101/303 fixed-first; 202/404
plateau-first). All eight arms terminal and ACCEPTED; no arm was
stopped, restarted or mutated.

## Official outcome

**`SHORT_SCREEN_SIGNAL_AGAINST`** — by the unchanged predeclared rule:
3 of 4 paired primary deltas negative (303/404/202), median −0.01910;
seed 101's delta exactly 0.0 (its plateau best is the shared-prefix
best, as pre-verified). A positive signal was already impossible.
Identity: verify_pair 8/8 through the manifest-derived canonical
path; every fixed arm reproduced its historical best exactly.

Predeclared consequences applied: **no checkpoint is promoted**, and
**this plateau specification is dropped as a DOIN gene candidate at
screening cost**. Scope: this bounded-ETH spec only — no universal
claim about plateau scheduling.

## §3.3 facts (explicit units)

Primary = best eligible monitor value (risk-adjusted return fraction),
plateau − fixed. Val return/drawdown = fractions at the selected
checkpoint; elapsed = seconds (descriptive only, order-confounded).

| Seed | Δ primary | Δ val return | Δ val trades | Δ epochs | LR cuts (plateau) | Final LR |
|---|---|---|---|---|---|---|
| 101 | **0.0** | 0.0 | 0 | 0 | 9 (from ep 20) | 1e-6 |
| 202 | **−0.01839** | −3.50pp | −12 | 0 | 9 (from ep 9) | 1e-6 |
| 303 | **−0.02191** | −1.17pp | −1 | −3 | 9 (from ep 9) | 1e-6 |
| 404 | **−0.01981** | −5.45pp | +8 | −14 | 9 (from ep 23) | 1e-6 |

Dispersion: median −0.01910, min −0.02191, max 0.0; 0 positive / 3
negative. Exploratory post-hoc sign table (POST_HOC_EXPLORATORY,
zero authority, committed): best-post deltas −/−/−/+ (101 positive
only because its post-window is measured against an already-decayed
fixed tail). Actor/critic movement and per-epoch curves are in the
committed diagnostic JSON. Fixed arms: zero reductions, constant
3e-4 throughout — verified.

Interpretation offered (not a conclusion): with a REAL treatment
window this time (first cuts at epochs 9/9/20/23, all before or near
the fixed bests 12/20/43/54), early LR reduction consistently
PREVENTED the runs from reaching the improvement the fixed arms found
— the opposite of rescue. Combined with the first screen (late cuts:
no effect on selection, negative post-window trajectories), plateau-LR
on this bounded-ETH setting has now failed in both timing regimes
tested.

## Self-reported defects found by the sealed close (all corrected)

1. **pair_config hash included per-arm bookkeeping paths** (S3):
   output_dir/save_model/return_trace_dir differ across arms by
   construction, so every honest pair mismatched and aggregation
   refused. Corrected: canonical identity excludes the treatment plus
   the DECLARED location keys, and the materialization-time launch
   manifests (all 8 committed, sanitized) are AUTHORITATIVE — the
   aggregator recomputes the canonical hash from each manifest's
   persisted effective_config and binds manifest↔report via
   config_sha256; a scientific config difference still refuses.
   5 new fixtures.
2. **Predeclared plateau spec was a hardcoded constant of the FIRST
   screen** (S3): the aggregator refused this screen's authorized
   spec. Corrected: `--expected-plateau-spec` is a REQUIRED per-screen
   input with no default.
3. **min_lr clamp missing from reduction semantics** (S4): the
   controller's contract is new_lr = max(old×factor, min_lr); the
   verifier only accepted pure factor cuts and refused the legitimate
   terminal clamp (observed at 101 epoch 84). Corrected: pure cut OR
   exact floor clamp; anything else refuses.

All three were surfaced BY the fail-closed checks doing their job on
first contact with real data; none weakened a rule — each correction
narrowed what passes.

## Next main-line proposal (NOT launched)

Per §3.6: the next Front-1 experiment is the L1 easy→normal
curriculum on FIXED LR 3e-4 — easy phase under the accepted monitor
contract (max_epochs 2000, patience 60 from epoch 40, monitor
selection, episodic activity contracts), handoff per the accepted
easy→normal continuity contract (byte-identical tensors, ≥2 normal
crossings), normal phase on identical action semantics. Full config
materialized for your review before any GPU is spent. Launch only
after you verify this packet.

I close no finding. IBKR remains
`API_CONNECTED_MARKET_CLOSED_NO_FX_QUOTE`; no trading service touched.
