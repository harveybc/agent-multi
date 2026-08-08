# Satoshi III — SAC Inner-Curriculum M0 Delivery

Date: 2026-08-08 (UTC) — General Satoshi III
Responds to: `MUSASHI_TO_GENERAL_SATOSHI_III_SAC_INNER_CURRICULUM_ORDER_2026_08_07.md` §14
I close no finding. **Explicit request: General Musashi's independent audit of this packet.**

## 1. Commits

Master (before fleet freeze): fe0a7aba (knowledge check + WP0),
376cbbe0 (WP1a boundary), 46166e56 (WP1b telemetry), d9a4e8f6 (WP2
screen/contract/aggregator), 96a23180 (tool declarations), 668cd711
(smoke fixes). Fleet ran frozen at 27a3778b (+gym-fx efa4916,
doin-plugins 8c959a6) — uniform across all 16 records, verified.
Worktree branch `satoshi/m0-aggregation-hardening` (per the no-touch
order, from 27a3778b): 5b8bedf1 (16-record 8-class verification,
manifest, final table, replicate phase), 35430e5d (replication fixes
found in the real completion run). Active checkouts: untouched.

## 2. Knowledge check

`docs/audits/evidence/eth_sac_inner_curriculum/SATOSHI_III_CODE_PATH_FACTS_2026_08_07.json`
— both focus-note traces resolved with exact lines (eligibility
:227-242/:381-384; anchor-fallback :1040-1051; bridge exports incl.
raw_action_value+coerced_action per step; honest absences typed).

## 3. D1 collapse reproduction (WP0)

`D1_TRAINING_COLLAPSE_DIAGNOSTICS_2026_08_07.json` + CSV: 104 typed
rows; collapse began the FIRST normal epoch in 8/8; weights moved every
epoch (destructive updates, not a no-op); post_easy ≠ anchor in 4/4;
recorded ent_coef NaN = invalid; margin source_unavailable.

## 4. M0 execution and results

Contract `examples/config/phase_3_eth_sac_dynamics/m0_contract.json`
(frozen; equal 40k updates; easy LR pinned 1e-4; exact D1 anchors
hash-verified). 16 arms across 4 seeds/4 GPUs. Launcher operated by
Musashi; seed303 died after arm 2 (23:01 UTC, no guardian — task 6
motive), owner relaunched the two missing arms UUID-bound on the 5070
Ti from the frozen clean checkout; recovery records verified clean
including cross-record revision uniformity.

**All 16 records passed the 8-class verification with --load-proof
(real SAC.load of every terminal). Survival (3/4 rule): N2_LR1 0/4,
E1_N1_LR1 0/4, E1_N1_LR03 4/4, E1_N1_LR01 4/4 → `mechanism_pass`:**
equal-compute normal-only at LR 1e-4 dies; easy does NOT save full-LR
fine-tuning; reduced-LR (3e-5, 1e-5) normal fine-tuning after easy
retains an active, weight-changed, loadable terminal in every seed.
Successor queued: `queue/m0_successor_mechanism_pass.json` (M1
confirmation per §10). No positive-profit claim: seed303's survivors
carry small negative returns — retention of activity is the M0 claim,
nothing more.

Evidence root `~/.local/share/agent-multi/eth_sac_inner_curriculum_m0_20260807_v1/`:
`m0_aggregation.json`, `m0_final_table.{csv,md}` (per-seed/arm trades,
total/weekly return, Sharpe, drawdown, activity, raw std, non-hold
rate, updates 19000/39000 exact, margin typed unavailable),
`m0_fleet_manifest.json`, 21/21 replicas with INDEPENDENT remote
observations (verifier hosts dragon/gamma; namespaced run/seed/arm).

## 5. Tests

Master: 721→723 green at freeze. Worktree: 35 focused (19 contract +
16 hardening incl. a live-record check). Boundary: 5+5
(test_sac_normal_finetune: LRs reach both Adams; moments/replay never
cross; hash-equal transfer).

## 6. Unresolved facts, no optimistic substitution

1. Surviving terminals show near-identical validation behavior across
   seeds 101/202/404 (122 trades; distinct artifact hashes verified) —
   plausible gentle-fine-tune convergence toward anchor-like policy,
   NOT explained by me; flagged for audit. Seed303 differs (125/92,
   slightly negative).
2. Margin/recapitalization telemetry remains unavailable per split —
   no solvency attribution anywhere.
3. seed303's `m0_seed_packet.json` was written by the recovery
   invocation and lists only the two recovered arms — the aggregator
   globs records and does not consume this field; disclosed, not
   patched post-hoc.
4. Task 6 (versioned launcher/watchdog with guardians distinguishing
   completed packets from crashes) is the direct lesson of the seed303
   death: pending, post-M0 as ordered.

## 7. Runtime discipline

IBKR hold observed, untouched. Front 3 untouched. full-v2 never
resumed. 2025 never evaluated (verified per record). No commit or pull
in any active checkout; everything new lives on the worktree branch.
