# Satoshi III — M0 Code-Path Map, Implementation Boundary, Fixtures, Unknowns

Date: 2026-08-07 — General Satoshi III
Responds to: `MUSASHI_TO_GENERAL_SATOSHI_III_M0_FOCUS_AND_IBKR_RECOVERY_NOTE_2026_08_07.md` §4
IBKR: hold observed and untouched; no order, no clear, no ownership taken.
Front 3: untouched. No curriculum result is claimed — no paired measurement exists yet.

## 1. Code-path map (complete, exact lines)

Full map with source refs:
`docs/audits/evidence/eth_sac_inner_curriculum/SATOSHI_III_CODE_PATH_FACTS_2026_08_07.json`
— both previously-unresolved facts are now RESOLVED:

**Fact 1 — checkpoint eligibility / promotion path**
(`pipeline_plugins/rl_pipeline_with_validation.py`):
- `_checkpoint_is_eligible` **:227-242** — `num_timesteps >=
  l1_min_checkpoint_timesteps AND trade_gate_passed`.
- `trade_gate_passed` **:381-384** — `train_tail_trades >=
  train_tail_min AND val_trades >= validation_min`; failing composite
  is penalized (:385).
- **The anchor-fallback mechanism :1040-1051** — before any epoch, the
  warm-start model itself is evaluated
  (`checkpoint_source="warm_start_normal_baseline"`); if its gate
  passes, the ANCHOR is saved as `best_model_path`
  (`best_checkpoint_saved=True`). Every later ineligible epoch leaves
  that selection standing — this is precisely how D1's selected N14 and
  EN4_10 reported the anchor's metrics while all their trained
  terminals were inactive.
- Per-epoch application **:1187-1218** (`_update_l1_checkpoint_state`
  :213-224 refuses ineligible epochs; activity-ineligible streaks
  consume their own budget, AUD-127).
- Nothing eligible at all → RuntimeError **:1369-1381** (no silent
  publication).
- Terminal weights preserved separately **:1383-1390**
  (`.terminal.zip`, AUD-129) — the artifact M0's `terminal_usable`
  facts will be computed from.

**Fact 2 — bridge diagnostics actually exported**
(`gym-fx/app/env.py`, `gym-fx/app/bt_bridge.py`):
- Per step (`_make_info`, env.py:860-877): equity, position,
  position_units, open_order_count, price, bar_index, trades,
  commission_paid, **raw_action_value**, **coerced_action**,
  action_diagnostics, execution_diagnostics.
- Reward-step extension (env.py:395-407): trade_cost, solvency_mode,
  economic_equity, recapitalization_debt/count,
  **would_margin_call_count**, **termination_cause**.
- execution_diagnostics families (bt_bridge.py:87-100+):
  entry_actions_seen, entry_orders_submitted, blocked_* (session/ATR/
  size/price), protected_entry_rejections,
  protected_market/limit/stop_entries, plugin_apply_errors.
- Raw-vs-threshold suppression is decidable from EXISTING exports
  (raw_action_value + coerced_action are both per-step).
- **Honest absences (not invented into the map):**
  per-event margin details (only the count is exported; event
  equity/debt stays internal, bt_bridge.py:240-268) →
  `source_unavailable` unless WP1 exports them; SAC actor/critic/
  entropy losses and optimizer-observed LR → `not_instrumented` in D1;
  raw-action stats on fixed validation observations →
  `not_instrumented` (WP1 adds).

## 2. Proposed minimal implementation boundary

**Modified (WP1, agent-multi only):**
1. `agent_plugins/sac_agent.py::load_for_training` — extend the
   already-returned transfer record with: source/target artifact
   sha256, policy tensor hash before/after transfer, actor/critic/
   target-critic distances, target optimizer LRs,
   `replay_transitions_transferred=0`. Pure additions to an existing
   dict; no behavior change.
2. `pipeline_plugins/rl_pipeline_with_validation.py` — epoch-loop
   telemetry only: replay size (before collection / after
   learning_starts / at end), gradient update count, actor/critic/
   entropy losses from SB3 logger, raw/thresholded action stats on a
   fixed validation observation batch, per-epoch actor/critic param
   sums. No selection-logic change.
3. `pipeline_plugins/rl_pipeline_with_solvency_curriculum.py` —
   boundary record emission (post_easy hash + transfer record + probe
   facts) into the run summary. No schedule change.

**New (WP0/WP2):** `tools/eth_sac_training_diagnostics.py` (read-only
D1 collector), `tools/eth_sac_inner_curriculum_screen.py`,
`tools/eth_sac_inner_curriculum_fleet.py`,
`tools/aggregate_eth_sac_inner_curriculum.py`,
`examples/config/phase_3_eth_sac_dynamics/m0_contract.json`.

**Not touched:** gym-fx (unless margin-event export proves impossible
from existing info — then its own commit+tests), doin-* repos, D1
records, live/LTS runners, Front 3.

## 3. Test fixtures (planned, per §12 of the M0 order)

- `tests/test_eth_sac_inner_curriculum_contract.py` — contract fixtures:
  four frozen arm specs (equal 40,000 updates asserted arithmetically);
  invalid LR/multiplier/schedule/negative-epoch fixtures that must fail
  BEFORE model construction; 2025-leak fixture (any test-split date in a
  trace/selection → refusal); duplicate id/replica-path fixtures;
  GPU-UUID mismatch fixture; D1-record immutability check (hash
  re-verification of decision_summary.json + 4 seed packets).
- `tests/unit/test_sac_normal_finetune.py` — boundary fixtures on a
  tiny in-memory SAC: policy tensors equal immediately after transfer;
  optimizer moments differ (fresh); replay empty at boundary and fills
  from normal only; normal LR observed inside BOTH actor and critic
  optimizers (the LR03/LR01 arms are meaningless if the multiplier
  never reaches Adam); fixed-entropy path asserted.
- `tests/unit/test_solvency_curriculum_pipeline.py` — extended, not
  rewritten: easy-is-train-only and forced-normal evaluation already
  covered there; add boundary-record presence.
- Raw-vs-threshold distinguishability fixture: synthetic trace where
  |raw| dispersion > threshold with zero entries → classified
  suppression, and |raw| ≈ 0 → classified collapse; anchor-traded/
  terminal-inactive fixture must yield `terminal_usable=false`.

## 4. Explicit unknowns (no optimistic substitution)

1. Whether D1 per-epoch action traces were persisted anywhere on disk —
   if not, WP0 reports raw-action questions as `not_instrumented` for
   D1 and M0 answers them prospectively; WP0's collector will establish
   this from the seed packets, not assume it.
2. Whether SB3's logger values (losses, updates) are recoverable from
   D1 logs post-hoc — likely `source_unavailable`; will be typed as
   found.
3. Margin-event detail export path (per-event equity/debt) — decided in
   WP1 after checking whether existing `would_margin_call_count` +
   recapitalization fields satisfy the M0 decision facts (they likely
   do for "no margin events → no solvency attribution").
4. Gamma dual-GPU contention under two simultaneous M0 workers —
   handled by the existing UUID binding (finding 163) but throughput
   asymmetry is unmeasured; ETA fields will reflect it, not hide it.

## 5. IBKR / Front 3 observance

The recovery hold (`halted:hold`, TWS Paper 7497, read_only=false,
model action `short`, decision refused) is observed as FACT and left
authoritative. I did not touch the runner, the hold, or any order path.
Front 3's CPU worker was not inspected beyond confirming I ran nothing
against it.

Next: WP0 collector (CPU, read-only, running next), then WP1 edits
behind the fixtures above, then WP2/WP3 to the four-GPU M0 launch.
