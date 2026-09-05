"""PRE freeze for order @61622469 (B4-E1..E6): unequal economic
envelopes, a non-B4 preflight command, a design the final code cannot
satisfy, incomplete cells, summary-trusting comparator verification,
and contradictory authority language."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sb = _load("sbb", "tools/screen_b_baselines.py")
b4m = _load("b4mat", "tools/materialize_b4_causal_sac.py")

print("== E1: unequal economic envelopes from the public paths ==")
cost_sets, _sha = sb.load_cost_sets()
alp = cost_sets["alpaca_ethusd"]["binding"]
df = None  # base_config does not need data
origin_stub = {"csv": "unused.csv", "year": 2024,
               "scored_start_index": 0}
geom = {"envelope_mode": "atr", "atr_window": 14, "atr_sl_mult": 3.0,
        "atr_tp_mult": 6.0, "collision_rule": "stop_first_pessimistic",
        "sizing_mode": "portfolio_fraction", "leverage_cap": 1.0}
comparator_cfg = sb.base_config(origin_stub, alp, dict(geom))
cells = json.loads((REPO / "docs/audits/evidence/"
                    "b4_materialization_20260905/"
                    "B4_CELL_CONFIGS.json").read_text())
cell_cfg = cells["o2024_seed101"]["effective_config"]
h_comp = comparator_cfg["execution_envelope"]["entry_cost_headroom"]
h_cell = cell_cfg["execution_envelope"]["entry_cost_headroom"]
print(f"comparator headroom: {h_comp}")
print(f"B4 cell headroom:    {h_cell}")
assert h_comp == 0.012102 and h_cell == 0.007102
assert h_comp != h_cell, "expected the PRE inequality"
print("UNEQUAL — shared_execution_envelope scales exposure by "
      "(1 - headroom): the comparison is economically biased")

print("\n== E2: the proposed GPU command is not a B4 command ==")
smoke = (REPO / "tools/wp4_cpu_smoke.py").read_text()
print("old gym-fx pin present:",
      "634c3fd3c344cae3c4048b334158185c8bf4e1ef" in smoke)
for token in ("B4_CELL_CONFIGS", "SUPERSEDING_DESIGN",
              "GENESIS_BINDING", "SCREEN_B_RESULTS"):
    print(f"references {token}:", token in smoke)
assert "634c3fd3c344cae3c4048b334158185c8bf4e1ef" in smoke
assert "B4_CELL_CONFIGS" not in smoke

print("\n== E3: the final code cannot satisfy its own sealed design ==")
own = hashlib.sha256(
    (REPO / "tools/screen_b_baselines.py").read_bytes()).hexdigest()
mat = hashlib.sha256(
    (REPO / "tools/materialize_b4_causal_sac.py").read_bytes()
).hexdigest()
print("live screen_b sha:", own[:12], "| live materializer:", mat[:12])
try:
    sb.bind_superseding_design()
    print("ACCEPTED (unexpected)")
    refused = False
except SystemExit as exc:
    print("bind refuses:", str(exc)[:80])
    refused = True
assert refused, "expected the drift refusal"

print("\n== E4: a cell is not a complete runnable recipe ==")
missing = [k for k in (
    "agent_plugin", "pipeline_plugin", "preprocessor_plugin",
    "learning_rate", "net_arch", "ent_coef", "batch_size",
    "buffer_size", "learning_starts", "train_freq", "gradient_steps",
    "epoch_timesteps", "max_epochs", "l1_patience",
    "selection_metric", "budget_max_env_steps", "budget_max_updates",
    "budget_max_wall_seconds", "input_data_file")
    if k not in cell_cfg]
print("training-semantic keys ABSENT from the hashed cell:", missing)
assert len(missing) >= 15
runner = (REPO / "tools/b4_mechanics_cell.py").read_text()
print("mechanics runner fills them elsewhere:",
      "base_config" in runner and "build_cfg.update" in runner)

print("\n== E5: comparator verification trusts a summary ==")
live_lineage = json.loads((REPO / "docs/audits/evidence/"
                           "b4_materialization_20260905/"
                           "GYMFX_LINEAGE_MANIFEST.json").read_text())
forged = {"population_label":
          "SCREEN_B_CURRENT_EXECUTION_TRUTH_OPTION_B",
          "gymfx_lineage_manifest_sha256":
          live_lineage["manifest_sha256"],
          "results": []}  # NO results, NO ledger, NO digests
ok_label = (forged.get("population_label")
            == "SCREEN_B_CURRENT_EXECUTION_TRUTH_OPTION_B")
try:
    b4m.check_lineage_match(cell_cfg, forged)
    ok_lineage = True
except SystemExit:
    ok_lineage = False
print("forged empty summary passes the label check:", ok_label)
print("forged empty summary passes the lineage check:", ok_lineage)
assert ok_label and ok_lineage, "expected summary-trust to reproduce"
print("ACCEPTED — the materializer's entire comparator verification "
      "grants cells against zero verified results")

print("\n== E6: authority language contradicts the decision ==")
v5 = json.loads((REPO / "docs/audits/evidence/"
                 "screen_b_rule_arms_v5_current_truth_20260905/"
                 "SCREEN_B_RESULTS.json").read_text())
auth = v5["results"][0].get("cost_authority", "")
print("v5 result cost_authority:", repr(auth))
src = (REPO / "tools/screen_b_baselines.py").read_text()
print("'pending ratification' in executing code:",
      "pending ratification" in src)
sysm = json.loads((REPO / "examples/config/phase_3_eth_sac_dynamics/"
                   "systems/ethusdt_4h_l1_system_v2.json").read_text())
print("owner act ratified observation v2 + build 6140; Alpaca costs "
      "were selected by Musashi review, not ratified by the owner")
assert "pending ratification" in auth or \
    "pending ratification" in src

print("\nPRE CONFIRMED: all six findings reproduce")
