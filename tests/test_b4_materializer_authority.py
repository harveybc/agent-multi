"""WP4 negative tests (finding 326/327): B4 cell configs must embed
envelope, venue cost and observation authority at materialization."""
import importlib.util
import json
from pathlib import Path

import pytest

MOD = Path(__file__).resolve().parents[1] / "tools" / "materialize_b4_causal_sac.py"
spec = importlib.util.spec_from_file_location("b4mat", MOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

OBS = json.loads(m.V2_SYSTEM.read_text())["observation"]
COST = json.loads(m.COST_MANIFEST.read_text())
ORIGIN = {"year": 2024, "path": "/x/contract.json", "sha256": "a" * 64}
GEOM = {"envelope_mode": "atr", "atr_window": 14, "atr_sl_mult": 2.0,
        "atr_tp_mult": 3.0, "collision_rule": "stop_first_pessimistic",
        "sizing_mode": "portfolio_fraction", "leverage_cap": 1.0}


def test_full_cell_config_binds_everything():
    cfg = m.build_cell_config(ORIGIN, 101, GEOM, COST, OBS, "e" * 64)
    assert cfg["strategy_plugin"] == "shared_execution_envelope"
    assert cfg["execution_envelope"]["atr_sl_mult"] == GEOM["atr_sl_mult"]
    assert cfg["execution_envelope"]["entry_cost_headroom"] > 0.006
    assert cfg["commission"] == COST["alpaca_ethusd"]["env_binding"][
        "commission"]
    assert cfg["cost_contract_id"] == "alpaca_ethusd"
    assert len(cfg["cost_manifest_sha256"]) == 64
    assert cfg["cost_g1_eligible"] is True
    assert cfg["cost_maker_taker_assumption"] == "taker"
    assert cfg["execution_envelope_sha256"] == "e" * 64
    assert cfg["require_observation_declaration"] is True
    oc = cfg["observation_contract"]
    assert oc["feature_columns_sha256"] == OBS["feature_columns_sha256"]
    assert oc["expected_flattened_dimension"] == 2660
    assert cfg["nested_split_contract_sha256"] == "a" * 64


def test_omitted_envelope_refused():
    with pytest.raises(SystemExit, match="frozen execution"):
        m.build_cell_config(ORIGIN, 101, None, COST, OBS)


def test_omitted_cost_contract_refused():
    with pytest.raises(SystemExit, match="alpaca_ethusd"):
        m.build_cell_config(ORIGIN, 101, GEOM, {"zero_cost": {}}, OBS)


def test_mt5_or_zero_forced_contract_refused():
    forced = dict(COST, _force_contract="mt5_ethusd")
    with pytest.raises(SystemExit, match="not\s+G1-eligible|not \nG1"):
        m.build_cell_config(ORIGIN, 101, GEOM, forced, OBS)


def test_omitted_observation_refused():
    with pytest.raises(SystemExit, match="observation declaration"):
        m.build_cell_config(ORIGIN, 101, GEOM, COST, None)
