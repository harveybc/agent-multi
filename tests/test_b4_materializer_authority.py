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
    cfg = m.build_cell_config(ORIGIN, 101, GEOM, COST, OBS, "e" * 64,
                              "f" * 64)
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


def test_d1_session_exposure_explicitly_off_and_tamper_refuses():
    cfg = m.build_cell_config(ORIGIN, 101, GEOM, COST, OBS,
                              "e" * 64, "f" * 64)
    assert cfg["session_exposure_enabled"] is False
    bad = dict(cfg)
    bad["session_exposure_enabled"] = True
    try:
        m.validate_cell_config(bad)
        raise AssertionError("weekly-session overlay accepted")
    except SystemExit as exc:
        assert "session_exposure_enabled" in str(exc)


def test_d1_mixed_gymfx_lineage_refuses():
    cfg = m.build_cell_config(ORIGIN, 101, GEOM, COST, OBS,
                              "e" * 64, "f" * 64)
    try:
        m.check_lineage_match(cfg,
                              {"gymfx_lineage_manifest_sha256":
                               "0" * 64})
        raise AssertionError("mixed lineage accepted")
    except SystemExit as exc:
        assert "mixed GymFxEnv lineage" in str(exc)
    # equal lineages pass
    m.check_lineage_match(cfg,
                          {"gymfx_lineage_manifest_sha256":
                           "f" * 64})


def test_d1_missing_lineage_identity_refuses():
    try:
        m.build_cell_config(ORIGIN, 101, GEOM, COST, OBS,
                            "e" * 64, "")
        raise AssertionError("cell without lineage accepted")
    except SystemExit as exc:
        assert "lineage" in str(exc)


def test_d1_wrong_gymfx_checkout_refuses(monkeypatch):
    monkeypatch.setattr(m, "GYMFX_PINNED_COMMIT", "0" * 40)
    try:
        m.gymfx_lineage_manifest()
        raise AssertionError("foreign gym-fx lineage accepted")
    except SystemExit as exc:
        assert "not the accepted lineage" in str(exc)


# --- Order @0b4d2748 B4-D2: sealed-design binding refusals ---
SB = (Path(__file__).resolve().parents[1] / "tools" /
      "screen_b_baselines.py")
_sb_spec = importlib.util.spec_from_file_location("sbb", SB)
sb = importlib.util.module_from_spec(_sb_spec)
_sb_spec.loader.exec_module(sb)


def _design(tmp_path, monkeypatch, **overrides):
    """A minimal design whose pins default to CORRECT for this
    checkout, with monkeypatched lineage; overrides poison pins."""
    import hashlib as _h
    d = {
        "sealed_code_identity": {
            "screen_b_baselines_py_sha256":
                _h.sha256(SB.read_bytes()).hexdigest()},
        "source_data_sha256": sb.DATA_SHA,
        "cost_manifest_sha256": sb._sha_file(sb.COST_MANIFEST),
        "calibration_rule": {
            "grid_sha256": sb._sha_obj(sb.CALIBRATION_GRID)},
        "execution_truth_binding": {
            "gymfx_point_of_use_manifest_sha256": "b" * 64},
    }
    for k, v in overrides.items():
        d[k] = v
    f = tmp_path / "design.json"
    f.write_text(json.dumps(d))
    monkeypatch.setattr(sb, "DESIGN_PATH", f)
    import materialize_b4_causal_sac as b4m
    monkeypatch.setattr(
        b4m, "gymfx_lineage_manifest",
        lambda: {"commit": "c" * 40, "manifest_sha256": "b" * 64})
    return d


def test_d2_unsealed_design_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "DESIGN_PATH", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="not sealed"):
        sb.bind_superseding_design()


def test_d2_drifted_code_refuses(tmp_path, monkeypatch):
    _design(tmp_path, monkeypatch,
            sealed_code_identity={
                "screen_b_baselines_py_sha256": "0" * 64})
    with pytest.raises(SystemExit, match="drifted"):
        sb.bind_superseding_design()


def test_d2_foreign_dataset_pin_refuses(tmp_path, monkeypatch):
    _design(tmp_path, monkeypatch, source_data_sha256="0" * 64)
    with pytest.raises(SystemExit, match="different source dataset"):
        sb.bind_superseding_design()


def test_d2_foreign_calibration_grid_refuses(tmp_path, monkeypatch):
    _design(tmp_path, monkeypatch,
            calibration_rule={"grid_sha256": "0" * 64})
    with pytest.raises(SystemExit,
                       match="different calibration grid"):
        sb.bind_superseding_design()


def test_d2_mixed_execution_truth_refuses(tmp_path, monkeypatch):
    _design(tmp_path, monkeypatch,
            execution_truth_binding={
                "gymfx_point_of_use_manifest_sha256": "a" * 64})
    with pytest.raises(SystemExit, match="mixed execution truth"):
        sb.bind_superseding_design()


def test_d2_correct_pins_bind(tmp_path, monkeypatch):
    _design(tmp_path, monkeypatch)
    out = sb.bind_superseding_design()
    assert out["lineage"]["manifest_sha256"] == "b" * 64
    assert len(out["design_sha256"]) == 64
