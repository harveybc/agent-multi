"""B4 authority battery.

Orders @0b4d2748 (D1/D2 regressions kept) and @61622469 (E1-E7):
one economic envelope, complete immutable cells, an executable
append-only amendment chain, evidence-complete comparator
verification, the single runner's closed CLI, corrected authority
language. Mutation tests strike the ACTUAL verifier module
(b4_authority), the materializer and the runner."""
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import b4_authority as a  # noqa: E402

MOD = REPO / "tools" / "materialize_b4_causal_sac.py"
spec = importlib.util.spec_from_file_location("b4mat", MOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SB = REPO / "tools" / "screen_b_baselines.py"
_sb_spec = importlib.util.spec_from_file_location("sbb", SB)
sb = importlib.util.module_from_spec(_sb_spec)
_sb_spec.loader.exec_module(sb)

RUN = REPO / "tools" / "b4_run_cell.py"
_run_spec = importlib.util.spec_from_file_location("b4run", RUN)
runner = importlib.util.module_from_spec(_run_spec)
_run_spec.loader.exec_module(runner)

OBS = json.loads(m.V2_SYSTEM.read_text())["observation"]
COST = json.loads(m.COST_MANIFEST.read_text())
ORIGIN = {"year": 2024, "path": "/x/contract.json", "sha256": "a" * 64}
GEOM = {"envelope_mode": "atr", "atr_window": 14, "atr_sl_mult": 2.0,
        "atr_tp_mult": 3.0, "collision_rule": "stop_first_pessimistic",
        "sizing_mode": "portfolio_fraction", "leverage_cap": 1.0}
RECIPE = {
    "env_plugin": "gym_fx_env", "agent_plugin": "sac_agent",
    "pipeline_plugin": "rl_pipeline_with_validation",
    "preprocessor_plugin": "feature_window_preprocessor",
    "learning_rate": 3e-4, "batch_size": 256, "learning_starts": 100,
    "train_days": 1, "epoch_timesteps": 20000, "max_epochs": 2000,
    "l1_patience": 60, "l1_patience_start_epoch": 40,
    "l1_min_delta": 0.0, "selection_metric":
        "paired_generalization_weekly_v1",
    "action_space_mode": "continuous",
    "continuous_action_threshold": 0.0,
    "continuous_action_contract": "target_exposure_hysteresis_v2",
    "initial_cash": 10000.0, "solvency_mode": "strict",
    "feature_scaling": None, "feature_scaling_window": None,
    "net_arch": [256, 256], "ent_coef": "auto",
    "buffer_size": 100000, "train_freq": 1, "gradient_steps": 1,
    "gamma": 0.99, "tau": 0.005, "use_sde": False,
    "genesis_construction_buffer_size": 100,
}


def _cell(**over):
    cfg = m.build_cell_config(ORIGIN, 101, GEOM, COST, OBS, "e" * 64,
                              "f" * 64, recipe=dict(RECIPE))
    cfg.update(over)
    return cfg


# ------------------------- historical D1/WP4 ----------------------
def test_full_cell_config_binds_everything():
    cfg = _cell()
    assert cfg["session_exposure_enabled"] is False
    assert cfg["cost_contract_id"] == "alpaca_ethusd"
    assert cfg["gymfx_lineage_manifest_sha256"] == "f" * 64
    assert cfg["require_observation_declaration"] is True
    assert cfg["execution_envelope"]["entry_cost_headroom"] == \
        a.entry_cost_headroom(COST["alpaca_ethusd"]["env_binding"])


def test_omitted_envelope_refused():
    with pytest.raises(SystemExit):
        m.build_cell_config(ORIGIN, 101, None, COST, OBS,
                            recipe=dict(RECIPE))


def test_omitted_cost_contract_refused():
    with pytest.raises(SystemExit):
        m.build_cell_config(ORIGIN, 101, GEOM, {"zero_cost": {}}, OBS,
                            recipe=dict(RECIPE))


def test_mt5_or_zero_forced_contract_refused():
    forced = dict(COST)
    forced["_force_contract"] = "mt5_ethusd"
    with pytest.raises(SystemExit):
        m.build_cell_config(ORIGIN, 101, GEOM, forced, OBS,
                            recipe=dict(RECIPE))


def test_omitted_observation_refused():
    with pytest.raises(SystemExit):
        m.build_cell_config(ORIGIN, 101, GEOM, COST, None,
                            recipe=dict(RECIPE))


def test_d1_session_exposure_explicitly_off_and_tamper_refuses():
    cfg = _cell()
    assert cfg["session_exposure_enabled"] is False
    bad = dict(cfg)
    bad["session_exposure_enabled"] = True
    with pytest.raises(SystemExit,
                       match="session_exposure_enabled"):
        m.validate_cell_config(bad)


def test_d1_mixed_gymfx_lineage_refuses():
    cfg = _cell()
    with pytest.raises(SystemExit, match="mixed GymFxEnv lineage"):
        m.check_lineage_match(
            cfg, {"gymfx_lineage_manifest_sha256": "0" * 64})
    m.check_lineage_match(
        cfg, {"gymfx_lineage_manifest_sha256": "f" * 64})


def test_d1_missing_lineage_identity_refuses():
    with pytest.raises(SystemExit, match="lineage"):
        m.build_cell_config(ORIGIN, 101, GEOM, COST, OBS, "e" * 64,
                            "", recipe=dict(RECIPE))


def test_d1_wrong_gymfx_checkout_refuses(monkeypatch):
    monkeypatch.setattr(a, "GYMFX_PINNED_COMMIT", "0" * 40)
    with pytest.raises(SystemExit, match="not the accepted lineage"):
        a.gymfx_lineage_manifest()


# ------------------------- E1: one envelope -----------------------
ALP = COST["alpaca_ethusd"]["env_binding"]


def test_e1_one_headroom_rule_is_012102():
    assert a.entry_cost_headroom(ALP) == 0.012102


def test_e1_comparator_and_b4_envelopes_are_equal():
    """The E1 parity regression: both sides build the SAME complete
    envelope and the SAME canonical digest from the same geometry
    and cost binding."""
    comp_env = a.complete_execution_envelope(GEOM, ALP)
    cfg = _cell()
    assert cfg["execution_envelope"] == comp_env
    assert cfg["complete_envelope_digest"] == \
        a.complete_envelope_digest(comp_env, ALP)


def test_e1_old_headroom_0007102_refuses():
    env = a.complete_execution_envelope(GEOM, ALP)
    env["entry_cost_headroom"] = 0.007102
    with pytest.raises(SystemExit, match="differs from the one"):
        a.verify_envelope(env, ALP)


def test_e1_missing_headroom_refuses():
    env = a.complete_execution_envelope(GEOM, ALP)
    del env["entry_cost_headroom"]
    with pytest.raises(SystemExit, match="omits economic field"):
        a.verify_envelope(env, ALP)


def test_e1_wrong_primitive_type_refuses():
    env = a.complete_execution_envelope(GEOM, ALP)
    env["entry_cost_headroom"] = True
    with pytest.raises(SystemExit, match="finite float"):
        a.verify_envelope(env, ALP)
    env["entry_cost_headroom"] = 1
    with pytest.raises(SystemExit, match="finite float"):
        a.verify_envelope(env, ALP)


def test_e1_cell_digest_must_rederive():
    cfg = _cell()
    bad = dict(cfg)
    bad["complete_envelope_digest"] = "0" * 64
    with pytest.raises(SystemExit, match="re-derive"):
        a.verify_cell_complete(bad)


# ------------------------- E4: complete cell ----------------------
def test_e4_every_required_key_missing_refuses():
    cfg = _cell()
    for key in a.REQUIRED_CELL_KEYS:
        bad = dict(cfg)
        del bad[key]
        with pytest.raises(SystemExit):
            a.verify_cell_complete(bad)


def test_e4_hidden_pretrained_replay_resume_refuse():
    cfg = _cell()
    for key in a.FORBIDDEN_CELL_KEYS:
        bad = dict(cfg)
        bad[key] = "/tmp/foreign_artifact"
        with pytest.raises(SystemExit, match="hidden"):
            a.verify_cell_complete(bad)


def test_e4_genesis_policy_must_forbid_warm_start():
    cfg = _cell()
    bad = dict(cfg)
    bad["genesis_policy"] = dict(bad["genesis_policy"],
                                 warm_start="allowed")
    with pytest.raises(SystemExit, match="warm_start"):
        a.verify_cell_complete(bad)


def test_e4_mechanics_mode_budget_fields_required():
    cfg = _cell()
    bad = json.loads(json.dumps(cfg))
    del bad["execution_modes"]["cpu_mechanics_replay"][
        "budget_max_updates"]
    with pytest.raises(SystemExit, match="omits budget field"):
        a.verify_cell_complete(bad)


# ------------------------- E6/E2: language ------------------------
def test_e6_forbidden_authority_language_refuses():
    for phrase in ("pending ratification",
                   "owner-ratified Alpaca cost",
                   "owner-ratified venue path"):
        with pytest.raises(SystemExit, match="forbidden authority"):
            a.verify_language({"cost_authority": phrase})


def test_e2_wp4_smoke_path_refused_in_b4_artifacts():
    with pytest.raises(SystemExit, match="forbidden authority"):
        a.verify_language(
            {"cmd": "python tools/wp4_cpu_smoke.py --seed 101"})


def test_e2_old_gymfx_commit_refused_at_point_of_use(monkeypatch):
    monkeypatch.setattr(
        a, "GYMFX_PINNED_COMMIT",
        "634c3fd3c344cae3c4048b334158185c8bf4e1ef")
    with pytest.raises(SystemExit, match="not the accepted lineage"):
        a.gymfx_lineage_manifest()


def test_e6_cell_with_forbidden_language_refuses():
    cfg = _cell()
    bad = dict(cfg)
    bad["cost_authority"] = "alpaca venue primary (pending ratification)"
    with pytest.raises(SystemExit, match="forbidden authority"):
        a.verify_cell_complete(bad)


# ------------------------- E3: amendment chain --------------------
def _pins(*rels):
    return {rel: a._sha_file(REPO / rel) for rel in rels}


def _fake_a4(tmp_path, monkeypatch, a5_over=None, a6_over=None,
             a7_over=None, a8_over=None, a9_over=None,
             a10_over=None, **over):
    a4 = {"amends_design_sha256": a.DESIGN_SHA,
          "supersedes_amendment_shas": list(a.AMENDMENT_SHAS),
          "final_code_pins": _pins(
              "tools/b4_authority.py",
              "tools/screen_b_baselines.py",
              "tools/materialize_b4_causal_sac.py",
              "tools/b4_run_cell.py")}
    a4.update(over)
    f = tmp_path / "a4.json"
    f.write_text(json.dumps(a4))
    monkeypatch.setattr(a, "AMENDMENT_4_PATH", f)
    a5 = {"amends_amendment_4_sha256": a._sha_file(f),
          "owner_authorization_sha256": a.OWNER_GPU_AUTH_SHA,
          "scientific_change": "NONE",
          "final_code_pins": _pins(
              "tools/b4_authority.py", "tools/b4_run_cell.py",
              "tests/test_b4_materializer_authority.py")}
    a5.update(a5_over or {})
    f5 = tmp_path / "a5.json"
    f5.write_text(json.dumps(a5))
    monkeypatch.setattr(a, "AMENDMENT_5_PATH", f5)
    a6 = {"amends_amendment_5_sha256": a._sha_file(f5),
          "data_role_change_disclosure": "test fixture disclosure",
          "proposed_campaign_population": {
              "cell_population_sha256": "1" * 64,
              "materialization_sha256": "2" * 64,
              "genesis_binding_sha256": "3" * 64},
          "final_code_pins": _pins(
              "tools/b4_authority.py", "tools/b4_run_cell.py",
              "tools/b4_campaign_executor.py",
              "tools/b4_campaign_ledger.py",
              "tools/b4_adjudicator.py",
              "tools/materialize_b4_causal_sac.py",
              "tests/test_b4_materializer_authority.py")}
    a6.update(a6_over or {})
    f6 = tmp_path / "a6.json"
    f6.write_text(json.dumps(a6))
    monkeypatch.setattr(a, "AMENDMENT_6_PATH", f6)
    a7 = {"amends_amendment_6_sha256": a._sha_file(f6),
          "change_disclosure": "test fixture disclosure",
          "proposed_campaign_population": {
              "cell_population_sha256": "4" * 64,
              "materialization_sha256": "5" * 64,
              "genesis_binding_sha256": "6" * 64},
          "final_code_pins": _pins(
              "tools/b4_authority.py", "tools/b4_run_cell.py",
              "tools/b4_campaign_executor.py",
              "tools/b4_campaign_ledger.py",
              "tools/b4_campaign_orchestrator.py",
              "tools/b4_adjudicator.py",
              "tools/materialize_b4_causal_sac.py",
              "pipeline_plugins/rl_pipeline_with_validation.py",
              "tests/test_b4_materializer_authority.py")}
    a7.update(a7_over or {})
    f7 = tmp_path / "a7.json"
    f7.write_text(json.dumps(a7))
    monkeypatch.setattr(a, "AMENDMENT_7_PATH", f7)
    a8 = {"amends_amendment_7_sha256": a._sha_file(f7),
          "change_disclosure": "test fixture disclosure",
          "proposed_campaign_population": {
              "cell_population_sha256": "7" * 64,
              "materialization_sha256": "8" * 64,
              "genesis_binding_sha256": "9" * 64},
          "final_code_pins": _pins(
              "tools/b4_authority.py", "tools/b4_run_cell.py",
              "tools/b4_campaign_executor.py",
              "tools/b4_campaign_ledger.py",
              "tools/b4_campaign_orchestrator.py",
              "tools/b4_adjudicator.py",
              "tools/materialize_b4_causal_sac.py",
              "pipeline_plugins/rl_pipeline_with_validation.py",
              "tests/test_b4_materializer_authority.py")}
    a8.update(a8_over or {})
    f8 = tmp_path / "a8.json"
    f8.write_text(json.dumps(a8))
    monkeypatch.setattr(a, "AMENDMENT_8_PATH", f8)
    a9 = {"amends_amendment_8_sha256": a._sha_file(f8),
          "change_disclosure": "test fixture disclosure",
          "proposed_campaign_population": {
              "cell_population_sha256": "a" * 64,
              "materialization_sha256": "b" * 64,
              "genesis_binding_sha256": "c" * 64},
          "final_code_pins": _pins(
              "tools/b4_authority.py", "tools/b4_run_cell.py",
              "tools/b4_campaign_executor.py",
              "tools/b4_campaign_ledger.py",
              "tools/b4_campaign_orchestrator.py",
              "tools/b4_adjudicator.py",
              "tools/materialize_b4_causal_sac.py",
              "pipeline_plugins/rl_pipeline_with_validation.py",
              "tests/test_b4_materializer_authority.py")}
    a9.update(a9_over or {})
    f9 = tmp_path / "a9.json"
    f9.write_text(json.dumps(a9))
    monkeypatch.setattr(a, "AMENDMENT_9_PATH", f9)
    monkeypatch.setattr(a, "AMENDMENT_9_SHA", a._sha_file(f9))
    a10 = {"amends_amendment_9_sha256": a._sha_file(f9),
           "change_disclosure": "test fixture disclosure",
           "scientific_change":
               "NONE — runtime authority (C23-C25) only",
           "proposed_campaign_population": {
               "cell_population_sha256": "d" * 64,
               "materialization_sha256": "e" * 64,
               "genesis_binding_sha256": "f" * 64},
           "final_code_pins": _pins(
               "tools/b4_authority.py", "tools/b4_run_cell.py",
               "tools/b4_campaign_executor.py",
               "tools/b4_campaign_ledger.py",
               "tools/b4_campaign_orchestrator.py",
               "tools/b4_adjudicator.py",
               "tools/materialize_b4_causal_sac.py",
               "pipeline_plugins/rl_pipeline_with_validation.py",
               "tests/test_b4_materializer_authority.py")}
    a10.update(a10_over or {})
    f10 = tmp_path / "a10.json"
    f10.write_text(json.dumps(a10))
    monkeypatch.setattr(a, "AMENDMENT_10_PATH", f10)
    return a4


def test_e3_valid_chain_passes(tmp_path, monkeypatch):
    _fake_a4(tmp_path, monkeypatch)
    chain = a.verify_amendment_chain()
    assert chain["design_sha256"] == a.DESIGN_SHA
    assert len(chain["amendment_shas"]) == 10


def test_e3_missing_final_amendment_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "AMENDMENT_4_PATH",
                        tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="final amendment"):
        a.verify_amendment_chain()


def test_e3_reordered_prior_chain_refuses(tmp_path, monkeypatch):
    _fake_a4(tmp_path, monkeypatch,
             supersedes_amendment_shas=list(
                 reversed(a.AMENDMENT_SHAS)))
    with pytest.raises(SystemExit, match="reordered"):
        a.verify_amendment_chain()


def test_e3_drifted_code_refuses(tmp_path, monkeypatch):
    pins = _pins("tools/b4_authority.py", "tools/b4_run_cell.py",
                 "tools/b4_campaign_executor.py",
                 "tools/b4_campaign_ledger.py",
                 "tools/b4_adjudicator.py",
                 "tools/materialize_b4_causal_sac.py",
                 "tests/test_b4_materializer_authority.py")
    pins = _pins("tools/b4_authority.py", "tools/b4_run_cell.py",
                 "tools/b4_campaign_executor.py",
                 "tools/b4_campaign_ledger.py",
                 "tools/b4_campaign_orchestrator.py",
                 "tools/b4_adjudicator.py",
                 "tools/materialize_b4_causal_sac.py",
                 "pipeline_plugins/rl_pipeline_with_validation.py",
                 "tests/test_b4_materializer_authority.py")
    pins["tools/b4_authority.py"] = "0" * 64
    _fake_a4(tmp_path, monkeypatch,
             a10_over={"final_code_pins": pins})
    with pytest.raises(SystemExit, match="differs from the final"):
        a.verify_amendment_chain()


def test_e3_incomplete_pin_surface_refuses(tmp_path, monkeypatch):
    pins = _pins("tools/b4_authority.py",
                 "tools/screen_b_baselines.py",
                 "tools/materialize_b4_causal_sac.py")
    _fake_a4(tmp_path, monkeypatch, final_code_pins=pins)
    with pytest.raises(SystemExit, match="full executing surface"):
        a.verify_amendment_chain()


def test_e3_altered_design_bytes_refuse(tmp_path, monkeypatch):
    fake = tmp_path / "design.json"
    fake.write_text(json.dumps({"tampered": True}))
    monkeypatch.setattr(a, "DESIGN_PATH", fake)
    with pytest.raises(SystemExit, match="immutable"):
        a.verify_amendment_chain()


def test_e3_altered_amendment_bytes_refuse(tmp_path, monkeypatch):
    bad = tmp_path / "amendment1.json"
    bad.write_text("{}")
    paths = list(a.AMENDMENT_PATHS)
    paths[0] = bad
    monkeypatch.setattr(a, "AMENDMENT_PATHS", tuple(paths))
    with pytest.raises(SystemExit, match="amendment 1 bytes"):
        a.verify_amendment_chain()


def test_e3_bind_passes_at_final_tip():
    """B4-E3 §5: at the final tip bind_superseding_design() ITSELF
    must pass — through the real chain, the real design and the live
    gym-fx checkout."""
    out = sb.bind_superseding_design()
    assert out["design_sha256"] == a.DESIGN_SHA
    assert out["chain"]["final_code_pins"][
        "tools/screen_b_baselines.py"] == a._sha_file(SB)


def test_e3_bind_foreign_dataset_refuses(monkeypatch):
    monkeypatch.setattr(sb, "DATA_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="source dataset"):
        sb.bind_superseding_design()


def test_e3_bind_foreign_grid_refuses(monkeypatch):
    monkeypatch.setattr(sb, "CALIBRATION_GRID", [{"rogue": 1}])
    with pytest.raises(SystemExit, match="calibration grid"):
        sb.bind_superseding_design()


def test_e3_bind_mixed_execution_truth_refuses(monkeypatch):
    monkeypatch.setattr(
        a, "gymfx_lineage_manifest",
        lambda: {"commit": "c" * 40, "manifest_sha256": "9" * 64})
    with pytest.raises(SystemExit, match="mixed execution truth"):
        sb.bind_superseding_design()


# ------------------------- E5: comparator forgeries ---------------
STUB_LINEAGE = {"commit": "c" * 40, "manifest_sha256": "b" * 64}
LABEL = "SCREEN_B_CURRENT_EXECUTION_TRUTH_OPTION_B"


def _population(tmp_path, monkeypatch):
    """A MINIMAL fully-valid comparator population; tests poison it."""
    monkeypatch.setattr(a, "gymfx_lineage_manifest",
                        lambda: dict(STUB_LINEAGE))
    d = tmp_path / "pop"
    d.mkdir()
    design = {"source_data_sha256": "d" * 64}
    frozen_sha = a._sha_obj(GEOM)
    for year in (2022, 2023, 2024):
        cells = [{"geometry_index": i, "envelope_sha256":
                  a._sha_obj(dict(GEOM,
                                  atr_sl_mult=100.0 + i)),
                  "criterion": {"eligible": False}}
                 for i in range(6)]
        cells.append({"geometry_index": 6,
                      "envelope_sha256": frozen_sha,
                      "criterion": {"eligible": True,
                                    "composite_median": 0.1}})
        (d / f"ENVELOPE_CALIBRATION_o{year}.json").write_text(
            json.dumps({"frozen_geometry": GEOM,
                        "frozen_envelope_sha256": frozen_sha,
                        "grid_cells": cells,
                        "calibration_year": year - 1}))
    rows = []
    for year in (2022, 2023, 2024):
        for gi in range(7):
            for arm in ("B1", "B2a", "B2b", "B3"):
                rows.append({"trial_id": f"cal{year}g{gi}{arm}",
                             "screen": "B_envelope_calibration",
                             "registered_before_results": True})
    results = []
    for year in (2022, 2023, 2024):
        for arm in ("B0", "B1", "B2a", "B2b", "B3"):
            tid = f"score{year}{arm}"
            rows.append({"trial_id": tid, "screen": "B",
                         "registered_before_results": True})
            pb = d / f"{arm}_{year}_per_bar.csv"
            pb.write_text(f"bar,{arm},{year}\n")
            results.append({
                "arm": arm, "origin": year,
                "cost_set": "alpaca_ethusd",
                "population_label": LABEL,
                "gymfx_lineage_manifest_sha256": "b" * 64,
                "execution_envelope_sha256": frozen_sha,
                "trial_id": tid,
                "per_bar_csv": str(pb),
                "per_bar_sha256": a._sha_file(pb),
                "complete_envelope_digest": "e" * 64,
                "cost_authority": a.COST_AUTHORITY})
    (d / "trial_ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    manifest = {"schema": "agent_multi.screen_b_run_manifest.v3",
                "superseding_design_sha256": a.DESIGN_SHA,
                "gymfx_lineage_manifest_sha256": "b" * 64,
                "source_data_sha256": "d" * 64}
    (d / "RUN_MANIFEST.json").write_text(json.dumps(manifest))
    packet = {"run_manifest_sha256":
              a._sha_file(d / "RUN_MANIFEST.json"),
              "population_label": LABEL,
              "results": results, "sealed_2025_used": False}
    (d / "SCREEN_B_RESULTS.json").write_text(json.dumps(packet))
    return d, design, packet


def test_e5_valid_population_verifies(tmp_path, monkeypatch):
    d, design, _ = _population(tmp_path, monkeypatch)
    facts = a.verify_comparator_population(d, design)
    assert facts["n_results"] == 15 and facts["n_ledger"] == 99


def _rewrite_packet(d, packet):
    (d / "SCREEN_B_RESULTS.json").write_text(json.dumps(packet))
    packet["run_manifest_sha256"] = a._sha_file(
        d / "RUN_MANIFEST.json")
    (d / "SCREEN_B_RESULTS.json").write_text(json.dumps(packet))


def test_e5_forged_summary_with_missing_results_refuses(
        tmp_path, monkeypatch):
    """The exact E5 counterexample: valid label AND valid lineage but
    the result records are gone — must refuse now."""
    d, design, packet = _population(tmp_path, monkeypatch)
    packet["results"] = []
    _rewrite_packet(d, packet)
    with pytest.raises(SystemExit, match="expected exactly 15"):
        a.verify_comparator_population(d, design)


def test_e5_extra_result_refuses(tmp_path, monkeypatch):
    d, design, packet = _population(tmp_path, monkeypatch)
    packet["results"].append(dict(packet["results"][0]))
    _rewrite_packet(d, packet)
    with pytest.raises(SystemExit):
        a.verify_comparator_population(d, design)


def test_e5_altered_per_bar_evidence_refuses(tmp_path, monkeypatch):
    d, design, packet = _population(tmp_path, monkeypatch)
    Path(packet["results"][3]["per_bar_csv"]).write_text("altered\n")
    with pytest.raises(SystemExit, match="digest-broken"):
        a.verify_comparator_population(d, design)


def test_e5_unregistered_trial_refuses(tmp_path, monkeypatch):
    d, design, packet = _population(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in
            (d / "trial_ledger.jsonl").read_text().splitlines()]
    rows = [r for r in rows
            if r["trial_id"] != packet["results"][0]["trial_id"]]
    (d / "trial_ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(SystemExit):
        a.verify_comparator_population(d, design)


def test_e5_foreign_result_lineage_refuses(tmp_path, monkeypatch):
    d, design, packet = _population(tmp_path, monkeypatch)
    packet["results"][7]["gymfx_lineage_manifest_sha256"] = "0" * 64
    _rewrite_packet(d, packet)
    with pytest.raises(SystemExit, match="foreign lineage"):
        a.verify_comparator_population(d, design)


def test_e5_non_causal_calibration_refuses(tmp_path, monkeypatch):
    d, design, _ = _population(tmp_path, monkeypatch)
    cal = json.loads(
        (d / "ENVELOPE_CALIBRATION_o2024.json").read_text())
    cal["calibration_year"] = 2024
    (d / "ENVELOPE_CALIBRATION_o2024.json").write_text(
        json.dumps(cal))
    with pytest.raises(SystemExit, match="year-1"):
        a.verify_comparator_population(d, design)


def test_e5_missing_envelope_digest_refuses(tmp_path, monkeypatch):
    d, design, packet = _population(tmp_path, monkeypatch)
    del packet["results"][0]["complete_envelope_digest"]
    _rewrite_packet(d, packet)
    with pytest.raises(SystemExit, match="complete-envelope"):
        a.verify_comparator_population(d, design)


def test_e5_forbidden_language_in_results_refuses(
        tmp_path, monkeypatch):
    d, design, packet = _population(tmp_path, monkeypatch)
    packet["results"][0]["cost_authority"] = \
        "alpaca venue primary (pending ratification)"
    _rewrite_packet(d, packet)
    with pytest.raises(SystemExit, match="forbidden authority"):
        a.verify_comparator_population(d, design)


# ------------------------- E7: runner CLI is closed ---------------
def test_e7_cli_cannot_override_scientific_values(tmp_path):
    for flag in ("--learning-rate", "--net-arch", "--seed",
                 "--budget-max-updates", "--train-year"):
        with pytest.raises(SystemExit):
            runner.main(["--cell-id", "o2024_seed101",
                         "--materialization-root", str(tmp_path),
                         "--output-root", str(tmp_path / "out"),
                         "--device", "cpu", flag, "1"])


def test_p1_gpu_wrong_cell_refuses(tmp_path):
    with pytest.raises(SystemExit,
                       match="approved exactly ONE GPU cell"):
        runner.main(["--cell-id", "o2023_seed202",
                     "--materialization-root", str(tmp_path),
                     "--output-root", str(tmp_path / "out"),
                     "--device", "cuda:0"])


def test_p2_foreign_root_refuses_any_device(tmp_path):
    for dev in ("cpu", "cuda:0"):
        with pytest.raises(SystemExit,
                           match="absent from the materialization"):
            runner.main(["--cell-id", "o2024_seed101",
                         "--materialization-root", str(tmp_path),
                         "--output-root", str(tmp_path / "out"),
                         "--device", dev])


def test_e7_unreviewed_cell_id_refuses(tmp_path):
    (tmp_path / "B4_CELL_CONFIGS.json").write_text(json.dumps({}))
    with pytest.raises(SystemExit, match="not a reviewed"):
        runner.load_cell(tmp_path, "o2024_seed999")


def test_e7_tampered_cell_digest_refuses(tmp_path):
    cfg = _cell()
    (tmp_path / "B4_CELL_CONFIGS.json").write_text(json.dumps(
        {"o2024_seed101": {"effective_config": cfg,
                           "config_sha256": "0" * 64}}))
    with pytest.raises(SystemExit, match="digest mismatch"):
        runner.load_cell(tmp_path, "o2024_seed101")


# ------------------------- genesis refusals (kept) ----------------
g = importlib.import_module("tools.p1lr_genesis_artifacts")


class _FakeBuf:
    def __init__(self, n):
        self._n = n

    def size(self):
        return self._n


class _FakeModel:
    def __init__(self, n_updates=0, num_timesteps=0, replay=0):
        self._n_updates = n_updates
        self.num_timesteps = num_timesteps
        self.replay_buffer = _FakeBuf(replay)
        self.policy = object()


def test_d1_nonzero_genesis_updates_refuse():
    with pytest.raises(RuntimeError, match="GENESIS_NOT_ZERO_UPDATE"):
        g._zero_update_proof(_FakeModel(n_updates=1))
    with pytest.raises(RuntimeError, match="GENESIS_NOT_ZERO_UPDATE"):
        g._zero_update_proof(_FakeModel(replay=5))
    g._zero_update_proof(_FakeModel())


def _fake_build(monkeypatch, hashes):
    class _FakePlugin:
        def save(self, model, path):
            Path(path).write_bytes(b"z")

    class _FakeEnv:
        def close(self):
            pass

    monkeypatch.setattr(
        g, "resolve_observation_dimension",
        lambda c, b: {"observation_dim": 4, "net_arch": [8],
                      "ent_coef": 0.1})
    monkeypatch.setattr(
        g, "_build_model",
        lambda seed, dim, facts: (_FakeModel(), _FakePlugin(),
                                  _FakeEnv()))
    sac = importlib.import_module("agent_plugins.sac_agent")
    it = iter(hashes)
    monkeypatch.setattr(sac, "_policy_tensor_hash",
                        lambda pol: next(it))


def test_d1_resume_artifact_refuses(tmp_path, monkeypatch):
    _fake_build(monkeypatch, ["a" * 64, "a" * 64])
    seed_dir = tmp_path / "seed7"
    seed_dir.mkdir()
    (seed_dir / "zero_update_genesis_seed7.zip").write_bytes(b"old")
    with pytest.raises(RuntimeError, match="GENESIS_EXISTS"):
        g.build_seed_genesis({}, {}, 7, tmp_path)


def test_d1_foreign_tensor_identity_refuses(tmp_path, monkeypatch):
    _fake_build(monkeypatch, ["a" * 64, "b" * 64])
    with pytest.raises(RuntimeError, match="GENESIS_NONDETERMINISTIC"):
        g.build_seed_genesis({}, {}, 8, tmp_path)


# ============ Order @9fb017e3: B4-P1..P4 battery ============
MAT_V2 = (Path.home() /
          ".local/share/agent-multi/b4_materialization_v2_20260905")
V6_DIR = (REPO / "docs/audits/evidence/"
          "screen_b_rule_arms_v6_e_corrected_20260905")
_mat_present = pytest.mark.skipif(
    not MAT_V2.is_dir(), reason="approved materialization root "
    "absent on this host")


def test_p1_exact_authorization_verifies():
    rec = a.verify_gpu_preflight_authorization()
    assert rec["decision"] == \
        "APPROVE_ONE_B4_BOUNDED_GPU_PREFLIGHT_ONLY"
    lim = rec["preflight_limits"]
    assert (lim["environment_steps_max"], lim["optimizer_updates_max"],
            lim["wall_seconds_max"]) == (20000, 20000, 7200)
    assert lim["attempts"] == 1


def test_p1_missing_authorization_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "OWNER_GPU_AUTH_PATH",
                        tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="authorization absent"):
        a.verify_gpu_preflight_authorization()


def test_p1_edited_or_self_rehashed_record_refuses(
        tmp_path, monkeypatch):
    rec = json.loads(a.OWNER_GPU_AUTH_PATH.read_text())
    rec["preflight_limits"]["environment_steps_max"] = 40_000_000
    f = tmp_path / "auth.json"
    f.write_text(json.dumps(rec))
    monkeypatch.setattr(a, "OWNER_GPU_AUTH_PATH", f)
    with pytest.raises(SystemExit, match="bytes differ"):
        a.verify_gpu_preflight_authorization()


def test_p1_wrong_carried_digest_refuses(monkeypatch):
    monkeypatch.setattr(a, "OWNER_GPU_AUTH_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="bytes differ"):
        a.verify_gpu_preflight_authorization()


def test_p1_wrong_decision_refuses(tmp_path, monkeypatch):
    rec = json.loads(a.OWNER_GPU_AUTH_PATH.read_text())
    rec["decision"] = "APPROVE_FULL_CAMPAIGN"
    f = tmp_path / "auth.json"
    f.write_text(json.dumps(rec))
    monkeypatch.setattr(a, "OWNER_GPU_AUTH_PATH", f)
    monkeypatch.setattr(a, "OWNER_GPU_AUTH_SHA",
                        a._sha_file(f))
    with pytest.raises(SystemExit, match="single "
                       "bounded GPU preflight"):
        a.verify_gpu_preflight_authorization()


def test_p1_unknown_field_refuses(tmp_path, monkeypatch):
    rec = json.loads(a.OWNER_GPU_AUTH_PATH.read_text())
    rec["attacker_extension"] = {"grants": "everything"}
    f = tmp_path / "auth.json"
    f.write_text(json.dumps(rec))
    monkeypatch.setattr(a, "OWNER_GPU_AUTH_PATH", f)
    monkeypatch.setattr(a, "OWNER_GPU_AUTH_SHA", a._sha_file(f))
    with pytest.raises(SystemExit, match="unknown top-level"):
        a.verify_gpu_preflight_authorization()


@_mat_present
def test_p2_approved_materialization_verifies():
    rec = a.verify_gpu_preflight_authorization()
    a.verify_approved_materialization(MAT_V2, rec)


@_mat_present
def test_p2_self_rebound_cell_refuses_on_external_identity(tmp_path):
    """The exact P2 reproduction with ALL internal digests repaired
    — must refuse on the externally pinned identity before any
    model construction."""
    rec = a.verify_gpu_preflight_authorization()
    fake = tmp_path / "mat"
    (fake / "genesis").mkdir(parents=True)
    cells = json.loads((MAT_V2 / "B4_CELL_CONFIGS.json").read_text())
    cfg = cells["o2024_seed101"]["effective_config"]
    cfg["learning_rate"] = 0.123
    new_digest = hashlib.sha256(json.dumps(
        cfg, sort_keys=True, default=str).encode()).hexdigest()
    cells["o2024_seed101"]["config_sha256"] = new_digest
    (fake / "B4_CELL_CONFIGS.json").write_text(json.dumps(cells))
    import shutil
    shutil.copy(MAT_V2 / "B4_MATERIALIZATION.json",
                fake / "B4_MATERIALIZATION.json")
    binding = json.loads(
        (MAT_V2 / "genesis" / "GENESIS_BINDING.json").read_text())
    binding["binding"]["o2024_seed101"] = new_digest
    (fake / "genesis" / "GENESIS_BINDING.json").write_text(
        json.dumps(binding))
    with pytest.raises(SystemExit,
                       match="owner-approved identity"):
        a.verify_approved_materialization(fake, rec)
    with pytest.raises(SystemExit,
                       match="owner-approved identity"):
        runner.main(["--cell-id", "o2024_seed101",
                     "--materialization-root", str(fake),
                     "--output-root", str(tmp_path / "out"),
                     "--device", "cpu"])


def test_p3_real_v6_digests_rederive():
    design = json.loads(a.DESIGN_PATH.read_text())
    binding = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/"
         "cost_manifest_eth_h4_v2_screen_b_20260826.json"
         ).read_text())["alpaca_ethusd"]["env_binding"]
    facts = a.verify_comparator_population(V6_DIR, design, binding)
    assert len(facts["complete_envelope_digest_by_origin"]) == 3


def test_p3_forged_complete_envelope_digest_refuses(
        tmp_path, monkeypatch):
    """The exact P3 counterexample: 64 zeroes in one result."""
    import shutil
    copy = tmp_path / "v6"
    copy.mkdir()
    for f in ("RUN_MANIFEST.json", "trial_ledger.jsonl",
              "ENVELOPE_CALIBRATION_o2022.json",
              "ENVELOPE_CALIBRATION_o2023.json",
              "ENVELOPE_CALIBRATION_o2024.json"):
        shutil.copy(V6_DIR / f, copy / f)
    packet = json.loads((V6_DIR / "SCREEN_B_RESULTS.json").read_text())
    packet["results"][0]["complete_envelope_digest"] = "0" * 64
    (copy / "SCREEN_B_RESULTS.json").write_text(json.dumps(packet))
    design = json.loads(a.DESIGN_PATH.read_text())
    binding = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/"
         "cost_manifest_eth_h4_v2_screen_b_20260826.json"
         ).read_text())["alpaca_ethusd"]["env_binding"]
    with pytest.raises(SystemExit, match="does not re-derive from "
                       "the frozen geometry"):
        a.verify_comparator_population(copy, design, binding)


def test_p3_changed_cost_field_changes_derivation():
    """Commission, slippage or a geometry field changes the derived
    digest — the factual field, not a label, decides."""
    binding = {"commission": 0.00295115, "slippage_perc": 0.0001}
    base = a.complete_envelope_digest(
        a.complete_execution_envelope(GEOM, binding), binding)
    for poison in ({"commission": 0.001},
                   {"slippage_perc": 0.002}):
        b2 = dict(binding, **poison)
        d2 = a.complete_envelope_digest(
            a.complete_execution_envelope(GEOM, b2), b2)
        assert d2 != base
    g2 = dict(GEOM, atr_sl_mult=9.9)
    assert a.complete_envelope_digest(
        a.complete_execution_envelope(g2, binding),
        binding) != base


def test_p4_gpu_mode_derives_only_from_owner_record():
    rec = a.verify_gpu_preflight_authorization()
    cfg = _cell()
    cfg["execution_modes"]["gpu_economic"][
        "budget_max_env_steps"] = 999_999_999
    mode = runner.gpu_mode_from_record(rec, cfg)
    assert mode["budget_max_env_steps"] == 20000
    assert mode["budget_max_updates"] == 20000
    assert mode["budget_max_wall_seconds"] == 7200.0
    assert mode["rss_cap_bytes"] == 8 * 1024 ** 3
    assert mode["cuda_cap_bytes"] == 6 * 1024 ** 3
    assert mode["thermal_cap_celsius"] == 87
    assert mode["learn_segments"] == [20000]
    assert mode["train_year"] == 2023


def test_p4_owner_train_year_mismatch_refuses():
    rec = json.loads(a.OWNER_GPU_AUTH_PATH.read_text())
    rec["execution_contract"]["training_year"] = 2024
    cfg = _cell()
    with pytest.raises(SystemExit, match="training year"):
        runner.gpu_mode_from_record(rec, cfg)


def test_p4_ambiguous_or_missing_gpu_telemetry_refuses(monkeypatch):
    monkeypatch.setattr(runner, "_nvidia_query",
                        lambda q, d=None: [])
    with pytest.raises(SystemExit, match="missing or ambiguous"):
        runner.gpu_inventory("0")
    monkeypatch.setattr(runner, "_nvidia_query",
                        lambda q, d=None: ["a,b,1,2,3", "c,d,4,5,6"])
    with pytest.raises(SystemExit, match="missing or ambiguous"):
        runner.gpu_inventory("0")


def _guard(peak, **kw):
    import types
    cb = runner.make_guard_callback(peak, **kw)
    cb.model = types.SimpleNamespace(_n_updates=0)
    cb.num_timesteps = 25
    return cb


def test_p4_guard_stops_on_rss_thermal_and_lost_telemetry(
        monkeypatch):
    peak = {"peak_rss_bytes": 0, "stop": None}
    cb = _guard(peak, rss_cap=1, thermal_cap=87.0)
    assert cb._on_step() is False and "RSS cap" in peak["stop"]
    monkeypatch.setattr(runner, "_gpu_temp", lambda d: 99.0)
    peak = {"peak_rss_bytes": 0, "stop": None}
    cb = _guard(peak, rss_cap=2 ** 40, thermal_cap=87.0,
                gpu_device="0")
    assert cb._on_step() is False
    assert "thermal cap 87.0C exceeded at 99.0C" in peak["stop"]
    monkeypatch.setattr(runner, "_gpu_temp", lambda d: None)
    peak = {"peak_rss_bytes": 0, "stop": None}
    cb = _guard(peak, rss_cap=2 ** 40, thermal_cap=87.0,
                gpu_device="0")
    assert cb._on_step() is False
    assert "telemetry lost" in peak["stop"]


def test_p4_heartbeat_emits_facts():
    peak = {"peak_rss_bytes": 0, "stop": None}
    beats = []
    cb = _guard(peak, rss_cap=2 ** 40, thermal_cap=200.0,
                heartbeat_seconds=0, heartbeats=beats)
    assert cb._on_step() is True
    assert beats and beats[0]["env_steps"] == 25


@_mat_present
def test_p4_second_attempt_refuses(tmp_path, monkeypatch):
    ledger = tmp_path / "attempt.json"
    ledger.write_text("{}")
    monkeypatch.setattr(a, "GPU_ATTEMPT_LEDGER", ledger)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        runner, "gpu_inventory",
        lambda d: {"uuid": "GPU-x", "name": "x",
                   "temperature_celsius": 40.0,
                   "memory_used_mib": 100.0,
                   "memory_total_mib": 8188.0})
    monkeypatch.setattr(runner, "gpu_compute_apps", lambda d: [])
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(SystemExit, match="already consumed"):
        runner.main(["--cell-id", "o2024_seed101",
                     "--materialization-root", str(MAT_V2),
                     "--output-root", str(tmp_path / "out"),
                     "--device", "cuda:0"])


@_mat_present
def test_p4_hot_device_resource_blocks_without_consuming(
        tmp_path, monkeypatch):
    ledger = tmp_path / "attempt.json"
    monkeypatch.setattr(a, "GPU_ATTEMPT_LEDGER", ledger)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        runner, "gpu_inventory",
        lambda d: {"uuid": "GPU-x", "name": "x",
                   "temperature_celsius": 90.0,
                   "memory_used_mib": 100.0,
                   "memory_total_mib": 8188.0})
    monkeypatch.setattr(runner, "gpu_compute_apps", lambda d: [])
    out = tmp_path / "out"
    rc = runner.main(["--cell-id", "o2024_seed101",
                      "--materialization-root", str(MAT_V2),
                      "--output-root", str(out),
                      "--device", "cuda:0"])
    assert rc == 0 and not ledger.exists()
    term = json.loads(
        (out / "B4_GPU_PREFLIGHT_TERMINAL.json").read_text())
    assert term["status"] == "B4_GPU_PREFLIGHT_RESOURCE_BLOCKED"
    assert term["attempt_consumed"] is False


@_mat_present
def test_p4_busy_device_resource_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "GPU_ATTEMPT_LEDGER",
                        tmp_path / "attempt.json")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        runner, "gpu_inventory",
        lambda d: {"uuid": "GPU-x", "name": "x",
                   "temperature_celsius": 40.0,
                   "memory_used_mib": 100.0,
                   "memory_total_mib": 8188.0})
    monkeypatch.setattr(
        runner, "gpu_compute_apps",
        lambda d: [{"pid": 1234, "used_memory_mib": 4000.0}])
    out = tmp_path / "out"
    rc = runner.main(["--cell-id", "o2024_seed101",
                      "--materialization-root", str(MAT_V2),
                      "--output-root", str(out),
                      "--device", "cuda:0"])
    assert rc == 0
    term = json.loads(
        (out / "B4_GPU_PREFLIGHT_TERMINAL.json").read_text())
    assert term["status"] == "B4_GPU_PREFLIGHT_RESOURCE_BLOCKED"
    assert "substantial CUDA compute workload" in term["reason"]


@_mat_present
def test_p4_multi_device_binding_refuses(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(SystemExit, match="exactly one explicit"):
        runner.main(["--cell-id", "o2024_seed101",
                     "--materialization-root", str(MAT_V2),
                     "--output-root", str(tmp_path / "out"),
                     "--device", "cuda:0"])


# ============ Order @e8bb500f: E8-E12 battery ============
EXEC = REPO / "tools" / "b4_campaign_executor.py"
_exec_spec = importlib.util.spec_from_file_location("b4exec", EXEC)
executor = importlib.util.module_from_spec(_exec_spec)
_exec_spec.loader.exec_module(executor)
LED = REPO / "tools" / "b4_campaign_ledger.py"
_led_spec = importlib.util.spec_from_file_location("b4led", LED)
ledger_mod = importlib.util.module_from_spec(_led_spec)
_led_spec.loader.exec_module(ledger_mod)
ADJ = REPO / "tools" / "b4_adjudicator.py"
_adj_spec = importlib.util.spec_from_file_location("b4adj", ADJ)
adj = importlib.util.module_from_spec(_adj_spec)
_adj_spec.loader.exec_module(adj)
import numpy as _np


def test_e12_chain_requires_amendment_6(tmp_path, monkeypatch):
    _fake_a4(tmp_path, monkeypatch)
    monkeypatch.setattr(a, "AMENDMENT_6_PATH",
                        tmp_path / "absent6.json")
    with pytest.raises(SystemExit, match="amendment 6 absent"):
        a.verify_amendment_chain()


def test_e12_a6_must_disclose_role_change(tmp_path, monkeypatch):
    _fake_a4(tmp_path, monkeypatch)
    a6 = json.loads(a.AMENDMENT_6_PATH.read_text())
    a6["data_role_change_disclosure"] = ""
    a.AMENDMENT_6_PATH.write_text(json.dumps(a6))
    with pytest.raises(SystemExit, match="disclose"):
        a.verify_amendment_chain()


def test_e12_campaign_tree_binds_a6_population(
        tmp_path, monkeypatch):
    _fake_a4(tmp_path, monkeypatch)
    fake = tmp_path / "mat"
    (fake / "genesis").mkdir(parents=True)
    (fake / "B4_CELL_CONFIGS.json").write_text("{}")
    (fake / "B4_MATERIALIZATION.json").write_text("{}")
    (fake / "genesis" / "GENESIS_BINDING.json").write_text("{}")
    with pytest.raises(SystemExit,
                       match="proposed population identity"):
        a.verify_campaign_materialization(fake)


def test_e12_contract_windows_are_causal(tmp_path):
    """Scored-year isolation at the contract seam: every fitting or
    selection role ends at/before score start; the sealed zone
    begins the instant scoring ends."""
    for year in (2022, 2023, 2024):
        c = m.author_origin_contract(year, tmp_path)
        roles = c["roles"]
        score_start = f"{year}-01-01T00:00:00"
        assert roles["fit_train"]["end"] <= score_start
        assert roles["train_monitor"]["end"] <= score_start
        assert roles["inner_validation"]["end"] == score_start
        assert roles["outer_validation"]["start"] == score_start
        assert roles["sealed_test"]["start"] == \
            roles["outer_validation"]["end"]


def test_e12_executor_cli_is_closed(tmp_path):
    with pytest.raises(SystemExit):
        executor.main(["--cell-id", "o2024_seed101",
                       "--materialization-root", str(tmp_path),
                       "--output-root", str(tmp_path),
                       "--action", "dry-run",
                       "--learning-rate", "0.9"])


def test_e12_execute_refuses_without_campaign_authorization(
        tmp_path):
    assert executor.CAMPAIGN_AUTH_SHA is None
    with pytest.raises(SystemExit,
                       match="campaign authorization record"):
        executor.execute_cell("o2024_seed101", tmp_path, tmp_path,
                              "cuda:0")


def test_e12_terminal_records_are_immutable(tmp_path):
    executor.write_terminal(tmp_path, "o2022_seed101", "FAILED",
                            {"reason": "x"})
    with pytest.raises(SystemExit, match="immutable"):
        executor.write_terminal(tmp_path, "o2022_seed101",
                                "FAILED", {"reason": "y"})
    with pytest.raises(SystemExit, match="unknown terminal"):
        executor.write_terminal(tmp_path, "o2023_seed101",
                                "PROMOTED", {})
    with pytest.raises(SystemExit, match="inadmissible terminal"):
        executor.write_terminal(tmp_path, "o2023_seed202",
                                "COMPLETED", {"attempt_id": "a1"})


def test_e12_stop_classification():
    f = executor.classify_stop
    assert f(None, "GPU thermal cap 87C exceeded", False) == \
        "THERMAL_STOP"
    assert f(None, "RSS cap exceeded", False) == "RESOURCE_STOP"
    assert f("external stop request (budget_stop_file present)",
             None, False) == "EXTERNALLY_STOPPED"
    assert f("wall budget 43200s exceeded", None, False) == \
        "TIMED_OUT"
    assert f(None, None, False) == "COMPLETED"


def _ledger_fixture(tmp_path):
    """C21 coherent universe: the fixture materializes its OWN
    frozen source per origin (540-bar context + the full year, 4h),
    a comparator population with the same bar identities, exact-
    schema claims/terminals, real checkpoint bytes and PHYSICAL
    intent/completion seals. Every factual field re-derives."""
    import numpy as _np2
    import pandas as _pd2
    gen = a.CAMPAIGN_GENERATION
    bars = {2022: 2190, 2023: 2190, 2024: 2196}
    (tmp_path / "contracts").mkdir(exist_ok=True)
    comp_dir = tmp_path / "comp_default"
    comp_dir.mkdir(exist_ok=True)
    frozen = {}
    arms = []
    for year, n in bars.items():
        ctx = _pd2.date_range(end=f"{year}-01-01",
                              periods=541, freq="4h")[:-1]
        yr = _pd2.date_range(f"{year}-01-01", periods=n, freq="4h")
        dts = ctx.append(yr)
        close = 1.0 + _np2.arange(len(dts)) * 1e-4
        srcp = tmp_path / f"source_{year}.csv"
        _pd2.DataFrame({"DATE_TIME": dts, "CLOSE": close}
                       ).to_csv(srcp, index=False)
        (tmp_path / "contracts" /
         f"b4_causal_origin_{year}_contract.json").write_text(
            json.dumps({"source_ref": f"fixture:{year}"}))
        sdt = [d.strftime("%Y-%m-%d %H:%M") for d in yr]
        scl = close[540:]
        frozen[year] = {
            "datetimes": sdt,
            "row_shas": [hashlib.sha256(
                f"{d}|{c:.10g}".encode()).hexdigest()
                for d, c in zip(sdt, scl)]}
        cp = comp_dir / f"B0_{year}.csv"
        _pd2.DataFrame({"datetime": yr,
                        "net_return": [0.0] * n}).to_csv(
            cp, index=False)
        arms.append({"arm": "B0", "origin": year,
                     "per_bar_csv": str(cp),
                     "per_bar_sha256": ledger_mod._sha_file(cp)})
    (comp_dir / "SCREEN_B_RESULTS.json").write_text(
        json.dumps({"results": arms}))
    cells = {f"o{y}_seed{s}": {"cell_config_sha256":
                               f"{y}{s}".ljust(64, "a")}
             for y in (2022, 2023, 2024)
             for s in (101, 202, 303, 404)}
    entries = {cid: {"cell_config_sha256": c["cell_config_sha256"]}
               for cid, c in cells.items()}
    led = {"schema": "agent_multi.b4_campaign_ledger.v1",
           "cells": entries,
           "campaign_digest": ledger_mod._sha_obj(
               {cid: e["cell_config_sha256"]
                for cid, e in entries.items()})}
    results = tmp_path / "results"
    for cid, e in entries.items():
        year = int(cid.split("_")[0][1:])
        seed = int(cid.split("seed")[1])
        d = results / cid
        d.mkdir(parents=True)
        n = bars[year]
        eq = _np2.linspace(1000.0, 1010.0, n)
        delta = _np2.zeros(n)
        delta[1:] = _np2.diff(eq)
        nr = _np2.zeros(n)
        nr[1:] = eq[1:] / eq[:-1] - 1.0
        df = _pd2.DataFrame({
            "origin": [year] * n,
            "seed": [seed] * n,
            "datetime_utc": frozen[year]["datetimes"],
            "scored_index": list(range(540, 540 + n)),
            "source_row_sha256": frozen[year]["row_shas"],
            "requested_exposure": [0.0] * n,
            "realized_exposure": [0.0] * n,
            "gross_equity": ["UNAVAILABLE_ENV_FACT"] * n,
            "economic_equity": eq,
            "net_equity_delta_observed": delta,
            "env_pnl_fact": delta,
            "commission_delta": [0.0] * n,
            "pre_commission_equity_delta_derived": delta,
            "slippage_declared": ["declared_zero"] * n,
            "net_return": nr})
        pb = d / f"per_bar_{cid}.csv"
        df.to_csv(pb, index=False)
        ident_sha = hashlib.sha256(
            "|".join(frozen[year]["datetimes"]).encode()
        ).hexdigest()
        ckp = d / f"checkpoint_{cid}.zip"
        ckp.write_bytes(f"fixture-checkpoint-{cid}".encode())
        os.chmod(ckp, 0o644)
        term = {"schema": "agent_multi.b4_cell_terminal.v1",
                "cell": cid, "terminal": "COMPLETED",
                "g1_eligible": False,
                "checkpoint_promotable": False,
                "cell_config_sha256": e["cell_config_sha256"],
                "attempt_id": f"attempt_{cid}",
                "artifact_class": "fixture",
                "per_bar_csv": str(pb),
                "per_bar_sha256": ledger_mod._sha_file(pb),
                "scored_index_sha256": ident_sha,
                "scored_bars": n,
                "counter_semantics": "fixture",
                "checkpoint_sha256": ledger_mod._sha_file(ckp),
                "checkpoint_path": str(ckp),
                "sealed_2025_used": False,
                "wall_seconds": 1.0,
                "effective_limits": {}}
        tp = d / "B4_CELL_TERMINAL.json"
        _ctl_write(tp, json.dumps(term))
        _write_claim_and_seal(results, cid, f"attempt_{cid}")
    return led, results


def _write_claim_and_seal(results, cid, att):
    """Exact-schema self-integral claim + PHYSICAL seal, all
    private-mode under 0700 directories."""
    gen = a.CAMPAIGN_GENERATION
    d = results / cid
    os.chmod(results, 0o700)
    os.chmod(d, 0o700)
    tp = d / "B4_CELL_TERMINAL.json"
    os.chmod(tp, 0o600)
    rec = _signed_claim(
        {"schema": "agent_multi.b4_attempt_claim.v2",
         "campaign_generation": gen,
         "attempt_id": att, "cell": cid,
         "claimed_wall": 0.0, "claimed_monotonic": 0.0,
         "holder_pid": os.getpid(),
         "terminal_sha256": None})
    _ctl_write(d / f"CLAIM_{gen}.json", json.dumps(rec))
    for w in d.glob("SEAL_*.json"):
        w.unlink()
    orch.seal_attempt(results, cid, att)


def _reseal(results, cid):
    tp = results / cid / "B4_CELL_TERMINAL.json"
    term = json.loads(tp.read_text())
    att = term.get("attempt_id", f"attempt_{cid}")
    _write_claim_and_seal(results, cid, att)


def _patch_universe(led, tmp_path, monkeypatch):
    monkeypatch.setattr(ledger_mod, "verify_ledger",
                        lambda lp, mr: led)
    monkeypatch.setattr(
        ledger_mod.b4a, "resolve_source_ref",
        lambda ref: tmp_path / f"source_{ref.split(':')[1]}.csv")
    monkeypatch.setattr(
        ledger_mod, "_derive_comparator_dir",
        lambda mr: tmp_path / "comp_default")


def _check_results(led, tmp_path, results, monkeypatch):
    _patch_universe(led, tmp_path, monkeypatch)
    return ledger_mod.verify_campaign_results(
        tmp_path / "ledger.json", tmp_path, results)


def test_e12_complete_population_verifies(tmp_path, monkeypatch):
    led, results = _ledger_fixture(tmp_path)
    facts = _check_results(led, tmp_path, results, monkeypatch)
    assert facts["n"] == 12


def test_e12_partial_population_refuses(tmp_path, monkeypatch):
    led, results = _ledger_fixture(tmp_path)
    import shutil
    shutil.rmtree(results / "o2024_seed404")
    with pytest.raises(SystemExit, match="partial population"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_e12_mechanics_result_as_scientific_refuses(
        tmp_path, monkeypatch):
    led, results = _ledger_fixture(tmp_path)
    tp = results / "o2024_seed101" / "B4_CELL_TERMINAL.json"
    base = json.loads(tp.read_text())
    # smuggled extra field dies on the EXACT schema first
    term = dict(base)
    term["status"] = "B4_GPU_PREFLIGHT_MECHANICS_AND_THROUGHPUT_ONLY"
    _ctl_write(tp, json.dumps(term))
    _reseal(results, "o2024_seed101")
    with pytest.raises(SystemExit, match="exact schema"):
        _check_results(led, tmp_path, results, monkeypatch)
    # a preflight token inside the terminal class dies as mechanics
    term = dict(base)
    term["terminal"] = "COMPLETED_B4_GPU_PREFLIGHT_MECHANICS"
    _ctl_write(tp, json.dumps(term))
    _reseal(results, "o2024_seed101")
    with pytest.raises(SystemExit, match="mechanics/preflight"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_e12_reused_attempt_refuses(tmp_path, monkeypatch):
    led, results = _ledger_fixture(tmp_path)
    for cid in ("o2022_seed101", "o2022_seed202"):
        tp = results / cid / "B4_CELL_TERMINAL.json"
        term = json.loads(tp.read_text())
        term["attempt_id"] = "attempt_SAME"
        _ctl_write(tp, json.dumps(term))
        _reseal(results, cid)
    with pytest.raises(SystemExit, match="attempt identity reused"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_e12_foreign_cell_digest_refuses(tmp_path, monkeypatch):
    led, results = _ledger_fixture(tmp_path)
    tp = results / "o2023_seed303" / "B4_CELL_TERMINAL.json"
    term = json.loads(tp.read_text())
    term["cell_config_sha256"] = "f" * 64
    _ctl_write(tp, json.dumps(term))
    _reseal(results, "o2023_seed303")
    with pytest.raises(SystemExit, match="foreign cell digest"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_e12_extra_cell_refuses(tmp_path, monkeypatch):
    led, results = _ledger_fixture(tmp_path)
    (results / "o2025_seed999").mkdir()
    with pytest.raises(SystemExit, match="extra/foreign"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_e12_sealed_read_refuses(tmp_path, monkeypatch):
    led, results = _ledger_fixture(tmp_path)
    tp = results / "o2022_seed303" / "B4_CELL_TERMINAL.json"
    term = json.loads(tp.read_text())
    term["sealed_2025_used"] = True
    _ctl_write(tp, json.dumps(term))
    _reseal(results, "o2022_seed303")
    with pytest.raises(SystemExit, match="sealed-period"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_e12_score_dependent_scheduling_refuses():
    led = {"cells": {cid: {"status": "PENDING"}
                     for cid in ledger_mod.EXPECTED_CELLS}}
    with pytest.raises(SystemExit, match="score-bearing"):
        ledger_mod.schedule_next(led, {"observed_sharpe": 1.0})
    with pytest.raises(SystemExit, match="unknown scheduler"):
        ledger_mod.schedule_next(led, {"favorite_color": "red"})
    nxt = ledger_mod.schedule_next(
        led, {"device_available": True,
              "stop_file_present": False,
              "compute_apps_active": False})
    assert nxt == "o2022_seed101"


def _synth_series(rng, n, drift):
    return rng.normal(0.0002, 0.004, n) + drift


def _synth_population(drift_b4):
    rng = _np.random.default_rng(7)
    rules = {}
    for arm in adj.RULE_ARMS:
        for y in adj.ORIGINS:
            rules[(arm, y)] = _synth_series(
                rng, adj.BARS_PER_YEAR[y], 0.0)
    b4 = {}
    for y in adj.ORIGINS:
        for s in adj.SEEDS:
            b4[(y, s)] = _synth_series(
                rng, adj.BARS_PER_YEAR[y], drift_b4)
    return b4, rules


def test_e12_adjudicator_strong_candidate_advances(monkeypatch):
    monkeypatch.setattr(adj, "BOOT_B", 300)
    b4, rules = _synth_population(0.004)
    out = adj.adjudicate(b4, rules, 111, 6, 1e-4, True)
    assert out["verdict"] == "ADVANCES"
    assert out["g1_votes"]["pass"] is True
    assert out["spa"]["p_consistent"] <= 0.05
    assert out["bootstrap"]["seed"] == 20260824


def test_e12_adjudicator_null_candidate_does_not_advance(
        monkeypatch):
    monkeypatch.setattr(adj, "BOOT_B", 300)
    b4, rules = _synth_population(0.0)
    out = adj.adjudicate(b4, rules, 111, 6, 1e-4, True)
    assert out["verdict"] == "DOES_NOT_ADVANCE"
    assert out["failed_conditions"]


def test_e12_broken_pairing_refuses(monkeypatch):
    b4, rules = _synth_population(0.004)
    b4[(2023, 202)] = b4[(2023, 202)][:100]
    with pytest.raises(SystemExit, match="broken per-bar pairing"):
        adj.adjudicate(b4, rules, 111, 6, 1e-4, True)


def test_e12_missing_comparator_arm_refuses():
    b4, rules = _synth_population(0.004)
    del rules[("B2a", 2023)]
    with pytest.raises(SystemExit, match="B2a@2023 missing"):
        adj.adjudicate(b4, rules, 111, 6, 1e-4, True)


def test_e12_nonfinite_refuses():
    b4, rules = _synth_population(0.004)
    b4[(2022, 101)][5] = float("nan")
    with pytest.raises(SystemExit, match="non-finite"):
        adj.adjudicate(b4, rules, 111, 6, 1e-4, True)


def test_e12_altered_per_bar_record_refuses(tmp_path):
    f = tmp_path / "pb.csv"
    f.write_text("net_return\n0.001\n0.002\n")
    good = adj._sha_file(f)
    f.write_text("net_return\n0.9\n0.9\n")
    with pytest.raises(SystemExit, match="ALTERED"):
        adj._per_bar_net(f, good, "x")


def test_e12_bootstrap_contract_constants_frozen():
    """Mutation anchor: doc-41 constants are the contract — altering
    seed, B or the block-length source is a chain-visible change."""
    assert adj.BOOT_B == 10_000
    assert adj.BOOT_SEED == 20260824
    assert adj.ALPHA == 0.05
    import inspect
    body = inspect.getsource(adj.adjudicate)
    assert "politis_white_block_length(control)" in body


def test_e12_block_length_from_control_only():
    rng = _np.random.default_rng(3)
    x = rng.normal(0, 0.01, 2190)
    b1 = adj.politis_white_block_length(x)
    b2 = adj.politis_white_block_length(x)
    assert b1 == b2 and b1["block_length"] >= 1.0
    with pytest.raises(SystemExit, match="too short"):
        adj.politis_white_block_length(x[:50])


# ============ Order @0ce52740: C1-C8 acceptance battery ============
ORCH = REPO / "tools" / "b4_campaign_orchestrator.py"
_orch_spec = importlib.util.spec_from_file_location("b4orch", ORCH)
orch = importlib.util.module_from_spec(_orch_spec)
_orch_spec.loader.exec_module(orch)
rl_mod = importlib.import_module(
    "pipeline_plugins.rl_pipeline_with_validation")


def _race(target, n, args_list):
    import multiprocessing as mp
    barrier = mp.Barrier(n)
    q = mp.Queue()
    procs = [mp.Process(target=target,
                        args=(barrier, q) + tuple(args_list[i]))
             for i in range(n)]
    [p.start() for p in procs]
    [p.join() for p in procs]
    return sorted(q.get() for _ in range(n))


def _terminal_racer(barrier, q, out_root, tag):
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4exec_race", str(EXEC))
    ex = ilu.module_from_spec(spec)
    spec.loader.exec_module(ex)
    barrier.wait()
    try:
        ex.write_terminal(Path(out_root), "o2022_seed101", "FAILED",
                          {"reason": tag})
        q.put(("success", tag))
    except SystemExit:
        q.put(("refused", tag))


def test_c5_terminal_race_exactly_one_winner(tmp_path):
    outcomes = _race(_terminal_racer, 2,
                     [(str(tmp_path), "A"), (str(tmp_path), "B")])
    kinds = [k for k, _ in outcomes]
    assert kinds.count("success") == 1 and \
        kinds.count("refused") == 1
    term = json.loads((tmp_path / "o2022_seed101" /
                       "B4_CELL_TERMINAL.json").read_text())
    winner = [t for k, t in outcomes if k == "success"][0]
    assert term["reason"] == winner


def _claim_racer(barrier, q, results_root, tag):
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("b4orch_race", str(ORCH))
    om = ilu.module_from_spec(spec)
    spec.loader.exec_module(om)
    barrier.wait()
    try:
        om.claim_attempt(Path(results_root), "o2023_seed101")
        q.put(("claimed", tag))
    except SystemExit:
        q.put(("refused", tag))


def test_c4_attempt_claim_race_exactly_one(tmp_path):
    outcomes = _race(_claim_racer, 2,
                     [(str(tmp_path), "A"), (str(tmp_path), "B")])
    kinds = [k for k, _ in outcomes]
    assert kinds.count("claimed") == 1 and \
        kinds.count("refused") == 1
    claims = list((tmp_path / "o2023_seed101").glob("CLAIM_*.json"))
    assert len(claims) == 1


def test_c10_two_hundred_fresh_root_races(tmp_path):
    """C9-C16 §11.3: 200 fresh-root two-process claim races, zero
    double winners."""
    doubles = 0
    for i in range(200):
        root = tmp_path / f"r{i}"
        outcomes = _race(_claim_racer, 2,
                         [(str(root), "A"), (str(root), "B")])
        kinds = [k for k, _ in outcomes]
        if kinds.count("claimed") != 1:
            doubles += 1
    assert doubles == 0


def test_c4_crash_after_claim_blocks_cell(tmp_path):
    claim = orch.claim_attempt(tmp_path, "o2024_seed202")
    assert claim["attempt_id"].startswith("attempt_")
    # crash: no terminal -> AMBIGUOUS; a second claim refuses on the
    # fixed per-cell path, and adjudication blocks
    with pytest.raises(SystemExit, match="already exists"):
        orch.claim_attempt(tmp_path, "o2024_seed202")
    assert orch.adjudicate_cell_state(
        tmp_path, "o2024_seed202") == "AMBIGUOUS_CLAIM"


def test_c13_global_boundary_exact_below_above(tmp_path):
    gen = a.CAMPAIGN_GENERATION
    limits = {"global_gpu_hours_ceiling": 96.0}
    d = tmp_path / "o2022_seed101"
    d.mkdir(parents=True)
    tp = d / "B4_CELL_TERMINAL.json"
    for hours, expect_dispatch in ((95.0, True),
                                   (96.0, False),
                                   (96.1, False)):
        os.chmod(d, 0o700)
        _ctl_write(tp, json.dumps(
            {"terminal": "FAILED",
             "wall_seconds": hours * 3600.0}))
        _ctl_write(d / f"CLAIM_{gen}.json", json.dumps(
            _signed_claim(
                {"schema": "agent_multi.b4_attempt_claim.v2",
                 "campaign_generation": gen,
                 "attempt_id": "attempt_x",
                 "cell": "o2022_seed101",
                 "claimed_wall": 0.0, "claimed_monotonic": 0.0,
                 "holder_pid": os.getpid(),
                 "terminal_sha256": None})))
        remaining = orch.remaining_global_seconds(tmp_path, limits)
        can = remaining >= orch.MIN_SEGMENT_SECONDS
        assert can is expect_dispatch, (hours, remaining)
    # intrasegment: the effective cell wall is min(cell, remaining)
    assert executor_effective_wall(43200.0, 3600.0) == 3600.0
    assert executor_effective_wall(43200.0, 50000.0) == 43200.0


def executor_effective_wall(cell_wall, remaining):
    return float(min(cell_wall, remaining))


def test_c13_malformed_duration_fails_closed(tmp_path):
    gen = a.CAMPAIGN_GENERATION
    d = tmp_path / "o2022_seed101"
    d.mkdir(parents=True)
    os.chmod(d, 0o700)
    _ctl_write(d / f"CLAIM_{gen}.json", json.dumps(_signed_claim(
        {"schema": "agent_multi.b4_attempt_claim.v2",
         "campaign_generation": gen, "attempt_id": "x",
         "cell": "o2022_seed101", "claimed_wall": 0.0,
         "claimed_monotonic": 0.0, "holder_pid": os.getpid(),
         "terminal_sha256": None})))
    _ctl_write(d / "B4_CELL_TERMINAL.json", json.dumps(
        {"terminal": "FAILED", "wall_seconds": "twelve"}))
    with pytest.raises(SystemExit, match="malformed terminal"):
        orch.gpu_seconds_spent(tmp_path)


def test_c4_global_lock_is_exclusive(tmp_path):
    with orch.GlobalLock(tmp_path):
        with pytest.raises(SystemExit, match="exactly one winner"):
            with orch.GlobalLock(tmp_path):
                pass


def test_c11_dry_run_zero_writes(tmp_path, monkeypatch):
    """§11.4: dry-run on fresh AND preexisting roots writes NOTHING;
    a nonexistent root still permits a pure dry-run."""
    mat = Path.home() / (".local/share/agent-multi/"
                         "b4_materialization_v5_20260906")
    dr = Path.home() / (".local/share/agent-multi/"
                        "b4_campaign_dryrun_v5_20260906")
    if not mat.is_dir() or not (dr / "CAMPAIGN_LEDGER.json"
                                ).is_file():
        pytest.skip("v5 materialization absent on this host")
    fresh = tmp_path / "fresh_root_never_created"
    rc = orch.run_campaign(mat, dr / "CAMPAIGN_LEDGER.json",
                           fresh, "cpu", execute=False)
    assert rc == 0 and not fresh.exists()
    pre_root = tmp_path / "pre"
    (pre_root / "o2022_seed101").mkdir(parents=True)
    marker = pre_root / "o2022_seed101" / "x.json"
    marker.write_text("{}")
    snap = orch._snapshot(pre_root)
    rc = orch.run_campaign(mat, dr / "CAMPAIGN_LEDGER.json",
                           pre_root, "cpu", execute=False)
    assert rc == 0 and orch._snapshot(pre_root) == snap


def test_c12_lease_bypass_impossible(tmp_path, monkeypatch):
    """§11.5: direct executor, fake claim, stale lease -> zero
    compute. Layer 1: no authorization refuses first. Layer 2: with
    a mocked authorization, the lease gate refuses next."""
    with pytest.raises(SystemExit,
                       match="campaign authorization record"):
        executor.execute_cell("o2024_seed101", tmp_path, tmp_path,
                              "cpu", lease_path=None)
    authf = tmp_path / "auth.json"
    authf.write_text("{}")
    monkeypatch.setattr(executor, "CAMPAIGN_AUTH_SHA", "f" * 64)
    monkeypatch.setattr(executor, "CAMPAIGN_AUTH_PATH", authf)
    monkeypatch.setattr(executor.b4a,
                        "verify_campaign_authorization_record",
                        lambda *a_, **k_: {})
    with pytest.raises(SystemExit, match="no execution lease"):
        executor.execute_cell("o2024_seed101", tmp_path, tmp_path,
                              "cpu", lease_path=None)
    fake = tmp_path / "LEASE_forged.json"
    fake.write_text(json.dumps(
        {"campaign_generation": a.CAMPAIGN_GENERATION,
         "cell": "o2024_seed101", "attempt_id": "attempt_forged"}))
    # C24: a public-mode lease dies at the earliest layer
    with pytest.raises(SystemExit, match="not the private"):
        orch.verify_lease(fake, tmp_path, "o2024_seed101", tmp_path)
    fake.unlink()
    _ctl_write(fake, json.dumps(
        {"campaign_generation": a.CAMPAIGN_GENERATION,
         "cell": "o2024_seed101", "attempt_id": "attempt_forged"}))
    with pytest.raises(SystemExit, match="exact schema"):
        orch.verify_lease(fake, tmp_path, "o2024_seed101", tmp_path)
    # a WELL-FORMED lease still refuses without its claim
    (tmp_path / "B4_MATERIALIZATION.json").write_text("{}")
    (tmp_path / "o2024_seed101").mkdir(mode=0o700,
                                       exist_ok=True)
    os.chmod(tmp_path / "o2024_seed101", 0o700)
    good = orch.issue_lease(
        tmp_path, "o2024_seed101",
        {"attempt_id": "attempt_foreign"}, "e" * 64, tmp_path)
    with pytest.raises(SystemExit,
                       match="claim for o2024_seed101 absent"):
        orch.verify_lease(good, tmp_path, "o2024_seed101", tmp_path)
    claim = orch.claim_attempt(tmp_path, "o2024_seed101")
    with pytest.raises(SystemExit, match="attempt differs"):
        orch.verify_lease(good, tmp_path, "o2024_seed101", tmp_path)
    # C17: matching claim but a FOREIGN authorization digest
    lease2 = orch.issue_lease(tmp_path, "o2024_seed101", claim,
                              "e" * 64, tmp_path)
    with pytest.raises(SystemExit,
                       match="authorization digest differs"):
        orch.verify_lease(lease2, tmp_path, "o2024_seed101",
                          tmp_path, expected_auth_sha="f" * 64)
    # C17: no live lock -> no capability
    with pytest.raises(SystemExit, match="live campaign"):
        orch.verify_lease(lease2, tmp_path, "o2024_seed101",
                          tmp_path)


def test_c15_resume_adjudicates_every_class(tmp_path):
    """§11.9: resume over every terminal class and uncertain
    state."""
    gen = a.CAMPAIGN_GENERATION
    st = orch.adjudicate_cell_state
    assert st(tmp_path, "o2022_seed101") == "PENDING"
    c1 = orch.claim_attempt(tmp_path, "o2022_seed101")
    assert st(tmp_path, "o2022_seed101") == "AMBIGUOUS_CLAIM"
    executor.write_terminal(tmp_path, "o2022_seed101", "FAILED",
                            {"attempt_id": c1["attempt_id"],
                             "reason": "x", "wall_seconds": 1.0})
    assert st(tmp_path, "o2022_seed101") == "UNCERTAIN"  # unsealed
    orch.seal_attempt(tmp_path, "o2022_seed101",
                      c1["attempt_id"])
    assert st(tmp_path,
              "o2022_seed101") == "TERMINAL_FAILED"
    # malformed terminal -> UNCERTAIN (rewrite AFTER seal)
    tp = tmp_path / "o2022_seed101" / "B4_CELL_TERMINAL.json"
    tp.chmod(0o644)
    tp.write_text("{broken")
    assert st(tmp_path, "o2022_seed101") == "UNCERTAIN"


def test_c16_seal_without_terminal_refuses(tmp_path):
    claim = orch.claim_attempt(tmp_path, "o2023_seed202")
    with pytest.raises(SystemExit, match="UNCERTAIN"):
        orch.seal_attempt(tmp_path, "o2023_seed202",
                          claim["attempt_id"])
    leftover = list((tmp_path / "o2023_seed202"
                     ).glob("CLAIM_*.json"))
    assert len(leftover) == 1   # the claim persists as AMBIGUOUS


def test_c16_telemetry_cadence_governed(monkeypatch):
    """§11.10: with a controlled clock the resource probe fires at
    the declared cadence only — never per environment step."""
    calls = {"n": 0}
    monkeypatch.setattr(rl_mod, "_check_resource_budget",
                        lambda cfg, **k: calls.__setitem__(
                            "n", calls["n"] + 1))
    clock = {"t": 100.0}
    monkeypatch.setattr(rl_mod._time_mod, "monotonic",
                        lambda: clock["t"])
    cfg = {"budget_max_env_steps": 10 ** 9,
           "resource_sample_seconds": 5.0}
    cb = rl_mod.make_executing_budget_callback(cfg, 0.0)
    import types
    cb.model = types.SimpleNamespace(_n_updates=0, num_timesteps=0)
    for step in range(100):
        clock["t"] += 0.05          # 100 steps over 5 seconds
        cb.num_timesteps = step
        assert cb._on_step() is True
    assert calls["n"] == 1, calls   # ONE due sample, not 100


def test_c16_presegment_forces_resource_check(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(rl_mod, "_check_resource_budget",
                        lambda cfg, **k: calls.__setitem__(
                            "n", calls["n"] + 1))
    rl_mod._check_executing_budget(
        {"budget_max_env_steps": 10 ** 9},
        __import__("types").SimpleNamespace(_n_updates=0,
                                            num_timesteps=0),
        started_wall=0.0, next_segment_timesteps=1000)
    assert calls["n"] == 1
    rl_mod._check_executing_budget(
        {"budget_max_env_steps": 10 ** 9},
        __import__("types").SimpleNamespace(_n_updates=0,
                                            num_timesteps=0),
        started_wall=0.0, next_segment_timesteps=0)
    assert calls["n"] == 1          # NOT called on the step path


def test_c3_resource_guard_rss_and_lost_telemetry(monkeypatch):
    with pytest.raises(rl_mod.ExecutingBudgetExceeded,
                       match="RSS budget"):
        rl_mod._check_resource_budget(
            {"budget_max_rss_bytes": 1})
    monkeypatch.setattr(rl_mod, "_read_gpu_temp", lambda d: None)
    with pytest.raises(rl_mod.ExecutingBudgetExceeded,
                       match="failing CLOSED"):
        rl_mod._check_resource_budget(
            {"budget_max_gpu_temp_celsius": 87.0,
             "budget_gpu_device": "0"})
    with pytest.raises(rl_mod.ExecutingBudgetExceeded,
                       match="without a bound"):
        rl_mod._check_resource_budget(
            {"budget_max_gpu_temp_celsius": 87.0})


def test_c3_poisoned_cell_limit_is_invisible(tmp_path):
    """A mutation putting laxer limits into the cell's gpu_economic
    mode must be INVISIBLE: effective limits derive from the
    authorized contract only."""
    limits = a.load_resource_contract()
    assert limits["budget_max_wall_seconds"] == 43200.0
    assert limits["budget_max_rss_bytes"] == 8 * 1024 ** 3
    assert limits["budget_max_cuda_bytes"] == 6 * 1024 ** 3
    assert limits["budget_max_gpu_temp_celsius"] == 87.0
    assert limits["global_gpu_hours_ceiling"] == 96.0
    src_exec = EXEC.read_text()
    body = src_exec[src_exec.index("def build_economic_config"):
                    src_exec.index("def reconcile_per_bar")]
    assert 'econ["budget_max' not in body
    assert "load_resource_contract" in body


def test_c2_shifted_series_refuses_by_identity():
    idents_ok = {(y, s): [f"2024-01-0{(i % 9) + 1} 00:00"
                          for i in range(10)]
                 for y in adj.ORIGINS for s in adj.SEEDS}
    rule_idents = {(arm, y): list(idents_ok[(y, 101)])
                   for arm in adj.RULE_ARMS for y in adj.ORIGINS}
    b4, rules = _synth_population(0.004)
    b4 = {k: v[:10] for k, v in b4.items()}
    rules = {k: v[:10] for k, v in rules.items()}
    b4["__identities__"] = dict(idents_ok)
    b4["__identities__"][(2023, 202)] = \
        ["2024-01-02 00:00"] + idents_ok[(2023, 202)][:-1]
    rules["__identities__"] = rule_idents
    with pytest.raises(SystemExit, match="bar-identity vector"):
        adj.adjudicate(b4, rules, 111, 6, 1e-4, True)


def test_c16_independent_reconciliation_refuses(tmp_path):
    import pandas as pd
    good = pd.DataFrame({
        "economic_equity": [100.0, 101.0, 102.5],
        "net_equity_delta_observed": [0.0, 1.0, 1.5],
        "env_pnl_fact": [0.0, 1.0, 1.5],
        "commission_delta": [0.0, 0.1, 0.1]})
    executor.reconcile_per_bar(good)
    bad = good.copy()
    bad.loc[2, "env_pnl_fact"] = 9.9   # env fact disagrees
    with pytest.raises(SystemExit, match="conservation broken"):
        executor.reconcile_per_bar(bad)
    neg = good.copy()
    neg.loc[1, "commission_delta"] = -0.5
    with pytest.raises(SystemExit, match="commission counter"):
        executor.reconcile_per_bar(neg)


def test_c2_one_artifact_cannot_be_two_results(tmp_path,
                                               monkeypatch):
    d = tmp_path / "res"
    pb = tmp_path / "shared.csv"
    pb.write_text("datetime_utc,net_return\n2024-01-01 00:00,0.0\n")
    sha = a._sha_file(pb)
    for cid in ("o2022_seed101", "o2022_seed202"):
        cd = d / cid
        cd.mkdir(parents=True)
        (cd / "B4_CELL_TERMINAL.json").write_text(json.dumps(
            {"per_bar_csv": str(pb), "per_bar_sha256": sha}))
    monkeypatch.setattr(
        adj, "ORIGINS", (2022,))
    monkeypatch.setattr(adj, "SEEDS", (101, 202))
    import b4_campaign_ledger as lm
    monkeypatch.setattr(lm, "verify_campaign_results",
                        lambda *a_, **k_: None)
    with pytest.raises(SystemExit, match="cannot be two results"):
        adj.load_campaign_results(tmp_path / "ledger.json",
                                  tmp_path, d)


def test_c7_executable_path_facts_all_true():
    facts = executor._executable_path_facts()
    assert all(facts.values()), facts


def test_f7_no_absolute_paths_guard():
    with pytest.raises(SystemExit, match="absolute local path"):
        a.verify_no_absolute_paths(
            {"csv": "/home/someone/data.csv"})
    a.verify_no_absolute_paths(
        {"csv": "contracts/x.json",
         "ref": "predictor:examples/data/x.csv"})


def test_c14_null_seal_refuses(tmp_path, monkeypatch):
    """A6 POST: an unsealed attempt is UNCERTAIN, never accepted."""
    led, results = _ledger_fixture(tmp_path)
    for w in (results / "o2022_seed101").glob("SEAL_COMPLETE_*"):
        w.unlink()
    with pytest.raises(SystemExit, match="UNSEALED or UNCERTAIN"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_c14_one_row_forgery_refuses(tmp_path, monkeypatch):
    """A6 POST: the fabricated one-row per-bar file dies on exact
    cardinality for every origin."""
    led, results = _ledger_fixture(tmp_path)
    cid = "o2024_seed101"
    pb = results / cid / f"per_bar_{cid}.csv"
    pb.write_text("net_return\n999\n")
    tp = results / cid / "B4_CELL_TERMINAL.json"
    term = json.loads(tp.read_text())
    term["per_bar_sha256"] = ledger_mod._sha_file(pb)
    _ctl_write(tp, json.dumps(term))
    _reseal(results, cid)
    with pytest.raises(SystemExit,
                       match="required columns|scored rows"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_c14_shifted_identity_refuses_vs_comparator(
        tmp_path, monkeypatch):
    """§11.8: a shifted bar-identity vector refuses against the
    comparator arms."""
    led, results = _ledger_fixture(tmp_path)
    comp_dir = tmp_path / "comp"
    comp_dir.mkdir()
    import pandas as _pd3
    arms = []
    for y in (2022, 2023, 2024):
        n = {2022: 2190, 2023: 2190, 2024: 2196}[y]
        cp = comp_dir / f"B0_{y}.csv"
        _pd3.DataFrame({
            "datetime": _pd3.date_range("2000-01-01", periods=n,
                                        freq="4h"),
            "net_return": [0.0] * n}).to_csv(cp, index=False)
        arms.append({"arm": "B0", "origin": y,
                     "per_bar_csv": str(cp),
                     "per_bar_sha256": ledger_mod._sha_file(cp)})
    (comp_dir / "SCREEN_B_RESULTS.json").write_text(json.dumps(
        {"results": arms}))
    with pytest.raises(SystemExit, match="differs from comparator"):
        _check_results_with_comp(led, tmp_path, results,
                                 comp_dir, monkeypatch)


def _check_results_with_comp(led, tmp_path, results, comp,
                             monkeypatch):
    _patch_universe(led, tmp_path, monkeypatch)
    return ledger_mod.verify_campaign_results(
        tmp_path / "ledger.json", tmp_path, results,
        comparator_dir=comp)


# ================= C17-C22 acceptance battery (§7) =================

def _ctl_write(path, text):
    """Test harness: (re)write a control object PRIVATE (0600)."""
    p = Path(path)
    if p.exists():
        p.unlink()
    fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, text.encode()
                 if isinstance(text, str) else text)
    finally:
        os.close(fd)


def _signed_claim(rec):
    body = {k: rec[k] for k in sorted(rec) if k != "claim_sha256"}
    rec["claim_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    return rec


def _live_lock(root, orch_mod=None):
    om = orch_mod or orch
    Path(root).mkdir(mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    rec = {"schema": om.LOCK_SCHEMA_NAME,
           "generation": a.CAMPAIGN_GENERATION,
           "epoch": 1, "holder_pid": os.getpid(),
           "acquire_id": "harness0000000000"}
    rec["lock_sha256"] = om._self_sha(rec, "lock_sha256")
    _ctl_write(Path(root) / "LOCK_EPOCH_1.json",
               json.dumps(rec, indent=1))


def _resign_lease(doc):
    body = {k: doc[k] for k in sorted(doc) if k != "lease_sha256"}
    doc["lease_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    return doc


def test_c17_foreign_lease_variants_die(tmp_path):
    """§7.1: the exact PRE forgery and every re-signed foreign
    variant refuse BEFORE any compute construction."""
    (tmp_path / "B4_MATERIALIZATION.json").write_text("{}")
    _live_lock(tmp_path)
    claim = orch.claim_attempt(tmp_path, "o2024_seed101")
    lease_p = orch.issue_lease(tmp_path, "o2024_seed101", claim,
                               "a" * 64, tmp_path)
    # sanity: the honest lease verifies
    ok = orch.verify_lease(lease_p, tmp_path, "o2024_seed101",
                           tmp_path, expected_auth_sha="a" * 64)
    assert ok["attempt_id"] == claim["attempt_id"]
    base = json.loads(lease_p.read_text())
    # 1) the PRE mutation verbatim: the foreign schema token now
    # dies at the earliest layer
    doc = dict(base)
    doc["schema"] = "attacker.anything.v9"
    doc["authorization_sha256"] = "b" * 64
    doc["holder_pid"] = 999999
    _ctl_write(lease_p, json.dumps(doc))
    with pytest.raises(SystemExit, match="foreign lease schema"):
        orch.verify_lease(lease_p, tmp_path, "o2024_seed101",
                          tmp_path)
    # 1b) unsigned content mutation (schema intact): digest breaks
    doc = dict(base)
    doc["authorization_sha256"] = "b" * 64
    doc["holder_pid"] = 999999
    _ctl_write(lease_p, json.dumps(doc))
    with pytest.raises(SystemExit, match="altered"):
        orch.verify_lease(lease_p, tmp_path, "o2024_seed101",
                          tmp_path)
    # 2) re-signed foreign schema
    doc = _resign_lease({**base, "schema": "attacker.v9"})
    _ctl_write(lease_p, json.dumps(doc))
    with pytest.raises(SystemExit, match="foreign lease schema"):
        orch.verify_lease(lease_p, tmp_path, "o2024_seed101",
                          tmp_path)
    # 3) re-signed foreign authorization digest
    doc = _resign_lease({**base, "authorization_sha256": "b" * 64})
    _ctl_write(lease_p, json.dumps(doc))
    with pytest.raises(SystemExit,
                       match="authorization digest differs"):
        orch.verify_lease(lease_p, tmp_path, "o2024_seed101",
                          tmp_path, expected_auth_sha="a" * 64)
    # 4) re-signed foreign holder pid
    doc = _resign_lease({**base, "holder_pid": 999999})
    _ctl_write(lease_p, json.dumps(doc))
    with pytest.raises(SystemExit, match="holder identity"):
        orch.verify_lease(lease_p, tmp_path, "o2024_seed101",
                          tmp_path)
    # 5) restored honest lease but the lock vanished
    _ctl_write(lease_p, json.dumps(base))
    (tmp_path / "LOCK_EPOCH_1.json").unlink()
    with pytest.raises(SystemExit, match="live campaign"):
        orch.verify_lease(lease_p, tmp_path, "o2024_seed101",
                          tmp_path)


def _seal_root(tmp_path, name):
    root = tmp_path / name
    c = orch.claim_attempt(root, "o2022_seed101")
    executor.write_terminal(root, "o2022_seed101", "FAILED", {
        "attempt_id": c["attempt_id"], "reason": "x",
        "wall_seconds": 1.0})
    return root, c


def test_c18_seal_fsync_outcome_matrix(tmp_path, monkeypatch):
    """§7.2: BOTH physical outcomes of a failed completion write.
    The caller always sees failure; a fresh adjudication reads the
    PHYSICAL state — persisted bytes seal, absent/partial bytes
    stay UNCERTAIN, and uncertain never becomes success."""
    real = orch._excl_write

    def failing(path, payload, mode=0o600, physical="persist"):
        if "SEAL_COMPLETE" in Path(path).name:
            if physical == "persist":
                real(path, payload, mode)
            elif physical == "partial":
                real(path, payload[: len(payload) // 2], mode)
            raise OSError("injected completion write failure")
        return real(path, payload, mode)

    # outcome A: bytes persisted although the caller saw OSError
    rootA, cA = _seal_root(tmp_path, "A")
    monkeypatch.setattr(
        orch, "_excl_write",
        lambda p, b, m=0o600: failing(p, b, m, "persist"))
    with pytest.raises(OSError, match="injected"):
        orch.seal_attempt(rootA, "o2022_seed101", cA["attempt_id"])
    monkeypatch.setattr(orch, "_excl_write", real)
    assert orch.seal_state(rootA, "o2022_seed101") == "SEALED"
    assert orch.adjudicate_cell_state(
        rootA, "o2022_seed101") == "TERMINAL_FAILED"
    # outcome B: bytes did NOT persist -> UNCERTAIN, blocked
    rootB, cB = _seal_root(tmp_path, "B")
    monkeypatch.setattr(
        orch, "_excl_write",
        lambda p, b, m=0o600: failing(p, b, m, "absent"))
    with pytest.raises(OSError, match="injected"):
        orch.seal_attempt(rootB, "o2022_seed101", cB["attempt_id"])
    monkeypatch.setattr(orch, "_excl_write", real)
    assert orch.seal_state(rootB, "o2022_seed101") == "UNCERTAIN"
    assert orch.adjudicate_cell_state(
        rootB, "o2022_seed101") == "UNCERTAIN"
    # outcome C: HALF the bytes persisted -> UNCERTAIN, blocked
    rootC, cC = _seal_root(tmp_path, "C")
    monkeypatch.setattr(
        orch, "_excl_write",
        lambda p, b, m=0o600: failing(p, b, m, "partial"))
    with pytest.raises(OSError, match="injected"):
        orch.seal_attempt(rootC, "o2022_seed101", cC["attempt_id"])
    monkeypatch.setattr(orch, "_excl_write", real)
    assert orch.seal_state(rootC, "o2022_seed101") == "UNCERTAIN"
    assert orch.adjudicate_cell_state(
        rootC, "o2022_seed101") == "UNCERTAIN"
    # transplanted completion from A into B -> UNCERTAIN
    rootD, cD = _seal_root(tmp_path, "D")
    src_c = next((rootA / "o2022_seed101").glob("SEAL_COMPLETE_*"))
    dst = (rootD / "o2022_seed101" /
           f"SEAL_COMPLETE_{cD['attempt_id']}.json")
    intent = (rootD / "o2022_seed101" /
              f"SEAL_INTENT_{cD['attempt_id']}.json")
    _ctl_write(intent, src_c.read_text())
    _ctl_write(dst, src_c.read_text())
    assert orch.seal_state(rootD, "o2022_seed101") == "UNCERTAIN"


def test_c19_lock_acquire_release_matrix(tmp_path):
    """§7.2/§7.3 (C23): monotone epochs — one holder; witnessed
    in-place release; uncertain release blocks; released epoch
    reclaimable exactly once per contender; nothing is ever
    unlinked or auto-stolen."""
    os.chmod(tmp_path, 0o700)
    lk = orch.GlobalLock(tmp_path)
    lk.__enter__()
    assert lk.epoch == 1
    # second contender refuses while held
    with pytest.raises(SystemExit, match="HELD"):
        orch.GlobalLock(tmp_path).__enter__()
    # a NON-holder object cannot release
    thief = orch.GlobalLock(tmp_path)
    thief.held = True
    thief.epoch = 1
    thief.acquire_id = "0" * 16
    with pytest.raises(SystemExit, match="non-holder"):
        thief.__exit__()
    assert orch.current_lock_epoch(tmp_path)["state"] == "HELD"
    # owned release transitions IN PLACE to RELEASED (no unlink)
    lk.__exit__()
    st = orch.current_lock_epoch(tmp_path)
    assert st["state"] == "RELEASED" and st["epoch"] == 1
    assert (tmp_path / "LOCK_EPOCH_1.json").exists()
    assert (tmp_path / "LOCK_RELEASE_INTENT_1.json").exists()
    assert (tmp_path / "LOCK_RELEASE_COMPLETE_1.json").exists()
    # released epoch is reclaimable as epoch 2
    lk2 = orch.GlobalLock(tmp_path)
    lk2.__enter__()
    assert lk2.epoch == 2
    # same-uid deletion of the CURRENT lock (the 0700 root
    # excludes everyone else): the HOLDER can never silently
    # succeed — its release adjudicates the epoch and fails closed
    (tmp_path / "LOCK_EPOCH_2.json").unlink()
    with pytest.raises(SystemExit, match="non-holder|uncertain"):
        lk2.__exit__()          # holder fails closed, no silent ok
    # crashed holder (held epoch, dead pid) is never auto-stolen
    root2 = tmp_path / "crashed"
    root2.mkdir(mode=0o700)
    rec = {"schema": orch.LOCK_SCHEMA_NAME,
           "generation": a.CAMPAIGN_GENERATION,
           "epoch": 1, "holder_pid": 999999,
           "acquire_id": "dead000000000000"}
    rec["lock_sha256"] = orch._self_sha(rec, "lock_sha256")
    _ctl_write(root2 / "LOCK_EPOCH_1.json", json.dumps(rec))
    with pytest.raises(SystemExit, match="HELD"):
        orch.GlobalLock(root2).__enter__()


def test_c23_uncertain_release_blocks_two_real_processes(
        tmp_path, monkeypatch):
    """C23 acceptance: the exact PRE sequence — failed FINAL
    durability during release — leaves the epoch RELEASING; a
    second REAL process cannot enter; a fully durable release
    remains reclaimable by a real process."""
    import multiprocessing as mp2
    os.chmod(tmp_path, 0o700)
    lk = orch.GlobalLock(tmp_path)
    lk.__enter__()
    real = orch._excl_write

    def completion_lost(path, payload, mode=0o600):
        if "LOCK_RELEASE_COMPLETE" in Path(path).name:
            raise OSError("injected completion write failure")
        return real(path, payload, mode)

    monkeypatch.setattr(orch, "_excl_write", completion_lost)
    with pytest.raises(SystemExit, match="uncertain release"):
        lk.__exit__()
    monkeypatch.setattr(orch, "_excl_write", real)
    assert orch.current_lock_epoch(tmp_path)["state"] == \
        "RELEASING"
    ctx = mp2.get_context("fork")
    q = ctx.Queue()

    def _enter(q2):
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "b4orch_c23", Path(__file__).resolve().parents[1]
            / "tools/b4_campaign_orchestrator.py")
        om = ilu.module_from_spec(spec)
        spec.loader.exec_module(om)
        try:
            om.GlobalLock(tmp_path).__enter__()
            q2.put(("entered", None))
        except SystemExit as exc:
            q2.put(("refused", str(exc)))

    pr = ctx.Process(target=_enter, args=(q,))
    pr.start()
    pr.join(30)
    kind, msg = q.get(timeout=5)
    assert kind == "refused" and "UNCERTAIN" in msg
    # the OTHER physical outcome: completion bytes persisted even
    # though the caller saw an error -> RELEASED and reclaimable
    # by a fresh real process
    root2 = tmp_path / "persisted"
    root2.mkdir(mode=0o700)
    lkB = orch.GlobalLock(root2)
    lkB.__enter__()

    def completion_persists_then_errors(path, payload,
                                        mode=0o600):
        real(path, payload, mode)
        if "LOCK_RELEASE_COMPLETE" in Path(path).name:
            raise OSError("injected post-persist failure")

    monkeypatch.setattr(orch, "_excl_write",
                        completion_persists_then_errors)
    with pytest.raises(SystemExit, match="uncertain release"):
        lkB.__exit__()
    monkeypatch.setattr(orch, "_excl_write", real)
    assert orch.current_lock_epoch(root2)["state"] == "RELEASED"
    q2 = ctx.Queue()

    def _enter2(qq):
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "b4orch_c23b", Path(__file__).resolve().parents[1]
            / "tools/b4_campaign_orchestrator.py")
        om = ilu.module_from_spec(spec)
        spec.loader.exec_module(om)
        try:
            got = om.GlobalLock(root2)
            got.__enter__()
            qq.put(("entered", got.epoch))
        except SystemExit as exc:
            qq.put(("refused", str(exc)))

    pr2 = ctx.Process(target=_enter2, args=(q2,))
    pr2.start()
    pr2.join(30)
    kind2, epoch2 = q2.get(timeout=5)
    assert kind2 == "entered" and epoch2 == 2


def test_c24_control_plane_private_and_descriptor_bound(tmp_path):
    """C24 acceptance: modes, foreign objects, symlink swaps and
    smuggled fields all refuse; nothing is silently chmodded."""
    os.chmod(tmp_path, 0o700)
    _live_lock(tmp_path)
    claim = orch.claim_attempt(tmp_path, "o2024_seed101")
    (tmp_path / "B4_MATERIALIZATION.json").write_text("{}")
    lease_p = orch.issue_lease(tmp_path, "o2024_seed101", claim,
                               "a" * 64, tmp_path)
    import stat as _st
    for path in (tmp_path / "LOCK_EPOCH_1.json",
                 tmp_path / "o2024_seed101" /
                 f"CLAIM_{a.CAMPAIGN_GENERATION}.json",
                 lease_p):
        assert _st.S_IMODE(os.stat(path).st_mode) == 0o600
    assert _st.S_IMODE(os.stat(
        tmp_path / "o2024_seed101").st_mode) == 0o700
    # a permissive claim is REFUSED, never chmodded
    cp = (tmp_path / "o2024_seed101" /
          f"CLAIM_{a.CAMPAIGN_GENERATION}.json")
    os.chmod(cp, 0o644)
    with pytest.raises(SystemExit, match="not the private"):
        orch.load_claim(tmp_path, "o2024_seed101")
    assert _st.S_IMODE(os.stat(cp).st_mode) == 0o644  # untouched
    os.chmod(cp, 0o600)
    # symlink substitution refuses at open (O_NOFOLLOW)
    donor = tmp_path / "donor.json"
    donor.write_text(cp.read_text())
    cp.unlink()
    cp.symlink_to(donor)
    with pytest.raises(SystemExit, match="unopenable|symlink"):
        orch.load_claim(tmp_path, "o2024_seed101")
    cp.unlink()
    _ctl_write(cp, donor.read_text())
    # smuggled extra field refuses (exact schema)
    rec = json.loads(cp.read_text())
    rec["attacker"] = True
    _ctl_write(cp, json.dumps(rec))
    with pytest.raises(SystemExit, match=r"exact\s+schema"):
        orch.load_claim(tmp_path, "o2024_seed101")
    # tampered self-digest refuses
    rec.pop("attacker")
    rec["claimed_wall"] = 999.0
    _ctl_write(cp, json.dumps(rec))
    with pytest.raises(SystemExit, match=r"does not\s+re-derive"):
        orch.load_claim(tmp_path, "o2024_seed101")
    # a permissive control DIRECTORY refuses
    root2 = tmp_path / "lax"
    root2.mkdir(mode=0o755)
    with pytest.raises(SystemExit, match="not the private"):
        orch.claim_attempt(root2, "o2022_seed101")


def _contender(root, q):
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4orch_child",
        Path(__file__).resolve().parents[1]
        / "tools/b4_campaign_orchestrator.py")
    om = ilu.module_from_spec(spec)
    spec.loader.exec_module(om)
    try:
        om.claim_attempt(root, "o2022_seed101")
        q.put(("claimed", None))
    except SystemExit as exc:
        q.put(("refused", str(exc)))
    q.put(("state", om.adjudicate_cell_state(root,
                                             "o2022_seed101")))


def test_c18_two_process_contention_after_uncertainty(tmp_path,
                                                      monkeypatch):
    """§7.4: after an uncertain seal boundary, two REAL processes
    both refuse to re-claim and both adjudicate UNCERTAIN."""
    import multiprocessing as mp
    root, c = _seal_root(tmp_path, "U")
    real = orch._excl_write
    monkeypatch.setattr(
        orch, "_excl_write",
        lambda p, b, m=0o600: (_ for _ in ()).throw(
            OSError("injected")) if "SEAL_COMPLETE" in Path(p).name
        else real(p, b, m))
    with pytest.raises(OSError):
        orch.seal_attempt(root, "o2022_seed101", c["attempt_id"])
    monkeypatch.setattr(orch, "_excl_write", real)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    ps = [ctx.Process(target=_contender, args=(root, q))
          for _ in range(2)]
    [p.start() for p in ps]
    [p.join(timeout=60) for p in ps]
    out = [q.get(timeout=5) for _ in range(4)]
    claims = [v for k, v in out if k in ("claimed", "refused")]
    states = [v for k, v in out if k == "state"]
    assert all("one winner" in c or "exactly one" in c
               for c in claims if c)
    assert len([k for k, _ in out if k == "refused"]) == 2
    assert states == ["UNCERTAIN", "UNCERTAIN"]


def test_c20_minimal_terminal_cannot_complete(tmp_path,
                                              monkeypatch):
    """§7.5: the pre-C21 minimal terminal (7 evidence keys) is no
    longer admissible at the final gate."""
    led, results = _ledger_fixture(tmp_path)
    cid = "o2022_seed101"
    tp = results / cid / "B4_CELL_TERMINAL.json"
    term = json.loads(tp.read_text())
    minimal = {k: term[k] for k in (
        "schema", "cell", "terminal", "attempt_id",
        "cell_config_sha256", "per_bar_csv", "per_bar_sha256",
        "scored_index_sha256", "checkpoint_sha256",
        "sealed_2025_used")}
    _ctl_write(tp, json.dumps(minimal))
    _reseal(results, cid)
    with pytest.raises(SystemExit, match="exact schema"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_c20_comparator_not_omittable(tmp_path, monkeypatch):
    """§7.6: comparator evidence is derived and mandatory — absent
    derivation refuses; the CLI carries no omission path; a None
    argument DERIVES instead of skipping."""
    (tmp_path / "B4_MATERIALIZATION.json").write_text(
        json.dumps({"no_comparator": True}))
    with pytest.raises(SystemExit,
                       match="no comparator_ref|comparator"):
        ledger_mod._derive_comparator_dir(tmp_path)
    lsrc = (Path(__file__).resolve().parents[1]
            / "tools/b4_campaign_ledger.py").read_text()
    assert "comparator_dir" not in lsrc[lsrc.index("def main"):]
    body = lsrc[lsrc.index("def verify_campaign_results"):
                lsrc.index("def verify_single_cell_result")]
    assert "_derive_comparator_dir" in body


def test_c20_completion_requires_full_verifier_source():
    """§7.8: run_campaign consumes verify_campaign_results BEFORE
    it may report CAMPAIGN_COMPLETE (the integrated run proves the
    live path end to end)."""
    osrc = (Path(__file__).resolve().parents[1]
            / "tools/b4_campaign_orchestrator.py").read_text()
    body = osrc[osrc.index("def run_campaign"):]
    i_verify = body.index("verify_campaign_results")
    i_complete = body.index('"CAMPAIGN_COMPLETE"')
    assert i_verify < i_complete


def _mutate_perbar(results, cid, mutfn):
    import pandas as _pd4
    d = results / cid
    pb = d / f"per_bar_{cid}.csv"
    df = _pd4.read_csv(pb)
    df = mutfn(df)
    df.to_csv(pb, index=False)
    tp = d / "B4_CELL_TERMINAL.json"
    term = json.loads(tp.read_text())
    term["per_bar_sha256"] = ledger_mod._sha_file(pb)
    _ctl_write(tp, json.dumps(term))
    _reseal(results, cid)


def test_c21_factual_mutations_each_fail(tmp_path, monkeypatch):
    """§7.7: every factual field mutation independently refuses at
    the final gate — presence is not evidence."""
    cid = "o2023_seed202"

    def fresh():
        import shutil as _sh
        for child in tmp_path.iterdir():
            if child.is_dir():
                _sh.rmtree(child)
            else:
                child.unlink()
        return _ledger_fixture(tmp_path)

    # 1. seed column lies
    led, results = fresh()

    def m1(df):
        df["seed"] = df["seed"] + 1
        return df
    _mutate_perbar(results, cid, m1)
    with pytest.raises(SystemExit, match="seed column"):
        _check_results(led, tmp_path, results, monkeypatch)
    # 2. scored_index relative instead of absolute
    led, results = fresh()

    def m2(df):
        df["scored_index"] = range(len(df))
        return df
    _mutate_perbar(results, cid, m2)
    with pytest.raises(SystemExit, match="absolute sequence"):
        _check_results(led, tmp_path, results, monkeypatch)
    # 3. timestamps shifted off the frozen source
    led, results = fresh()

    def m3(df):
        df["datetime_utc"] = list(df["datetime_utc"][1:]) + \
            ["2099-01-01 00:00"]
        return df
    _mutate_perbar(results, cid, m3)
    with pytest.raises(SystemExit, match="frozen source|comparator"):
        _check_results(led, tmp_path, results, monkeypatch)
    # 4. source_row_sha256 fabricated
    led, results = fresh()

    def m4(df):
        df["source_row_sha256"] = ["s" * 64] * len(df)
        return df
    _mutate_perbar(results, cid, m4)
    with pytest.raises(SystemExit, match="does not\\s+recompute"):
        _check_results(led, tmp_path, results, monkeypatch)
    # 5. net_return decoupled from the equity path
    led, results = fresh()

    def m5(df):
        df.loc[10, "net_return"] = 0.5
        return df
    _mutate_perbar(results, cid, m5)
    with pytest.raises(SystemExit, match="net_return does not"):
        _check_results(led, tmp_path, results, monkeypatch)
    # 6. checkpoint bytes differ from the declared digest
    led, results = fresh()
    ck = results / cid / f"checkpoint_{cid}.zip"
    ck.write_bytes(b"tampered-checkpoint-bytes")
    with pytest.raises(SystemExit, match="checkpoint bytes"):
        _check_results(led, tmp_path, results, monkeypatch)
    # 7. checkpoint reused across two cells
    led, results = fresh()
    other = "o2023_seed303"
    tp = results / other / "B4_CELL_TERMINAL.json"
    term = json.loads(tp.read_text())
    donor = json.loads((results / cid /
                        "B4_CELL_TERMINAL.json").read_text())
    term["checkpoint_sha256"] = donor["checkpoint_sha256"]
    term["checkpoint_path"] = donor["checkpoint_path"]
    _ctl_write(tp, json.dumps(term))
    _reseal(results, other)
    with pytest.raises(SystemExit, match="reuses the checkpoint"):
        _check_results(led, tmp_path, results, monkeypatch)


def _c22_harness(tmp_path, monkeypatch, exec_mod, ceiling_hours):
    (tmp_path / "B4_MATERIALIZATION.json").write_text("{}")
    monkeypatch.setattr(exec_mod, "CAMPAIGN_AUTH_SHA", "f" * 64)
    authf = tmp_path / "auth.json"
    authf.write_text("{}")
    monkeypatch.setattr(exec_mod, "CAMPAIGN_AUTH_PATH", authf)
    monkeypatch.setattr(exec_mod.b4a,
                        "verify_campaign_authorization_record",
                        lambda *a_, **k_: {})
    monkeypatch.setattr(
        exec_mod.b4a, "load_resource_contract",
        lambda: {"global_gpu_hours_ceiling": ceiling_hours})
    _live_lock(tmp_path)
    claim = orch.claim_attempt(tmp_path, "o2024_seed101")
    lease = orch.issue_lease(tmp_path, "o2024_seed101", claim,
                             "f" * 64, tmp_path)
    return lease


def test_c22_global_budget_cannot_be_enlarged(tmp_path,
                                              monkeypatch):
    """§7.9: the executor recomputes the remainder at the point of
    use; a caller value can only TIGHTEN it."""
    # exhausted ceiling + huge caller -> still refused
    lease = _c22_harness(tmp_path, monkeypatch, executor, 0.0)
    with pytest.raises(SystemExit, match="hard bound"):
        executor.execute_cell(
            "o2024_seed101", tmp_path, tmp_path, "cpu",
            lease_path=lease,
            global_wall_remaining_seconds=1e12)
    # huge ceiling + small caller -> the caller TIGHTENS
    for child in tmp_path.iterdir():
        import shutil as _sh
        _sh.rmtree(child) if child.is_dir() else child.unlink()
    lease = _c22_harness(tmp_path, monkeypatch, executor, 1000.0)
    with pytest.raises(SystemExit, match="hard bound"):
        executor.execute_cell(
            "o2024_seed101", tmp_path, tmp_path, "cpu",
            lease_path=lease,
            global_wall_remaining_seconds=100.0)
    # huge ceiling + None -> the derived remainder passes this
    # layer and the NEXT gate (materialization) refuses instead
    for child in tmp_path.iterdir():
        import shutil as _sh
        _sh.rmtree(child) if child.is_dir() else child.unlink()
    lease = _c22_harness(tmp_path, monkeypatch, executor, 1000.0)
    try:
        executor.execute_cell(
            "o2024_seed101", tmp_path, tmp_path, "cpu",
            lease_path=lease,
            global_wall_remaining_seconds=None)
        raise AssertionError("must not reach execution")
    except BaseException as exc:
        assert "hard bound" not in str(exc)


# ---------------- §7.12: guard-removal mutations ------------------

def _mutant_module(rel, old, new, name):
    import tempfile
    srcp = Path(__file__).resolve().parents[1] / rel
    text = srcp.read_text()
    assert old in text, f"mutation anchor missing in {rel}"
    mp = Path(tempfile.mkdtemp()) / f"{name}.py"
    mp.write_text(text.replace(old, new))
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(name, mp)
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    if hasattr(m, "REPO"):
        m.REPO = Path(__file__).resolve().parents[1]
    return m


def test_mut_c17_holder_binding_is_the_guard(tmp_path):
    """Removing the holder-identity comparison lets a foreign-pid
    lease through — proving test_c17 bites that exact guard."""
    m = _mutant_module(
        "tools/b4_campaign_orchestrator.py",
        '''    if not (lease["holder_pid"] == lockrec["holder_pid"]
            == claim.get("holder_pid") == me):''',
        '''    if False and not (lease["holder_pid"] == lockrec["holder_pid"]
            == claim.get("holder_pid") == me):''',
        "orch_mut_c17")
    (tmp_path / "B4_MATERIALIZATION.json").write_text("{}")
    _live_lock(tmp_path)
    claim = orch.claim_attempt(tmp_path, "o2024_seed101")
    lease_p = orch.issue_lease(tmp_path, "o2024_seed101", claim,
                               "a" * 64, tmp_path)
    doc = _resign_lease({**json.loads(lease_p.read_text()),
                         "holder_pid": 999999})
    _ctl_write(lease_p, json.dumps(doc))
    with pytest.raises(SystemExit, match="holder identity"):
        orch.verify_lease(lease_p, tmp_path, "o2024_seed101",
                          tmp_path)
    assert m.verify_lease(lease_p, tmp_path, "o2024_seed101",
                          tmp_path)["holder_pid"] == 999999


def test_mut_c18_completion_integrity_is_the_guard(tmp_path):
    """Removing the completion self-digest check accepts a tampered
    completion — proving seal_state's integrity check bites."""
    m = _mutant_module(
        "tools/b4_campaign_orchestrator.py",
        '''        if hashlib.sha256(json.dumps(
                body, sort_keys=True).encode()).hexdigest() != \\
                completion.get("completion_sha256"):
            return "UNCERTAIN"''',
        '''        if False:
            return "UNCERTAIN"''',
        "orch_mut_c18")
    root, c = _seal_root(tmp_path, "M")
    orch.seal_attempt(root, "o2022_seed101", c["attempt_id"])
    cp = next((root / "o2022_seed101").glob("SEAL_COMPLETE_*"))
    doc = json.loads(cp.read_text())
    doc["completion_sha256"] = "0" * 64
    _ctl_write(cp, json.dumps(doc))
    assert orch.seal_state(root, "o2022_seed101") == "UNCERTAIN"
    assert m.seal_state(root, "o2022_seed101") == "SEALED"


def test_mut_c20_seal_gate_is_the_guard(tmp_path, monkeypatch):
    """Removing the ledger's physical-seal gate lets an unsealed
    cell into completion — proving the C20 gate bites."""
    m = _mutant_module(
        "tools/b4_campaign_ledger.py",
        '''        if orch.seal_state(results_root, cid) != "SEALED":
            raise LedgerRefusal(''',
        '''        if False:
            raise LedgerRefusal(''',
        "ledger_mut_c20")
    led, results = _ledger_fixture(tmp_path)
    for w in (results / "o2022_seed101").glob("SEAL_COMPLETE_*"):
        w.unlink()
    with pytest.raises(SystemExit, match="UNSEALED or UNCERTAIN"):
        _check_results(led, tmp_path, results, monkeypatch)
    monkeypatch.setattr(m, "verify_ledger", lambda lp, mr: led)
    monkeypatch.setattr(
        m, "_derive_comparator_dir",
        lambda mr: tmp_path / "comp_default")
    assert m.verify_campaign_results(
        tmp_path / "ledger.json", tmp_path, results)["n"] == 12


def test_mut_c21_source_row_recompute_is_the_guard(tmp_path,
                                                   monkeypatch):
    """Removing the source-row recomputation lets a fabricated
    source_row_sha256 column through — proving C21 bites."""
    m = _mutant_module(
        "tools/b4_campaign_ledger.py",
        '''        if list(df["source_row_sha256"].astype(str)) != \\
                exp["row_shas"]:
            raise LedgerRefusal(''',
        '''        if False:
            raise LedgerRefusal(''',
        "ledger_mut_c21")
    led, results = _ledger_fixture(tmp_path)
    cid = "o2023_seed101"

    def m4(df):
        df["source_row_sha256"] = ["s" * 64] * len(df)
        return df
    _mutate_perbar(results, cid, m4)
    with pytest.raises(SystemExit, match="does not\\s+recompute"):
        _check_results(led, tmp_path, results, monkeypatch)
    monkeypatch.setattr(m, "verify_ledger", lambda lp, mr: led)
    monkeypatch.setattr(
        m, "_derive_comparator_dir",
        lambda mr: tmp_path / "comp_default")
    assert m.verify_campaign_results(
        tmp_path / "ledger.json", tmp_path, results)["n"] == 12


def test_mut_c22_min_recompute_is_the_guard(tmp_path, monkeypatch):
    """Reverting to caller-trusting remainder acceptance lets an
    exhausted budget run — proving the C22 recompute bites."""
    m = _mutant_module(
        "tools/b4_campaign_executor.py",
        '''    recomputed = _orch.remaining_global_seconds(
        Path(out_root), b4a.load_resource_contract())
    if global_wall_remaining_seconds is None:
        global_wall_remaining_seconds = recomputed
    else:
        global_wall_remaining_seconds = min(
            float(global_wall_remaining_seconds), recomputed)''',
        '''    if global_wall_remaining_seconds is None:
        global_wall_remaining_seconds = _orch.\\
            remaining_global_seconds(Path(out_root),
                                     b4a.load_resource_contract())''',
        "exec_mut_c22")
    m.b4a = executor.b4a
    lease = _c22_harness(tmp_path, monkeypatch, executor, 0.0)
    with pytest.raises(SystemExit, match="hard bound"):
        executor.execute_cell(
            "o2024_seed101", tmp_path, tmp_path, "cpu",
            lease_path=lease,
            global_wall_remaining_seconds=1e12)
    monkeypatch.setattr(m, "CAMPAIGN_AUTH_SHA", "f" * 64)
    monkeypatch.setattr(m, "CAMPAIGN_AUTH_PATH",
                        tmp_path / "auth.json")
    try:
        m.execute_cell("o2024_seed101", tmp_path, tmp_path, "cpu",
                       lease_path=lease,
                       global_wall_remaining_seconds=1e12)
        raise AssertionError("must not reach execution")
    except BaseException as exc:
        assert "hard bound" not in str(exc)


def test_c25_checkpoint_adversaries(tmp_path, monkeypatch):
    """§5.6 (C25): missing, symlinked, non-regular, permissive and
    altered checkpoints each block the strongest verifier."""
    cid = "o2022_seed202"

    def fresh():
        import shutil as _sh
        for child in tmp_path.iterdir():
            if child.is_dir():
                _sh.rmtree(child)
            else:
                child.unlink()
        return _ledger_fixture(tmp_path)

    # (a) ABSENT — the exact PRE
    led, results = fresh()
    (results / cid / f"checkpoint_{cid}.zip").unlink()
    with pytest.raises(SystemExit, match="ABSENT"):
        _check_results(led, tmp_path, results, monkeypatch)
    # (b) symlink substitution
    led, results = fresh()
    ck = results / cid / f"checkpoint_{cid}.zip"
    donor = results / cid / "donor.zip"
    donor.write_bytes(ck.read_bytes())
    ck.unlink()
    ck.symlink_to(donor)
    with pytest.raises(SystemExit, match="unopenable"):
        _check_results(led, tmp_path, results, monkeypatch)
    # (c) directory in place of the artifact
    led, results = fresh()
    ck = results / cid / f"checkpoint_{cid}.zip"
    ck.unlink()
    ck.mkdir()
    with pytest.raises(SystemExit,
                       match="unopenable|not a regular"):
        _check_results(led, tmp_path, results, monkeypatch)
    # (d) group/world-writable artifact
    led, results = fresh()
    os.chmod(results / cid / f"checkpoint_{cid}.zip", 0o666)
    with pytest.raises(SystemExit, match="writable"):
        _check_results(led, tmp_path, results, monkeypatch)
    # (e) altered bytes (kept from C21, now descriptor-read)
    led, results = fresh()
    (results / cid / f"checkpoint_{cid}.zip").write_bytes(
        b"tampered")
    os.chmod(results / cid / f"checkpoint_{cid}.zip", 0o644)
    with pytest.raises(SystemExit, match="checkpoint bytes"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_mut_c25_mandatory_existence_is_the_guard(tmp_path,
                                                  monkeypatch):
    """Reverting to `if exists: verify` re-admits the removed
    checkpoint — proving C25 bites."""
    m = _mutant_module(
        "tools/b4_campaign_ledger.py",
        '''        try:
            ckfd = os.open(ck, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            raise LedgerRefusal(
                f"REFUSED: {cid} checkpoint artifact is ABSENT — "
                "a declared digest of missing bytes is not "
                "evidence")''',
        '''        try:
            ckfd = os.open(ck, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            facts[cid] = {"terminal": term["terminal"],
                          "attempt_id": att,
                          "per_bar_sha256": term["per_bar_sha256"],
                          "checkpoint_sha256_verified": None}
            continue''',
        "ledger_mut_c25")
    led, results = _ledger_fixture(tmp_path)
    cid = "o2023_seed101"
    (results / cid / f"checkpoint_{cid}.zip").unlink()
    with pytest.raises(SystemExit, match="ABSENT"):
        _check_results(led, tmp_path, results, monkeypatch)
    monkeypatch.setattr(m, "verify_ledger", lambda lp, mr: led)
    monkeypatch.setattr(
        m, "_derive_comparator_dir",
        lambda mr: tmp_path / "comp_default")
    assert m.verify_campaign_results(
        tmp_path / "ledger.json", tmp_path, results)["n"] == 12


def test_mut_c23_releasing_guard_bites(tmp_path, monkeypatch):
    """Treating RELEASING as RELEASED re-admits the second holder
    after an uncertain release — proving C23 bites."""
    m = _mutant_module(
        "tools/b4_campaign_orchestrator.py",
        '''            if st["state"] == "RELEASING":
                raise OrchestratorRefusal(
                    f"REFUSED: lock epoch {cur} release is "
                    "UNCERTAIN (intent without durable completion) "
                    "— operator disposition, no second holder")''',
        '''            if st["state"] == "RELEASING":
                st = dict(st, state="RELEASED")''',
        "orch_mut_c23")
    os.chmod(tmp_path, 0o700)
    lk = orch.GlobalLock(tmp_path)
    lk.__enter__()
    real = orch._excl_write
    monkeypatch.setattr(
        orch, "_excl_write",
        lambda p, b, mm=0o600: (_ for _ in ()).throw(
            OSError("injected"))
        if "LOCK_RELEASE_COMPLETE" in Path(p).name
        else real(p, b, mm))
    with pytest.raises(SystemExit, match="uncertain release"):
        lk.__exit__()
    monkeypatch.setattr(orch, "_excl_write", real)
    with pytest.raises(SystemExit, match="UNCERTAIN"):
        orch.GlobalLock(tmp_path).__enter__()
    # defense in depth: killing ONLY the scan-layer guard is not
    # enough — the post-choice revalidation still refuses
    with pytest.raises(SystemExit, match="no longer RELEASED"):
        m.GlobalLock(tmp_path).__enter__()
    # both layers removed -> the forged reclaim finally enters,
    # proving each guard is live and necessary
    m2 = _mutant_module(
        "tools/b4_campaign_orchestrator.py",
        '''            if st["state"] == "RELEASING":
                raise OrchestratorRefusal(
                    f"REFUSED: lock epoch {cur} release is "
                    "UNCERTAIN (intent without durable completion) "
                    "— operator disposition, no second holder")''',
        '''            if st["state"] == "RELEASING":
                st = dict(st, state="RELEASED")''',
        "orch_mut_c23_full")
    src2 = Path(m2.__file__).read_text()
    old2 = '''            if prev["state"] != "RELEASED":'''
    assert old2 in src2
    Path(m2.__file__).write_text(src2.replace(
        old2, '''            if False:'''))
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("orch_mut_c23_full2",
                                       m2.__file__)
    m3 = ilu.module_from_spec(spec)
    spec.loader.exec_module(m3)
    # the single-layer attempt above created epoch 2 and then
    # self-released it in order on refusal; clear every epoch-2
    # residue so the double mutant contends on the uncertain
    # epoch 1 alone
    for stale in list(tmp_path.glob("LOCK_EPOCH_2.json")) + \
            list(tmp_path.glob("LOCK_RELEASE_INTENT_2.json")) + \
            list(tmp_path.glob("LOCK_RELEASE_COMPLETE_2.json")):
        stale.unlink()
    got = m3.GlobalLock(tmp_path)
    got.__enter__()                      # only the DOUBLE mutant
    assert got.epoch == 2


def test_mut_c24_mode_guard_bites(tmp_path):
    """Removing the exact-mode check re-admits a public claim —
    proving C24 bites."""
    m = _mutant_module(
        "tools/b4_campaign_orchestrator.py",
        '''        if expected_mode is not None and \\
                stat.S_IMODE(st.st_mode) != expected_mode:''',
        '''        if False and expected_mode is not None and \\
                stat.S_IMODE(st.st_mode) != expected_mode:''',
        "orch_mut_c24")
    os.chmod(tmp_path, 0o700)
    _live_lock(tmp_path)
    orch.claim_attempt(tmp_path, "o2024_seed101")
    cp = (tmp_path / "o2024_seed101" /
          f"CLAIM_{a.CAMPAIGN_GENERATION}.json")
    os.chmod(cp, 0o644)
    with pytest.raises(SystemExit, match="not the private"):
        orch.load_claim(tmp_path, "o2024_seed101")
    assert m.load_claim(
        tmp_path, "o2024_seed101")["cell"] == "o2024_seed101"


# ================== C26: append-only chain adversaries =============

def test_c26_rewritten_a9_refuses_without_a10(tmp_path,
                                              monkeypatch):
    """The exact C26 finding: an in-place rewritten amendment 9
    refuses even when no amendment 10 exists yet."""
    _fake_a4(tmp_path, monkeypatch)
    f9 = tmp_path / "a9.json"
    doc = json.loads(f9.read_text())
    doc["final_code_pins"]["tools/b4_authority.py"] = "1" * 64
    f9.write_text(json.dumps(doc))          # rewrite IN PLACE
    with pytest.raises(SystemExit,
                       match="never edited in place"):
        a.verify_amendment_chain()


def test_c26_restored_a9_without_a10_refuses(tmp_path,
                                             monkeypatch):
    _fake_a4(tmp_path, monkeypatch)
    monkeypatch.setattr(a, "AMENDMENT_10_PATH",
                        tmp_path / "absent_a10.json")
    with pytest.raises(SystemExit, match="amendment 10 absent"):
        a.verify_amendment_chain()


def test_c26_a10_naming_rewritten_a9_refuses(tmp_path,
                                             monkeypatch):
    """Amendment 10 must name the ORIGINAL reviewed amendment-9
    bytes, never the rewritten ones."""
    _fake_a4(tmp_path, monkeypatch,
             a10_over={"amends_amendment_9_sha256": "9" * 64})
    with pytest.raises(SystemExit,
                       match="exact reviewed bytes"):
        a.verify_amendment_chain()


def test_c26_altered_code_after_a10_refuses(tmp_path, monkeypatch):
    pins = _pins("tools/b4_authority.py", "tools/b4_run_cell.py",
                 "tools/b4_campaign_executor.py",
                 "tools/b4_campaign_ledger.py",
                 "tools/b4_campaign_orchestrator.py",
                 "tools/b4_adjudicator.py",
                 "tools/materialize_b4_causal_sac.py",
                 "pipeline_plugins/rl_pipeline_with_validation.py",
                 "tests/test_b4_materializer_authority.py")
    pins["tools/b4_campaign_orchestrator.py"] = "2" * 64
    _fake_a4(tmp_path, monkeypatch,
             a10_over={"final_code_pins": pins})
    with pytest.raises(SystemExit, match="differs from the final"):
        a.verify_amendment_chain()


def test_c26_record_naming_a9_refuses(tmp_path, monkeypatch):
    """An authorization candidate naming amendment_9_sha256 instead
    of the truthful amendment_10_sha256 refuses."""
    _fake_a4(tmp_path, monkeypatch)
    binds = a.campaign_record_required_bindings()
    assert "amendment_10_sha256" in binds
    assert "amendment_9_sha256" not in binds
    limits = a.load_resource_contract()
    rec = {"schema": "agent_multi.owner_campaign_authorization.v2",
           "recorded_at_date": "2026-09-06",
           "authority": "project_owner",
           "recorded_by": "General Musashi",
           "decision": "APPROVE_B4_TWELVE_CELL_CAMPAIGN",
           "bindings": dict(binds),
           "per_cell_limits": {k: limits[k] for k in (
               "budget_max_env_steps", "budget_max_updates",
               "budget_max_wall_seconds", "budget_max_rss_bytes",
               "budget_max_cuda_bytes",
               "budget_max_gpu_temp_celsius")},
           "owner_decision": {
               "intent_record_sha256": "3" * 64,
               "owner_words": "ok yo autorizo"}}
    good = tmp_path / "rec_good.json"
    good.write_text(json.dumps(rec))
    got = a.verify_campaign_authorization_record(
        good, a._sha_file(good))
    assert got["bindings"]["amendment_10_sha256"] == \
        binds["amendment_10_sha256"]
    # naming the (rewritten or original) amendment 9 field refuses
    bad = dict(rec)
    bad_binds = dict(binds)
    bad_binds.pop("amendment_10_sha256")
    bad_binds["amendment_9_sha256"] = a._sha_file(
        tmp_path / "a9.json")
    bad["bindings"] = bad_binds
    badp = tmp_path / "rec_bad.json"
    badp.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        a.verify_campaign_authorization_record(
            badp, a._sha_file(badp))


def test_c26_self_rehashed_replacement_a10_refuses(tmp_path,
                                                   monkeypatch):
    """A coherent REPLACEMENT amendment 10 (self-consistent but
    naming a different predecessor) grants nothing."""
    _fake_a4(tmp_path, monkeypatch)
    f10 = tmp_path / "a10.json"
    doc = json.loads(f10.read_text())
    doc["amends_amendment_9_sha256"] = hashlib.sha256(
        b"attacker history").hexdigest()
    f10.write_text(json.dumps(doc))
    with pytest.raises(SystemExit,
                       match="exact reviewed bytes"):
        a.verify_amendment_chain()


def test_c26_missing_or_reordered_amendment_refuses(tmp_path,
                                                    monkeypatch):
    _fake_a4(tmp_path, monkeypatch)
    monkeypatch.setattr(a, "AMENDMENT_8_PATH",
                        tmp_path / "gone_a8.json")
    with pytest.raises(SystemExit, match="amendment 8 absent"):
        a.verify_amendment_chain()
    # duplicated content in the wrong slot (a8 bytes as a9) breaks
    # the exact link
    _fake_a4(tmp_path, monkeypatch)
    (tmp_path / "a9.json").write_text(
        (tmp_path / "a8.json").read_text())
    with pytest.raises(SystemExit,
                       match="never edited in place|does not name"):
        a.verify_amendment_chain()


def test_c26_git_history_regression():
    """The PRE reproduction stays executable: the reviewed original
    and the rewritten blob hash to the recorded values."""
    import subprocess
    rel = ("docs/audits/evidence/"
           "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_9_2026_09_06.json")
    repo = Path(__file__).resolve().parents[1]

    def blob(commit):
        return subprocess.run(
            ["git", "cat-file", "-p", f"{commit}:{rel}"],
            cwd=repo, capture_output=True, check=True).stdout
    assert hashlib.sha256(blob("d97c3f62")).hexdigest() == \
        a.AMENDMENT_9_SHA
    assert hashlib.sha256(blob("d8f25438")).hexdigest() == (
        "01aeee957c993ee764b8e4753860d9c03b6467973eed5a3ec07bb42"
        "5e1e4a337")
    # the live file is the RESTORED original
    assert a._sha_file(repo / rel) == a.AMENDMENT_9_SHA
