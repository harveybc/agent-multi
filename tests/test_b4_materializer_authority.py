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
             **over):
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
    return a4


def test_e3_valid_chain_passes(tmp_path, monkeypatch):
    _fake_a4(tmp_path, monkeypatch)
    chain = a.verify_amendment_chain()
    assert chain["design_sha256"] == a.DESIGN_SHA
    assert len(chain["amendment_shas"]) == 6


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
    pins["tools/b4_authority.py"] = "0" * 64
    _fake_a4(tmp_path, monkeypatch,
             a6_over={"final_code_pins": pins})
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
                       match="amendment-6 proposed population"):
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
                       match="no owner campaign authorization"):
        executor.execute_cell("o2024_seed101", tmp_path, tmp_path,
                              "cuda:0")


def test_e12_terminal_records_are_immutable(tmp_path):
    executor.write_terminal(tmp_path, "o2022_seed101", "FAILED",
                            {"reason": "x"})
    with pytest.raises(SystemExit, match="immutable"):
        executor.write_terminal(tmp_path, "o2022_seed101",
                                "COMPLETED", {})
    with pytest.raises(SystemExit, match="unknown terminal"):
        executor.write_terminal(tmp_path, "o2023_seed101",
                                "PROMOTED", {})


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
        d = results / cid
        d.mkdir(parents=True)
        pb = d / "per_bar.csv"
        pb.write_text("net_return\n0.0\n")
        term = {"schema": "agent_multi.b4_cell_terminal.v1",
                "cell": cid, "terminal": "COMPLETED",
                "cell_config_sha256": e["cell_config_sha256"],
                "attempt_id": f"attempt_{cid}",
                "per_bar_csv": str(pb),
                "per_bar_sha256": ledger_mod._sha_file(pb),
                "sealed_2025_used": False}
        (d / "B4_CELL_TERMINAL.json").write_text(json.dumps(term))
    return led, results


def _check_results(led, tmp_path, results, monkeypatch):
    monkeypatch.setattr(ledger_mod, "verify_ledger",
                        lambda lp, mr: led)
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
    term = json.loads(tp.read_text())
    term["status"] = "B4_GPU_PREFLIGHT_MECHANICS_AND_THROUGHPUT_ONLY"
    tp.write_text(json.dumps(term))
    with pytest.raises(SystemExit, match="mechanics/preflight"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_e12_reused_attempt_refuses(tmp_path, monkeypatch):
    led, results = _ledger_fixture(tmp_path)
    for cid in ("o2022_seed101", "o2022_seed202"):
        tp = results / cid / "B4_CELL_TERMINAL.json"
        term = json.loads(tp.read_text())
        term["attempt_id"] = "attempt_SAME"
        tp.write_text(json.dumps(term))
    with pytest.raises(SystemExit, match="attempt identity reused"):
        _check_results(led, tmp_path, results, monkeypatch)


def test_e12_foreign_cell_digest_refuses(tmp_path, monkeypatch):
    led, results = _ledger_fixture(tmp_path)
    tp = results / "o2023_seed303" / "B4_CELL_TERMINAL.json"
    term = json.loads(tp.read_text())
    term["cell_config_sha256"] = "f" * 64
    tp.write_text(json.dumps(term))
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
    tp.write_text(json.dumps(term))
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
