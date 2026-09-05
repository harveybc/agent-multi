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
def _fake_a4(tmp_path, monkeypatch, **over):
    a4 = {"amends_design_sha256": a.DESIGN_SHA,
          "supersedes_amendment_shas": list(a.AMENDMENT_SHAS),
          "final_code_pins": {
              "tools/b4_authority.py":
                  a._sha_file(REPO / "tools/b4_authority.py"),
              "tools/screen_b_baselines.py":
                  a._sha_file(REPO / "tools/screen_b_baselines.py"),
              "tools/materialize_b4_causal_sac.py":
                  a._sha_file(REPO /
                              "tools/materialize_b4_causal_sac.py"),
              "tools/b4_run_cell.py":
                  a._sha_file(REPO / "tools/b4_run_cell.py")}}
    a4.update(over)
    f = tmp_path / "a4.json"
    f.write_text(json.dumps(a4))
    monkeypatch.setattr(a, "AMENDMENT_4_PATH", f)
    return a4


def test_e3_valid_chain_passes(tmp_path, monkeypatch):
    _fake_a4(tmp_path, monkeypatch)
    chain = a.verify_amendment_chain()
    assert chain["design_sha256"] == a.DESIGN_SHA
    assert len(chain["amendment_shas"]) == 4


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
    a4 = _fake_a4(tmp_path, monkeypatch)
    a4["final_code_pins"]["tools/b4_run_cell.py"] = "0" * 64
    (a.AMENDMENT_4_PATH).write_text(json.dumps(a4))
    with pytest.raises(SystemExit, match="differs from the final"):
        a.verify_amendment_chain()


def test_e3_incomplete_pin_surface_refuses(tmp_path, monkeypatch):
    a4 = _fake_a4(tmp_path, monkeypatch)
    del a4["final_code_pins"]["tools/b4_run_cell.py"]
    (a.AMENDMENT_4_PATH).write_text(json.dumps(a4))
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


def test_e7_gpu_refuses_without_authorization(tmp_path):
    with pytest.raises(SystemExit, match="Musashi GPU authorization"):
        runner.main(["--cell-id", "o2024_seed101",
                     "--materialization-root", str(tmp_path),
                     "--output-root", str(tmp_path / "out"),
                     "--device", "cuda:0"])


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
