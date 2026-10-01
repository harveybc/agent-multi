"""Admission tests: no ML imports, training, GPU, or remote process access."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from rl_temporal.replica_policy import ReplicaPolicy, declaration

ROOT = Path(__file__).resolve().parents[2]


def cell(seed=101, arm="RL-S0"):
    return {"arm": arm, "train_seed": seed, "eval_seed": seed,
            "max_epochs": 200, "task": {"dataset_id": "fixture"}}


def test_screening_once_per_arm_and_persisted_claim(tmp_path):
    policy = ReplicaPolicy(tmp_path / "runs")
    for arm in ("RL-S0", "RL-S1", "RL-D0", "RL-D1"):
        out = tmp_path / "runs" / f"{arm}_seed101"
        assert policy.admit(cell(arm=arm), out, claim=True)["status"] == "ALLOW_SCREENING"
        assert ReplicaPolicy(policy.root).admit(cell(arm=arm), out)["status"] == "HELD_EXISTING_ATTEMPT"
    assert len(json.loads(policy.state_path.read_text())["claims"]) == 4


@pytest.mark.parametrize("seed", [202, 303])
def test_extra_requires_predeclared_persisted_exact_config_justification(tmp_path, seed):
    policy = ReplicaPolicy(tmp_path / "runs")
    cfg, out = cell(seed), tmp_path / "runs" / f"RL-S0_seed{seed}"
    assert policy.admit(cfg, out)["status"] == "HELD_EXTRA_REPLICA"
    with pytest.raises(ValueError):
        policy.authorize(cfg, out, reason_code="final_contrast", reason=" ", declared_by="reviewer")
    policy.authorize(cfg, out, reason_code="final_contrast", reason="Predeclared close finalist contrast", declared_by="reviewer")
    persisted = policy.state_path.read_bytes()
    changed = {**cfg, "max_epochs": 201}
    assert policy.admit(changed, out)["status"] == "HELD_EXTRA_REPLICA"
    assert ReplicaPolicy(policy.root).admit(cfg, out, claim=True)["status"] == "ALLOW_JUSTIFIED_REPLICA"
    claim = next(iter(json.loads(policy.state_path.read_text())["claims"].values()))
    assert claim["authorization"]["declared_at"] <= claim["claimed_at"]
    assert b"Predeclared close finalist contrast" in persisted
    with pytest.raises(ValueError, match="attempt"):
        policy.authorize(cfg, out, reason_code="final_contrast", reason="Too late", declared_by="reviewer")


def test_fourth_rejected_and_results_never_overwritten(tmp_path):
    policy = ReplicaPolicy(tmp_path / "runs")
    cfg, out = cell(404), tmp_path / "runs" / "RL-S0_seed404"
    assert policy.admit(cfg, out)["status"] == "REJECTED_FOURTH_REPLICA"
    with pytest.raises(ValueError):
        policy.authorize(cfg, out, reason_code="published_protocol", reason="Literature", declared_by="reviewer")
    out.mkdir(parents=True)
    result = out / "RESULT.json"
    result.write_text('{"status":"historical"}\n')
    before = result.read_bytes()
    assert policy.admit(cfg, out, claim=True)["status"] == "PRESERVED_EXISTING_RESULT"
    assert result.read_bytes() == before


def test_partial_attempt_cannot_be_authorized_retroactively(tmp_path):
    policy = ReplicaPolicy(tmp_path / "runs")
    out = policy.root / "RL-S0_seed202"
    out.mkdir(parents=True)
    (out / "best_policy.zip").write_bytes(b"prior checkpoint")
    assert policy.admit(cell(202), out)["status"] == "HELD_EXISTING_ATTEMPT"
    with pytest.raises(ValueError):
        policy.authorize(cell(202), out, reason_code="final_contrast", reason="Late", declared_by="reviewer")


def test_bad_state_and_changed_seed_fail_closed(tmp_path):
    policy = ReplicaPolicy(tmp_path)
    policy.state_path.write_text('{"schema":"unknown"}')
    with pytest.raises(ValueError):
        policy.admit(cell(), tmp_path / "out")
    assert declaration(404)["status"] == "REJECTED_FOURTH_REPLICA"
    assert declaration(202)["status"] == "HELD_EXTRA_REPLICA"


def test_direct_runner_refuses_before_importing_training_stack(tmp_path):
    cfg = tmp_path / "cell.json"
    cfg.write_text(json.dumps(cell(202)))
    proc = subprocess.run([sys.executable, str(ROOT / "tools/run_rl_temporal_cell.py"),
                           "--cell", str(cfg), "--data-root", str(tmp_path),
                           "--out", str(tmp_path / "runs/RL-S0_seed202")],
                          capture_output=True, text=True, timeout=10)
    assert proc.returncode != 0
    assert "HELD_EXTRA_REPLICA" in proc.stdout + proc.stderr
    assert not (tmp_path / "runs/RL-S0_seed202").exists()


@pytest.mark.parametrize("seed,expected", [(202, "HELD_EXTRA_REPLICA"), (303, "HELD_EXTRA_REPLICA"), (404, "REJECTED_FOURTH_REPLICA")])
def test_queue_handoff_never_launches_s0_202_without_justification(tmp_path, seed, expected):
    home = tmp_path / "home"
    checkout = home / ".local/state/scratch/g-rl/wt/agent-multi"
    checkout.mkdir(parents=True)
    shutil.copytree(ROOT / "rl_temporal", checkout / "rl_temporal", ignore=shutil.ignore_patterns("__pycache__"))
    cfg = checkout / f"RL-S0_seed{seed}.json"
    cfg.write_text(json.dumps(cell(seed)))
    # No crispdm-run or trainer is installed in this sandbox: reaching it is a failure.
    proc = subprocess.run(["bash", str(ROOT / "tools/rl_temporal_queue.sh"),
                           "fixture", "cpu", "", "128M", str(tmp_path / "runs"), str(cfg)],
                          env={**os.environ, "HOME": str(home), "PILOT_ARGS": ""},
                          capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert expected in proc.stdout
    assert " start " not in proc.stdout
    assert not (tmp_path / f"runs/RL-S0_seed{seed}").exists()


def test_default_matrix_and_fourth_refusal_without_ml(tmp_path):
    # Extract just build_matrix and its lightweight globals; no torch model is built.
    import ast
    tree = ast.parse((ROOT / "rl_temporal/arms.py").read_text())
    default = next(n.value for n in tree.body if isinstance(n, ast.AnnAssign) and n.target.id == "PAIRED_SEEDS")
    assert ast.literal_eval(default) == (101,)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_matrix"]
    ns = dict(SelectedFeatureBinding=object, Sequence=list, PAIRED_SEEDS=(101,), Any=object,
              List=list, Dict=dict, ARMS={a: {} for a in ("RL-S0", "RL-S1", "RL-D0", "RL-D1")},
              build_arm_config=lambda arm, binding, seed, **kw: {"arm": arm, "train_seed": seed})
    from rl_temporal.replica_policy import validate_matrix_seeds
    ns["validate_matrix_seeds"] = validate_matrix_seeds
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "arms.py", "exec"), ns)
    assert len(ns["build_matrix"](None, out_root=str(tmp_path))) == 4
    with pytest.raises(ValueError):
        ns["build_matrix"](None, out_root=str(tmp_path), seeds=(101, 202, 303, 404))


def test_concurrent_claims_only_one_can_train(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def claim(_):
        return ReplicaPolicy(tmp_path).admit(cell(), tmp_path / "run", claim=True)["status"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        decisions = list(pool.map(claim, range(4)))
    assert decisions.count("ALLOW_SCREENING") == 1
    assert decisions.count("HELD_EXISTING_ATTEMPT") == 3


def test_direct_runner_preserves_legacy_result_without_dependencies(tmp_path):
    cfg = tmp_path / "cell.json"
    cfg.write_text(json.dumps(cell(404)))
    out = tmp_path / "runs/legacy"
    out.mkdir(parents=True)
    result = out / "RESULT.json"
    result.write_text('{"status":"legacy result retained"}')
    before = result.read_bytes()
    proc = subprocess.run([sys.executable, str(ROOT / "tools/run_rl_temporal_cell.py"),
                           "--cell", str(cfg), "--data-root", str(tmp_path), "--out", str(out)],
                          capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert "PRESERVED_EXISTING_RESULT" in proc.stdout
    assert result.read_bytes() == before


def test_future_authorization_and_bad_reason_cannot_admit(tmp_path):
    policy = ReplicaPolicy(tmp_path)
    cfg, out = cell(202), tmp_path / "out"
    with pytest.raises(ValueError):
        policy.authorize(cfg, out, reason_code="already_in_grid", reason="Old grid", declared_by="owner")
    policy.authorize(cfg, out, reason_code="final_contrast", reason="Close finalists", declared_by="owner")
    state = json.loads(policy.state_path.read_text())
    state["authorizations"]["RL-S0:202"]["declared_at"] = "2999-01-01T00:00:00+00:00"
    policy.state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="precede"):
        policy.admit(cfg, out, claim=True)


def test_existing_result_adopted_from_another_output_name(tmp_path):
    policy = ReplicaPolicy(tmp_path)
    out = tmp_path / "RL-S0_seed101"
    out.mkdir()
    (out / "RESULT.json").write_text('{}')
    assert policy.admit(cell(), tmp_path / "different-name", claim=True)["status"] == "PRESERVED_EXISTING_RESULT"


def test_legacy_result_identity_is_found_even_in_renamed_directory(tmp_path):
    out = tmp_path / "historical-device-name"
    out.mkdir()
    result = out / "RESULT.json"
    result.write_text(json.dumps({"arm": "RL-S0", "seed": 202, "status": "RESULT"}))
    policy = ReplicaPolicy(tmp_path)
    assert policy.admit(cell(202), tmp_path / "RL-S0_seed202")["status"] == "PRESERVED_EXISTING_RESULT"
    with pytest.raises(ValueError):
        policy.authorize(cell(202), tmp_path / "RL-S0_seed202", reason_code="final_contrast",
                         reason="Do not retrain historical result", declared_by="owner")
