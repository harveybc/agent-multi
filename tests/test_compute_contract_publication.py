"""Whether the compute contract survives the journey to a receipt, and the evaluation ceiling.

S1: "Record SB3 version and resolved plugin settings. [...] Use a separate evaluation cap
<=384 transitions and distinguish evaluation from training counters. [...] config alias
regression and actual governed input consumption."

The alias defect this guards against is the one that actually happened: a budget written only
under `training` never reached the agent, which then used its own default of 10,000 and spent
10,240 transitions against a request for 64. A ceiling stranded the same way would be worse —
it would read as enforced while enforcing nothing.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tools" / "governed_offline_replay.py"

spec = importlib.util.spec_from_file_location("governed_offline_replay_publication", RUNNER)
replay = importlib.util.module_from_spec(spec)
sys.modules["governed_offline_replay_publication"] = replay
spec.loader.exec_module(replay)


def template_at(tmp_path: Path) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"experiment": {"name": "s1"}, "data": {},
                                "training": {"total_timesteps": 4}, "environment": {}}),
                    encoding="utf-8")
    return path


def flat_for(tmp_path: Path, **extra) -> dict:
    body = {"experiment_config": str(template_at(tmp_path)),
            "runtime_overlay": str(tmp_path / "o.json"),
            "input_data_file": str(tmp_path / "delivered.csv"),
            "total_timesteps": 64, "training_transition_cap": 64,
            "evaluation_transition_cap": 384, "n_steps": 64, "batch_size": 64, "n_epochs": 1}
    body.update(extra)
    return body


def test_the_declared_ceilings_reach_the_level_the_runtime_reads(tmp_path):
    """The alias regression, for limits rather than for the budget."""
    flat = flat_for(tmp_path)
    merged = replay.merged_config(flat, Path(flat["experiment_config"]), tmp_path)

    assert merged["training_transition_cap"] == 64
    assert merged["evaluation_transition_cap"] == 384
    assert merged["n_steps"] == 64 and merged["batch_size"] == 64 and merged["n_epochs"] == 1
    assert merged["total_timesteps"] == 64, "the budget regression stays green"
    assert merged["training"]["total_timesteps"] == 64, "both levels, as before"


def test_an_undeclared_ceiling_is_not_invented(tmp_path):
    """No run silently acquires a limit it never declared, which would then look enforced."""
    flat = flat_for(tmp_path)
    for key in ("training_transition_cap", "evaluation_transition_cap"):
        flat.pop(key)
    merged = replay.merged_config(flat, Path(flat["experiment_config"]), tmp_path)
    assert "training_transition_cap" not in merged
    assert "evaluation_transition_cap" not in merged


def test_only_measurements_are_flattened_into_metrics():
    """A counter the runtime could not produce is absent from the metrics, never a zero."""
    contract = {"schema": "compute_contract.v1", "algorithm": "PPO",
                "requested_training_transitions": 64, "training_transition_cap": 64,
                "training_transitions_observed": 64, "collected_rollouts": 1,
                "optimization_epochs_completed": 1, "gradient_updates": None,
                "optimizer_step_calls": 1, "sb3_n_updates_delta": 1,
                "cap_respected": True, "partial_rollout": False,
                "evaluation_transitions": None,
                "counters_unavailable": ["evaluation_transitions"]}
    metrics = replay.compute_metrics({"compute_contract": contract})

    assert metrics["compute_optimizer_step_calls"] == 1.0
    assert metrics["compute_optimization_epochs_completed"] == 1.0
    assert metrics["compute_cap_respected"] == 1.0
    assert "compute_gradient_updates" not in metrics, "PPO makes no gradient-update claim"
    assert "compute_evaluation_transitions" not in metrics, "unmeasured stays absent"
    assert "compute_sb3_n_updates_delta" not in metrics, (
        "the raw library counter is kept in the receipt, not published as a named cost")


def test_a_run_without_a_contract_publishes_no_compute_metrics():
    assert replay.compute_metrics({"observed_timesteps": 64}) == {}
    assert replay.compute_metrics({"compute_contract": "not a mapping"}) == {}


def test_the_receipt_carries_the_whole_contract_beside_the_flat_metrics(tmp_path):
    """End to end through the real runner: meanings and gaps must not be lost to flattening."""
    (tmp_path / "o.json").write_text("{}", encoding="utf-8")
    (tmp_path / "delivered.csv").write_text("DATE_TIME,CLOSE\n2024-01-01 00:00:00,1\n",
                                            encoding="utf-8")
    entry = tmp_path / "entry.py"
    entry.write_text(
        "import json, pathlib, sys\n"
        "config = json.loads(pathlib.Path(sys.argv[sys.argv.index('--load_config') + 1])"
        ".read_text())\n"
        "json.dump({'observed_timesteps': 64, 'compute_contract': {\n"
        "  'schema': 'compute_contract.v1', 'algorithm': 'PPO', 'library_version': '2.9.0',\n"
        "  'requested_training_transitions': config['total_timesteps'],\n"
        "  'training_transition_cap': config['training_transition_cap'],\n"
        "  'training_transitions_observed': 64, 'collected_rollouts': 1,\n"
        "  'optimization_epochs_completed': 1, 'optimizer_step_calls': 1,\n"
        "  'sb3_n_updates_meaning': 'optimization_epochs', 'cap_respected': True,\n"
        "  'counters_unavailable': ['evaluation_transitions']}},\n"
        "  open(config['results_file'], 'w'))\n", encoding="utf-8")

    out = tmp_path / "out"
    out.mkdir()
    flat = flat_for(tmp_path, entry_point=str(entry), save_log=str(out / "replay_log.json"))
    flat_path = out / "flat.json"
    flat_path.write_text(json.dumps(flat), encoding="utf-8")
    result = subprocess.run([sys.executable, str(RUNNER), "--load_config", str(flat_path)],
                            capture_output=True, text=True, cwd=str(tmp_path), timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr

    receipt = json.loads((out / "replay_log.json").read_text(encoding="utf-8"))
    assert receipt["compute_training_transitions_observed"] == 64.0
    assert receipt["compute_optimizer_step_calls"] == 1.0
    assert receipt["requested_timesteps"] == 64.0, "requested and observed stay distinct"
    contract = receipt["compute_contract"]
    assert contract["sb3_n_updates_meaning"] == "optimization_epochs"
    assert contract["counters_unavailable"] == ["evaluation_transitions"], (
        "what could not be measured is part of the receipt, and cannot survive as a float")
    assert contract["library_version"] == "2.9.0", "the library version is on the record"


# --- the evaluation ceiling, on the deployed evaluation loop ----------------------------

class StubEnv:
    """The smallest thing `_evaluate` can walk: it never terminates, so only a cap stops it."""

    def __init__(self):
        self.steps = 0

    def reset(self, seed=None):
        self.steps = 0
        return {"obs": 0.0}, {"equity": 100.0, "bar_index": 0}

    def step(self, action):
        self.steps += 1
        return ({"obs": 0.0}, 0.0, False, False,
                {"equity": 100.0, "bar_index": self.steps})

    def summary(self):
        return {"trades_total": 0}


class StubAgent:
    def predict(self, model, obs, deterministic=True):
        return 0


@pytest.mark.parametrize("cap", [8, 64])
def test_evaluation_stops_at_its_declared_ceiling(cap):
    """An endless environment is stopped by the cap, and the truncation is declared."""
    from pipeline_plugins.rl_pipeline import PipelinePlugin

    summary = PipelinePlugin()._evaluate(
        StubEnv(), StubAgent(), object(),
        {"eval_seed": 0, "evaluation_transition_cap": cap, "asset": "none", "timeframe": "none"})

    assert summary["episode_length"] == cap
    assert summary["evaluation_truncated_by_cap"] is True


def test_an_evaluation_that_ends_on_its_own_is_not_reported_as_truncated():
    from pipeline_plugins.rl_pipeline import PipelinePlugin

    class Ends(StubEnv):
        def step(self, action):
            obs, reward, _t, _tr, info = super().step(action)
            return obs, reward, self.steps >= 5, False, info

    summary = PipelinePlugin()._evaluate(
        Ends(), StubAgent(), object(),
        {"eval_seed": 0, "evaluation_transition_cap": 384, "asset": "none", "timeframe": "none"})

    assert summary["episode_length"] == 5
    assert summary["evaluation_truncated_by_cap"] is False
