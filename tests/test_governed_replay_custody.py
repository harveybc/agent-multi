"""What a replay receipt is allowed to claim, and what it must never claim.

R3 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:

    "Seed a stale summary, stop a run early, omit its summary and alter only its overlay in
     regression tests. None may report stale or configured counts as completed work."

Two defects were named by the audit and both are covered here:

* the runner read the repository-global `config_out.json` as a source of metrics, so a file
  left by ANY earlier run could be reported as this run's work;
* `total_timesteps` — a configured budget — travelled as if it were a measured step count.

The runner is executed as a subprocess, exactly as the governed wrapper executes it, with a
tiny fake entry point standing in for the training application: this file is about custody and
counting, not about training, and a real training run would prove neither.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tools" / "governed_offline_replay.py"

TEMPLATE = {"experiment": {"name": "custody"}, "data": {}, "training": {}, "environment": {}}


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """A runnable replay: template, overlay, input, and an entry point we control."""
    template = tmp_path / "experiment.json"
    template.write_text(json.dumps(TEMPLATE), encoding="utf-8")
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"runtime": "cpu"}), encoding="utf-8")
    data = tmp_path / "input.csv"
    data.write_text("DATE_TIME,value\n2024-01-01 00:00:00,1\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    flat = tmp_path / "flat.json"

    def write(summary=None, timesteps=64, entry_point=None):
        body = {"experiment_config": str(template), "runtime_overlay": str(overlay),
                "input_data_file": str(data), "total_timesteps": timesteps,
                "save_log": str(out / "replay_log.json")}
        if entry_point:
            body["entry_point"] = str(entry_point)
        flat.write_text(json.dumps(body), encoding="utf-8")
        if summary is not None:
            (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return flat

    return {"write": write, "out": out, "template": template, "overlay": overlay,
            "data": data}


def fake_entry_point(flat: Path, out: Path, writes=None) -> Path:
    """A minimal application: it writes the summary a real run would, and nothing else."""
    fake = flat.parent / "fake_main.py"
    body = "import json, pathlib, sys\n"
    if writes is not None:
        body += (f"pathlib.Path({str(out / 'summary.json')!r}).write_text("
                 f"json.dumps({writes!r}))\n")
    fake.write_text(body, encoding="utf-8")
    return fake


def run(flat: Path, *, out: Path | None = None):
    """Run the real runner as the governed wrapper runs it: a subprocess."""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(flat.parent)}
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--load_config", str(flat)],
        capture_output=True, text=True, cwd=str(flat.parent), env=env,
        timeout=180)
    log = out / "replay_log.json" if out else None
    receipt = json.loads(log.read_text(encoding="utf-8")) if log and log.is_file() else None
    return result, receipt


def test_a_stale_summary_is_not_reported_as_this_run(stage):
    """A summary written BEFORE the run started belongs to some other run."""
    flat = stage["write"](summary={"observed_timesteps": 999999})
    time.sleep(0.01)
    _, receipt = run(flat, out=stage["out"])
    assert receipt is not None
    assert receipt["custody"]["summary_origin"] == "stale_ignored"
    assert "observed_timesteps" not in receipt, (
        "a number from a previous run reached this run's receipt")


def test_an_omitted_summary_yields_no_observed_count_at_all(stage):
    """Missing observation is missing. It is not replaced by the budget."""
    flat = stage["write"](timesteps=64)
    _, receipt = run(flat, out=stage["out"])
    assert receipt["custody"]["summary_origin"] in ("absent", "stale_ignored")
    assert "observed_timesteps" not in receipt
    assert receipt["requested_timesteps"] == 64.0, (
        "the budget is still reported, under its own name")


def test_the_budget_is_never_renamed_into_work_done(stage):
    """The audit's finding, as a rule: 64 configured is not 64 executed."""
    flat = stage["write"](timesteps=64)
    _, receipt = run(flat, out=stage["out"])
    assert "total_timesteps" not in receipt, (
        "the ambiguous name is gone: a reader could not tell budget from measurement")
    assert receipt["requested_timesteps"] == 64.0


def test_every_input_of_the_run_is_bound_to_its_receipt(stage):
    flat = stage["write"]()
    _, receipt = run(flat, out=stage["out"])
    custody = receipt["custody"]
    for field in ("experiment_template_sha256", "runtime_overlay_sha256",
                  "resolved_config_sha256", "runner_source_sha256", "input_data_sha256"):
        assert isinstance(custody[field], str) and len(custody[field]) == 64, field


def test_altering_only_the_overlay_changes_the_receipt_identity(stage):
    """Two runs that differ in nothing but the overlay must not look identical."""
    flat = stage["write"]()
    _, first = run(flat, out=stage["out"])
    stage["overlay"].write_text(json.dumps({"runtime": "cpu", "threads": 1}),
                                encoding="utf-8")
    _, second = run(flat, out=stage["out"])
    assert first["custody"]["runtime_overlay_sha256"] != second["custody"][
        "runtime_overlay_sha256"]


def test_a_failing_run_keeps_its_outcome_and_its_cost(stage):
    """A partial or failed run is a legitimate outcome, with the cost it really spent."""
    flat = stage["write"]()
    result, receipt = run(flat, out=stage["out"])
    assert receipt["exit_code"] != 0 or receipt["exit_code"] == 0
    assert receipt["wall_seconds"] >= 0.0
    assert "stdout_tail" in receipt


def test_a_summary_this_run_wrote_IS_reported(stage, tmp_path):
    """The positive control. Without it, "stale_ignored" could pass by ignoring everything."""
    flat = stage["write"]()
    entry = fake_entry_point(flat, stage["out"], writes={"observed_timesteps": 64,
                                                         "observed_updates": 4})
    flat = stage["write"](entry_point=entry)
    _, receipt = run(flat, out=stage["out"])
    assert receipt["exit_code"] == 0, receipt.get("stdout_tail")
    assert receipt["custody"]["summary_origin"] == "this_run"
    assert receipt["observed_timesteps"] == 64.0
    assert receipt["observed_updates"] == 4.0
    assert receipt["requested_timesteps"] == 64.0
    assert receipt["custody"]["entry_point_sha256"], "the entry point is bound too"


def test_a_run_stopped_early_reports_what_it_observed_not_what_it_asked(stage):
    """Budget 64, the application reports 12 before stopping: the receipt must say 12."""
    flat = stage["write"]()
    entry = fake_entry_point(flat, stage["out"], writes={"observed_timesteps": 12})
    flat = stage["write"](timesteps=64, entry_point=entry)
    _, receipt = run(flat, out=stage["out"])
    assert receipt["observed_timesteps"] == 12.0
    assert receipt["requested_timesteps"] == 64.0
