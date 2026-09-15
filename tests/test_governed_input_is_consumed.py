"""Whether the delivered file is READ, or merely named in a configuration.

§3 of `predictor/docs/handoffs/MUSASHI_WORKER_ACTIVATION_AND_R1_R6_COMPLETION_2026_09_15.md`:

    "Prove that the governed file is actually read, not just named in a config: change the
     delivered fixture deterministically and verify consumed row identities/digest; make the
     old sample input unavailable in an isolated test and exercise the same entry point.
     Retain the regression for flat versus nested input configuration."

The defect this comes from is real and was found by running the thing: the runner wrote the
governed path only under `data`, while `app/config.py` reads the flat `input_data_file`, so a
replay opened `examples/data/eurusd_sample.csv` — a file no campaign ever delivered — and
nobody would have noticed from the receipt.

Three questions are separated here, because they fail differently:

* does the path the runner writes reach the place the application reads? (flat vs nested);
* does the process actually open the delivered bytes? (traced on the real file system);
* if the sample file is gone, does the same entry point still work? (it must, and a run that
  silently fell back would now fail loudly).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tools" / "governed_offline_replay.py"
SAMPLE = REPO / "examples" / "data" / "eurusd_sample.csv"

spec = importlib.util.spec_from_file_location("governed_offline_replay_under_test", RUNNER)
replay = importlib.util.module_from_spec(spec)
sys.modules["governed_offline_replay_under_test"] = replay
spec.loader.exec_module(replay)


def fixture(tmp_path, marker: float):
    """A delivered file whose content is decided by `marker`, so two deliveries differ."""
    rows = ["DATE_TIME,CLOSE,HIGH,LOW,OPEN"]
    for index in range(8):
        value = marker + index
        rows.append(f"2024-01-01 0{index}:00:00,{value},{value + 1},{value - 1},{value}")
    path = tmp_path / f"delivered_{marker:.0f}.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merged_for(tmp_path, delivered: Path) -> dict:
    template = tmp_path / "experiment.json"
    template.write_text(json.dumps({"experiment": {"name": "consumption"},
                                    "data": {}, "training": {"total_timesteps": 4},
                                    "environment": {}}), encoding="utf-8")
    flat = {"experiment_config": str(template), "runtime_overlay": str(tmp_path / "o.json"),
            "input_data_file": str(delivered), "total_timesteps": 4,
            "save_log": str(tmp_path / "out" / "replay_log.json")}
    (tmp_path / "out").mkdir(exist_ok=True)
    return replay.merged_config(flat, template, tmp_path / "out")


def test_the_delivered_path_reaches_the_key_the_application_reads(tmp_path):
    """The regression for the defect itself: flat AND nested, not one or the other."""
    delivered = fixture(tmp_path, 100)
    merged = merged_for(tmp_path, delivered)
    assert merged["input_data_file"] == str(delivered), (
        "app/config.py reads the flat key; writing only the nested one sent the run to the "
        "sample file")
    assert merged["data"]["input_data_file"] == str(delivered), (
        "the nested key is what the experiment template documents; both must agree")


def test_the_run_writes_its_summary_into_its_own_directory(tmp_path):
    """Observed counters can only be read from this run if the app writes them here."""
    delivered = fixture(tmp_path, 100)
    merged = merged_for(tmp_path, delivered)
    assert merged["results_file"] == str(tmp_path / "out" / "summary.json")


def test_two_different_deliveries_produce_two_different_resolved_configs(tmp_path):
    """Determinism with a difference: the same code, a changed fixture, a changed identity."""
    first = merged_for(tmp_path, fixture(tmp_path, 100))
    second = merged_for(tmp_path, fixture(tmp_path, 500))
    assert first["input_data_file"] != second["input_data_file"]
    assert digest(Path(first["input_data_file"])) != digest(Path(second["input_data_file"]))


def reader_entry(tmp_path: Path) -> Path:
    """A tiny stand-in application that records exactly what it opened."""
    entry = tmp_path / "reader_main.py"
    entry.write_text(
        "import hashlib, json, pathlib, sys\n"
        "config = json.loads(pathlib.Path(sys.argv[sys.argv.index('--load_config') + 1])"
        ".read_text())\n"
        "path = pathlib.Path(config['input_data_file'])\n"
        "rows = path.read_text().strip().split('\\n')[1:]\n"
        "json.dump({'opened': str(path),\n"
        "           'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),\n"
        "           'row_ids': [r.split(',')[0] for r in rows],\n"
        "           'values': [r.split(',')[1] for r in rows],\n"
        "           'observed_timesteps': len(rows)},\n"
        "          open(config['results_file'], 'w'))\n", encoding="utf-8")
    return entry


def run_replay(tmp_path: Path, delivered: Path, entry: Path, out: Path, cwd: Path | None = None):
    template = tmp_path / "experiment.json"
    if not template.is_file():
        template.write_text(json.dumps({"experiment": {}, "data": {}, "training": {},
                                        "environment": {}}), encoding="utf-8")
    overlay = tmp_path / "o.json"
    if not overlay.is_file():
        overlay.write_text("{}", encoding="utf-8")
    out.mkdir(parents=True, exist_ok=True)
    flat_path = out / "flat.json"
    flat_path.write_text(json.dumps({
        "experiment_config": str(template), "runtime_overlay": str(overlay),
        "input_data_file": str(delivered), "entry_point": str(entry),
        "total_timesteps": 8, "save_log": str(out / "replay_log.json")}), encoding="utf-8")
    home = cwd or tmp_path
    return subprocess.run([sys.executable, str(RUNNER), "--load_config", str(flat_path)],
                          capture_output=True, text=True, cwd=str(home),
                          env={"PATH": "/usr/bin:/bin", "HOME": str(home)}, timeout=180)


def read_rows(path: Path):
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    return [line.split(",")[0] for line in lines[1:]]


def test_the_process_opens_the_delivered_file_and_not_the_sample(tmp_path):
    """Traced on the file system rather than inferred: the entry point records what it opened."""
    delivered = fixture(tmp_path, 700)
    out = tmp_path / "out"
    result = run_replay(tmp_path, delivered, reader_entry(tmp_path), out)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    assert summary["opened"] == str(delivered), "the process opened another file"
    assert summary["input_sha256"] == digest(delivered), "the bytes read are the bytes delivered"
    assert summary["row_ids"] == read_rows(delivered), (
        "row identities must match the delivered file, not merely the row count")

    receipt = json.loads((out / "replay_log.json").read_text(encoding="utf-8"))
    assert receipt["custody"]["input_data_sha256"] == summary["input_sha256"], (
        "the receipt's custody digest and the bytes actually read must be the same file")
    assert receipt["observed_timesteps"] == 8.0, (
        "what the application reported is what the receipt carries")
    assert receipt["requested_timesteps"] == 8.0


def test_a_changed_delivery_changes_what_was_consumed(tmp_path):
    """Change the fixture deterministically and the consumed CONTENT moves with it."""
    entry = reader_entry(tmp_path)
    outcomes = {}
    for marker in (100, 900):
        delivered = fixture(tmp_path, marker)
        out = tmp_path / f"out_{marker}"
        result = run_replay(tmp_path, delivered, entry, out)
        assert result.returncode == 0, result.stdout + result.stderr
        outcomes[marker] = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert outcomes[100]["input_sha256"] != outcomes[900]["input_sha256"]
    assert outcomes[100]["values"] != outcomes[900]["values"], (
        "the values the process read must follow the delivery")
    assert outcomes[100]["row_ids"] == outcomes[900]["row_ids"], (
        "the identities are timestamps and do NOT change here: identity alone is not proof, "
        "which is exactly why the digest and the values are checked too")


def test_a_decoy_sample_is_never_opened_even_when_it_exists(tmp_path):
    """The isolated case, with the fallback made possible on purpose so it can be caught.

    A decoy `examples/data/eurusd_sample.csv` is planted in the working directory the process
    runs in. If the run ever resolved that relative default instead of the delivered path, the
    entry point would see it and refuse.
    """
    isolated = tmp_path / "isolated"
    (isolated / "examples" / "data").mkdir(parents=True)
    decoy = isolated / "examples" / "data" / "eurusd_sample.csv"
    decoy.write_text("DATE_TIME,CLOSE\n1999-01-01 00:00:00,0\n", encoding="utf-8")

    delivered = fixture(tmp_path, 300)
    entry = tmp_path / "strict_main.py"
    entry.write_text(
        "import hashlib, json, pathlib, sys\n"
        "config = json.loads(pathlib.Path(sys.argv[sys.argv.index('--load_config') + 1])"
        ".read_text())\n"
        "path = pathlib.Path(config['input_data_file'])\n"
        "assert 'eurusd_sample' not in str(path), 'the run fell back to the sample input'\n"
        "assert path.is_file(), f'the delivered input is not there: {path}'\n"
        "rows = path.read_text().strip().split('\\n')[1:]\n"
        "json.dump({'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),\n"
        "           'observed_timesteps': len(rows)}, open(config['results_file'], 'w'))\n",
        encoding="utf-8")

    out = tmp_path / "out"
    result = run_replay(tmp_path, delivered, entry, out, cwd=isolated)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["input_sha256"] == digest(delivered)
    assert digest(decoy) != summary["input_sha256"], "the decoy was never the file read"
    receipt = json.loads((out / "replay_log.json").read_text(encoding="utf-8"))
    assert receipt["observed_timesteps"] == 8.0
