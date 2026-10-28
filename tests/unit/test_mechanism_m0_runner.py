import json
import subprocess
import sys
from pathlib import Path


def test_runner_two_processes_byte_identical_and_business_no_daily_cap():
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "tools/mechanism_m0.py")]
    first = subprocess.check_output(command, cwd=root)
    second = subprocess.check_output(command, cwd=root)
    assert first == second
    report = json.loads(first)
    assert report["unilateral_20_by_27"]["distinct_profiles"] == 521
    assert report["training_runs"] == report["market_rollouts"] == 0
    contract = json.loads((root / "docs/work_plan/mechanism_synthesis/BUSINESS_CONTRACT.json").read_text())
    assert contract["max_entries_per_day"] is None
    assert contract["max_simultaneous_positions"] == 1
    assert contract["entry_wait_after_session_open_hours"] == 12
    assert contract["forced_flat_before_session_close_hours"] == 12
