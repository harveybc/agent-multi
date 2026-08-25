"""Focused tests for tools/warmup_context_probe.py (SOTA-R06)."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[1] / "tools" / "warmup_context_probe.py"
spec = importlib.util.spec_from_file_location("warmup_probe", PROBE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_verify_role_csv_refuses_hash_mismatch(tmp_path):
    csv = tmp_path / "role.csv"
    csv.write_text("a,b\n1,2\n")
    role = {"csv": str(csv), "csv_sha256": "0" * 64}
    with pytest.raises(SystemExit) as e:
        mod.verify_role_csv("fit_train", role)
    assert "REFUSED" in str(e.value)


def test_verify_role_csv_accepts_matching_hash(tmp_path):
    csv = tmp_path / "role.csv"
    csv.write_text("a,b\n1,2\n")
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    assert mod.verify_role_csv("fit_train",
                               {"csv": str(csv), "csv_sha256": digest}) == digest


def test_source_head_zero_fraction_distinguishes_data_zeros(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n0,0\n0,1\n1,1\n1,1\n")
    frac = mod.raw_head_zero_fraction(csv, 4)
    assert frac == pytest.approx(3 / 8)


def test_zeros_profile_metrics_first_dense_step():
    m = mod.zeros_profile_metrics([1.0, 0.6, 0.02, 0.01])
    assert m["zero_fraction_at_reset"] == 1.0
    assert m["first_step_below_50pct_zeros"] == 2
    assert m["first_step_below_5pct_zeros"] == 2


def test_canonical_config_sha_is_order_independent():
    a = mod.canonical_config_sha({"x": 1, "y": [2, 3]})
    b = mod.canonical_config_sha({"y": [2, 3], "x": 1})
    assert a == b and len(a) == 64


def test_load_arm_contract_reads_launch_and_manifest(tmp_path):
    arm = tmp_path / "arm"
    (arm / "normal" / "nested_splits").mkdir(parents=True)
    (arm / "normal_report.launch_manifest.json").write_text(
        json.dumps({"effective_config": {"k": 1}}))
    (arm / "normal" / "nested_splits" / "nested_split_manifest.json").write_text(
        json.dumps({"roles": {"fit_train": {"csv": None}}}))
    launch, manifest = mod.load_arm_contract(arm, "normal")
    assert launch["effective_config"] == {"k": 1}
    assert "fit_train" in manifest["roles"]
