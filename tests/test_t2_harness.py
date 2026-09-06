"""T2.1/T2.4 adversarial battery: leakage, bytes decoupling,
duplication, seed-as-unit, forbidden claims, cost, identity and
the confirmatory gate — frozen while the public bank is absent."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault(
    "B4_T1_PREPROCESSOR_ROOT",
    str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))

import t2_public_data_census as census_mod  # noqa: E402
import t2_assay_harness as hz  # noqa: E402

CENSUS = census_mod.build_census()
_dev_present = pytest.mark.skipif(
    not any(v["status"] == "PRESENT_DEVELOPMENT_ONLY"
            for v in CENSUS["development_only_units"].values()),
    reason="no development units in this environment")


def test_census_verdict_and_exclusions():
    assert CENSUS["verdict"] == "CONFIRMATORY_BANK_UNAVAILABLE"
    assert "financial" in CENSUS["excluded_classes"]
    assert "synthetic_own" in CENSUS["excluded_classes"]
    ded = CENSUS["deficit"]["operator_must_supply"]
    assert {"monash_subset", "etth1", "weather_jena"} <= set(ded)


@_dev_present
def test_bytes_decoupling_refuses():
    c = copy.deepcopy(CENSUS)
    c["development_only_units"]["sm_co2"]["bytes_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="decoupled"):
        hz.load_task_unit(c, "sm_co2")


def test_no_dataframe_entry_point():
    """A dataframe supplied apart from its bytes has no entry: the
    only loader takes (census, unit_id) and re-reads the bytes."""
    import inspect
    sig = inspect.signature(hz.load_task_unit)
    assert list(sig.parameters) == ["census", "unit_id"]
    src = (REPO / "tools/t2_assay_harness.py").read_text()
    assert "read_csv(io.BytesIO(raw))" in src


@_dev_present
def test_future_rows_cannot_reach_training(tmp_path):
    """Leakage adversary: mutating validation/score rows changes
    neither the train-role features of any arm nor the fitted
    causal transform on the train role."""
    co = hz.load_co()
    unit = hz.load_task_unit(CENSUS, "sm_co2")
    y = unit["y"]
    roles = hz.roles_of(len(y))
    lo_t, hi_t = roles["train"]
    y_mut = y.copy()
    y_mut[hi_t:] = 999.0
    den_a = hz.causal_denoise(co, y, roles, "sm_co2")
    den_b = hz.causal_denoise(co, y_mut, roles, "sm_co2")
    assert den_a["d"][:hi_t].tobytes() == \
        den_b["d"][:hi_t].tobytes()
    n_a = hz.nuisance(y, roles, "sm_co2|ewma")
    n_b = hz.nuisance(y_mut, roles, "sm_co2|ewma")
    assert all(a.tobytes() == b.tobytes()
               for a, b in zip(n_a, n_b))
    for series_pair in (([y], [y_mut]),
                        ([y, den_a["d"], y - den_a["d"]],
                         [y_mut, den_b["d"], y_mut - den_b["d"]])):
        Xa = hz._lag_matrix(series_pair[0], lo_t, hi_t, 1)
        Xb = hz._lag_matrix(series_pair[1], lo_t, hi_t, 1)
        assert Xa.tobytes() == Xb.tobytes()


@_dev_present
def test_record_schema_and_forbidden_claims():
    co = hz.load_co()
    unit = hz.load_task_unit(CENSUS, "sm_sunspots")
    rec = hz.assay_unit(co, unit)
    hz.check_record_schema(rec)
    # smuggled eligibility refuses
    bad = copy.deepcopy(rec)
    bad["publicly_eligible"] = True
    with pytest.raises(SystemExit, match="exact schema"):
        hz.check_record_schema(bad)
    # a noise/SNR claim refuses even inside nested content
    bad2 = copy.deepcopy(rec)
    bad2["results"]["X"]["ridge"]["true_snr"] = 10.0
    bad2["record_sha256"] = hashlib.sha256(json.dumps(
        {k: bad2[k] for k in sorted(bad2)
         if k != "record_sha256"}, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    with pytest.raises(SystemExit, match="forbidden claim"):
        hz.check_record_schema(bad2)
    # cost omission refuses
    bad3 = copy.deepcopy(rec)
    bad3.pop("cost")
    with pytest.raises(SystemExit, match="exact schema"):
        hz.check_record_schema(bad3)
    # tampered record digest refuses
    bad4 = copy.deepcopy(rec)
    bad4["results"]["X"]["ridge"]["mae"] = 0.0
    with pytest.raises(SystemExit, match="does not re-derive"):
        hz.check_record_schema(bad4)
    # the statistical unit is the task, never the seed
    assert rec["unit_is_the_statistical_unit"] is True
    assert set(rec["claim_classes_only"]) == {
        "utility", "calibration", "extreme_preservation", "cost"}
    # operator selection is bound to the T1 record and the fitted
    # artifact identity travels in the record
    assert rec["operator"]["selection_source"] == \
        "T1_v4_record_LAB_CALIBRATED"
    assert len(rec["operator"]["artifact_sha256"]) == 64
    # seasonal period source is the design, not the data
    assert rec["results"]["seasonal_naive"]["period_source"] == \
        "predeclared_design_constant"


def test_confirmatory_gate_refuses_public_data_required(tmp_path):
    cpath = tmp_path / "census.json"
    cpath.write_text(json.dumps(CENSUS))
    rc = subprocess.run(
        [sys.executable, str(REPO / "tools/t2_assay_harness.py"),
         "--census", str(cpath),
         "--output", str(tmp_path / "out.json"),
         "--confirmatory"],
        capture_output=True, text=True,
        env={**os.environ})
    assert rc.returncode != 0
    assert "PUBLIC_DATA_REQUIRED" in rc.stderr + rc.stdout
    assert not (tmp_path / "out.json").exists()


def test_task_duplication_refuses(tmp_path):
    c = copy.deepcopy(CENSUS)
    u = c["development_only_units"]
    if "sm_co2" in u and u["sm_co2"].get("bytes_sha256"):
        u["sm_dup"] = dict(u["sm_co2"])
        cpath = tmp_path / "census.json"
        cpath.write_text(json.dumps(c))
        rc = subprocess.run(
            [sys.executable,
             str(REPO / "tools/t2_assay_harness.py"),
             "--census", str(cpath),
             "--output", str(tmp_path / "out.json"),
             "--development-only"],
            capture_output=True, text=True, env={**os.environ})
        assert rc.returncode != 0
        assert "duplication" in rc.stderr + rc.stdout


def test_financial_and_synthetic_have_no_path():
    """Excluded classes carry no loadable unit and the census
    never lists them as consumable."""
    for uid in CENSUS["development_only_units"]:
        assert not uid.startswith(("fx_", "eth_", "synthetic_"))
    with pytest.raises(SystemExit, match="not in the census"):
        hz.load_task_unit(CENSUS, "eurusd_hour")


def test_no_authority_tokens_in_t2_tools():
    for tool in ("t2_public_data_census.py",
                 "t2_assay_harness.py"):
        text = (REPO / "tools" / tool).read_text()
        for token in ("PUBLICLY_ELIGIBLE\"", "REPRODUCED_UNDER",
                      '"AUTHORIZED"'):
            assert token not in text, (tool, token)
