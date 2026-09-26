"""The M4 CONFIRMATION execution BODY (2026-09-26).

`execute_confirmation` verified its gates, wrote the pre-result
ledger and returned; the 3024 units were never driven. This
battery covers the body that now drives them and, above all, that
the body changed NOTHING about who may run it:

- the gate chain still refuses FIRST, with the records absent,
  and the body is never even reached;
- no test in this file installs either external record, and no
  CONFIRMATION array, score or ledger is created anywhere — the
  body is exercised over DEVELOPMENT units only;
- the bank's C33 kill-17 construction guard still refuses
  CONFIRMATION bytes for every caller that does not explicitly
  pass the new pass-through flag.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import m4_confirmation_protocol as cp  # noqa: E402
import m4_confirmation_runner as cr  # noqa: E402
import m4_generator_bank as gb  # noqa: E402
import m4_v5_runner as rn  # noqa: E402

CLI = [sys.executable, str(REPO / "tools"
                           / "m4_confirmation_runner.py")]


@pytest.fixture(scope="module")
def design():
    return cp.bind_calibration_evidence(REPO)["design"]


def _dev_units(design, n=None):
    u = rn.intervention_units_v5(design, "DEVELOPMENT")
    return u if n is None else u[:n]


def _run_cli(*args):
    return subprocess.run([*CLI, *args], capture_output=True,
                          text=True, timeout=900,
                          env={"CUDA_VISIBLE_DEVICES": "",
                               "PATH": "/usr/bin:/bin",
                               "HOME": str(Path.home())})


# ---- PROOF 1: the gate still refuses, and creates nothing ----

def test_execution_path_refuses_without_records(tmp_path,
                                                monkeypatch):
    monkeypatch.setattr(cp, "MUSASHI_REVIEW_RECORD_PATH",
                        tmp_path / "absent.json")
    out = tmp_path / "run"
    with pytest.raises(SystemExit, match="ABSENT"):
        cr.run_confirmation(REPO, out)
    assert not out.exists(), (
        "the execution path created artifacts before refusing")


def test_gate_refusal_never_reaches_the_body(tmp_path,
                                             monkeypatch):
    """The body is not merely harmless without records — it is
    never entered."""
    monkeypatch.setattr(cp, "MUSASHI_REVIEW_RECORD_PATH",
                        tmp_path / "absent.json")

    def _never(*a, **k):
        raise AssertionError(
            "the unit body ran with the records absent")

    monkeypatch.setattr(cr, "execute_confirmation_units",
                        _never)
    with pytest.raises(SystemExit, match="ABSENT"):
        cr.run_confirmation(REPO, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_owner_record_absent_alone_still_refuses(
        tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "OWNER_EXECUTION_RECORD_PATH",
                        tmp_path / "absent_exec.json")
    out = tmp_path / "run"
    with pytest.raises(SystemExit, match="ABSENT|record"):
        cr.run_confirmation(REPO, out)
    assert not out.exists()


def test_no_confirmation_artifact_is_created_by_this_battery():
    state = Path.home() / ".local/share/agent-multi"
    assert list(state.glob("*m4*confirmation*")) == []
    assert not cp.MUSASHI_REVIEW_RECORD_PATH.exists()
    assert not cp.OWNER_EXECUTION_RECORD_PATH.exists()


# ---- the kill-17 construction guard is still closed ----

def test_confirmation_unit_still_refuses_by_default(design,
                                                    tmp_path):
    u = rn._iv_unit("CONFIRMATION", "sine", "white", 16, 0, 0)
    out = tmp_path / "o"
    (out / "intervention").mkdir(parents=True)
    acct = cr.new_accounting()
    with pytest.raises(SystemExit, match="RESERVED"):
        rn._run_intervention_unit_v5(design, u, out, acct)


def test_prior_role_census_never_builds_confirmation(design):
    cells = [("sine", "white")]
    d = cr.role_array_digests(design, cells)
    conf = gb.generate("CONFIRMATION", "sine", "white", 0,
                       allow_confirmation=True)
    assert cr.generator_array_digests(conf).isdisjoint(d)
    assert len(d) == 6 * (
        design["populations_v5"]["DEVELOPMENT_per_cell"]
        + design["populations_v5"]["CALIBRATION_per_cell"])


# ---- PROOF 2: the body over DEVELOPMENT units ----

def test_body_runs_development_units_end_to_end(tmp_path):
    out = tmp_path / "probe"
    r = _run_cli("development-execution-probe", "--out",
                 str(out), "--units", "2")
    assert r.returncode == 0, r.stderr[-800:]
    body = json.loads(r.stdout)
    assert body["role"] == "DEVELOPMENT"
    assert body["units_total"] == 2
    assert body["units_complete"] == 2
    assert body["units_new_this_session"] == 2
    assert body["census_complete"] is True
    assert body["session_status"] == "CENSUS_COMPLETE"
    assert body["confirmation_artifacts"] == 0
    assert body["accounting"]["optimization_updates"] > 0
    assert body["accounting"]["descriptor_evals"] == 2 * 4
    assert body["generators_disjointness_verified"] == 2
    assert body["prior_role_digests"] > 0
    assert [p for p in out.rglob("*")
            if "CONFIRMATION" in p.name] == []
    recs = sorted((out / "intervention").glob("*_summary.json"))
    assert len(recs) == 2
    for p in recs:
        assert oct(p.stat().st_mode)[-3:] == "600"


def test_body_resumes_and_never_reruns_a_complete_unit(tmp_path):
    out = tmp_path / "probe"
    first = _run_cli("development-execution-probe", "--out",
                     str(out), "--units", "2",
                     "--batch-units", "1")
    assert first.returncode == 0, first.stderr[-800:]
    b1 = json.loads(first.stdout)
    assert b1["units_new_this_session"] == 1
    assert b1["units_pending"] == 1
    assert b1["census_complete"] is False
    assert b1["session_status"] == "BATCH_FILLED_RESUMABLE"
    done = sorted((out / "intervention").glob("*_summary.json"))
    assert len(done) == 1
    before = hashlib.sha256(done[0].read_bytes()).hexdigest()
    second = _run_cli("development-execution-probe", "--out",
                      str(out), "--units", "2")
    assert second.returncode == 0, second.stderr[-800:]
    b2 = json.loads(second.stdout)
    assert b2["units_complete"] == 2
    assert b2["units_new_this_session"] == 1
    assert b2["census_complete"] is True
    after = hashlib.sha256(done[0].read_bytes()).hexdigest()
    assert after == before, (
        "a completed unit's record was rewritten on resume")
    assert len(list(out.glob("SESSION_*_REPORT.json"))) == 2


def test_partial_record_is_never_read_as_complete(tmp_path,
                                                  design):
    out = tmp_path / "probe"
    r = _run_cli("development-execution-probe", "--out",
                 str(out), "--units", "1")
    assert r.returncode == 0, r.stderr[-800:]
    rec = sorted((out / "intervention").glob("*_summary.json"))[0]
    full = rec.read_bytes()
    unit = _dev_units(design, 1)[0]
    # a truncated record at the FINAL name
    rec.write_bytes(full[:len(full) // 2])
    with pytest.raises(SystemExit):
        cr.read_complete_unit_record(rec, unit, "DEVELOPMENT")
    # valid JSON whose self-identity no longer re-derives
    doc = json.loads(full.decode())
    doc["paired_primary_difference"] = 99.9
    rec.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="UNCERTAIN"):
        cr.read_complete_unit_record(rec, unit, "DEVELOPMENT")
    # a record that binds another unit
    rec.write_bytes(full)
    other = rn._iv_unit("DEVELOPMENT", "sine", "white", 16, 1, 0)
    with pytest.raises(SystemExit, match="binds"):
        cr.read_complete_unit_record(rec, other, "DEVELOPMENT")


# ---- the body's own guards bite ----

def test_array_domain_disjointness_bites(tmp_path, design):
    units = _dev_units(design, 1)
    g = gb.generate(units[0]["role"], units[0]["family"],
                    units[0]["noise_coord"],
                    units[0]["generator_index"])
    poisoned = cr.generator_array_digests(g)
    out = tmp_path / "o"
    out.mkdir()
    with pytest.raises(SystemExit, match="collide with prior"):
        cr.execute_confirmation_units(
            design, units, out, cr.new_accounting(),
            expect_role="DEVELOPMENT", allow_confirmation=False,
            prior_digests=poisoned)
    assert list((out / "intervention").glob("*_summary.json")) \
        == []


def test_empty_prior_census_refuses(tmp_path, design):
    out = tmp_path / "o"
    out.mkdir()
    with pytest.raises(SystemExit, match="EMPTY"):
        cr.execute_confirmation_units(
            design, _dev_units(design, 1), out,
            cr.new_accounting(), expect_role="DEVELOPMENT",
            allow_confirmation=False, prior_digests=set())


def test_one_census_one_role(tmp_path, design):
    out = tmp_path / "o"
    out.mkdir()
    with pytest.raises(SystemExit, match="one census, one role"):
        cr.execute_confirmation_units(
            design, _dev_units(design, 1), out,
            cr.new_accounting(), expect_role="CONFIRMATION",
            allow_confirmation=True, prior_digests={"x" * 64})


# ---- the sealed probe is unchanged ----

def test_sealed_development_probe_unchanged(tmp_path):
    p = cr.development_mechanics_probe(REPO, tmp_path / "sealed")
    assert p["records"] == 2
    assert p["confirmation_artifacts"] == 0
    assert len(p["units_run"]) == 2


# ---- the sealed resource limits still govern the body ----

def test_stop_file_stops_the_session_resumably(tmp_path,
                                               design):
    out = tmp_path / "probe"
    out.mkdir()
    (out / "M4_RUN_STOP").write_text("operator stop")
    units = _dev_units(design, 2)
    prior = cr.role_array_digests(design, cr.cells_of_units(units),
                                  roles=("CALIBRATION",))
    body = cr.execute_confirmation_units(
        design, units, out, cr.new_accounting(),
        expect_role="DEVELOPMENT", allow_confirmation=False,
        prior_digests=prior)
    assert body["session_status"] == "STOP_REQUESTED"
    assert body["units_new_this_session"] == 0
    assert body["units_pending"] == 2
    assert body["census_complete"] is False
    assert list((out / "intervention").glob("*_summary.json")) \
        == []
    (out / "M4_RUN_STOP").unlink()
    body2 = cr.execute_confirmation_units(
        design, units, out, cr.new_accounting(),
        expect_role="DEVELOPMENT", allow_confirmation=False,
        prior_digests=prior)
    assert body2["census_complete"] is True
    assert body2["units_new_this_session"] == 2


def test_heartbeat_is_written_by_the_sealed_limit_machinery(
        tmp_path, design):
    out = tmp_path / "probe"
    out.mkdir()
    units = _dev_units(design, 1)
    prior = cr.role_array_digests(design, cr.cells_of_units(units),
                                  roles=("CALIBRATION",))
    cr.execute_confirmation_units(
        design, units, out, cr.new_accounting(),
        expect_role="DEVELOPMENT", allow_confirmation=False,
        prior_digests=prior)
    hb = json.loads((out / "M4_RUN_HEARTBEAT.json").read_text())
    assert set(hb) == {"monotonic", "updates_done"}


# ---- the records the body writes are what the verifier eats ----

def test_body_records_satisfy_the_verifier_contract(tmp_path,
                                                    design):
    import m4_v5_protocol as pv
    out = tmp_path / "probe"
    r = _run_cli("development-execution-probe", "--out",
                 str(out), "--units", "2")
    assert r.returncode == 0, r.stderr[-800:]
    recs = sorted((out / "intervention").glob("*_summary.json"))
    assert len(recs) == 2
    for p in recs:
        rec = json.loads(p.read_text())
        assert len(rec["unit_id"].split("::")) == 7
        assert rec.get("unit_status") is None
        assert set(rec["arms"]) == set(pv.CHECKPOINTS)
        for arm in rec["arms"].values():
            assert isinstance(arm["restricted_endpoint"], int)
            assert isinstance(arm["updates_done"], int)
        eff = (rec["arms"]["calibration_stop"][
                   "restricted_endpoint"]
               - rec["arms"]["initialization"][
                   "restricted_endpoint"])
        assert abs(eff - rec["paired_primary_difference"]) \
            <= 1e-9, (
            "the verifier's re-derivation rule would refuse this "
            "record")
