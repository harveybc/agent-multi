"""B4 C55: the external-watchdog acceptance battery — real
subprocesses through the supervisor's production seam."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import b4_cell_supervisor as sup  # noqa: E402

PY = sys.executable
V7 = (Path.home() / ".local/share/agent-multi/"
      "b4_campaign_results_v7_20260907")


def _world(tmp_path, cell="oTEST_seed1"):
    out = tmp_path / "root"
    cdir = out / cell
    (cdir / "cell_runtime").mkdir(parents=True)
    os.chmod(out, 0o700)
    return out, cdir


def _body(script):
    """A child body honoring the production contract: last two
    argv items are `--capability <path>`."""
    return [PY, "-c",
            "import sys\n"
            "cap = sys.argv[sys.argv.index('--capability')+1]\n"
            + script]


STALL_BODY = """
import json, os, time
cell = os.path.dirname(cap)
# observable durable progress, then a callback-starved stall
for i in range(2):
    with open(os.path.join(cell, 'cell_runtime',
                           'status.json'), 'w') as fh:
        json.dump({'epoch_completed': i}, fh)
    time.sleep(0.1)
x = 1.0
while True:
    x = x * 1.0000000001
"""

IGNORE_STOP_BODY = STALL_BODY   # same shape: never reads STOP

GRACEFUL_BODY = """
import json, os, time
cell = os.path.dirname(cap)
stop = os.path.join(cell, 'STOP')
i = 0
while True:
    with open(os.path.join(cell, 'cell_runtime',
                           'status.json'), 'w') as fh:
        json.dump({'epoch_completed': i}, fh)
    if os.path.exists(stop):
        with open(os.path.join(cell,
                               'B4_CELL_TERMINAL.json'),
                  'w') as fh:
            json.dump({'terminal': 'EXTERNALLY_STOPPED',
                       'cell': os.path.basename(cell)}, fh)
        raise SystemExit(0)
    i += 1
    time.sleep(0.2)
"""

LATE_COMPLETED_BODY = """
import json, os, time, signal
cell = os.path.dirname(cap)
stop = os.path.join(cell, 'STOP')
def onterm(sig, frm):
    with open(os.path.join(cell, 'B4_CELL_TERMINAL.json'),
              'w') as fh:
        json.dump({'terminal': 'COMPLETED',
                   'cell': os.path.basename(cell)}, fh)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, onterm)
with open(os.path.join(cell, 'cell_runtime', 'status.json'),
          'w') as fh:
    json.dump({'epoch_completed': 0}, fh)
x = 1.0
while True:
    x = x * 1.0000000001
"""

GRANDCHILD_BODY = """
import json, os, subprocess, sys, time
cell = os.path.dirname(cap)
with open(os.path.join(cell, 'cell_runtime', 'status.json'),
          'w') as fh:
    json.dump({'epoch_completed': 0}, fh)
subprocess.Popen([sys.executable, '-c',
                  'import time\\nwhile True: time.sleep(1)'])
x = 1.0
while True:
    x = x * 1.0000000001
"""


def _run(out, cdir, body, **kw):
    args = dict(cell_id=cdir.name, out_root=out, cell_dir=cdir,
                attempt_id="attempt_test01",
                child_cmd=_body(body), device="cpu",
                wall_s=kw.pop("wall_s", 60.0),
                liveness_silence_s=kw.pop("liveness", 1.0),
                grace_stop_s=kw.pop("grace_stop", 1.0),
                grace_term_s=kw.pop("grace_term", 1.0),
                poll_s=kw.pop("poll", 0.3))
    args.update(kw)
    return sup.run_cell_supervised(**args)


def test_1_2_stall_terminated_reaped_escalation(tmp_path):
    """Kills 1+2: a child that stops consuming callbacks and
    ignores the stop file is terminated by the parent, requires
    escalation, and is fully reaped."""
    out, cdir = _world(tmp_path)
    r = _run(out, cdir, STALL_BODY)
    assert r["outcome"] == "QUARANTINED_RUNTIME_STALL"
    assert r["stop_reason"] == "TELEMETRY_STALL"
    assert "stop_request_durable" in r["escalation"]
    assert "sigterm_group" in r["escalation"]
    assert "reaped_group_empty" in r["escalation"]
    assert r["group_empty"] is True
    assert (cdir / "STOP").exists()
    assert list(cdir.glob("SUPERVISOR_INCIDENT_*"))


def test_3_graceful_stop_typed_terminal(tmp_path):
    """Kill 3: a cooperative child writes a genuine typed
    terminal during the graceful interval — the supervisor
    reports the FACT (graceful ack), never fabricates."""
    out, cdir = _world(tmp_path)
    r = _run(out, cdir, GRACEFUL_BODY, wall_s=1.2,
             liveness=30.0, grace_stop=5.0)
    assert r["stop_reason"] == "HARD_WALL"
    assert r["outcome"] == "TERMINAL_PRESENT_GRACEFUL_ACK"
    term = json.loads((cdir / "B4_CELL_TERMINAL.json"
                       ).read_text())
    assert term["terminal"] == "EXTERNALLY_STOPPED"


def test_4_missing_terminal_quarantined(tmp_path):
    """Kill 4: no terminal -> quarantined, with a supervisor
    class record and no fabricated terminal."""
    out, cdir = _world(tmp_path)
    r = _run(out, cdir, STALL_BODY)
    assert r["outcome"] == "QUARANTINED_RUNTIME_STALL"
    assert not (cdir / "B4_CELL_TERMINAL.json").exists()
    cls = json.loads(next(cdir.glob(
        "SUPERVISOR_ATTEMPT_CLASS_*.json")).read_text())
    assert cls["classification"] == "QUARANTINED_RUNTIME_STALL"


def test_5_late_completed_never_completes(tmp_path):
    """Kill 5: a COMPLETED terminal written AFTER the stop
    request (on SIGTERM) can never become a completion."""
    out, cdir = _world(tmp_path)
    r = _run(out, cdir, LATE_COMPLETED_BODY, liveness=1.0,
             grace_stop=0.5, grace_term=3.0)
    assert (cdir / "B4_CELL_TERMINAL.json").exists()
    assert r["outcome"] == "QUARANTINED_RUNTIME_STALL"
    cls = json.loads(next(cdir.glob(
        "SUPERVISOR_ATTEMPT_CLASS_*.json")).read_text())
    assert cls["terminal_late_completed"] is True


def test_6_campaign_stop_blocks_next_claim():
    """Kill 6: CAMPAIGN_STOP holds the scheduler before any next
    claim (productive schedule_next path)."""
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4led", REPO / "tools/b4_campaign_ledger.py")
    led = ilu.module_from_spec(spec)
    spec.loader.exec_module(led)
    nxt = led.schedule_next(
        {"cells": {c: {"status": "PENDING"}
                   for c in led.EXPECTED_CELLS}},
        {"device_available": True, "stop_file_present": True,
         "compute_apps_active": False})
    assert nxt.startswith("HOLD")


def test_7_restart_never_resumes_quarantined(tmp_path):
    """Kill 7: a restart adjudicates the quarantined attempt
    AMBIGUOUS_CLAIM (claim without terminal) and the orchestrator
    refuses it — never an automatic resume."""
    out, cdir = _world(tmp_path, cell="o2022_seed303")
    (cdir.parent / "o2022_seed303" /
     "CLAIM_b4_campaign_generation_v8_20260910.json"
     ).write_text(json.dumps({"attempt_id": "a1"}))
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4orch_t", REPO / "tools/b4_campaign_orchestrator.py")
    orch = ilu.module_from_spec(spec)
    spec.loader.exec_module(orch)
    st = orch.adjudicate_cell_state(out, "o2022_seed303")
    assert st == "AMBIGUOUS_CLAIM"
    osrc = (REPO / "tools/b4_campaign_orchestrator.py"
            ).read_text()
    assert 'in ("AMBIGUOUS_CLAIM", "UNCERTAIN")' in osrc


def test_8_no_group_member_survives(tmp_path):
    """Kill 8: a grandchild spawned by the stalled child dies
    with the process group — the group-empty proof holds."""
    out, cdir = _world(tmp_path)
    r = _run(out, cdir, GRANDCHILD_BODY)
    assert r["group_empty"] is True
    assert "reaped_group_empty" in r["escalation"]


def test_9_gpu_close_once_includes_failed_time(tmp_path):
    """Kill 9: the GPU close record is written exactly once and
    quarantined time counts against the ceiling."""
    out, cdir = _world(tmp_path, cell="o2099_seed1")
    (cdir / "CLAIM_b4_campaign_generation_v8_20260910.json"
     ).write_text(json.dumps({"attempt_id": "attempt_test01"}))
    r = _run(out, cdir, STALL_BODY)
    closes = list(cdir.glob("SUPERVISOR_GPU_CLOSE_*.json"))
    assert len(closes) == 1
    doc = json.loads(closes[0].read_text())
    assert doc["counts_against_global_ceiling"] is True
    assert doc["charged_seconds"] > 0
    with pytest.raises(FileExistsError):
        sup._excl_append_json(closes[0], {"x": 1})
    charges = sup.recompute_gpu_charges([out])
    key = f"{out.name}/{cdir.name}"
    assert charges["per_cell"][key] >= doc["charged_seconds"] - 1
    assert charges["remaining_seconds"] < \
        charges["ceiling_seconds"]


def test_12_recovery_generation_immutable_and_closed():
    """Kill 12 + launch boundary: amendment 16 declares
    scientific_change NONE with the population identity
    untouched; regenerating a published amendment refuses; the
    launch gate demands BOTH external records."""
    import b4_authority as b4a
    a16 = json.loads(b4a.AMENDMENT_16_PATH.read_text())
    assert a16["scientific_change"].startswith("NONE")
    assert "same twelve cells" in a16["population_identity"]
    assert a16["campaign_generation_v8"] == b4a.V8_GENERATION
    assert a16["amends_amendment_15_sha256"] == \
        b4a.AMENDMENT_15_SHA
    gen = subprocess.run(
        [PY, str(REPO / "tools/b4_gen_amendment_16.py")],
        capture_output=True, text=True)
    assert gen.returncode != 0
    assert "REFUSED" in (gen.stderr + gen.stdout)
    with pytest.raises(SystemExit,
                       match="does not exist|READY_FOR_"
                             "EXTERNAL_MUSASHI_ACTA"):
        b4a.require_v6_launch_open()


def test_13_completed_v7_pairs_byte_identical():
    """Kill 13: the two completed v7 digest pairs remain
    byte-identical to the quarantine audit."""
    import hashlib
    pairs = {
        "o2022_seed101/B4_CELL_TERMINAL.json":
            "3a836df218702adfa8eb8d436b81064e3f517e59f16ab9a75"
            "0b58304a8c921d5",
        "o2022_seed202/B4_CELL_TERMINAL.json":
            "cd3e61e7e33772663e808745a1135a1d4cebf6ddd7c4b67f5"
            "4c134f0c3c5fd7c"}
    if not V7.exists():
        pytest.skip("v7 root absent on this host")
    for rel, want in pairs.items():
        got = hashlib.sha256((V7 / rel).read_bytes()).hexdigest()
        assert got == want


def test_14_mechanics_write_nothing_to_v7(tmp_path):
    """Kill 14: every supervised mechanics probe runs in its own
    tmp root; the real v7 root inventory is unchanged."""
    if not V7.exists():
        pytest.skip("v7 root absent on this host")
    def inv():
        return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(V7.rglob("*")) if p.is_file()}
    before = inv()
    out, cdir = _world(tmp_path)
    _run(out, cdir, STALL_BODY)
    assert inv() == before


def test_capability_binds_parent_and_cell(tmp_path):
    """C51: the capability self-derives and a transplanted
    capability (foreign parent) refuses in the child loader."""
    out, cdir = _world(tmp_path)
    path, cap = sup.build_capability(
        "gen", cdir.name, "a1", "mat", "0" * 64, "0" * 64,
        cdir, 60.0, 60.0)
    assert cap["capability_sha256"] == sup._self_sha(
        cap, "capability_sha256")
    # transplant: a capability issued by a FOREIGN parent
    cap2 = dict(cap)
    cap2["parent_pid"] = 1
    cap2.pop("capability_sha256")
    cap2["capability_sha256"] = sup._self_sha(
        cap2, "capability_sha256")
    path.write_text(json.dumps(cap2, indent=1, sort_keys=True))
    child = subprocess.run(
        [PY, "-c",
         "import sys; sys.path.insert(0, 'tools');\n"
         "import b4_cell_child as c\n"
         "c.load_capability(sys.argv[1])",
         str(path)], capture_output=True, text=True,
        cwd=str(REPO))
    assert child.returncode != 0
    assert "parent PID" in (child.stderr + child.stdout)
