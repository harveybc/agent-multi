"""R1 adversarial battery (Musashi order @65ee8488 §3, 2026-09-03):
the seven runtime counterexamples die typed — including the
watchdog-vs-worker race with a REAL child and a real timeout with a
late completer."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.experiment_runtime import (  # noqa: E402
    RunDirectory, RuntimePreflightError, aggregate, sha_obj,
    unit_id)


@pytest.fixture()
def run(tmp_path):
    r = RunDirectory(tmp_path / "phase",
                     allow_volatile_for_tests=True)
    idents = [{"experiment": "e", "family": "f", "window": 32,
               "latent": 16, "budget": 300, "seed": 100 + i,
               "origin": 1, "treatment": "cell"} for i in range(3)]
    ledger = {"schema": "s", "experiment": "e",
              "units": [{"unit_id": unit_id(x), "identity": x}
                        for x in idents],
              "digests": {"code": "c" * 64},
              "campaign_wall_ceiling_s": 600.0,
              "unit_timeout_s": 30.0}
    r.write_ledger(ledger)
    return r, [unit_id(x) for x in idents]


class TestCE4LedgerAndResults:

    def test_tampered_ledger_refuses_on_read(self, run):
        r, _uids = run
        raw = json.loads((r.root / "ledger.json").read_text())
        raw["digests"]["code"] = "e" * 64
        (r.root / "ledger.json").write_text(json.dumps(raw))
        with pytest.raises(RuntimePreflightError,
                           match="self-digest mismatch"):
            r.ledger()

    def test_tampered_result_refuses_aggregation(self, run):
        r, uids = run
        for uid in uids:
            r.claim(uid, expected_digests={})
            r.release(uid, "COMPLETED", result={"monitor_r2": 0.5})
        path = r.root / "units" / f"{uids[0]}.result.json"
        res = json.loads(path.read_text())
        res["monitor_r2"] = 999.0
        path.write_text(json.dumps(res))
        with pytest.raises(RuntimePreflightError,
                           match="digest mismatch"):
            aggregate(r, uids)

    def test_result_unit_correspondence_enforced(self, run):
        r, uids = run
        for uid in uids:
            r.claim(uid, expected_digests={})
            r.release(uid, "COMPLETED", result={"monitor_r2": 0.1})
        # swap one result file for another unit's (digest still
        # internally consistent, binding broken)
        a = r.root / "units" / f"{uids[0]}.result.json"
        b = r.root / "units" / f"{uids[1]}.result.json"
        a.write_text(b.read_text())
        with pytest.raises(RuntimePreflightError,
                           match="correspondence"):
            aggregate(r, uids)

    def test_release_stamps_unit_binding(self, run):
        r, uids = run
        r.claim(uids[0], expected_digests={})
        r.release(uids[0], "COMPLETED", result={"monitor_r2": 0.2})
        result = r.result(uids[0])
        assert result["unit_id"] == uids[0]
        recomputed = sha_obj({k: v for k, v in result.items()
                              if k != "result_digest"})
        assert result["result_digest"] == recomputed


class TestCE5RaceAndCAS:

    def test_terminal_never_overwrites_terminal(self, run):
        r, uids = run
        r.claim(uids[0], expected_digests={})
        r.release(uids[0], "TIMED_OUT", note="watchdog")
        with pytest.raises(RuntimePreflightError,
                           match="never\noverwritten|never "
                                 "overwritten|late completer"):
            r.release(uids[0], "COMPLETED",
                      result={"monitor_r2": 0.1})
        assert r.unit_state(uids[0])["state"] == "TIMED_OUT"

    def test_stale_attempt_cannot_finalize(self, run):
        r, uids = run
        state1 = r.claim(uids[0], expected_digests={})
        assert state1["attempt"] == 1
        r.release(uids[0], "INTERRUPTED", note="crash",
                  attempt=1)
        state2 = r.claim(uids[0], expected_digests={})
        assert state2["attempt"] == 2
        with pytest.raises(RuntimePreflightError, match="stale"):
            r.release(uids[0], "COMPLETED",
                      result={"monitor_r2": 0.3}, attempt=1)
        r.release(uids[0], "COMPLETED",
                  result={"monitor_r2": 0.3}, attempt=2)
        assert r.unit_state(uids[0])["state"] == "COMPLETED"

    def test_watchdog_never_marks_alive_process_without_killer(
            self, run):
        r, uids = run
        state = r.claim(uids[0], expected_digests={})
        # make it stale instantly but the pid (this test process)
        # is very much alive
        state["timeout_s"] = 0.0
        state["claimed_at"] = time.time() - 10
        from agent_plugins.experiment_runtime import (
            atomic_write_json)
        atomic_write_json(r._state_path(uids[0]), state)
        alerts = r.watchdog(temperature_reader=lambda: {})
        kinds = [a["type"] for a in alerts]
        assert "unit_timeout_alive" in kinds
        assert r.unit_state(uids[0])["state"] == "RUNNING"

    def test_watchdog_vs_worker_race_with_real_child(self, run):
        """The required race: a REAL child holds the unit; the
        watchdog kills AND reaps it BEFORE releasing TIMED_OUT; the
        late completer (stale actor) then refuses."""
        r, uids = run
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"])
        state = r.claim(uids[0], expected_digests={})
        from agent_plugins.experiment_runtime import (
            atomic_write_json)
        state["pid"] = child.pid
        state["timeout_s"] = 0.0
        state["claimed_at"] = time.time() - 10
        atomic_write_json(r._state_path(uids[0]), state)

        def kill_child(pid):
            assert pid == child.pid
            child.terminate()
            child.wait(timeout=30)
            return True

        alerts = r.watchdog(kill_child=kill_child,
                            temperature_reader=lambda: {})
        kinds = {a["type"] for a in alerts}
        assert "unit_timeout" in kinds
        assert child.poll() is not None, "child must be reaped"
        assert r.unit_state(uids[0])["state"] == "TIMED_OUT"
        # the zombie's buffered completion refuses
        with pytest.raises(RuntimePreflightError):
            r.release(uids[0], "COMPLETED",
                      result={"monitor_r2": 0.9}, attempt=1)
        assert r.unit_state(uids[0])["state"] == "TIMED_OUT"

    def test_real_timeout_with_late_completer_subprocess(
            self, tmp_path):
        """A REAL worker subprocess exceeds its timeout, the
        supervisor-side watchdog reaps it, a retry completes; the
        late attempt to write over it refuses (end-to-end)."""
        r = RunDirectory(tmp_path / "phase",
                         allow_volatile_for_tests=True)
        ident = {"experiment": "e", "family": "f", "window": 32,
                 "latent": 16, "budget": 300, "seed": 7,
                 "origin": 1, "treatment": "cell"}
        uid = unit_id(ident)
        r.write_ledger({"schema": "s", "experiment": "e",
                        "units": [{"unit_id": uid,
                                   "identity": ident}],
                        "digests": {}, "campaign_wall_ceiling_s": 60,
                        "unit_timeout_s": 1.0})
        # child claims and sleeps past its timeout
        code = (
            "import sys, time; sys.path.insert(0, %r);\n"
            "from agent_plugins.experiment_runtime import "
            "RunDirectory\n"
            "r = RunDirectory(%r, allow_volatile_for_tests=True)\n"
            "r.claim(%r, expected_digests={})\n"
            "time.sleep(60)\n" % (str(REPO), str(tmp_path / "phase"),
                                  uid))
        child = subprocess.Popen([sys.executable, "-c", code])
        for _ in range(100):
            if r.unit_state(uid)["state"] == "RUNNING":
                break
            time.sleep(0.1)
        state = r.unit_state(uid)
        assert state["state"] == "RUNNING"
        # the child claimed directly (run_one_unit is what stamps
        # timeout_s); stamp the 1.0s timeout the ledger declares
        from agent_plugins.experiment_runtime import (
            atomic_write_json)
        state["timeout_s"] = 1.0
        atomic_write_json(r._state_path(uid), state)
        time.sleep(1.2)  # exceed the 1.0s unit timeout

        def kill_child(pid):
            child.terminate()
            child.wait(timeout=30)
            return True

        alerts = r.watchdog(kill_child=kill_child,
                            temperature_reader=lambda: {})
        assert any(a["type"] == "unit_timeout" for a in alerts)
        assert r.unit_state(uid)["state"] == "TIMED_OUT"
        # retry completes at attempt 2
        r.claim(uid, expected_digests={})
        r.release(uid, "COMPLETED", result={"x": 1}, attempt=2)
        # the reaped child's ghost (attempt 1) refuses
        with pytest.raises(RuntimePreflightError):
            r.release(uid, "COMPLETED", result={"x": 2}, attempt=1)


class TestCE6ThermalAndDrift:

    def test_thermal_alert_over_limit(self, run):
        r, _uids = run
        alerts = r.watchdog(
            temperature_reader=lambda: {"gpu_max_c": 91.0,
                                        "cpu_max_c": 50.0})
        thermal = [a for a in alerts if a["type"] == "thermal"]
        assert thermal and thermal[0]["device"] == "gpu"
        assert thermal[0]["temperature_c"] == 91.0

    def test_no_thermal_alert_below_limit(self, run):
        r, _uids = run
        alerts = r.watchdog(
            temperature_reader=lambda: {"gpu_max_c": 60.0,
                                        "cpu_max_c": 50.0})
        assert not [a for a in alerts if a["type"] == "thermal"]

    def test_identity_drift_alert(self, run):
        r, _uids = run
        alerts = r.watchdog(
            expected_digests={"code": "d" * 64},
            temperature_reader=lambda: {})
        drift = [a for a in alerts if a["type"] == "identity_drift"]
        assert drift and "code" in drift[0]["drift"]

    def test_no_drift_when_digests_match(self, run):
        r, _uids = run
        alerts = r.watchdog(
            expected_digests={"code": "c" * 64},
            temperature_reader=lambda: {})
        assert not [a for a in alerts
                    if a["type"] == "identity_drift"]


class TestCE1CE7StratifiedEtaAndStatus:

    def test_eta_is_stratified_and_worker_aware(self, tmp_path):
        r = RunDirectory(tmp_path / "phase",
                         allow_volatile_for_tests=True)
        units = []
        for i in range(4):   # cheap stratum: persistence-like
            units.append({"experiment": "e", "family": "f",
                          "window": 32, "latent": 16, "budget": 0,
                          "seed": i, "origin": 0,
                          "treatment": "persistence"})
        for i in range(4):   # heavy stratum: w256 training
            units.append({"experiment": "e", "family": "f",
                          "window": 256, "latent": 128,
                          "budget": 2700, "seed": i, "origin": 0,
                          "treatment": "survivor_trained"})
        r.write_ledger({"schema": "s", "experiment": "e",
                        "units": [{"unit_id": unit_id(x),
                                   "identity": x} for x in units],
                        "digests": {},
                        "campaign_wall_ceiling_s": 600,
                        "unit_timeout_s": 60})
        from agent_plugins.experiment_runtime import (
            atomic_write_json)
        # complete half of each stratum with very different times
        for x, dur in ((units[0], 0.1), (units[1], 0.1),
                       (units[4], 600.0), (units[5], 600.0)):
            uid = unit_id(x)
            r.claim(uid, expected_digests={})
            state = r.unit_state(uid)
            state["claimed_at"] = 1000.0
            atomic_write_json(r._state_path(uid), state)
            r.release(uid, "COMPLETED", result={"ok": 1})
            state = r.unit_state(uid)
            state["finished_at"] = 1000.0 + dur
            atomic_write_json(r._state_path(uid), state)
        status = r.heartbeat(current_unit=None, workers=2)
        eta = status["eta"]
        lo, hi = eta["eta_interval_s"]
        # 2 cheap remaining * 0.1 + 2 heavy remaining * 600 = 1200.2
        # divided by 2 workers = 600.1
        assert lo == pytest.approx(600.1, rel=0.01)
        assert hi == pytest.approx(600.1, rel=0.01)
        assert eta["workers_assumed"] == 2
        assert len(eta["stratified"]) == 2
        assert eta["assumptions"]
        # the pooled diagnostic would have said ~0.1*4=0.4s (median)
        pooled = eta["pooled_unstratified_diagnostic"]
        assert pooled["median_unit_s"] < 601

    def test_status_lists_active_units_and_device(self, run):
        r, uids = run
        r.claim(uids[0], expected_digests={})
        status = r.heartbeat(current_unit=uids[0], workers=3,
                             device_class="cuda")
        assert status["device_class"] == "cuda"
        active = status["active_units"]
        assert len(active) == 1
        assert active[0]["unit"] == uids[0]
        assert active[0]["pid"] == os.getpid()
        assert "stratum" in active[0]
