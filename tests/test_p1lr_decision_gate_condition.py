"""R3 (order 2026-09-11): a refused P1LR gate must be a STABLE refusal.

The defect these tests pin down is a systemd semantics error, not a
Python one, so they are exercised against a REAL `systemd --user`
manager. Two throwaway template units are written into the runtime unit
directory ($XDG_RUNTIME_DIR/systemd/user) — transient by construction,
wiped when the session ends, and never touching the installed
p1lr-decision@ units or their drop-ins. Their ExecStart writes a marker
file instead of training anything.

  * the PRE shape calls the gate check from ExecStartPre and claims
    RestartPreventExitStatus=4 stops the loop. It does not: the main
    process never runs, ExecMainStatus stays 0, and Restart=on-failure
    keeps retrying a configuration refusal;
  * the POST shape calls it from ExecCondition. A refusal skips the
    unit — not failed, not restarted — and only a viable gate reaches
    ExecStart.

Four gate states are covered: absent, wrong schema, non-viable outcome,
and viable.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATE_CHECK = REPO / "examples/systemd/p1lr_decision_gate_check.sh"
SCHEMA = "agent_multi.p1_difficulty_lr_screen_verdict.v1"
RUNTIME_UNITS = Path(os.environ.get("XDG_RUNTIME_DIR", "/run/user/1000")) \
    / "systemd/user"


def _systemd_available() -> bool:
    if not shutil.which("systemctl") or not os.environ.get("XDG_RUNTIME_DIR"):
        return False
    probe = subprocess.run(("systemctl", "--user", "show", "-p", "Version"),
                           capture_output=True, text=True)
    return probe.returncode == 0


requires_systemd = pytest.mark.skipif(
    not _systemd_available(),
    reason="no reachable systemd --user manager in this environment")


def sc(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("systemctl", "--user", *args),
                          capture_output=True, text=True, timeout=60)


def prop(unit: str, name: str) -> str:
    out = sc("show", unit, "-p", name).stdout.strip()
    return out.split("=", 1)[1] if "=" in out else ""


# ----------------------------------------------------------------- gates
def write_contract(tmp: Path) -> Path:
    contract = tmp / "contract.json"
    contract.write_text(json.dumps({"contract": "p1lr-r3-selftest"},
                                   sort_keys=True))
    return contract


def write_gate(tmp: Path, contract: Path, *, kind: str) -> Path:
    """kind: viable | wrong_schema | not_viable | absent"""
    gate = tmp / "screen_verdict.json"
    if kind == "absent":
        return tmp / "gate_that_was_never_produced.json"
    digest = hashlib.sha256(contract.read_bytes()).hexdigest()
    body = {
        "schema": SCHEMA if kind != "wrong_schema" else "some.other.schema.v9",
        "outcome": ("SCREEN_VIABLE_REGION" if kind != "not_viable"
                    else "SCREEN_NO_VIABLE_REGION"),
        "gates": {"replica_terminal_loads": True},
        "contract_sha256": digest,
    }
    gate.write_text(json.dumps(body, sort_keys=True))
    return gate


# ----------------------------------------------------------------- units
PRE_UNIT = """[Unit]
Description=R3 selftest, PRE shape (gate check as ExecStartPre)
StartLimitIntervalSec=20
StartLimitBurst=2

[Service]
Type=oneshot
Environment=P1LR_GATE_STATE_DIR={state}
Environment=P1LR_GATE_INSTANCE=%i
Environment=P1LR_PYTHON={python}
ExecStartPre={check} {gate} {contract}
ExecStart=/bin/sh -c 'printf started > {marker}'
Restart=on-failure
RestartSec=1
RestartPreventExitStatus=4
"""

POST_UNIT = """[Unit]
Description=R3 selftest, POST shape (gate check as ExecCondition)
StartLimitIntervalSec=20
StartLimitBurst=2

[Service]
Type=oneshot
Environment=P1LR_GATE_STATE_DIR={state}
Environment=P1LR_GATE_INSTANCE=%i
Environment=P1LR_PYTHON={python}
ExecCondition={check} {gate} {contract}
ExecStart=/bin/sh -c 'printf started > {marker}'
Restart=on-failure
RestartSec=1
RestartPreventExitStatus=4
"""


class Harness:
    """One throwaway template unit in the runtime unit directory."""

    def __init__(self, tmp: Path, name: str, body: str, gate: Path,
                 contract: Path) -> None:
        self.name = name
        self.instance = f"{name}@1.service"
        self.marker = tmp / "EXECSTART_RAN"
        self.state = tmp / "gate_state"
        self.state.mkdir(exist_ok=True)
        RUNTIME_UNITS.mkdir(parents=True, exist_ok=True)
        self.path = RUNTIME_UNITS / f"{name}@.service"
        self.path.write_text(body.format(
            state=self.state, python=sys.executable, check=GATE_CHECK,
            gate=gate, contract=contract, marker=self.marker))
        sc("daemon-reload")

    def start(self) -> subprocess.CompletedProcess:
        return sc("start", self.instance)

    def settle(self, seconds: float = 6.0) -> None:
        """Let systemd finish any restart schedule it decided on."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            if prop(self.instance, "ActiveState") in ("failed", "inactive"):
                # inactive can still be a pause between restarts
                if prop(self.instance, "NRestarts") != "0":
                    time.sleep(0.5)
                    continue
                time.sleep(1.5)
                break
            time.sleep(0.3)

    def cleanup(self) -> None:
        sc("stop", self.instance)
        sc("reset-failed", self.instance)
        self.path.unlink(missing_ok=True)
        sc("daemon-reload")


@pytest.fixture
def harness(tmp_path):
    made: list[Harness] = []

    def make(kind: str, body: str, name: str) -> Harness:
        contract = write_contract(tmp_path)
        gate = write_gate(tmp_path, contract, kind=kind)
        h = Harness(tmp_path, name, body, gate, contract)
        made.append(h)
        return h

    yield make
    for h in made:
        h.cleanup()


# ------------------------------------------------------------- PRE proof
@requires_systemd
def test_pre_shape_restarts_a_configuration_refusal(harness):
    """The defect: RestartPreventExitStatus cannot see ExecStartPre."""
    h = harness("absent", PRE_UNIT, "p1lr-r3-pre-absent")
    h.start()
    h.settle()
    assert not h.marker.exists(), "ExecStart must never run on a bad gate"
    assert prop(h.instance, "ExecMainStatus") == "0", (
        "the main process never ran, which is exactly why "
        "RestartPreventExitStatus=4 has nothing to compare against")
    assert int(prop(h.instance, "NRestarts")) > 0, (
        "PRE: systemd retried a configuration refusal")


# ------------------------------------------------------------ POST proof
@pytest.mark.parametrize("kind", ["absent", "wrong_schema", "not_viable"])
@requires_systemd
def test_refused_gate_is_skipped_not_restarted(harness, kind):
    h = harness(kind, POST_UNIT, f"p1lr-r3-post-{kind.replace('_', '-')}")
    h.start()
    h.settle()
    assert not h.marker.exists(), (
        f"{kind}: ExecStart must not be reached")
    assert prop(h.instance, "NRestarts") == "0", (
        f"{kind}: a refusal must schedule no restart")
    assert prop(h.instance, "ActiveState") != "failed", (
        f"{kind}: a refusal is a skip, not a failure")


@requires_systemd
def test_viable_gate_reaches_execstart(harness):
    h = harness("viable", POST_UNIT, "p1lr-r3-post-viable")
    result = h.start()
    h.settle(3.0)
    assert h.marker.exists(), (
        f"a viable gate must reach ExecStart; systemctl said: "
        f"{result.stderr.strip()}")
    assert h.marker.read_text() == "started"
    assert prop(h.instance, "NRestarts") == "0"


# ------------------------------------------------- the typed record survives
@pytest.mark.parametrize(
    "kind,expected",
    [("absent", "REFUSED_SCREEN_GATE_MISSING"),
     ("wrong_schema", "REFUSED_SCREEN_GATE_SCHEMA"),
     ("not_viable", "REFUSED_SCREEN_NOT_VIABLE")])
def test_refusal_is_typed_in_a_durable_record(tmp_path, kind, expected):
    """A skip that says nothing is a silent failure. Run the check
    directly — no systemd needed to prove the record it leaves."""
    contract = write_contract(tmp_path)
    gate = write_gate(tmp_path, contract, kind=kind)
    state = tmp_path / "state"
    env = {**os.environ, "P1LR_GATE_STATE_DIR": str(state),
           "P1LR_GATE_INSTANCE": "77", "P1LR_PYTHON": sys.executable}
    run = subprocess.run(("bash", str(GATE_CHECK), str(gate), str(contract)),
                         capture_output=True, text=True, env=env, timeout=60)
    assert run.returncode == 4, run.stderr
    record = json.loads(
        (state / "p1lr-decision@77.gate.json").read_text())
    assert record["schema"] == "agent_multi.p1lr_decision_gate_check.v1"
    assert record["disposition"] == "REFUSED_NOT_EXECUTABLE"
    assert record["verified"] is False
    assert expected in record["refusal_classes"]
    assert record["gate_path"] == str(gate)


def test_verified_gate_records_its_own_disposition(tmp_path):
    contract = write_contract(tmp_path)
    gate = write_gate(tmp_path, contract, kind="viable")
    state = tmp_path / "state"
    env = {**os.environ, "P1LR_GATE_STATE_DIR": str(state),
           "P1LR_GATE_INSTANCE": "77", "P1LR_PYTHON": sys.executable}
    run = subprocess.run(("bash", str(GATE_CHECK), str(gate), str(contract)),
                         capture_output=True, text=True, env=env, timeout=60)
    assert run.returncode == 0, run.stderr
    record = json.loads((state / "p1lr-decision@77.gate.json").read_text())
    assert record["disposition"] == "GATE_VERIFIED"
    assert record["refusal_classes"] == []


def test_a_foreign_gate_is_refused_even_when_viable(tmp_path):
    """The gate must bind THIS contract, not merely be viable."""
    contract = write_contract(tmp_path)
    gate = write_gate(tmp_path, contract, kind="viable")
    other = tmp_path / "other_contract.json"
    other.write_text(json.dumps({"contract": "someone-elses"}))
    state = tmp_path / "state"
    env = {**os.environ, "P1LR_GATE_STATE_DIR": str(state),
           "P1LR_GATE_INSTANCE": "77", "P1LR_PYTHON": sys.executable}
    run = subprocess.run(("bash", str(GATE_CHECK), str(gate), str(other)),
                         capture_output=True, text=True, env=env, timeout=60)
    assert run.returncode == 4
    record = json.loads((state / "p1lr-decision@77.gate.json").read_text())
    assert "REFUSED_SCREEN_GATE_FOREIGN" in record["refusal_classes"]


# ---------------------------------------------- the shipped unit's shape
def test_shipped_template_uses_execcondition_not_execstartpre():
    body = (REPO / "examples/systemd/p1lr-decision@.service").read_text()
    directives = [ln for ln in body.splitlines()
                  if ln.startswith(("ExecStartPre=", "ExecCondition="))]
    assert any(d.startswith("ExecCondition=") and "gate_check" in d
               for d in directives), directives
    assert not any(d.startswith("ExecStartPre=") for d in directives), (
        "a non-empty ExecStartPre reintroduces the restart loop")


def test_start_limit_directives_sit_in_the_unit_section():
    """systemd silently ignores StartLimit* under [Service]."""
    section = None
    placed: dict[str, str] = {}
    for raw in (REPO / "examples/systemd/p1lr-decision@.service").read_text() \
            .splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
        elif line.startswith("StartLimit"):
            placed[line.split("=")[0]] = section
    assert placed, "the unit declares no start limit at all"
    assert set(placed.values()) == {"[Unit]"}, placed


def test_repair_script_rewrites_an_installed_dropin(tmp_path):
    """An installed drop-in still carrying ExecStartPre is the one way
    the corrected template can be undone."""
    unit_dir = tmp_path / "systemd/user"
    dropin_dir = unit_dir / "p1lr-decision@.service.d"
    dropin_dir.mkdir(parents=True)
    conf = dropin_dir / "20-explicit-close-v2.conf"
    conf.write_text(
        "[Service]\n"
        "WorkingDirectory=/somewhere/pinned\n"
        "ExecStartPre=\n"
        "ExecStartPre=/somewhere/pinned/p1lr_decision_gate_check.sh /g /c\n")
    env = {**os.environ, "P1LR_UNIT_DIR": str(unit_dir)}
    subprocess.run(
        ("bash", str(REPO / "examples/systemd"
                     / "repair_p1lr_decision_condition.sh"),),
        capture_output=True, text=True, env=env, timeout=120)
    body = conf.read_text()
    lines = [ln for ln in body.splitlines()
             if ln.startswith(("ExecStartPre=", "ExecCondition="))]
    assert "ExecCondition=/somewhere/pinned/p1lr_decision_gate_check.sh /g /c" \
        in lines, lines
    assert [ln for ln in lines if ln.startswith("ExecStartPre=")] \
        == ["ExecStartPre="], (
        "the only surviving ExecStartPre must be the empty clearing one")
    assert list(dropin_dir.glob("*.bak")), "the original must be preserved"


# ------------------------------------------------- seat disposition tool
def _seat_tool():
    sys.path.insert(0, str(REPO))
    import importlib
    return importlib.import_module("tools.p1lr_seat_disposition")


def test_a_missing_gate_declares_the_seat_historical(tmp_path, monkeypatch):
    tool = _seat_tool()
    contract = write_contract(tmp_path)
    absent = write_gate(tmp_path, contract, kind="absent")
    monkeypatch.setattr(
        tool, "effective_condition",
        lambda unit: (str(GATE_CHECK), [str(absent), str(contract)]))
    monkeypatch.setattr(tool, "_prop", lambda unit, name: "")
    record = tool.evaluate("p1lr-decision@99.service",
                           state_dir=tmp_path / "state")
    assert record["disposition"] == tool.HISTORICAL
    assert record["pinned_gate_exists"] is False
    assert record["gate_reconstructed"] is False
    assert not absent.exists(), "the tool must never produce the gate"
    assert "REFUSED_SCREEN_GATE_MISSING" in record["refusal_classes"]
    assert "disable --now" in record["owner_disposition"]


def test_a_viable_gate_declares_the_seat_executable(tmp_path, monkeypatch):
    tool = _seat_tool()
    contract = write_contract(tmp_path)
    gate = write_gate(tmp_path, contract, kind="viable")
    monkeypatch.setattr(
        tool, "effective_condition",
        lambda unit: (str(GATE_CHECK), [str(gate), str(contract)]))
    monkeypatch.setattr(tool, "_prop", lambda unit, name: "")
    record = tool.evaluate("p1lr-decision@99.service",
                           state_dir=tmp_path / "state")
    assert record["disposition"] == tool.EXECUTABLE
    assert "owner_disposition" not in record


def test_seat_records_are_append_only(tmp_path):
    tool = _seat_tool()
    state = tmp_path / "state"
    for i in range(3):
        tool.write_record({"schema": tool.SCHEMA, "unit": "u@1.service",
                           "disposition": tool.HISTORICAL, "i": i}, state)
    lines = (state / "SEAT_DISPOSITIONS.jsonl").read_text().splitlines()
    assert len(lines) == 3, "a disposition history is never overwritten"
    assert json.loads(lines[-1])["i"] == 2


def test_effective_condition_takes_the_last_dropin_not_the_template(
        monkeypatch):
    """Drop-in Exec* directives accumulate; the pinned one is last."""
    tool = _seat_tool()
    shown = (
        "{ path=/a/check.sh ; argv[]=/a/check.sh /a/gate ; "
        "ignore_errors=no }{ path=/b/check.sh ; argv[]=/b/check.sh /b/gate ; "
        "ignore_errors=no }")
    monkeypatch.setattr(tool, "_prop", lambda unit, name: shown)
    check, argv = tool.effective_condition("u@1.service")
    assert check == "/b/check.sh"
    assert argv == ["/b/gate"]
