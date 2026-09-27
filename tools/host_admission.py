#!/usr/bin/env python3
"""RR02 (order 2026-09-26): this repository's launchers ask the fleet's ONE memory authority.

Why this file exists
--------------------
`predictor/tools/crispdm_admission.py` is the single per-host memory admission on this fleet: one
exclusive lock covering the capacity reading, the decision and the write of a durable reservation
that outlives the reading and covers the whole load.  It is installed on every host of the fleet at
``~/.local/libexec/crispdm/crispdm_admission.py`` with the launcher ``~/.local/bin/crispdm-run``.

Two launchers in THIS repository called ``systemd-run --user`` directly, with no admission call at
all (predictor's DR01 follow-on return, §5.4).  They bypassed the host reservation entirely: their
loads were invisible to every other launcher's capacity reading, which is exactly the defect that
admitted two 8 GiB requests against one 12 GiB reading.

This module is a CLIENT, not a second authority.  It does not read memory, decide anything or hold
a policy of its own: it locates the deployed authority and asks it, locally or over the caller's own
ssh transport, and returns what it said.  If the authority is not installed it REFUSES -- an
unenforceable cap is not a cap, and that is the same direction predictor's own dispatcher takes.

It never invents a cap.  A caller that cannot state the measured memory need of the work it is about
to start is refused, because a number chosen to get past a gate is worse than no gate.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: where the fleet installs the one authority, on every host
DEPLOYED_MODULE = "~/.local/libexec/crispdm/crispdm_admission.py"
DEPLOYED_LAUNCHER = "~/.local/bin/crispdm-run"
REFUSED_EXIT = 75

ADMITTED = "ADMITTED"
QUEUED = "QUEUED"
REFUSED = "REFUSED"
NO_AUTHORITY = "NO_ADMISSION_AUTHORITY"
NO_CAP = "NO_MEASURED_CAP_DECLARED"


class AdmissionRefused(RuntimeError):
    """Raised instead of launching.  ``decision`` carries every reading the verdict used."""

    def __init__(self, decision: dict):
        super().__init__(f"{decision.get('verdict')} {decision.get('code')}: {decision.get('reason')}")
        self.decision = decision


@dataclass
class HostAdmission:
    """The authority as this repository reaches it.

    ``run`` is how a command is executed on the target host: the caller supplies its own transport,
    so nothing here opens a connection, learns a host name or writes one anywhere.  A local host
    passes ``run=None``.
    """

    run: object = None                 # callable(argv: list[str]) -> CompletedProcess, or None
    module: str = DEPLOYED_MODULE
    python: str = "python3"
    extra_env: dict = field(default_factory=dict)

    def _exec(self, argv: list[str]) -> subprocess.CompletedProcess:
        if self.run is None:
            return subprocess.run([sys.executable if self.python == "python3" else self.python,
                                   str(Path(self.module).expanduser()), *argv],
                                  capture_output=True, text=True, timeout=180)
        # the caller's transport: the command is quoted for a remote shell so `~` expands there
        return self.run([self.python, self.module, *argv])

    def available(self) -> bool:
        r = self._exec(["policy"])
        return r.returncode == 0

    def acquire(self, *, name: str, cap_bytes, wall_seconds: int, label: str = "",
                detached: bool = True) -> dict:
        """Reserve before the load starts, or say why not.  Never retries and never lowers a cap."""
        if not cap_bytes:
            return {"verdict": REFUSED, "code": NO_CAP,
                    "reason": ("this launcher will not start heavy work without the measured memory "
                               "need of that work. Declare it; a cap invented to pass a gate is not "
                               "a cap, and lowering a request below its measured need is forbidden.")}
        argv = ["acquire", "-n", name, "--cap-bytes", str(int(cap_bytes)),
                "-t", str(int(wall_seconds))]
        if label:
            argv += ["--label", label]
        if detached:
            argv += ["--detached"]
        r = self._exec(argv)
        try:
            decision = json.loads((r.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"verdict": REFUSED, "code": NO_AUTHORITY,
                    "reason": (f"the deployed admission authority did not answer (exit {r.returncode}); "
                               f"an unenforceable cap is not a cap, so nothing was started. "
                               f"{(r.stderr or '')[-200:]}")}
        return decision

    def arm(self, lease_id: str, *, unit: str = "", cgroup: str = "", pid=None) -> dict:
        argv = ["arm", lease_id]
        if pid:
            argv += ["--pid", str(int(pid))]
        if unit:
            argv += ["--unit", unit]
        if cgroup:
            argv += ["--cgroup", cgroup]
        r = self._exec(argv)
        try:
            return json.loads((r.stdout or "{}").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "code": "ARM_UNREADABLE", "stderr": (r.stderr or "")[-200:]}

    def release(self, lease_id: str) -> dict:
        r = self._exec(["release", lease_id])
        try:
            return json.loads((r.stdout or "{}").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "code": "RELEASE_UNREADABLE"}


def reserve_or_refuse(admission: HostAdmission, **kw) -> dict:
    """Admit, or raise.  QUEUED and REFUSED are both refusals to start: the caller waits for
    capacity or moves the work to a host that admits it -- it never asks again for less."""
    decision = admission.acquire(**kw)
    if decision.get("verdict") != ADMITTED:
        raise AdmissionRefused(decision)
    return decision


def unit_cgroup(run, unit: str) -> str:
    """This unit's OWN cgroup, read on the host the unit runs on.

    A detached unit that no process waits for is witnessed by its cgroup, and it must be the unit's
    own cgroup, never an ancestor it merely inherited -- an ancestor outlives the load, so the
    reservation could never be released.  Arming with the unit name alone is not enough either: a
    lease with no witness is protected only by the arming grace window, and once that expires the
    sweep reclaims it while the load is still running.
    """
    argv = ["systemctl", "--user", "show", unit, "-p", "ControlGroup", "--value"]
    r = run(argv) if run is not None else subprocess.run(argv, capture_output=True, text=True,
                                                         timeout=60)
    value = (r.stdout or "").strip()
    return value.lstrip("/") if value and value != "" else ""
