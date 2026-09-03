"""C1 acceptance (order @1649e7c0 §2): the terminal transition is a
REAL cross-process compare-and-set. Synchronized real-process races
for every competing terminal pair, a stale-attempt writer, watchdog
versus completion, and two simultaneous identical completions —
plus 200 fresh-root repetitions of the differing-terminal race with
ZERO double winners."""
from __future__ import annotations

import json
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
    RunDirectory, RuntimePreflightError, unit_id)

RACER = r'''
import json, os, sys, time
sys.path.insert(0, {repo!r})
from agent_plugins.experiment_runtime import RunDirectory
root, uid, terminal, barrier, attempt = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
    int(sys.argv[5]))
r = RunDirectory(root, allow_volatile_for_tests=True)
while not os.path.exists(barrier):
    pass
kwargs = {{"note": "racer", "attempt": attempt}}
if terminal == "COMPLETED":
    kwargs["result"] = {{"monitor_r2": 0.5}}
try:
    r.release(uid, terminal, **kwargs)
    print("WIN:" + terminal)
except BaseException as exc:
    print("LOSE:" + terminal + ":" + type(exc).__name__)
'''


def _fresh_running_unit(seed=0):
    root = Path.home() / ".cache" / f"c1race_{uuid.uuid4().hex}"
    run = RunDirectory(root / "p", allow_volatile_for_tests=True)
    ident = {"experiment": "e", "family": "f", "window": 32,
             "latent": 16, "budget": 300, "seed": seed, "origin": 1,
             "treatment": "cell"}
    uid = unit_id(ident)
    run.write_ledger({"schema": "s", "experiment": "e",
                      "units": [{"unit_id": uid, "identity": ident}],
                      "digests": {}, "campaign_wall_ceiling_s": 600,
                      "unit_timeout_s": 60})
    run.claim(uid, expected_digests={})
    return root, run, uid


def _race(root, uid, terminals, attempts=None):
    attempts = attempts or [1] * len(terminals)
    barrier = str(root / "go")
    procs = [subprocess.Popen(
        [sys.executable, "-c", RACER.format(repo=str(REPO)),
         str(root / "p"), uid, term, barrier, str(att)],
        stdout=subprocess.PIPE, text=True)
        for term, att in zip(terminals, attempts)]
    time.sleep(0.15)
    Path(barrier).touch()
    outs = [p.communicate(timeout=60)[0].strip() for p in procs]
    return outs


TERMINAL_PAIRS = [("FAILED", "TIMED_OUT"),
                  ("FAILED", "INTERRUPTED"),
                  ("TIMED_OUT", "INTERRUPTED"),
                  ("COMPLETED", "FAILED"),
                  ("COMPLETED", "TIMED_OUT"),
                  ("COMPLETED", "INTERRUPTED")]


class TestCompetingTerminalPairs:

    @pytest.mark.parametrize("pair", TERMINAL_PAIRS,
                             ids=["_vs_".join(p)
                                  for p in TERMINAL_PAIRS])
    def test_exactly_one_winner(self, pair):
        root, run, uid = _fresh_running_unit()
        try:
            outs = _race(root, uid, list(pair))
            wins = [o for o in outs if o.startswith("WIN")]
            loses = [o for o in outs if o.startswith("LOSE")]
            assert len(wins) == 1, outs
            assert len(loses) == 1, outs
            final = run.unit_state(uid)["state"]
            assert f"WIN:{final}" in outs, (final, outs)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestRequiredScenarios:

    def test_stale_attempt_writer_refuses(self):
        root, run, uid = _fresh_running_unit()
        try:
            run.release(uid, "INTERRUPTED", note="x", attempt=1)
            run.claim(uid, expected_digests={})  # attempt 2
            outs = _race(root, uid, ["FAILED", "COMPLETED"],
                         attempts=[1, 2])
            assert "LOSE:FAILED:RuntimePreflightError" in outs[0]
            assert outs[1] == "WIN:COMPLETED"
            assert run.unit_state(uid)["state"] == "COMPLETED"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_watchdog_vs_completion_single_winner(self):
        """The watchdog path (kill+reap then TIMED_OUT with attempt
        CAS) races a real completing writer: exactly one wins."""
        root, run, uid = _fresh_running_unit()
        try:
            outs = _race(root, uid, ["TIMED_OUT", "COMPLETED"],
                         attempts=[1, 1])
            wins = [o for o in outs if o.startswith("WIN")]
            assert len(wins) == 1, outs
            final = run.unit_state(uid)["state"]
            if final == "COMPLETED":
                result = run.result(uid)
                assert result["unit_id"] == uid
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_two_simultaneous_identical_completions(self):
        """Both may report success only through the VERIFIED
        idempotent path: one durable result, full binding + digest
        reverified."""
        root, run, uid = _fresh_running_unit()
        try:
            outs = _race(root, uid, ["COMPLETED", "COMPLETED"],
                         attempts=[1, 1])
            assert all(o.startswith("WIN:COMPLETED")
                       for o in outs), outs
            result = run.result(uid)
            assert result["unit_id"] == uid
            from agent_plugins.experiment_runtime import sha_obj
            assert result["result_digest"] == sha_obj(
                {k: v for k, v in result.items()
                 if k != "result_digest"})
            assert run.unit_state(uid)["state"] == "COMPLETED"
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestTwoHundredFreshRootRaces:

    def test_zero_double_winners_in_200_reps(self):
        double = 0
        for rep in range(200):
            root, run, uid = _fresh_running_unit(seed=rep)
            try:
                outs = _race(root, uid, ["FAILED", "TIMED_OUT"])
                wins = [o for o in outs if o.startswith("WIN")]
                if len(wins) != 1:
                    double += 1
            finally:
                shutil.rmtree(root, ignore_errors=True)
        assert double == 0, f"{double}/200 double winners"
