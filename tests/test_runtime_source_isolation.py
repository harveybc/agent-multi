"""WP0 runtime source isolation (order 2026-08-15 §2) — socket-free.

THE INCIDENT: the corrected P1LR v2 screen rejected all four omega
seed-101 cells because a third agent wrote an untracked handoff into
the CANONICAL checkout while the experiment executed from it. The
source-identity guard refused correctly, but the experiment should
never have executed from a shared writable checkout, the failure
surfaced as an untyped SEED_FAILED, and recovery was manual.

The five rules proven here:

  Rule 1  every long-running experiment executes from a DEDICATED
          DETACHED worktree bound to one commit and VERIFIED CLEAN
          before launch; a launch from a non-isolated or dirty tree
          REFUSES typed before any GPU/model work
          (tools/runtime_worktree.py + the runner launch gate);
  Rule 2  agents write only to separate named worktrees — convention
          documented in docs/ops/RUNTIME_SOURCE_ISOLATION.md, and the
          experiment is IMMUNE to violations via rule 1;
  Rule 3  every cell record binds worktree path, commit, tracked-diff
          digest and untracked digest at MATERIALIZATION and TERMINAL
          CUSTODY, plus explicit clean-at-launch facts;
  Rule 4  status and the idle guard expose source drift as a FAILED
          CELL with failure_class source_drift and a scheduled retry —
          never silent progress; a REFUSED_SOURCE_* launch refusal is
          source_isolation_refused and never blindly restarted;
  Rule 5  a missing cell is retried AFTER ITS SEED BATCH without
          rerunning valid cells (ALREADY_COMPLETE reuse; typed
          --retry-missing plan).

Worktree proofs use REAL temp git repos; everything else injects fakes
(the proven test_p1_difficulty_lr_factorial / test_p1lr_factorial_v2 /
test_p1lr_idle_guard harnesses).
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as sysid  # noqa: E402
from tools import multifront_status as ms  # noqa: E402
from tools import p1_difficulty_lr_factorial as p1  # noqa: E402
from tools import p1lr_idle_guard as guard_mod  # noqa: E402
from tools import runtime_worktree as rtw  # noqa: E402
from tests.test_p1_difficulty_lr_factorial import (  # noqa: E402
    FakePipeline,
)
from tests.test_p1lr_factorial_v2 import (  # noqa: E402
    _records_of,
    _run_seed,
    _v2_factory,
    bindings,  # noqa: F401  (fixture, used by rt)
    rt,  # noqa: F401  (fixture)
)
from tests.test_p1lr_idle_guard import (  # noqa: E402
    IDENTITY as GUARD_IDENTITY,
    NOW as GUARD_NOW,
    ORDER as GUARD_ORDER,
    _contract as _guard_contract,
    _poll as _guard_poll,
    _write as _guard_write,
)

# The 2026-08-15 incident, verbatim shape: the legacy (pre-WP0) error
# string a CELL_FAILED heartbeat carried when the canonical checkout
# grew an untracked handoff mid-cell.
INCIDENT_ERROR = (
    "RuntimeError: executing source tree moved during the cell: "
    "dirty_untracked_digest None -> 'aadbeca8deadbeefdeadbeefdeadbeef'")

CELLS = ("P1N_LR1E4", "P1N_LR3E5", "P1E_LR1E4", "P1E_LR3E5")

# Hermetic code identity carrying the WP0 split digests: agent-multi is
# a clean pinned worktree; gym-fx is deliberately dirty so the record
# tests prove the tracked/untracked digests FLOW THROUGH to custody.
WP0_SOURCES = {
    "agent-multi": {
        "repo_root": "/home/user/Documents/GitHub/.runtime/"
                     "agent-multi-p1lr-v2-924910fe",
        "commit": "9" * 40, "dirty": False, "dirty_entries": [],
        "dirty_untracked_digest": None,
        "tracked_diff_digest": None, "untracked_digest": None,
    },
    "gym-fx": {
        "repo_root": "/repo/gym-fx", "commit": "2" * 40, "dirty": True,
        "dirty_entries": [{"status": "??", "path": "scratch.txt",
                           "sha256": "0" * 64}],
        "dirty_untracked_digest": "c" * 64,
        "tracked_diff_digest": "a" * 64,
        "untracked_digest": "b" * 64,
    },
}

CLEAN_LAUNCH_FACTS = {
    "schema": rtw.LAUNCH_FACTS_SCHEMA,
    "worktree_path": WP0_SOURCES["agent-multi"]["repo_root"],
    "commit": "9" * 40,
    "detached": True,
    "linked_worktree": True,
    "runtime_root": "/home/user/Documents/GitHub/.runtime",
    "under_runtime_root": True,
    "clean": True,
    "dirty_entries": [],
    "dirty_untracked_digest": None,
    "tracked_diff_digest": None,
    "untracked_digest": None,
    "verified_clean_at_launch": True,
}


@pytest.fixture(autouse=True)
def pinned_wp0_sources(monkeypatch):
    """Hermetic code identity for every runner proof in this module
    (same rationale as the v1/v2 modules), extended with the WP0 split
    digests so custody flow-through is assertable."""
    monkeypatch.setattr(p1.ladder, "source_identities",
                        lambda: copy.deepcopy(WP0_SOURCES))


# ---------------------------------------------------------------------------
# real temp git repos for rule-1 worktree proofs
# ---------------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    """A real canonical repo with two commits, plus a runtime root."""
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    subprocess.run(["git", "init", "-q", str(canonical)], check=True)
    _git(canonical, "config", "user.email", "test@example.com")
    _git(canonical, "config", "user.name", "Test")
    (canonical / "code.py").write_text("print('v1')\n")
    _git(canonical, "add", "code.py")
    _git(canonical, "commit", "-q", "-m", "c1")
    first = _git(canonical, "rev-parse", "HEAD")
    (canonical / "code.py").write_text("print('v2')\n")
    _git(canonical, "add", "code.py")
    _git(canonical, "commit", "-q", "-m", "c2")
    second = _git(canonical, "rev-parse", "HEAD")
    runtime_root = tmp_path / ".runtime"
    return {"canonical": canonical, "runtime_root": runtime_root,
            "first": first, "second": second}


class TestRule1EnsureRuntimeWorktree:
    def test_creates_detached_clean_worktree_pinned_to_the_commit(
            self, repo):
        path = rtw.ensure_runtime_worktree(
            repo["first"], repo_root=repo["canonical"],
            runtime_root=repo["runtime_root"], label="p1lr-v2")
        assert path.parent == repo["runtime_root"]
        assert path.name == (f"canonical-p1lr-v2-{repo['first'][:12]}")
        assert _git(path, "rev-parse", "HEAD") == repo["first"]
        # detached, clean, and a LINKED worktree (its .git is a file)
        assert subprocess.run(
            ["git", "-C", str(path), "symbolic-ref", "-q", "HEAD"],
            capture_output=True).returncode != 0
        assert _git(path, "status", "--porcelain") == ""
        assert (path / ".git").is_file()

    def test_existing_worktree_is_verified_not_recreated(self, repo):
        path1 = rtw.ensure_runtime_worktree(
            repo["first"], repo_root=repo["canonical"],
            runtime_root=repo["runtime_root"])
        path2 = rtw.ensure_runtime_worktree(
            repo["first"], repo_root=repo["canonical"],
            runtime_root=repo["runtime_root"])
        assert path1 == path2

    def test_existing_worktree_on_another_commit_refuses_typed(
            self, repo):
        path = rtw.ensure_runtime_worktree(
            repo["first"], repo_root=repo["canonical"],
            runtime_root=repo["runtime_root"])
        _git(path, "checkout", "-q", "--detach", repo["second"])
        with pytest.raises(rtw.RuntimeWorktreeRefusal) as err:
            rtw.ensure_runtime_worktree(
                repo["first"], repo_root=repo["canonical"],
                runtime_root=repo["runtime_root"])
        assert err.value.code == rtw.REFUSED_WORKTREE_COMMIT_MISMATCH

    def test_existing_dirty_worktree_refuses_typed(self, repo):
        path = rtw.ensure_runtime_worktree(
            repo["first"], repo_root=repo["canonical"],
            runtime_root=repo["runtime_root"])
        # the incident shape: an UNTRACKED file appears in the tree
        (path / "RETSU_HANDOFF.md").write_text("untracked\n")
        with pytest.raises(rtw.RuntimeWorktreeRefusal) as err:
            rtw.ensure_runtime_worktree(
                repo["first"], repo_root=repo["canonical"],
                runtime_root=repo["runtime_root"])
        assert err.value.code == rtw.REFUSED_SOURCE_DIRTY

    def test_unknown_revision_refuses_typed(self, repo):
        with pytest.raises(rtw.RuntimeWorktreeRefusal) as err:
            rtw.ensure_runtime_worktree(
                "f" * 40, repo_root=repo["canonical"],
                runtime_root=repo["runtime_root"])
        assert err.value.code == rtw.REFUSED_WORKTREE_COMMIT_MISMATCH

    def test_relative_roots_resolve_against_the_caller(
            self, repo, tmp_path, monkeypatch):
        """A relative runtime root must not split between the git
        subprocess cwd (the repo) and the caller's cwd."""
        monkeypatch.chdir(tmp_path)
        path = rtw.ensure_runtime_worktree(
            repo["first"], repo_root=Path("canonical"),
            runtime_root=Path(".runtime"))
        assert path.is_absolute()
        assert path.parent == repo["runtime_root"].resolve()
        assert _git(path, "rev-parse", "HEAD") == repo["first"]
        # a stray nested copy under the repo would prove the split
        assert not (repo["canonical"] / ".runtime").exists()


class TestRule1AssertIsolatedLaunch:
    def _facts(self, repo, path):
        return rtw.launch_tree_facts(anchor=path / ".git",
                                     runtime_root=repo["runtime_root"])

    def test_clean_pinned_worktree_passes_with_full_facts(self, repo):
        path = rtw.ensure_runtime_worktree(
            repo["first"], repo_root=repo["canonical"],
            runtime_root=repo["runtime_root"])
        facts = rtw.assert_isolated_launch(self._facts(repo, path))
        assert facts["verified_clean_at_launch"] is True
        assert facts["commit"] == repo["first"]
        assert facts["detached"] is True
        assert facts["under_runtime_root"] is True
        assert facts["clean"] is True
        assert facts["tracked_diff_digest"] is None
        assert facts["untracked_digest"] is None

    def test_canonical_branch_checkout_refuses_not_isolated(self, repo):
        facts = rtw.launch_tree_facts(
            anchor=repo["canonical"] / "code.py",
            runtime_root=repo["runtime_root"])
        with pytest.raises(rtw.RuntimeWorktreeRefusal) as err:
            rtw.assert_isolated_launch(facts)
        assert err.value.code == rtw.REFUSED_SOURCE_NOT_ISOLATED

    def test_branch_worktree_under_runtime_root_refuses(self, repo):
        # under the runtime root but on a BRANCH: the head can move
        path = repo["runtime_root"] / "canonical-branchy"
        repo["runtime_root"].mkdir(parents=True, exist_ok=True)
        _git(repo["canonical"], "worktree", "add", "-q", "-b",
             "branchy", str(path), repo["first"])
        with pytest.raises(rtw.RuntimeWorktreeRefusal) as err:
            rtw.assert_isolated_launch(self._facts(repo, path))
        assert err.value.code == rtw.REFUSED_SOURCE_NOT_ISOLATED

    def test_detached_worktree_outside_runtime_root_refuses(
            self, repo, tmp_path):
        path = tmp_path / "elsewhere" / "wt"
        path.parent.mkdir()
        _git(repo["canonical"], "worktree", "add", "-q", "--detach",
             str(path), repo["first"])
        with pytest.raises(rtw.RuntimeWorktreeRefusal) as err:
            rtw.assert_isolated_launch(self._facts(repo, path))
        assert err.value.code == rtw.REFUSED_SOURCE_NOT_ISOLATED

    def test_untracked_file_at_launch_refuses_dirty(self, repo):
        # THE incident, moved to launch time: the tree grows an
        # untracked handoff — the launch refuses instead of running
        # four cells into a guaranteed drift failure.
        path = rtw.ensure_runtime_worktree(
            repo["first"], repo_root=repo["canonical"],
            runtime_root=repo["runtime_root"])
        (path / "RETSU_HANDOFF.md").write_text("untracked\n")
        with pytest.raises(rtw.RuntimeWorktreeRefusal) as err:
            rtw.assert_isolated_launch(self._facts(repo, path))
        assert err.value.code == rtw.REFUSED_SOURCE_DIRTY
        assert err.value.facts["untracked_digest"] is not None

    def test_injected_facts_need_no_git(self):
        facts = dict(CLEAN_LAUNCH_FACTS)
        assert rtw.assert_isolated_launch(facts)[
            "verified_clean_at_launch"] is True
        facts["detached"] = False
        with pytest.raises(rtw.RuntimeWorktreeRefusal):
            rtw.assert_isolated_launch(facts)


class TestRule1RunnerLaunchGate:
    """The refusal is BUILT INTO the runner launch path: typed, before
    any GPU or model work."""

    def _refusing(self, code, reason="not isolated"):
        def isolation_facts_fn():
            raise rtw.RuntimeWorktreeRefusal(code, reason,
                                             {"worktree_path": "/x"})
        return isolation_facts_fn

    def test_non_isolated_launch_refuses_before_gpu_and_model_work(
            self, rt):
        summary = _run_seed(
            rt, enforce_isolation=True,
            isolation_facts_fn=self._refusing(
                rtw.REFUSED_SOURCE_NOT_ISOLATED))
        assert summary["outcome"] == rtw.REFUSED_SOURCE_NOT_ISOLATED
        assert summary["failure_class"] == "source_isolation_refused"
        assert FakePipeline.calls == []  # no model work
        # typed refusal heartbeat with the failure class, exit class 4
        root = Path(rt.contract["output_root"])
        hb = json.loads((root / summary["experiment_identity"] /
                         "seed101" / "runner_heartbeat.json").read_text())
        assert hb["terminal_state"] == rtw.REFUSED_SOURCE_NOT_ISOLATED
        assert hb["failure_class"] == "source_isolation_refused"
        assert p1.EXIT_CLASS[summary["outcome"]] == 4

    def test_dirty_launch_refuses_typed(self, rt):
        summary = _run_seed(
            rt, enforce_isolation=True,
            isolation_facts_fn=self._refusing(rtw.REFUSED_SOURCE_DIRTY,
                                              "untracked handoff"))
        assert summary["outcome"] == rtw.REFUSED_SOURCE_DIRTY
        assert FakePipeline.calls == []
        assert p1.EXIT_CLASS[summary["outcome"]] == 4

    def test_isolation_gate_precedes_every_other_gate(self, rt):
        # even the GPU-binding refusal (wrong host) never fires: the
        # isolation refusal comes FIRST on the launch path.
        contract = copy.deepcopy(rt.contract)
        contract["assignments"]["101"]["hostname"] = "not-this-host"
        summary = p1.run_seed(
            101, contract=contract, bindings=rt.bindings,
            enforce_gpu=True, enforce_isolation=True,
            isolation_facts_fn=self._refusing(
                rtw.REFUSED_SOURCE_NOT_ISOLATED))
        assert summary["outcome"] == rtw.REFUSED_SOURCE_NOT_ISOLATED

    def test_verified_launch_facts_land_on_summary_and_records(
            self, rt):
        summary = _run_seed(
            rt, enforce_isolation=True,
            isolation_facts_fn=lambda: dict(CLEAN_LAUNCH_FACTS))
        assert summary["outcome"] == "SEED_COMPLETE"
        launch = summary["launch_isolation"]
        assert launch["enforced"] is True
        assert launch["verified_clean_at_launch"] is True
        for record in _records_of(rt, summary).values():
            block = record["source_isolation"]["launch"]
            assert block["enforced"] is True
            assert block["worktree_path"] == \
                CLEAN_LAUNCH_FACTS["worktree_path"]

    def test_cli_launch_path_enforces_by_default(self):
        """The fleet launch path (CLI) wires the gate ON by default;
        --no-isolation-check exists for socket-free tests only, and
        --retry-missing demands --seed."""
        source = (REPO / "tools/p1_difficulty_lr_factorial.py"
                  ).read_text()
        assert "--no-isolation-check" in source
        assert ("enforce_isolation=not args.no_isolation_check"
                in source)
        assert "--retry-missing requires --seed" in source


# ---------------------------------------------------------------------------
# Rule 2 — the documented convention (enforcement is rule 1)
# ---------------------------------------------------------------------------

class TestRule2Convention:
    def test_ops_doc_names_the_convention_and_the_refusals(self):
        doc = REPO / "docs/ops/RUNTIME_SOURCE_ISOLATION.md"
        assert doc.exists()
        text = doc.read_text()
        assert ".runtime" in text
        assert ".worktrees" in text
        assert "REFUSED_SOURCE_NOT_ISOLATED" in text
        assert "REFUSED_SOURCE_DIRTY" in text
        assert "tools/runtime_worktree.py" in text
        # convention + immunity, stated as such
        assert "CONVENTION" in text
        assert "IMMUNE" in text

    def test_new_tool_is_declared(self):
        declarations = json.loads(
            (REPO / "tools/TOOL_DECLARATIONS.json").read_text())
        entry = declarations["tools"]["runtime_worktree.py"]
        assert entry["lifecycle"] == "supported"
        assert entry["mutability"] in ("read_only", "mutating", "mixed")


# ---------------------------------------------------------------------------
# Rule 3 — record custody at materialization and terminal custody
# ---------------------------------------------------------------------------

class TestRule3SourceCustody:
    def test_identity_separates_tracked_and_untracked_digests(
            self, repo):
        canonical = repo["canonical"]
        clean = sysid.source_tree_identity(canonical)
        assert clean["tracked_diff_digest"] is None
        assert clean["untracked_digest"] is None
        assert clean["dirty_untracked_digest"] is None
        (canonical / "code.py").write_text("print('modified')\n")
        (canonical / "untracked.md").write_text("new\n")
        dirty = sysid.source_tree_identity(canonical)
        assert dirty["dirty"] is True
        assert dirty["tracked_diff_digest"] is not None
        assert dirty["untracked_digest"] is not None
        assert dirty["tracked_diff_digest"] != dirty["untracked_digest"]
        # the sealed combined digest stays present (v1 identities)
        assert dirty["dirty_untracked_digest"] is not None

    def test_untracked_only_never_fabricates_a_tracked_digest(
            self, repo):
        (repo["canonical"] / "untracked.md").write_text("new\n")
        ident = sysid.source_tree_identity(repo["canonical"])
        assert ident["tracked_diff_digest"] is None
        assert ident["untracked_digest"] is not None

    def test_records_bind_worktree_facts_at_both_custody_points(
            self, rt):
        summary = _run_seed(rt, enforce_isolation=True,
                            isolation_facts_fn=lambda: dict(
                                CLEAN_LAUNCH_FACTS))
        records = _records_of(rt, summary)
        assert sorted(records) == sorted(CELLS)
        for record in records.values():
            block = record["source_isolation"]
            assert block["schema"] == p1.SOURCE_ISOLATION_SCHEMA
            for point in ("at_materialization", "at_terminal_custody"):
                snap = block[point]
                assert sorted(snap) == ["agent-multi", "gym-fx"]
                am = snap["agent-multi"]
                assert am["worktree_path"] == WP0_SOURCES[
                    "agent-multi"]["repo_root"]
                assert am["commit"] == "9" * 40
                assert am["clean"] is True
                assert am["tracked_diff_digest"] is None
                assert am["untracked_digest"] is None
                gx = snap["gym-fx"]
                assert gx["clean"] is False
                assert gx["tracked_diff_digest"] == "a" * 64
                assert gx["untracked_digest"] == "b" * 64

    def test_unenforced_launch_is_typed_never_silent(self, rt):
        summary = _run_seed(rt)  # enforce_isolation defaults off (API)
        assert summary["launch_isolation"]["enforced"] is False
        for record in _records_of(rt, summary).values():
            launch = record["source_isolation"]["launch"]
            assert launch["enforced"] is False
            assert "MUST enforce" in launch["reason"]

    def test_drift_before_materialization_fails_cells_pre_pipeline(
            self, rt, monkeypatch):
        calls = {"n": 0}

        def moving_sources():
            calls["n"] += 1
            sources = copy.deepcopy(WP0_SOURCES)
            if calls["n"] >= 2:  # after the seed-launch baseline
                sources["agent-multi"]["dirty"] = True
                sources["agent-multi"]["dirty_untracked_digest"] = \
                    "d" * 64
            return sources

        monkeypatch.setattr(p1.ladder, "source_identities",
                            moving_sources)
        summary = _run_seed(rt)
        assert summary["outcome"] == "SEED_FAILED"
        assert summary["failed_cells"] == summary["cell_order"]
        assert summary["failure_classes"] == ["source_drift"]
        assert FakePipeline.calls == []  # caught BEFORE any model work


# ---------------------------------------------------------------------------
# Rule 4 — drift is a typed FAILED CELL plus scheduled retry
# ---------------------------------------------------------------------------

class TestRule4Classification:
    def test_legacy_incident_error_classifies_source_drift(self):
        block = ms.classify_p1lr_failure(
            "CELL_FAILED", INCIDENT_ERROR,
            unit="p1lr-screen@101.service", cell="P1N_LR1E4")
        assert block["failure_class"] == "source_drift"
        assert block["retry_eligible"] is True
        assert block["failed_cell"] == "P1N_LR1E4"
        assert block["retry"]["retry_command"] == \
            "systemctl --user restart p1lr-screen@101.service"
        assert "ALREADY_COMPLETE" in block["retry"]["retry_semantics"]

    def test_typed_drift_error_and_declared_class_classify(self):
        typed = ms.classify_p1lr_failure(
            "SEED_FAILED",
            "SourceDriftError: executing source tree moved during the "
            "cell: commit 'aaa' -> 'bbb'")
        assert typed["failure_class"] == "source_drift"
        declared = ms.classify_p1lr_failure(
            "CELL_FAILED", "RuntimeError: anything",
            declared_class="source_drift")
        assert declared["failure_class"] == "source_drift"

    def test_other_failures_are_unclassified_but_still_failed(self):
        block = ms.classify_p1lr_failure(
            "CELL_FAILED", "RuntimeError: cuda out of memory")
        assert block["failure_class"] == "unclassified"
        assert block["retry_eligible"] is True

    def test_isolation_refusal_is_never_blindly_restarted(self):
        block = ms.classify_p1lr_failure(
            "REFUSED_SOURCE_DIRTY", "untracked handoff",
            unit="p1lr-screen@101.service")
        assert block["failure_class"] == "source_isolation_refused"
        assert block["retry_eligible"] is False
        assert "runtime_worktree.py ensure" in block["remediation"]

    def test_non_failure_states_classify_none(self):
        for state in ("RUNNING", "CELL_COMPLETE", "ALREADY_COMPLETE",
                      None, ""):
            assert ms.classify_p1lr_failure(state, None) is None

    def test_mid_cell_drift_is_typed_seed_failed_with_retry(
            self, rt, monkeypatch):
        """The incident, replayed: drift lands AFTER cell 1 trained —
        cell 1 fails at terminal custody, later cells refuse BEFORE
        spending any pipeline work, and everything is typed."""
        calls = {"n": 0}

        def moving_sources():
            calls["n"] += 1
            sources = copy.deepcopy(WP0_SOURCES)
            if calls["n"] >= 3:  # baseline + cell-1 materialization
                sources["agent-multi"]["dirty"] = True
                sources["agent-multi"]["dirty_untracked_digest"] = \
                    "d" * 64
            return sources

        monkeypatch.setattr(p1.ladder, "source_identities",
                            moving_sources)
        summary = _run_seed(rt)
        assert summary["outcome"] == "SEED_FAILED"
        assert len(FakePipeline.calls) == 1  # only cell 1 trained
        assert summary["failure_classes"] == ["source_drift"]
        first_cell = summary["cell_order"][0]
        outcome = summary["cells"][first_cell]
        assert outcome["outcome"] == "CELL_FAILED"
        assert outcome["failure_class"] == "source_drift"
        assert outcome["retry_eligible"] is True
        # the CELL_FAILED heartbeat carries the typed class on disk
        root = Path(rt.contract["output_root"])
        hb = json.loads(
            (root / summary["experiment_identity"] / "seed101" /
             first_cell / "heartbeat.json").read_text())
        assert hb["terminal_state"] == "CELL_FAILED"
        assert hb["failure_class"] == "source_drift"
        assert "executing source tree moved" in hb["error"]


class _FakeReader:
    """Read-only fake for collect_p1lr_factorial."""

    def __init__(self, files):
        self.files = dict(files)
        self.errors = {}

    def read_text(self, host, path):
        return self.files.get((host, path))

    def nrestarts(self, host, unit):
        return 0

    def unit_loaded(self, host, unit):
        return True


def _status_contract(tmp_path) -> Path:
    contract = {
        "schema": "agent_multi.p1_difficulty_lr_factorial.v2",
        "experiment": "p1lr_v2_test",
        "asset": "ETHUSD",
        "cells": {name: {} for name in CELLS},
        "seeds": [101],
        "assignments": {"101": {"hostname": "omega",
                                "gpu_uuid": "GPU-101"}},
        "cell_order": {"101": list(CELLS)},
        "output_root": str(tmp_path / "out"),
        "decision_run": {"output_root": str(tmp_path / "out_decision")},
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    return path


class TestRule4StatusExposure:
    NOW = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    IDENT = "14e7ce8208ac9776"

    def _collect(self, tmp_path, heartbeat, *, hb_name=None):
        contract_path = _status_contract(tmp_path)
        root = str(tmp_path / "out")
        seed_dir = f"{root}/{self.IDENT}/seed101"
        name = hb_name or f"{seed_dir}/{CELLS[0]}/heartbeat.json"
        reader = _FakeReader({("omega", name): json.dumps(heartbeat)})
        block, unavailable, _queue = ms.collect_p1lr_factorial(
            contract_path=contract_path, reader=reader,
            identity=self.IDENT, local_hostname="omega",
            now_fn=lambda: self.NOW, mode="screen",
            identity_presence_fn=lambda r, i: r == root,
            transition_queue_dir=None)
        return block

    def test_source_drift_renders_failed_cell_plus_scheduled_retry(
            self, tmp_path):
        block = self._collect(tmp_path, {
            "schema": ms.P1LR_HEARTBEAT_SCHEMA, "mode": "screen",
            "terminal_state": "CELL_FAILED", "cell": CELLS[0],
            "error": INCIDENT_ERROR,
            "updated_utc": self.NOW.isoformat(),
        })
        worker = block["workers"]["101"]
        failure = worker["failure"]
        assert failure["failure_class"] == "source_drift"
        assert failure["retry_eligible"] is True
        assert failure["failed_cell"] == CELLS[0]
        assert failure["retry"]["retry_command"] == \
            "systemctl --user restart p1lr-screen@101.service"
        # block level: NAMED, never silent progress
        assert block["failures"] == [{
            "seed": 101, "cell": CELLS[0],
            "failure_class": "source_drift",
            "terminal_state": "CELL_FAILED",
            "retry_eligible": True}]
        assert "FAILED cell state(s)" in block["state_basis"]
        assert "source_drift" in block["state_basis"]
        assert block["state"] != "active"

    def test_runner_declared_class_wins_over_string_matching(
            self, tmp_path):
        block = self._collect(tmp_path, {
            "schema": ms.P1LR_HEARTBEAT_SCHEMA, "mode": "screen",
            "terminal_state": "CELL_FAILED", "cell": CELLS[1],
            "error": "SomethingElse: opaque", "failure_class":
                "source_drift",
            "updated_utc": self.NOW.isoformat(),
        })
        assert block["workers"]["101"]["failure"][
            "failure_class"] == "source_drift"

    def test_isolation_refusal_heartbeat_renders_no_blind_retry(
            self, tmp_path):
        root = str(tmp_path / "out")
        block = self._collect(
            tmp_path, {
                "schema": ms.P1LR_HEARTBEAT_SCHEMA, "mode": "screen",
                "terminal_state": "REFUSED_SOURCE_DIRTY", "cell": None,
                "error": "untracked handoff in the executing tree",
                "failure_class": "source_isolation_refused",
                "updated_utc": self.NOW.isoformat(),
            },
            hb_name=f"{root}/{self.IDENT}/seed101/runner_heartbeat.json")
        failure = block["workers"]["101"]["failure"]
        assert failure["failure_class"] == "source_isolation_refused"
        assert failure["retry_eligible"] is False
        assert "runtime_worktree.py ensure" in failure["remediation"]

    def test_healthy_running_worker_carries_no_failure_block(
            self, tmp_path):
        block = self._collect(tmp_path, {
            "schema": ms.P1LR_HEARTBEAT_SCHEMA, "mode": "screen",
            "terminal_state": "RUNNING", "cell": CELLS[0],
            "progress": "training",
            "updated_utc": self.NOW.isoformat(),
        })
        assert "failure" not in block["workers"]["101"]
        assert "failures" not in block


class TestRule4IdleGuard:
    def test_seed_facts_carry_error_and_failure_class(self, tmp_path):
        _guard_write(
            tmp_path, 101, f"{GUARD_ORDER[101][0]}/heartbeat.json",
            {"terminal_state": "CELL_FAILED",
             "cell": GUARD_ORDER[101][0],
             "error": INCIDENT_ERROR,
             "failure_class": "source_drift"},
            age_seconds=1200)
        facts = guard_mod.seed_facts(
            _guard_contract(tmp_path), GUARD_IDENTITY, 101, GUARD_NOW)
        assert facts["last_heartbeat_terminal_state"] == "CELL_FAILED"
        assert facts["last_heartbeat_error"] == INCIDENT_ERROR
        assert facts["last_heartbeat_failure_class"] == "source_drift"
        assert facts["last_heartbeat_cell"] == GUARD_ORDER[101][0]

    def test_stalled_drift_seed_is_classified_and_still_retried(
            self, tmp_path):
        """SEED_FAILED-due-to-source-drift: distinctly classified AND
        the bounded restart (the scheduled retry) still fires — drift
        is retryable, unlike a REFUSED_* configuration refusal."""
        _guard_write(
            tmp_path, 101, f"{GUARD_ORDER[101][0]}/heartbeat.json",
            {"terminal_state": "CELL_FAILED",
             "cell": GUARD_ORDER[101][0],
             "error": INCIDENT_ERROR},
            age_seconds=1200)
        report, emitter, restarts = _guard_poll(tmp_path)
        entry = report["seeds"]["101"]
        assert entry["failure"]["failure_class"] == "source_drift"
        assert entry["failure"]["retry_eligible"] is True
        assert entry["stalled"] is True
        assert "restart_attempted" in entry["actions"]
        assert restarts == ["p1lr-screen@101.service"]

    def test_isolation_refusal_withholds_the_restart(self, tmp_path):
        _guard_write(
            tmp_path, 101, "runner_heartbeat.json",
            {"terminal_state": "REFUSED_SOURCE_DIRTY", "cell": None,
             "error": "untracked handoff",
             "failure_class": "source_isolation_refused"},
            age_seconds=1200)
        report, emitter, restarts = _guard_poll(tmp_path)
        entry = report["seeds"]["101"]
        assert entry["failure"]["failure_class"] == \
            "source_isolation_refused"
        assert entry["failure"]["retry_eligible"] is False
        assert "restart_withheld_typed_refusal" in entry["actions"]
        assert restarts == []


# ---------------------------------------------------------------------------
# Rule 5 — retry missing cells only, after the seed batch
# ---------------------------------------------------------------------------

class TestRule5RetryMissing:
    def test_rerun_executes_only_missing_cells_and_reuses_the_rest(
            self, rt):
        first = _run_seed(rt)
        assert first["outcome"] == "SEED_COMPLETE"
        assert len(FakePipeline.calls) == 4
        exp_dir = (Path(rt.contract["output_root"]) /
                   first["experiment_identity"] / "seed101")
        missing = list(first["cell_order"][1:3])
        for cell in missing:
            (exp_dir / cell / "cell_record.json").unlink()

        plan = p1.seed_retry_plan(rt.contract, 101,
                                  bindings=rt.bindings)
        assert plan["schema"] == p1.RETRY_PLAN_SCHEMA
        assert plan["experiment_identity"] == \
            first["experiment_identity"]
        assert sorted(plan["missing_cells"]) == sorted(missing)
        assert sorted(plan["complete_cells"]) == sorted(
            c for c in first["cell_order"] if c not in missing)
        assert plan["will_run"] == plan["missing_cells"]
        assert plan["invalid_records"] == []

        FakePipeline.calls = []
        second = _run_seed(rt)
        # ONLY the missing cells ran; the two complete records were
        # reused byte-identically without touching their pipelines.
        assert second["outcome"] == "SEED_COMPLETE"
        assert len(FakePipeline.calls) == 2
        ran = {c["phase1_mode"] for c in FakePipeline.calls}
        assert ran == {rt.contract["cells"][cell]["phase1_dynamics"]
                       for cell in missing}
        for cell in first["cell_order"]:
            expected = ("CELL_COMPLETE" if cell in missing
                        else "ALREADY_COMPLETE")
            assert second["cells"][cell]["outcome"] == expected
        assert second["cells_completed"] == 4

    def test_retried_cells_run_after_the_batch_in_contract_order(
            self, rt):
        first = _run_seed(rt)
        exp_dir = (Path(rt.contract["output_root"]) /
                   first["experiment_identity"] / "seed101")
        order = first["cell_order"]
        (exp_dir / order[0] / "cell_record.json").unlink()
        (exp_dir / order[3] / "cell_record.json").unlink()
        FakePipeline.calls = []
        second = _run_seed(rt)
        assert second["outcome"] == "SEED_COMPLETE"
        # the retry pass visits the seed's cells in CONTRACT order, so
        # the missing ones run first-to-last after the original batch
        ran = [c["phase1_mode"] for c in FakePipeline.calls]
        assert ran == [rt.contract["cells"][order[0]]["phase1_dynamics"],
                       rt.contract["cells"][order[3]]["phase1_dynamics"]]

    def test_invalid_record_is_named_and_refused_not_overwritten(
            self, rt):
        first = _run_seed(rt)
        exp_dir = (Path(rt.contract["output_root"]) /
                   first["experiment_identity"] / "seed101")
        order = first["cell_order"]
        record_path = exp_dir / order[1] / "cell_record.json"
        record = json.loads(record_path.read_text())
        record.pop("actor_liveness_binding")  # strip v2 evidence
        record_path.write_text(json.dumps(record))

        plan = p1.seed_retry_plan(rt.contract, 101,
                                  bindings=rt.bindings)
        assert plan["invalid_records"] == [order[1]]
        assert plan["will_refuse"] == [order[1]]
        assert order[1] not in plan["will_run"]

        FakePipeline.calls = []
        second = _run_seed(rt)
        assert second["outcome"] == "SEED_FAILED"
        assert second["cells"][order[1]]["outcome"] == "CELL_FAILED"
        assert "refusing to overwrite" in second["cells"][order[1]][
            "error"]
        assert FakePipeline.calls == []  # valid cells were NOT rerun
        for cell in (order[0], order[2], order[3]):
            assert second["cells"][cell]["outcome"] == \
                "ALREADY_COMPLETE"

    def test_retry_plan_refuses_unknown_mode(self, rt):
        with pytest.raises(ValueError, match="unknown execution mode"):
            p1.seed_retry_plan(rt.contract, 101, bindings=rt.bindings,
                               mode="bogus")
