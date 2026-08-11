"""Finding 217 checker negative tests (WP0, order 209-223 §4).

Socket-free: every test injects a fake git runner; no subprocess, no
network, no repository state is touched. Each adversarial fixture must
produce a NONZERO exit code from ``main_with`` and an incremented
``failure_total`` in the written result:

- missing README on the remote default tree;
- local feature-branch drift (local HEAD has the file, the remote
  default tree does not — the checker must consult only the remote tip);
- Git command failure (unresolvable default ref / cat-file hard error);
- an expected repository with no fully checked row.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECKER_PATH = (REPO / "docs/audits/evidence/"
                "README_LINK_RESOLUTION_CHECKER_2026_08_10.py")

_spec = importlib.util.spec_from_file_location(
    "readme_link_resolution_checker", CHECKER_PATH)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

REMOTE_TIP = "a" * 40
LOCAL_HEAD = "b" * 40  # never a valid revision for the fake: local-only state


class FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeGit:
    """Injectable stand-in for the git runner.

    ``repos`` maps repo name -> dict with keys:
      default_branch, tip, readme (str | None), tree (set of paths at the
      REMOTE default tip), local_tree (paths only on the local HEAD),
      ls_remote_rc (nonzero simulates a git/network command failure),
      cat_file_rc (nonzero forces every tree lookup into a hard git error).
    """

    def __init__(self, repos):
        self.repos = repos
        self.calls = []

    def __call__(self, repo_dir, *args):
        self.calls.append((repo_dir, args))
        name = repo_dir[len(checker.BASE):]
        cfg = self.repos.get(name)
        if cfg is None:
            return FakeProc(128, stderr=f"fatal: cannot change to '{repo_dir}'")
        if args[:2] == ("rev-parse", "--git-dir"):
            return FakeProc(0, stdout=".git\n")
        if args[0] == "ls-remote":
            rc = cfg.get("ls_remote_rc", 0)
            if rc:
                return FakeProc(rc, stderr="ssh: connect to host github.com "
                                           "port 22: Connection refused")
            tip = cfg["tip"]
            branch = cfg["default_branch"]
            return FakeProc(0, stdout=(
                f"ref: refs/heads/{branch}\tHEAD\n{tip}\tHEAD\n"))
        if args[0] == "fetch":
            return FakeProc(0)
        if args[0] == "cat-file":
            spec = args[-1]
            tip = cfg["tip"]
            if spec == f"{tip}^{{commit}}":
                return FakeProc(0)
            if args[1] == "-p":
                if spec == f"{tip}:README.md":
                    if cfg.get("readme") is None:
                        return FakeProc(128, stderr=(
                            f"fatal: path 'README.md' does not exist "
                            f"in '{tip}'"))
                    return FakeProc(0, stdout=cfg["readme"])
                return FakeProc(128, stderr=f"fatal: bad revision {spec!r}")
            if args[1] == "-e":
                rc = cfg.get("cat_file_rc", 0)
                if rc:
                    return FakeProc(rc, stderr="fatal: unable to read tree")
                rev, _, path = spec.partition(":")
                if rev != tip:
                    # The corrected checker must never resolve links against
                    # anything but the remote default tip (e.g. local HEAD).
                    return FakeProc(128, stderr=(
                        f"fatal: fake refuses non-remote-tip revision {rev!r}"))
                if path in cfg["tree"]:
                    return FakeProc(0)
                if path in cfg.get("local_tree", ()):
                    # Real git wording when the working tree has the file but
                    # the queried committed tree does not (rc=128).
                    return FakeProc(128, stderr=(
                        f"fatal: path '{path}' exists on disk, but not in "
                        f"'{tip}'"))
                return FakeProc(128, stderr=(
                    f"fatal: path '{path}' does not exist in '{tip}'"))
        raise AssertionError(f"unexpected git invocation: {args!r}")


def _run(tmp_path, fake, repos, declared=None):
    out = tmp_path / "result.json"
    argv = [str(out), "test-label", *repos]
    rc = checker.main_with(argv, git=fake, declared=declared or {})
    return rc, json.loads(out.read_text())


def _repo(readme="[ok](good.py)\n", tree=frozenset({"good.py"}), **kw):
    cfg = {"default_branch": "master", "tip": REMOTE_TIP,
           "readme": readme, "tree": set(tree)}
    cfg.update(kw)
    return cfg


def test_all_green_exits_zero(tmp_path):
    fake = FakeGit({"repo-a": _repo()})
    rc, result = _run(tmp_path, fake, ["repo-a"])
    assert rc == 0
    assert result["failure_total"] == 0
    assert result["error_rows_total"] == 0
    assert result["missing_rows_total"] == 0
    row = result["repositories"][0]
    assert row["repository"] == "repo-a"
    assert row["branch"] == "master"
    assert row["head"] == REMOTE_TIP  # exact remote default tip, not local
    assert row["broken_relative_count"] == 0
    assert result["subset_of_full_inventory"] is True
    assert result["expected_repositories"] == ["repo-a"]


def test_missing_readme_increments_failure_and_exits_nonzero(tmp_path):
    fake = FakeGit({"repo-a": _repo(readme=None)})
    rc, result = _run(tmp_path, fake, ["repo-a"])
    assert rc != 0
    assert result["failure_total"] == 1
    assert result["error_rows_total"] == 1
    row = result["repositories"][0]
    assert "README.md not in remote default tree" in row["error"]


def test_missing_repository_increments_failure(tmp_path):
    fake = FakeGit({})  # directory/repo absent entirely
    rc, result = _run(tmp_path, fake, ["ghost-repo"])
    assert rc != 0
    assert result["failure_total"] == 1
    assert "missing repository" in result["repositories"][0]["error"]


def test_local_feature_branch_drift_is_broken_on_remote_default(tmp_path):
    """Local HEAD carries the file; the remote default tree does not.

    The 2026-08-10 checker resolved against local HEAD and reported such a
    link green. The corrected checker must resolve against the remote
    default tip only and count the link broken.
    """
    readme = ("[nested](pipeline_plugins/_nested_splits.py)\n"
              "[ok](setup.py)\n")
    fake = FakeGit({"repo-a": _repo(
        readme=readme,
        tree={"setup.py"},  # remote default tree
        local_tree={"setup.py", "pipeline_plugins/_nested_splits.py"})})
    rc, result = _run(tmp_path, fake, ["repo-a"])
    assert rc != 0
    row = result["repositories"][0]
    assert row["head"] == REMOTE_TIP
    assert row["broken_relative_count"] == 1
    assert (row["broken_relative"][0]["target"]
            == "pipeline_plugins/_nested_splits.py")
    assert result["failure_total"] == 1
    # The checker never consulted any local revision: every cat-file lookup
    # was pinned to the remote tip and no local rev-parse HEAD was issued.
    for _, args in fake.calls:
        assert "HEAD" not in args or args[0] == "ls-remote"
        if args[0] == "cat-file" and ":" in args[-1]:
            assert args[-1].startswith(REMOTE_TIP)


def test_git_command_failure_unresolved_default_ref(tmp_path):
    fake = FakeGit({"repo-a": _repo(ls_remote_rc=128)})
    rc, result = _run(tmp_path, fake, ["repo-a"])
    assert rc != 0
    assert result["failure_total"] == 1
    assert "unresolved default ref" in result["repositories"][0]["error"]


def test_git_command_hard_error_during_link_check(tmp_path):
    fake = FakeGit({"repo-a": _repo(cat_file_rc=128)})
    rc, result = _run(tmp_path, fake, ["repo-a"])
    assert rc != 0
    assert result["failure_total"] == 1
    assert "cat-file" in result["repositories"][0]["error"]


def test_declared_default_branch_mismatch_fails(tmp_path):
    fake = FakeGit({"repo-a": _repo()})
    rc, result = _run(tmp_path, fake, ["repo-a"],
                      declared={"repo-a": "main"})  # remote says master
    assert rc != 0
    assert "default-ref mismatch" in result["repositories"][0]["error"]


def test_missing_row_counts_as_failure():
    missing = checker.verify_coverage(["repo-a", "repo-b"],
                                      [{"repository": "repo-a",
                                        "error": "x"}])
    assert missing == [{"repository": "repo-b",
                        "error": "no result row produced"}]
    result = checker.build_result(
        "l", ["repo-a", "repo-b"], [],
        [{"repository": "repo-a", "error": "x"}], [])
    # one error row + one missing row
    assert result["missing_rows_total"] == 1
    assert result["error_rows_total"] == 1
    assert result["failure_total"] == 2


def test_partial_row_counts_as_failure():
    rows = [{"repository": "repo-a", "branch": "master"}]  # incomplete
    missing = checker.verify_coverage(["repo-a"], rows)
    assert len(missing) == 1
    assert "partial row" in missing[0]["error"]
    result = checker.build_result("l", ["repo-a"], [], rows, [])
    assert result["failure_total"] == 1


def test_mixed_fleet_counts_every_failure_class(tmp_path):
    """One green repo, one broken link, one missing README, one git error."""
    fake = FakeGit({
        "green": _repo(),
        "drift": _repo(readme="[x](gone.py)\n", tree=set()),
        "noreadme": _repo(readme=None),
        "neterr": _repo(ls_remote_rc=128),
    })
    rc, result = _run(tmp_path, fake, ["green", "drift", "noreadme", "neterr"])
    assert rc != 0
    assert result["broken_relative_total"] == 1
    assert result["error_rows_total"] == 2
    assert result["missing_rows_total"] == 0
    assert result["failure_total"] == 3
    assert result["readmes_checked"] == 4  # every expected repo has a row
