#!/usr/bin/env python3
"""Runtime source isolation: pinned detached worktrees for experiments.

WP0 (order 2026-08-15 §2). THE INCIDENT THIS CORRECTS: the corrected
P1LR v2 screen rejected all four omega seed-101 cells because a third
agent wrote an untracked handoff file into the CANONICAL checkout while
the experiment executed from it. The source-identity guard
(``assert_source_identity_unmoved``) correctly refused — but the
experiment should never have been executing from a shared writable
checkout, the failure surfaced as SEED_FAILED killing all four cells,
and recovery was manual.

The rule, enforced here and consumed by the runner launch path:

1. every long-running experiment executes from a DEDICATED DETACHED
   WORKTREE bound to one commit and VERIFIED CLEAN before launch —
   :func:`assert_isolated_launch` refuses (typed, before any GPU or
   model work) a launch whose executing tree is (a) not a detached
   worktree under the declared runtime root, or (b) dirty / carrying
   untracked files at launch;
2. agents write only to separate NAMED worktrees (convention:
   ``~/Documents/GitHub/.worktrees/<repo>-<agent>-<topic>-<date>``) and
   never to the canonical checkout while anything might execute from
   it. That convention cannot be technically forced on other agents —
   the experiment is made IMMUNE to violations by rule 1. See
   docs/ops/RUNTIME_SOURCE_ISOLATION.md;
3. :func:`ensure_runtime_worktree` is the ONE shared mechanism that
   creates or verifies the pinned worktree, for operators (this
   module's CLI) and units (the pin_p1lr_decision_runtime.sh drop-in
   pattern) alike.

Declared runtime root (existing convention, the p1lr-decision drop-in
pattern): ``~/Documents/GitHub/.runtime``, overridable via the
``P1LR_RUNTIME_ROOT`` / ``AGENT_MULTI_RUNTIME_ROOT`` environment
variables or the ``runtime_root`` parameter.

Typed refusal codes (exit class 4 in the runner — configuration
refusals are recorded FAILED and never blindly restarted):

  REFUSED_SOURCE_NOT_ISOLATED   executing tree is not a detached
                                worktree under the runtime root
  REFUSED_SOURCE_DIRTY          executing/pinned tree carries tracked
                                modifications or untracked files
  REFUSED_WORKTREE_COMMIT_MISMATCH
                                an existing pinned worktree sits on a
                                different commit than requested

Socket-free by construction: everything is local git subprocess facts,
and every predicate accepts injected facts for tests.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as sysid  # noqa: E402

LAUNCH_FACTS_SCHEMA = "agent_multi.runtime_worktree_launch_facts.v1"

#: The declared runtime root (p1lr drop-in convention).
DEFAULT_RUNTIME_ROOT = Path.home() / "Documents/GitHub/.runtime"
RUNTIME_ROOT_ENV_VARS = ("P1LR_RUNTIME_ROOT", "AGENT_MULTI_RUNTIME_ROOT")

REFUSED_SOURCE_NOT_ISOLATED = "REFUSED_SOURCE_NOT_ISOLATED"
REFUSED_SOURCE_DIRTY = "REFUSED_SOURCE_DIRTY"
REFUSED_WORKTREE_COMMIT_MISMATCH = "REFUSED_WORKTREE_COMMIT_MISMATCH"

REFUSAL_CODES = (REFUSED_SOURCE_NOT_ISOLATED, REFUSED_SOURCE_DIRTY,
                 REFUSED_WORKTREE_COMMIT_MISMATCH)


class RuntimeWorktreeRefusal(RuntimeError):
    """A typed runtime-source-isolation refusal (never a silent pass).

    ``code`` is one of :data:`REFUSAL_CODES`; ``facts`` carries the
    observed launch-tree facts so the refusal is self-describing
    evidence, not a pointer to context.
    """

    def __init__(self, code: str, reason: str,
                 facts: Optional[Dict[str, Any]] = None) -> None:
        if code not in REFUSAL_CODES:  # fail closed on typos
            raise ValueError(f"unknown refusal code {code!r}")
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason
        self.facts = dict(facts or {})


def declared_runtime_root(runtime_root: Optional[Path] = None) -> Path:
    """The declared runtime root: explicit parameter, then the drop-in
    environment variables, then the home convention."""
    if runtime_root is not None:
        return Path(runtime_root).expanduser()
    for name in RUNTIME_ROOT_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return DEFAULT_RUNTIME_ROOT


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo_root), *args],
                          capture_output=True, text=True)


def _is_detached(repo_root: Path) -> bool:
    """True when HEAD is detached (no symbolic ref)."""
    proc = _run_git(repo_root, "symbolic-ref", "-q", "HEAD")
    return proc.returncode != 0


def _is_linked_worktree(repo_root: Path) -> bool:
    """True for a linked ``git worktree`` (its .git is a file, not the
    object-store directory of the canonical clone)."""
    return (Path(repo_root) / ".git").is_file()


def launch_tree_facts(anchor: Optional[Path] = None,
                      runtime_root: Optional[Path] = None
                      ) -> Dict[str, Any]:
    """Observed isolation facts of the tree that actually executes.

    Derives the repo root from the ACTUAL checkout containing
    ``anchor`` (this module by default — the code that is running),
    never from a hard-coded sibling path, and reuses the dirty-aware
    source identity so the recorded digests are the exact family the
    experiment identity already binds.
    """
    anchor = Path(anchor or __file__)
    repo_root = sysid.resolve_repo_root(anchor)
    identity = sysid.source_tree_identity(repo_root)
    root = declared_runtime_root(runtime_root).resolve()
    resolved = Path(identity["repo_root"]).resolve()
    under_runtime_root = root in resolved.parents
    return {
        "schema": LAUNCH_FACTS_SCHEMA,
        "worktree_path": identity["repo_root"],
        "commit": identity["commit"],
        "detached": _is_detached(resolved),
        "linked_worktree": _is_linked_worktree(resolved),
        "runtime_root": str(root),
        "under_runtime_root": under_runtime_root,
        "clean": not identity["dirty"],
        "dirty_entries": identity["dirty_entries"],
        "dirty_untracked_digest": identity["dirty_untracked_digest"],
        "tracked_diff_digest": identity.get("tracked_diff_digest"),
        "untracked_digest": identity.get("untracked_digest"),
    }


def assert_isolated_launch(facts: Optional[Dict[str, Any]] = None, *,
                           anchor: Optional[Path] = None,
                           runtime_root: Optional[Path] = None
                           ) -> Dict[str, Any]:
    """WP0 rule 1: refuse a launch that does not execute from a clean
    detached worktree under the declared runtime root.

    Returns the verified facts (for the launch record) on success;
    raises :class:`RuntimeWorktreeRefusal` with code
    ``REFUSED_SOURCE_NOT_ISOLATED`` or ``REFUSED_SOURCE_DIRTY``
    otherwise. ``facts`` is injectable for socket-free tests; by
    default the facts of the EXECUTING tree are gathered fresh.
    """
    facts = dict(facts if facts is not None
                 else launch_tree_facts(anchor, runtime_root))
    if not facts.get("detached") or not facts.get("under_runtime_root"):
        raise RuntimeWorktreeRefusal(
            REFUSED_SOURCE_NOT_ISOLATED,
            "the executing tree "
            f"{facts.get('worktree_path')!r} is not a DEDICATED DETACHED "
            "worktree under the declared runtime root "
            f"{facts.get('runtime_root')!r} (detached="
            f"{facts.get('detached')}, under_runtime_root="
            f"{facts.get('under_runtime_root')}). A long-running "
            "experiment never executes from a shared writable checkout "
            "(WP0, incident 2026-08-15); create the pinned worktree with "
            "tools/runtime_worktree.py ensure <commit> and launch from "
            "it", facts)
    if not facts.get("clean"):
        entries = facts.get("dirty_entries") or []
        listed = ", ".join(f"{e.get('status')} {e.get('path')}"
                           for e in entries[:5])
        raise RuntimeWorktreeRefusal(
            REFUSED_SOURCE_DIRTY,
            "the executing worktree "
            f"{facts.get('worktree_path')!r} is DIRTY at launch "
            f"({len(entries)} entr{'y' if len(entries) == 1 else 'ies'}: "
            f"{listed}{'…' if len(entries) > 5 else ''}; "
            f"tracked_diff_digest={facts.get('tracked_diff_digest')!r}, "
            f"untracked_digest={facts.get('untracked_digest')!r}) — an "
            "unclean tree can never masquerade as a commit; restore it "
            "or pin a fresh worktree (WP0)", facts)
    facts["verified_clean_at_launch"] = True
    return facts


def verify_runtime_worktree(path: Path, commit: str) -> Dict[str, Any]:
    """Verify an EXISTING pinned worktree: detached, clean, at exactly
    ``commit``. Returns its facts; raises the typed refusal."""
    path = Path(path)
    proc = _run_git(path, "rev-parse", "HEAD")
    if proc.returncode != 0:
        raise RuntimeWorktreeRefusal(
            REFUSED_SOURCE_NOT_ISOLATED,
            f"{path} exists but is not a git worktree: "
            f"{proc.stderr.strip()[:200]}",
            {"worktree_path": str(path)})
    head = proc.stdout.strip()
    if head != commit:
        raise RuntimeWorktreeRefusal(
            REFUSED_WORKTREE_COMMIT_MISMATCH,
            f"existing runtime worktree {path} sits on {head[:12]}…, not "
            f"the requested {commit[:12]}… — one pinned worktree binds "
            "ONE commit; never retarget it in place (WP0)",
            {"worktree_path": str(path), "commit": head,
             "requested_commit": commit})
    identity = sysid.source_tree_identity(path)
    if identity["dirty"]:
        raise RuntimeWorktreeRefusal(
            REFUSED_SOURCE_DIRTY,
            f"existing runtime worktree {path} is dirty "
            f"(digest {identity['dirty_untracked_digest']}); a pinned "
            "runtime worktree is read-only execution space (WP0)",
            {"worktree_path": str(path),
             "dirty_entries": identity["dirty_entries"],
             "tracked_diff_digest": identity.get("tracked_diff_digest"),
             "untracked_digest": identity.get("untracked_digest")})
    if not _is_detached(path):
        raise RuntimeWorktreeRefusal(
            REFUSED_SOURCE_NOT_ISOLATED,
            f"existing runtime worktree {path} is on a BRANCH, not "
            "detached — a branch head can move under a running "
            "experiment (WP0)",
            {"worktree_path": str(path), "commit": head})
    return {
        "worktree_path": str(path),
        "commit": head,
        "detached": True,
        "clean": True,
        "tracked_diff_digest": None,
        "untracked_digest": None,
    }


def ensure_runtime_worktree(commit: str, *,
                            repo_root: Optional[Path] = None,
                            runtime_root: Optional[Path] = None,
                            label: Optional[str] = None) -> Path:
    """Create-or-verify the DEDICATED DETACHED worktree pinned to
    ``commit`` under the declared runtime root, and return its path.

    The ONE shared mechanism (WP0 rule 1) for operators and units: an
    existing worktree is VERIFIED (detached, clean, exact commit —
    typed refusal otherwise, never silent reuse); a missing one is
    created with ``git worktree add --detach``. The name follows the
    drop-in convention ``<repo>-[<label>-]<commit12>``.
    """
    repo_root = Path(repo_root).resolve() if repo_root is not None \
        else sysid.resolve_repo_root(Path(__file__))
    proc = _run_git(repo_root, "rev-parse", "--verify",
                    f"{commit}^{{commit}}")
    if proc.returncode != 0:
        raise RuntimeWorktreeRefusal(
            REFUSED_WORKTREE_COMMIT_MISMATCH,
            f"revision {commit!r} does not resolve to a commit in "
            f"{repo_root}: {proc.stderr.strip()[:200]}",
            {"requested_commit": commit, "repo_root": str(repo_root)})
    full = proc.stdout.strip()
    # Absolute by construction: a relative root would resolve against
    # the GIT subprocess cwd (the repo) while Python resolved it
    # against the caller's cwd — two different trees.
    root = declared_runtime_root(runtime_root).resolve()
    name = (f"{repo_root.name}-{label}-{full[:12]}" if label
            else f"{repo_root.name}-{full[:12]}")
    path = root / name
    if path.exists():
        verify_runtime_worktree(path, full)
        return path
    root.mkdir(parents=True, exist_ok=True)
    proc = _run_git(repo_root, "worktree", "add", "--detach",
                    str(path), full)
    if proc.returncode != 0:
        raise RuntimeWorktreeRefusal(
            REFUSED_SOURCE_NOT_ISOLATED,
            f"git worktree add --detach {path} {full[:12]}… failed: "
            f"{proc.stderr.strip()[:300]}",
            {"worktree_path": str(path), "requested_commit": full})
    verify_runtime_worktree(path, full)  # post-condition, fail closed
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ensure = sub.add_parser(
        "ensure", help="create/verify the pinned detached worktree for "
                       "a commit under the runtime root; prints its "
                       "path and facts")
    ensure.add_argument("commit", help="git revision to pin")
    ensure.add_argument("--label", default=None,
                        help="optional name label, e.g. p1lr-v2")
    ensure.add_argument("--repo-root", type=Path, default=None)
    ensure.add_argument("--runtime-root", type=Path, default=None)
    verify = sub.add_parser(
        "verify", help="verify THIS executing tree satisfies the WP0 "
                       "launch rule (detached worktree under the "
                       "runtime root, clean); exit 0 facts / exit 4 "
                       "typed refusal")
    verify.add_argument("--runtime-root", type=Path, default=None)
    args = parser.parse_args()
    try:
        if args.command == "ensure":
            path = ensure_runtime_worktree(
                args.commit, repo_root=args.repo_root,
                runtime_root=args.runtime_root, label=args.label)
            # anchor on the worktree's own .git file so the facts are
            # gathered from the PINNED tree, not this executing one
            facts = launch_tree_facts(path / ".git", args.runtime_root)
            print(json.dumps({"outcome": "RUNTIME_WORKTREE_READY",
                              "path": str(path), "facts": facts}))
            return 0
        facts = assert_isolated_launch(runtime_root=args.runtime_root)
        print(json.dumps({"outcome": "LAUNCH_TREE_ISOLATED",
                          "facts": facts}))
        return 0
    except RuntimeWorktreeRefusal as refusal:
        print(json.dumps({"outcome": refusal.code,
                          "reason": refusal.reason,
                          "facts": refusal.facts}))
        return 4


if __name__ == "__main__":
    sys.exit(main())
