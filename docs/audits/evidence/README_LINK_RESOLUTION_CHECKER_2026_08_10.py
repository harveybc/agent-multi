#!/usr/bin/env python3
"""Finding 217 link checker (corrected 2026-08-11 per WP0, order 209-223).

Resolves EVERY relative link found in the delivered READMEs through
`git cat-file -e <remote-default-tip>:<target>` against the exact committed
tree at each repository's REMOTE DEFAULT ref. The default ref is resolved
from GitHub metadata (`git ls-remote --symref origin HEAD`) and cross-checked
against the `default_branch` declared in
`REPOSITORY_PRESENTATION_INVENTORY_2026_08_10.json`. An arbitrary local
`HEAD` is never consulted.

Failure accounting (the 2026-08-10 version under-counted; corrected here):

- a missing repository, unresolved/mismatched default ref, missing README or
  any Git command error produces an error row and increments failure;
- every broken relative link increments failure;
- every expected repository must produce a fully checked row; a missing or
  partial row increments failure;
- the process exits nonzero iff ``failure_total`` > 0.

External links (http/https/mailto) are listed separately and are NOT resolved
here; syntactic parsing is not equated with reachability.

Usage:
    python README_LINK_RESOLUTION_CHECKER_2026_08_10.py OUT.json LABEL [repo ...]

With explicit repo names only that subset is checked and the result records
``subset_of_full_inventory: true`` naming exactly the repositories covered.
"""
import json
import os
import re
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone

BASE = "/home/harveybc/Documents/GitHub/"
HERE = os.path.dirname(os.path.abspath(__file__))
INVENTORY_PATH = os.path.join(
    HERE, "REPOSITORY_PRESENTATION_INVENTORY_2026_08_10.json")

# The 20 WP-B delivered READMEs (inventory minus causal-inference).
REPOS = [
    "agent-multi", "doin-core", "doin-evaluator", "doin-node",
    "doin-optimizer", "doin-plugins", "feature-eng", "feature-extractor",
    "financial-data", "gym-fx", "heuristic-strategy", "lts",
    "prediction_provider", "predictor", "preprocessor", "rl-optimizer",
    "synthetic-datagen", "timeseries-gan", "trading-contracts",
    "trading-signal",
]
# Checked separately (finding 219 refresh; not part of the 20 WP-B READMEs).
# Resolved against its remote default ref exactly like every other repo; its
# canonical dirty checkout is never consulted.
EXTRA_REPOS = ["causal-inference"]

LINK_RE = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

REF_SOURCE = ("git ls-remote --symref origin HEAD, cross-checked against "
              "inventory default_branch")

ROW_REQUIRED_KEYS = frozenset({
    "repository", "branch", "head", "relative_links_checked",
    "external_links", "external_links_count", "broken_relative",
    "broken_relative_count",
})


class CheckError(Exception):
    """A condition that must yield an error row and a nonzero exit."""


def default_git_runner(repo_dir, *args):
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_SSH_COMMAND", "ssh -oBatchMode=yes")
    return subprocess.run(["git", "-C", repo_dir, *args],
                          capture_output=True, text=True, env=env)


def load_declared_default_branches(path=INVENTORY_PATH):
    with open(path) as f:
        inv = json.load(f)
    return {name: meta.get("default_branch")
            for name, meta in inv["repositories"].items()}


def _run(git, repo_dir, *args):
    try:
        return git(repo_dir, *args)
    except Exception as exc:  # runner itself blew up: a Git command error
        raise CheckError(f"git {' '.join(args)} raised {exc!r}")


def resolve_remote_default(repo, declared_branch, git):
    """Return (branch, tip_sha) for the repository's remote default ref."""
    repo_dir = BASE + repo
    r = _run(git, repo_dir, "rev-parse", "--git-dir")
    if r.returncode != 0:
        raise CheckError(
            f"missing repository or not a git repository: "
            f"{(r.stderr or '').strip()}")
    r = _run(git, repo_dir, "ls-remote", "--symref", "origin", "HEAD")
    if r.returncode != 0:
        raise CheckError(
            f"unresolved default ref: git ls-remote failed "
            f"(rc={r.returncode}): {(r.stderr or '').strip()}")
    branch = tip = None
    for line in (r.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        left, right = parts
        if right != "HEAD":
            continue
        if left.startswith("ref: refs/heads/"):
            branch = left[len("ref: refs/heads/"):]
        elif re.fullmatch(r"[0-9a-f]{40}", left):
            tip = left
    if not branch or not tip:
        raise CheckError(
            "unresolved default ref: could not parse ls-remote --symref "
            f"output: {(r.stdout or '').strip()!r}")
    if declared_branch and declared_branch != branch:
        raise CheckError(
            f"default-ref mismatch: inventory declares {declared_branch!r} "
            f"but remote HEAD is {branch!r}")
    # Make sure the exact remote tip object is available locally.
    r = _run(git, repo_dir, "cat-file", "-e", tip + "^{commit}")
    if r.returncode != 0:
        f = _run(git, repo_dir, "fetch", "--quiet", "origin", branch)
        if f.returncode != 0:
            raise CheckError(
                f"remote default tip {tip} not present locally and fetch "
                f"failed (rc={f.returncode}): {(f.stderr or '').strip()}")
        r = _run(git, repo_dir, "cat-file", "-e", tip + "^{commit}")
        if r.returncode != 0:
            raise CheckError(
                f"remote default tip {tip} still absent after fetch")
    return branch, tip


def _is_path_missing(returncode, stderr):
    """git reports tree-missing paths as rc=1 (bare objects) or rc=128 with
    one of two fatal messages; anything else is a real command error."""
    if returncode == 1:
        return True
    err = stderr or ""
    return ("does not exist in" in err
            or "exists on disk, but not in" in err)


def _cat_file_exists(git, repo_dir, spec):
    """True/False for object existence; CheckError on real git failure."""
    r = _run(git, repo_dir, "cat-file", "-e", spec)
    if r.returncode == 0:
        return True
    if _is_path_missing(r.returncode, r.stderr):
        return False
    raise CheckError(
        f"git cat-file -e {spec} error (rc={r.returncode}): "
        f"{(r.stderr or '').strip()}")


def check_repo(repo, declared_branch, git):
    repo_dir = BASE + repo
    try:
        branch, tip = resolve_remote_default(repo, declared_branch, git)
        blob = _run(git, repo_dir, "cat-file", "-p", f"{tip}:README.md")
        if blob.returncode != 0 and _is_path_missing(blob.returncode,
                                                    blob.stderr):
            raise CheckError(
                f"README.md not in remote default tree ({branch}@{tip})")
        if blob.returncode != 0:
            raise CheckError(
                f"git cat-file -p {tip}:README.md error "
                f"(rc={blob.returncode}): {(blob.stderr or '').strip()}")
        text = blob.stdout
        external, relative, broken = [], [], []
        for raw in LINK_RE.findall(text):
            link = raw.strip("<>")
            if link.startswith(("http://", "https://", "mailto:")):
                external.append(link)
                continue
            if link.startswith("#"):
                continue  # intra-document anchor
            target = urllib.parse.unquote(link.split("#")[0]).rstrip("/")
            if not target:
                continue
            relative.append(link)
            if not _cat_file_exists(git, repo_dir, f"{tip}:{target}"):
                broken.append({"link": link, "target": target})
        return {
            "repository": repo,
            "branch": branch,
            "ref_source": REF_SOURCE,
            "head": tip,
            "relative_links_checked": len(relative),
            "external_links": sorted(set(external)),
            "external_links_count": len(external),
            "broken_relative": broken,
            "broken_relative_count": len(broken),
        }
    except CheckError as exc:
        return {"repository": repo, "error": str(exc)}
    except Exception as exc:  # never silently drop a row
        return {"repository": repo,
                "error": f"unexpected checker exception: {exc!r}"}


def verify_coverage(expected, rows):
    """Every expected repository must have produced a fully checked row."""
    rows_by = {}
    for row in rows:
        if isinstance(row, dict) and "repository" in row:
            rows_by.setdefault(row["repository"], row)
    missing = []
    for name in expected:
        row = rows_by.get(name)
        if row is None:
            missing.append({"repository": name,
                            "error": "no result row produced"})
        elif "error" not in row and not ROW_REQUIRED_KEYS.issubset(row):
            absent = sorted(ROW_REQUIRED_KEYS - set(row))
            missing.append({"repository": name,
                            "error": f"partial row, missing keys: {absent}"})
    return missing


def build_result(label, repos, extra_repos, rows, extra_rows):
    missing = (verify_coverage(repos, rows)
               + verify_coverage(extra_repos, extra_rows))
    error_rows = [r for r in rows + extra_rows if "error" in r]
    broken_main = sum(r.get("broken_relative_count", 0) for r in rows)
    broken_extra = sum(r.get("broken_relative_count", 0) for r in extra_rows)
    failure_total = len(error_rows) + len(missing) + broken_main + broken_extra
    return {
        "label": label,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": ("git cat-file -e <remote-default-tip>:<target> per "
                   "repository (exact committed tree at the remote default "
                   "ref); " + REF_SOURCE),
        "expected_repositories": list(repos),
        "expected_extra_repositories": list(extra_repos),
        "subset_of_full_inventory": sorted(repos) != sorted(REPOS)
                                    or sorted(extra_repos) != sorted(EXTRA_REPOS),
        "readmes_checked": len(rows),
        "relative_links_checked_total": sum(
            r.get("relative_links_checked", 0) for r in rows),
        "external_links_total": sum(
            r.get("external_links_count", 0) for r in rows),
        "broken_relative_total": broken_main,
        "extra_broken_relative_total": broken_extra,
        "error_rows_total": len(error_rows),
        "missing_rows": missing,
        "missing_rows_total": len(missing),
        "failure_total": failure_total,
        "repositories": rows,
        "extra_non_wpb_repositories": extra_rows,
    }


def main_with(argv, git=default_git_runner, declared=None):
    if len(argv) < 2:
        print("usage: README_LINK_RESOLUTION_CHECKER OUT.json LABEL [repo ...]",
              file=sys.stderr)
        return 2
    out_path, label, subset = argv[0], argv[1], argv[2:]
    if declared is None:
        declared = load_declared_default_branches()
    if subset:
        repos, extra_repos = list(subset), []
    else:
        repos, extra_repos = list(REPOS), list(EXTRA_REPOS)
    rows = [check_repo(r, declared.get(r), git) for r in repos]
    extra_rows = [check_repo(r, declared.get(r), git) for r in extra_repos]
    result = build_result(label, repos, extra_repos, rows, extra_rows)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=False)
        f.write("\n")
    print(json.dumps({
        "label": label,
        "repositories_covered": repos + extra_repos,
        "subset_of_full_inventory": result["subset_of_full_inventory"],
        "readmes_checked": result["readmes_checked"],
        "relative_links_checked_total": result["relative_links_checked_total"],
        "external_links_total": result["external_links_total"],
        "broken_relative_total": result["broken_relative_total"],
        "extra_broken_relative_total": result["extra_broken_relative_total"],
        "error_rows_total": result["error_rows_total"],
        "missing_rows_total": result["missing_rows_total"],
        "failure_total": result["failure_total"],
        "errors": [
            {"repository": r["repository"], "error": r["error"]}
            for r in rows + extra_rows if "error" in r
        ],
        "broken": [
            {"repository": r["repository"], "broken": r["broken_relative"]}
            for r in rows + extra_rows if r.get("broken_relative_count")
        ],
    }, indent=1))
    return 0 if result["failure_total"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main_with(sys.argv[1:]))
