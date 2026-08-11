#!/usr/bin/env python3
"""Finding 217 link checker.

Resolves EVERY relative link found in the delivered READMEs through
`git cat-file -e <HEAD-commit>:<target>` against the exact committed tree
of each repository's checked-out default branch. External links
(http/https/mailto) are listed separately and are NOT resolved here;
syntactic parsing is not equated with reachability.
"""
import json
import re
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone

BASE = "/home/harveybc/Documents/GitHub/"
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
# causal-inference is resolved against origin/master because its canonical
# dirty checkout must remain untouched (its local HEAD is behind).
EXTRA_REPOS = [("causal-inference", "origin/master")]

LINK_RE = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def git(repo, *args):
    return subprocess.run(["git", "-C", BASE + repo, *args],
                          capture_output=True, text=True)


def check_repo(repo, rev="HEAD"):
    head = git(repo, "rev-parse", rev).stdout.strip()
    branch = git(repo, "rev-parse", "--abbrev-ref", rev).stdout.strip()
    blob = git(repo, "cat-file", "-p", f"{head}:README.md")
    if blob.returncode != 0:
        return {"repository": repo, "head": head, "error": "README.md not in HEAD tree"}
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
        r = git(repo, "cat-file", "-e", f"{head}:{target}")
        if r.returncode != 0:
            broken.append({"link": link, "target": target})
    return {
        "repository": repo,
        "branch": branch,
        "head": head,
        "relative_links_checked": len(relative),
        "external_links": sorted(set(external)),
        "external_links_count": len(external),
        "broken_relative": broken,
        "broken_relative_count": len(broken),
    }


def main(out_path, label):
    rows = [check_repo(r) for r in REPOS]
    extra = [check_repo(r, rev) for r, rev in EXTRA_REPOS]
    result = {
        "label": label,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "git cat-file -e <HEAD>:<target> per repository (exact committed tree)",
        "readmes_checked": len(rows),
        "relative_links_checked_total": sum(r.get("relative_links_checked", 0) for r in rows),
        "external_links_total": sum(r.get("external_links_count", 0) for r in rows),
        "broken_relative_total": sum(r.get("broken_relative_count", 0) for r in rows),
        "repositories": rows,
        "extra_non_wpb_repositories": extra,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=False)
        f.write("\n")
    print(json.dumps({
        "label": label,
        "readmes_checked": result["readmes_checked"],
        "relative_links_checked_total": result["relative_links_checked_total"],
        "external_links_total": result["external_links_total"],
        "broken_relative_total": result["broken_relative_total"],
        "broken": [
            {"repository": r["repository"], "broken": r["broken_relative"]}
            for r in rows if r.get("broken_relative_count")
        ],
        "extra_broken": [
            {"repository": r["repository"], "broken": r.get("broken_relative", [])}
            for r in extra if r.get("broken_relative_count")
        ],
    }, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
