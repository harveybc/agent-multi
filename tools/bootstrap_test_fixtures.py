#!/usr/bin/env python3
"""Test-fixture bootstrap for clean checkouts (audit 2026-08-08, req 6).

The suite has exactly two non-hermetic dependencies; this tool makes
them explicit, verifiable and reproducible instead of hiding them
behind a workstation-local pass count:

1. The pinned decision base contract
   ``examples/results/project3_ethusdt_4h_sac_train_val_test_v2/
   config_out.json`` — tracked in Git (whitelisted over the generated
   results ignore rule) precisely because its sha is a decision-bearing
   pin; this tool verifies presence and hash.
2. The sibling ``../doin-node`` checkout whose campaign template
   directories several materializer tests read. When absent, this tool
   clones it from origin; it always reports the revision in use.

Typed outcomes (stdout JSON):
  FIXTURES_READY        exit 0 — suite is runnable from this checkout
  FIXTURES_INCOMPLETE   exit 3 — something is missing; each item typed
  BOOTSTRAP_FAILED      exit 2 — an acquisition step itself failed

``--check-only`` never mutates anything (CI mode).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOIN_NODE_ORIGIN = "https://github.com/harveybc/doin-node.git"
# Exact full revision the suite's fixtures are proven against
# (order §6/WP5). A checkout at any other revision is reported as a
# mismatch — never silently used, never mutated in place.
DOIN_NODE_PIN = "5bd6d3966df37e98e0de6fb904d0ec81566866a6"
DOIN_TEMPLATE_DIRS = (
    "examples/trading/phase_1_asset_policy_usdcad_4h_protected_easy_v2",
    "examples/trading/phase_1_asset_policy_usdcad_4h_full_genome_v1",
    "examples/trading/phase_2_eth_anchored_smoke_v1",
)


def check_base_contract() -> dict:
    from tools import eth_curriculum_decision_experiment as d1

    path = Path(d1.ETH_BASE)
    item = {"fixture": "pinned_base_contract", "path": str(path)}
    if not path.is_file():
        item["status"] = "missing"
        item["remedy"] = ("file is tracked in Git as of this commit; a "
                          "checkout without it predates the whitelist — "
                          "update the checkout")
        return item
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    item["sha256"] = actual
    item["status"] = ("ready" if actual == d1.ETH_BASE_SHA256
                      else "hash_mismatch")
    if item["status"] == "hash_mismatch":
        item["expected_sha256"] = d1.ETH_BASE_SHA256
    return item


def check_doin_node(*, mutate: bool) -> dict:
    doin_root = ROOT.parent / "doin-node"
    item: dict = {"fixture": "sibling_doin_node", "path": str(doin_root)}
    if not doin_root.is_dir():
        if not mutate:
            item["status"] = "missing"
            item["remedy"] = (f"git clone {DOIN_NODE_ORIGIN} {doin_root} "
                             "(or run this tool without --check-only)")
            return item
        clone = subprocess.run(
            ["git", "clone", "--quiet", DOIN_NODE_ORIGIN, str(doin_root)],
            capture_output=True, text=True)
        if clone.returncode != 0:
            item["status"] = "clone_failed"
            item["stderr"] = clone.stderr.strip()[-400:]
            return item
        item["cloned_from"] = DOIN_NODE_ORIGIN
        pin = subprocess.run(
            ["git", "-C", str(doin_root), "checkout", "--quiet",
             "--detach", DOIN_NODE_PIN],
            capture_output=True, text=True)
        if pin.returncode != 0:
            item["status"] = "clone_failed"
            item["stderr"] = ("pin checkout failed: "
                             + pin.stderr.strip()[-300:])
            return item
    rev = subprocess.run(["git", "-C", str(doin_root), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    item["revision"] = rev.stdout.strip() or "unavailable"
    item["pinned_revision"] = DOIN_NODE_PIN
    if item["revision"] != DOIN_NODE_PIN:
        item["status"] = "revision_mismatch"
        item["remedy"] = (f"sibling checkout is at {item['revision']}, "
                          f"suite fixtures are pinned to {DOIN_NODE_PIN}; "
                          "update the sibling explicitly (this tool never "
                          "mutates an existing checkout)")
        return item
    missing = []
    for rel in DOIN_TEMPLATE_DIRS:
        tdir = doin_root / rel
        if not tdir.is_dir() or not any(tdir.glob("*_node.json")):
            missing.append(rel)
    if missing:
        item["status"] = "templates_missing"
        item["missing_template_dirs"] = missing
        item["remedy"] = ("sibling checkout lacks required campaign "
                          "templates; update it to a revision that "
                          "carries them")
    else:
        item["status"] = "ready"
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="verify only; never clone or write")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))

    items = [check_base_contract(),
             check_doin_node(mutate=not args.check_only)]
    failed = [i for i in items if i["status"] in ("clone_failed",)]
    incomplete = [i for i in items if i["status"] not in ("ready",)]
    if failed:
        outcome = "BOOTSTRAP_FAILED"
        code = 2
    elif incomplete:
        outcome = "FIXTURES_INCOMPLETE"
        code = 3
    else:
        outcome = "FIXTURES_READY"
        code = 0
    print(json.dumps({"schema": "agent_multi.test_fixture_bootstrap.v1",
                      "outcome": outcome, "fixtures": items}, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
