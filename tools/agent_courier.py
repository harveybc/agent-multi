#!/usr/bin/env python3
"""Agent courier — the barracks mail (owner order 2026-08-20).

The generals already speak through committed documents under
``docs/handoffs/`` whose filenames carry the addressing convention
(``MUSASHI_TO_GENERAL_SATOSHI_*.md``, ``SATOSHI_TO_SERGEANT_RETSU_*``…).
The owner has been carrying those documents by hand. This daemon is the
missing transport, and ONLY the transport:

- it polls ``git fetch`` and scans the fetched branches' handoff trees
  for documents addressed to the LOCAL general;
- a new document (deduplicated by blob sha, idempotent across restarts)
  is extracted to the local inbox and INJECTED into the local general
  through its own subscription CLI in headless mode (``claude -p`` here;
  ``codex exec`` / ``grok`` on the other machines — configurable);
- the general's reply is what it always was: a committed, pushed
  document. The counterpart courier delivers it. The git audit trail
  is therefore PRESERVED AND STRENGTHENED — nothing moves outside it.

What this deliberately does NOT do:
- no API keys, no third-party orchestration platform, no provider
  bridging: each general keeps its native harness and subscription;
- no permission bypass: the headless invocation inherits the local
  CLI's configured permission settings unchanged;
- no authority: delivery is transport. Owner-boundary approvals
  (capital, GPU campaigns, promotion signers) remain human by doctrine.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "agent_multi.agent_courier.v1"

#: filename-addressing: TO_<name> wins; aliases map to one identity.
IDENTITY_ALIASES = {
    "satoshi": ("SATOSHI",),
    "musashi": ("MUSASHI",),
    "retsu": ("RETSU",),
}
HANDOFF_DIRS = ("docs/handoffs", "docs/audits")
DOC_RE = re.compile(r"_TO_(?:GENERAL_|SERGEANT_)?([A-Z]+)")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout


def addressed_to(filename: str, identity: str) -> bool:
    matches = DOC_RE.findall(filename.upper())
    aliases = IDENTITY_ALIASES.get(identity, (identity.upper(),))
    return any(m in aliases for m in matches)


def scan_repo(repo: Path, identity: str, branch_prefixes) -> list[dict]:
    """Every handoff blob addressed to us on any fetched branch."""
    git(repo, "fetch", "--quiet", "origin")
    found = []
    refs = git(repo, "for-each-ref", "--format=%(refname)",
               "refs/remotes/origin/").split()
    for ref in refs:
        short = ref.replace("refs/remotes/origin/", "")
        if branch_prefixes and not any(
                short.startswith(p) for p in branch_prefixes):
            continue
        for base in HANDOFF_DIRS:
            try:
                tree = git(repo, "ls-tree", "-r", ref, "--", base)
            except RuntimeError:
                continue
            for line in tree.splitlines():
                meta, path = line.split("\t", 1)
                blob = meta.split()[2]
                name = Path(path).name
                if name.endswith(".md") and addressed_to(name, identity):
                    found.append({"repo": str(repo), "ref": short,
                                  "path": path, "blob": blob,
                                  "name": name})
    return found


def deliver(doc: dict, *, repo: Path, inbox: Path,
            command_template: list[str], dry_run: bool) -> dict:
    content = git(repo, "cat-file", "-p", doc["blob"])
    inbox.mkdir(parents=True, exist_ok=True)
    local = inbox / f"{doc['blob'][:12]}_{doc['name']}"
    local.write_text(content)
    prompt = (f"Nuevo documento entregado por el correo del cuartel.\n"
              f"Origen: {doc['repo']} rama {doc['ref']}\n"
              f"Ruta: {doc['path']}\n"
              f"Copia local: {local}\n\n"
              f"Léelo completo y actúa según las órdenes vigentes; "
              f"responde, como siempre, con un documento comprometido "
              f"y empujado.")
    command = [arg.replace("{prompt}", prompt)
               for arg in command_template]
    record = {"delivered_utc": utc(), "doc": doc,
              "local_copy": str(local), "command": command[0],
              "dry_run": dry_run}
    if dry_run:
        record["result"] = "DRY_RUN"
        return record
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=3600)
    record["result"] = ("DELIVERED" if result.returncode == 0
                        else f"DELIVERY_FAILED:{result.returncode}")
    record["stdout_tail"] = result.stdout[-500:]
    return record


def load_state(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "seen_blobs": [], "log": []}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--identity", required=True,
                        choices=sorted(IDENTITY_ALIASES))
    parser.add_argument("--repo", type=Path, action="append",
                        required=True)
    parser.add_argument("--branch-prefix", action="append", default=[])
    parser.add_argument("--inbox", type=Path,
                        default=Path.home() / ".local/state/"
                        "agent-courier/inbox")
    parser.add_argument("--state", type=Path,
                        default=Path.home() / ".local/state/"
                        "agent-courier/state.json")
    parser.add_argument("--deliver-cmd", nargs="+",
                        default=["claude", "-p", "{prompt}"],
                        help="headless CLI of the LOCAL general; "
                             "{prompt} is replaced. codex/grok on "
                             "their machines")
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    state = load_state(args.state)
    seen = set(state["seen_blobs"])
    while True:
        for repo in args.repo:
            try:
                docs = scan_repo(repo, args.identity,
                                 args.branch_prefix)
            except RuntimeError as error:
                print(json.dumps({"at": utc(),
                                  "error": str(error)[:200]}))
                continue
            for doc in docs:
                if doc["blob"] in seen:
                    continue
                record = deliver(doc, repo=repo, inbox=args.inbox,
                                 command_template=args.deliver_cmd,
                                 dry_run=args.dry_run)
                print(json.dumps(record))
                seen.add(doc["blob"])
                state["seen_blobs"] = sorted(seen)
                state["log"] = (state["log"] + [record])[-200:]
                args.state.parent.mkdir(parents=True, exist_ok=True)
                args.state.write_text(json.dumps(state, indent=1))
        if args.once:
            return 0
        time.sleep(max(30, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
