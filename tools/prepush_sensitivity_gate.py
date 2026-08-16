#!/usr/bin/env python3
"""Pre-push sensitivity gate (finding AUD-SEC-20260810-215).

Scans the OUTGOING commits of a push — the exact ranges git is about to
publish — for sensitivity classes before they can reach any remote:

- ``credentials``: private-key blocks, API keys/secrets/passwords/tokens
  with values, bearer values, AWS/GitHub/Slack token shapes;
- ``account_identifier``: broker account codes (IBKR ``DU…``/``U…``),
  numeric account/login assignments, Alpaca key IDs;
- ``signer_key_path``: ssh/gnupg key material paths, signer/keystore/
  wallet files, ``id_rsa``-family names;
- ``third_party_content``: social-packet content markers (third-party
  text never leaves private operator storage; scoped to ``docs/``);
- ``sl_tp_live_levels``: live stop-loss/take-profit protection levels in
  evidence documents (scoped to ``docs/``);
- ``topology``: IP addresses, hostname assignments and GPU UUIDs beyond
  the explicit allowlist.

Findings are TYPED — class, rule, commit, path, line — and the matched
span is REDACTED in every excerpt: the gate never republishes the value
it found. Any finding (or a truncated scan) exits nonzero and blocks the
push; a clean scan exits 0.

Install the hook (template in ``tools/hooks/pre-push``)::

    cp tools/hooks/pre-push .git/hooks/pre-push
    chmod +x .git/hooks/pre-push

The allowlist lives at ``tools/prepush_sensitivity_allowlist.json`` in
the scanned repository. ``allowed_path_globs`` maps a sensitivity class
(or ``"*"`` for all classes) to fnmatch globs of exempt paths; it exists
for narrowly named fixture/config files and must stay minimal.

Limitations, stated plainly: the gate scans ADDED text lines of each
outgoing commit. Binary payloads, already-pushed history and encrypted
blobs are out of scope — history repair is governed by the owner-gated
plan in ``docs/audits/evidence/HISTORY_SCRUB_PLAN_2026_08_10.md``, and
``git push --no-verify`` bypasses any hook, which the standing evidence
contract forbids for sensitive content.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA = "agent_multi.prepush_sensitivity_gate.v1"
ALLOWLIST_BASENAME = "prepush_sensitivity_allowlist.json"
ZERO_SHA = re.compile(r"^0+$")

# Values a hostname-assignment capture may take without naming topology.
_HOSTNAME_STOPWORDS = {
    "localhost", "none", "null", "true", "false", "unknown", "host",
    "hostname", "hosts", "str", "self", "value", "name", "example",
    "test", "optional",
}

DEFAULT_ALLOWLIST: dict[str, Any] = {
    "allowed_hostnames": ["localhost", "github.com"],
    "allowed_ips": ["127.0.0.1", "0.0.0.0"],
    "allowed_path_globs": {},
}


class Rule:
    """One typed detection rule. ``path_globs`` (if set) restricts the
    rule to matching paths; ``check`` (if set) receives the regex match
    and the allowlist and returns True when the match is a finding."""

    def __init__(self, sensitivity_class: str, name: str, pattern: str,
                 *, path_globs: Optional[list[str]] = None,
                 check: Optional[Any] = None):
        self.sensitivity_class = sensitivity_class
        self.name = name
        self.regex = re.compile(pattern)
        self.path_globs = path_globs
        self.check = check

    def applies_to(self, path: str) -> bool:
        if not self.path_globs:
            return True
        return any(fnmatch.fnmatch(path, glob) for glob in self.path_globs)


def _hostname_not_allowed(match: re.Match, allowlist: dict) -> bool:
    value = next(
        (group for group in match.groups() if group is not None),
        match.group(0),
    ).lower()
    if value in _HOSTNAME_STOPWORDS:
        return False
    return value not in {h.lower() for h in
                         allowlist.get("allowed_hostnames", [])}


def _ip_not_allowed(match: re.Match, allowlist: dict) -> bool:
    return match.group(0) not in set(allowlist.get("allowed_ips", []))


RULES: list[Rule] = [
    # ── credentials ──
    Rule("credentials", "private_key_block",
         r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    Rule("credentials", "aws_access_key_id", r"\bAKIA[0-9A-Z]{16}\b"),
    Rule("credentials", "github_token", r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    Rule("credentials", "slack_token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    Rule("credentials", "secret_assignment",
         r"(?i)\b(?:api[_-]?key|api[_-]?secret|client[_-]?secret|password|"
         r"passwd|auth[_-]?token|access[_-]?token|secret[_-]?key|"
         r"signing[_-]?key)\b[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9+/_=.\-]{8,}"),
    Rule("credentials", "bearer_value",
         r"(?i)\bbearer\s+[a-z0-9_\-.=+/]{16,}"),
    # ── account identifiers ──
    Rule("account_identifier", "ibkr_account_code",
         r"\b(?:DU|DF)\d{6,8}\b|\bU\d{7,8}\b"),
    Rule("account_identifier", "alpaca_key_id", r"\bPK[A-Z0-9]{16,20}\b"),
    Rule("account_identifier", "numeric_account_assignment",
         r"(?i)\b(?:account[_-]?(?:id|number|login)|mt5[_-]?login|login)\b"
         r"[^\n=:]{0,6}[:=]{1,2}[\s\"']*\d{5,12}\b"),
    # ── signer / key material paths ──
    Rule("signer_key_path", "ssh_gnupg_path",
         r"(?:~|/home/[A-Za-z0-9_-]+|/root)?/\.(?:ssh|gnupg)/[^\s\"']+"),
    Rule("signer_key_path", "keystore_wallet_file",
         r"(?i)\b[\w./-]*(?:signer|keystore|wallet)[\w-]*"
         r"\.(?:json|pem|key|p12)\b"),
    Rule("signer_key_path", "ssh_identity_name",
         r"\bid_(?:rsa|ed25519|ecdsa|dsa)\b"),
    # ── third-party content markers (evidence documents) ──
    Rule("third_party_content", "social_content_marker",
         r"(?i)\b(?:third[_-]party[_-]content|author_handle|post_text|"
         r"post_body|quoted_post|social_post)\b",
         path_globs=["docs/*"]),
    # ── live SL/TP protection levels (evidence documents) ──
    Rule("sl_tp_live_levels", "protection_level_value",
         r"(?i)\b(?:stop_loss|take_profit|sl_price|tp_price|sl_level|"
         r"tp_level)\b[^\n]{0,6}[:=][^\n]{0,4}\d+(?:\.\d+)?",
         path_globs=["docs/*"]),
    # ── topology beyond allowlist ──
    Rule("topology", "ip_address",
         r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
         r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
         check=_ip_not_allowed),
    # Colon style accepts a complete quoted JSON value or a complete bare
    # YAML value. Equals style requires a quoted literal. Requiring the bare
    # value to end at a value delimiter prevents dynamic expressions such as
    # ``hostname: socket.gethostname()`` and prose such as ``host: the
    # marker`` from being misclassified as topology.
    #
    # NARROWED 2026-08-16 (false-positive class, detection unchanged): a
    # QUOTED key with an UNQUOTED value is a variable reference in code —
    # ``"hostname": local_hostname`` — never a literal, because a literal
    # in JSON or a dict is always quoted (``"hostname": "dragon"`` still
    # fires). The bare-key YAML form ``hostname: dragon`` is untouched.
    # This narrows the RULE, not the allowlist, which stays shrink-only.
    Rule("topology", "hostname_assignment",
         r"(?i)(?:"
         r"(?<![\"'])\b(?:hostname|host|ssh_host|machine)\b\s*"
         r"(?::\s*(?:[\"']([a-z][a-z0-9_.-]{1,40})[\"']|"
         r"([a-z][a-z0-9_.-]{1,40})(?=\s*(?:[,}\]#]|$)))|"
         r"=\s*[\"']([a-z][a-z0-9_.-]{1,40})[\"'])"
         r"|"
         r"[\"']\b(?:hostname|host|ssh_host|machine)\b[\"']\s*"
         r":\s*[\"']([a-z][a-z0-9_.-]{1,40})[\"']"
         r")",
         check=_hostname_not_allowed),
    Rule("topology", "gpu_uuid",
         r"\bGPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
         r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
]


def load_allowlist(repo: Path, explicit: Optional[Path]) -> dict[str, Any]:
    path = explicit or (repo / "tools" / ALLOWLIST_BASENAME)
    merged = {key: (dict(value) if isinstance(value, dict) else list(value))
              for key, value in DEFAULT_ALLOWLIST.items()}
    try:
        loaded = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return merged
    for key in ("allowed_hostnames", "allowed_ips"):
        values = loaded.get(key)
        if isinstance(values, list):
            merged[key] = sorted(set(merged[key]) | set(map(str, values)))
    globs = loaded.get("allowed_path_globs")
    if isinstance(globs, dict):
        merged["allowed_path_globs"] = {
            str(cls): [str(g) for g in v]
            for cls, v in globs.items() if isinstance(v, list)}
    return merged


def _path_exempt(path: str, sensitivity_class: str,
                 allowlist: dict[str, Any]) -> bool:
    globs = allowlist.get("allowed_path_globs") or {}
    for cls in (sensitivity_class, "*"):
        for glob in globs.get(cls, []):
            if fnmatch.fnmatch(path, glob):
                return True
    return False


def _redact_all(line: str, spans: list[tuple[int, int, str]],
                limit: int = 160) -> str:
    """The excerpt NEVER carries ANY matched value: every matched span on
    the line — of every rule and class — is replaced, so one finding's
    excerpt cannot republish a neighbouring finding's value."""
    merged: list[list[Any]] = []
    for start, end, sensitivity_class in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, sensitivity_class])
    result = line
    for start, end, sensitivity_class in reversed(merged):
        result = (result[:start] + f"[REDACTED:{sensitivity_class}]"
                  + result[end:])
    return result.strip()[:limit]


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


def outgoing_commits_from_refs(repo: Path, ref_lines: Iterable[str],
                               max_commits: int) -> tuple[list[str], bool]:
    """Parse pre-push stdin lines '<local_ref> <local_sha> <remote_ref>
    <remote_sha>' into the union of outgoing commits (newest first)."""
    commits: list[str] = []
    seen: set[str] = set()
    truncated = False
    for line in ref_lines:
        parts = line.split()
        if len(parts) != 4:
            continue
        _local_ref, local_sha, _remote_ref, remote_sha = parts
        if ZERO_SHA.match(local_sha):
            continue  # deletion pushes publish nothing new
        if ZERO_SHA.match(remote_sha):
            rev_args = ["rev-list", local_sha, "--not", "--remotes"]
        else:
            rev_args = ["rev-list", f"{remote_sha}..{local_sha}"]
        listed = _git(repo, *rev_args).split()
        if len(listed) > max_commits:
            listed = listed[:max_commits]
            truncated = True
        for sha in listed:
            if sha not in seen:
                seen.add(sha)
                commits.append(sha)
    return commits, truncated


def added_lines(repo: Path, commit: str) -> Iterable[tuple[str, int, str]]:
    """Yield (path, new_file_line_number, added_line) for one commit."""
    diff = _git(repo, "show", "--format=", "--unified=0", "--no-color",
                commit)
    path: Optional[str] = None
    line_no = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            path = None if target == "/dev/null" else \
                target[2:] if target.startswith("b/") else target
        elif raw.startswith("@@"):
            match = re.search(r"\+(\d+)", raw)
            line_no = int(match.group(1)) if match else 1
        elif raw.startswith("+") and not raw.startswith("+++"):
            if path is not None:
                yield path, line_no, raw[1:]
            line_no += 1


def scan_commits(repo: Path, commits: list[str],
                 allowlist: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for commit in commits:
        for path, line_no, line in added_lines(repo, commit):
            hits: list[tuple[Rule, tuple[int, int]]] = []
            for rule in RULES:
                if not rule.applies_to(path):
                    continue
                if _path_exempt(path, rule.sensitivity_class, allowlist):
                    continue
                for match in rule.regex.finditer(line):
                    if rule.check and not rule.check(match, allowlist):
                        continue
                    hits.append((rule, match.span()))
            if not hits:
                continue
            excerpt = _redact_all(
                line, [(start, end, rule.sensitivity_class)
                       for rule, (start, end) in hits])
            for rule, _span in hits:
                findings.append({
                    "class": rule.sensitivity_class,
                    "rule": rule.name,
                    "commit": commit,
                    "path": path,
                    "line": line_no,
                    "excerpt": excerpt,
                })
    return findings


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-push sensitivity gate (AUD-SEC-20260810-215)")
    parser.add_argument("--repo", type=Path, default=Path.cwd(),
                        help="repository to scan")
    parser.add_argument("--range", dest="range_expr", default=None,
                        help="explicit commit range A..B to scan")
    parser.add_argument("--stdin-refs", action="store_true",
                        help="read pre-push '<local_ref> <local_sha> "
                             "<remote_ref> <remote_sha>' lines from stdin")
    parser.add_argument("--allowlist", type=Path, default=None,
                        help=f"allowlist path (default: "
                             f"<repo>/tools/{ALLOWLIST_BASENAME})")
    parser.add_argument("--max-commits", type=int, default=200,
                        help="commit cap per pushed ref; exceeding it "
                             "fails closed as a truncated scan")
    parser.add_argument("--json", action="store_true",
                        help="emit the full typed report as JSON")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    allowlist = load_allowlist(repo, args.allowlist)
    truncated = False
    try:
        if args.range_expr:
            commits = _git(repo, "rev-list", args.range_expr).split()
            if len(commits) > args.max_commits:
                commits, truncated = commits[:args.max_commits], True
        elif args.stdin_refs:
            commits, truncated = outgoing_commits_from_refs(
                repo, sys.stdin, args.max_commits)
        else:
            parser.error("one of --range or --stdin-refs is required")
        findings = scan_commits(repo, commits, allowlist)
    except RuntimeError as exc:
        print(f"prepush-sensitivity-gate: git error: {exc}",
              file=sys.stderr)
        return 2

    blocked = bool(findings) or truncated
    report = {
        "schema": SCHEMA,
        "outcome": "BLOCK" if blocked else "CLEAN",
        "scanned_commits": commits,
        "truncated_scan": truncated,
        "findings": findings,
    }
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        for finding in findings:
            print(f"BLOCK [{finding['class']}/{finding['rule']}] "
                  f"{finding['commit'][:12]} {finding['path']}:"
                  f"{finding['line']} {finding['excerpt']}")
        if truncated:
            print("BLOCK [gate/truncated_scan] outgoing range exceeded "
                  f"--max-commits={args.max_commits}; refusing an "
                  "unscanned push")
        if not blocked:
            print(f"prepush-sensitivity-gate: CLEAN "
                  f"({len(commits)} outgoing commit(s) scanned)")
    if blocked:
        print("prepush-sensitivity-gate: push BLOCKED — remove the "
              "sensitive content or narrow the allowlist deliberately "
              "(AUD-SEC-20260810-215)", file=sys.stderr)
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
