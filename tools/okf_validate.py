#!/usr/bin/env python3
"""Deterministic validator for the project-owned OKF v0.2 bundle (K1).

Scope (doc 31 K1, Musashi order 2026-08-03): validate frontmatter schema
and provenance, freshness semantics, duplicate identities, contradictions,
prohibited secret/account patterns and missing canonical sources; produce
and verify a reproducible hash manifest.

Design constraints:

- zero external dependencies (no YAML library): the frontmatter dialect is
  deliberately flat and strict — scalars and dash-lists only; anything the
  parser cannot prove is a refusal, never a guess;
- deterministic output: errors are sorted, the manifest is byte-stable and
  `--as-of` pins the freshness clock for reproducible runs;
- Git remains canonical: this tool never rewrites a concept, it only
  validates and hashes.

Exit code 0 means the bundle is conformant; 1 means violations (listed one
per line, sorted); 2 means usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

BUNDLE_SCHEMA = "agent_multi.okf_bundle.v1"
MANIFEST_NAME = "MANIFEST.sha256"

REQUIRED_KEYS = (
    "type", "id", "title", "status", "producer", "verified_by",
    "created", "updated", "review_by", "canonical_for", "supersedes",
    "sources", "tags",
)
LIST_KEYS = ("sources", "tags")
STATUS_VALUES = ("verified", "draft")
TYPE_VALUES = ("concept",)

# Prohibited content: credentials, tokens, broker account identifiers,
# private keys. A hit anywhere in a concept file is a refusal.
PROHIBITED_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passphrase)\s*[:=]\s*\S+"),
    re.compile(r"\bDU[0-9]{5,}\b"),
    re.compile(r"\b\d{8,}:[A-Za-z0-9_\-]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
# Sources must be reviewed canonical documents, never state or bulk data.
FORBIDDEN_SOURCE_SUFFIXES = (
    ".sqlite", ".db", ".log", ".bin", ".pt", ".onnx", ".zip", ".json",
)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _parse_date(value: str, field: str, errors: list[str], concept: str):
    if not _DATE_RE.match(value):
        errors.append(f"{concept}: {field} is not an ISO date: {value!r}")
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        errors.append(f"{concept}: {field} is not a valid date: {value!r}")
        return None


def parse_frontmatter(text: str, concept: str, errors: list[str]):
    """Strict flat frontmatter: `key: scalar` or `key:` + `  - item` lines."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        errors.append(f"{concept}: missing frontmatter opening '---'")
        return None, ""
    data: dict[str, object] = {}
    current_list: str | None = None
    end_index = None
    for index, raw in enumerate(lines[1:], start=1):
        if raw.strip() == "---":
            end_index = index
            break
        if not raw.strip():
            errors.append(f"{concept}: blank line inside frontmatter")
            return None, ""
        if raw.startswith("  - "):
            if current_list is None:
                errors.append(f"{concept}: list item outside a list key: {raw!r}")
                return None, ""
            data[current_list].append(raw[4:].strip())  # type: ignore[union-attr]
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):(.*)$", raw)
        if not match:
            errors.append(f"{concept}: unparseable frontmatter line: {raw!r}")
            return None, ""
        key, rest = match.group(1), match.group(2).strip()
        if key in data:
            errors.append(f"{concept}: duplicate frontmatter key {key!r}")
            return None, ""
        if key in LIST_KEYS:
            if rest:
                errors.append(f"{concept}: {key} must be a dash-list")
                return None, ""
            data[key] = []
            current_list = key
        else:
            if not rest:
                errors.append(f"{concept}: empty value for {key!r}")
                return None, ""
            data[key] = rest
            current_list = None
    if end_index is None:
        errors.append(f"{concept}: missing frontmatter closing '---'")
        return None, ""
    body = "\n".join(lines[end_index + 1:]).strip()
    return data, body


def validate_concept(path: Path, repo_root: Path, as_of: date, errors: list[str]):
    concept = path.name
    text = path.read_text(encoding="utf-8")
    for pattern in PROHIBITED_PATTERNS:
        if pattern.search(text):
            errors.append(f"{concept}: prohibited secret/account pattern present")
    data, body = parse_frontmatter(text, concept, errors)
    if data is None:
        return None
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        errors.append(f"{concept}: missing keys {missing}")
        return None
    unknown = sorted(set(data) - set(REQUIRED_KEYS))
    if unknown:
        errors.append(f"{concept}: unknown keys {unknown}")
    if data["type"] not in TYPE_VALUES:
        errors.append(f"{concept}: type must be one of {TYPE_VALUES}")
    if data["status"] not in STATUS_VALUES:
        errors.append(f"{concept}: status must be one of {STATUS_VALUES}")
    identifier = str(data["id"])
    if not _ID_RE.match(identifier):
        errors.append(f"{concept}: id {identifier!r} is not kebab-case")
    if identifier != path.stem:
        errors.append(f"{concept}: id {identifier!r} != filename stem {path.stem!r}")
    created = _parse_date(str(data["created"]), "created", errors, concept)
    updated = _parse_date(str(data["updated"]), "updated", errors, concept)
    review_by = _parse_date(str(data["review_by"]), "review_by", errors, concept)
    if created and updated and updated < created:
        errors.append(f"{concept}: updated precedes created")
    if updated and review_by and review_by <= updated:
        errors.append(f"{concept}: review_by must be after updated")
    if review_by and review_by < as_of:
        errors.append(
            f"{concept}: STALE — review_by {review_by} is before as-of {as_of}"
        )
    sources = data.get("sources") or []
    if not isinstance(sources, list) or not sources:
        errors.append(f"{concept}: sources must be a non-empty list")
        sources = []
    for source in sources:
        source_path = repo_root / str(source)
        if str(source).startswith(("/", "~")) or ".." in str(source):
            errors.append(f"{concept}: source escapes the repository: {source}")
        elif not source_path.is_file():
            errors.append(f"{concept}: missing source {source}")
        elif source_path.suffix.lower() in FORBIDDEN_SOURCE_SUFFIXES:
            errors.append(f"{concept}: forbidden source class {source}")
    if not body:
        errors.append(f"{concept}: empty body")
    return data


def validate_bundle(bundle_dir: Path, repo_root: Path, as_of: date):
    errors: list[str] = []
    concepts: dict[str, dict] = {}
    files = sorted(
        p for p in bundle_dir.glob("*.md") if p.name not in ("README.md",)
    )
    if not files:
        errors.append("bundle: no concept files found")
    for path in files:
        data = validate_concept(path, repo_root, as_of, errors)
        if data is None:
            continue
        identifier = str(data["id"])
        if identifier in concepts:
            errors.append(f"bundle: duplicate id {identifier!r}")
        else:
            concepts[identifier] = data
    superseded = set()
    for identifier, data in sorted(concepts.items()):
        target = str(data["supersedes"])
        if target != "none":
            if target not in concepts:
                errors.append(
                    f"{identifier}: supersedes unknown concept {target!r}"
                )
            else:
                superseded.add(target)
    canonical_owners: dict[str, str] = {}
    for identifier, data in sorted(concepts.items()):
        if identifier in superseded:
            continue
        topic = str(data["canonical_for"])
        if topic in canonical_owners:
            errors.append(
                f"bundle: CONTRADICTION — {canonical_owners[topic]!r} and "
                f"{identifier!r} both claim canonical_for {topic!r} without "
                "supersession"
            )
        else:
            canonical_owners[topic] = identifier
    return sorted(errors), files


def compute_manifest(bundle_dir: Path) -> str:
    lines = [f"schema: {BUNDLE_SCHEMA}"]
    entries = []
    for path in sorted(bundle_dir.glob("*.md")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.name}")
    lines.extend(entries)
    bundle_digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    lines.append(f"bundle_sha256: {bundle_digest}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="knowledge/okf")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--as-of", default=None,
                        help="ISO date pinning the freshness clock")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--check-manifest", action="store_true")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    bundle_dir = (repo_root / args.bundle).resolve()
    if not bundle_dir.is_dir():
        print(f"bundle directory not found: {bundle_dir}", file=sys.stderr)
        return 2
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    errors, files = validate_bundle(bundle_dir, repo_root, as_of)
    for line in errors:
        print(f"VIOLATION {line}")
    if errors:
        print(f"result: {len(errors)} violation(s) in {len(files)} file(s)")
        return 1
    manifest = compute_manifest(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_NAME
    if args.write_manifest:
        manifest_path.write_text(manifest, encoding="utf-8")
        print(f"manifest written: {manifest_path}")
    if args.check_manifest:
        if not manifest_path.is_file():
            print("VIOLATION bundle: manifest missing")
            return 1
        if manifest_path.read_text(encoding="utf-8") != manifest:
            print("VIOLATION bundle: manifest does not match current content")
            return 1
        print("manifest verified")
    print(f"result: clean — {len(files)} concept(s), as-of {as_of}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
