#!/usr/bin/env python3
"""Engineering surface index — first-cycle pilot (agent-multi only).

Implements P1 of MUSASHI_DISPOSITION_SATOSHI_III_DETERMINISTIC_TOOLING
_2026_08_06 with the T-1/T-5 redesign: machine-DISCOVERED facts are kept
strictly separate from human-REVIEWED semantic declarations, and unknown
semantics stay UNCLASSIFIED — they are never guessed from prose.

Discovered (AST and metadata only — no tool is imported or executed):
  per tool file: path, sha256, main-guard/main-function presence, argparse
  argument names, top-level imports; per entry point: source-declared
  group/name/target from setup.py, installed group/name/target from the
  named environment's package metadata, and source-vs-installed drift.

Declared (reviewed, from tools/TOOL_DECLARATIONS.json): purpose,
lifecycle, mutability, authority class, owner, replacement, plugin
protocol/example/config-key annotations.

The generated JSON is OUTPUT, never a source of truth. The declaration
file is the semantic source. `importlib.metadata.entry_points()` reads
installed metadata without importing any plugin module.

Typed outcomes: OK | BLOCK (structural contradiction such as duplicate
ids or an invalid import target) | UNAVAILABLE (a required fact could
not be established, e.g. the named environment is missing).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trading_contracts import content_hash

GENERATOR_VERSION = "engineering_surface_index.v1"
REPO = Path(__file__).resolve().parent.parent

LIFECYCLES = {
    "supported",
    "campaign_frozen",
    "experimental",
    "historical",
    "deprecated",
}
MUTABILITIES = {"read_only", "mutating", "mixed"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- discovery


def discover_tool(path: Path) -> dict:
    """AST-only facts about one tool file. The file is never imported."""
    source = path.read_text(encoding="utf-8")
    facts: dict = {
        "path": str(path.relative_to(REPO)),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "parse_error": None,
        "has_main_guard": False,
        "defines_main": False,
        "argparse_arguments": [],
        "imports": [],
    }
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        facts["parse_error"] = f"{exc.__class__.__name__}: {exc}"
        return facts

    arguments: set[str] = set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                facts["has_main_guard"] = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "main":
                facts["defines_main"] = True
        elif isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        arguments.add(arg.value)
                        break
    facts["argparse_arguments"] = sorted(arguments)
    facts["imports"] = sorted(imports)
    return facts


def discover_source_entry_points(setup_py: Path) -> dict[str, dict[str, str]]:
    """Entry points as declared in setup.py, via AST literal extraction."""
    tree = ast.parse(setup_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "entry_points":
            declared = ast.literal_eval(node.value)
            result: dict[str, dict[str, str]] = {}
            for group, entries in declared.items():
                result[group] = {}
                for entry in entries:
                    name, _, target = entry.partition("=")
                    result[group][name.strip()] = target.strip()
            return result
    raise ValueError(f"no entry_points keyword found in {setup_py}")


def discover_installed_entry_points(
    python_executable: str, groups: list[str]
) -> dict:
    """Installed metadata from the NAMED environment. Loads metadata only;
    no plugin module is imported."""
    probe = (
        "import json, sys\n"
        "from importlib.metadata import entry_points\n"
        "groups = json.loads(sys.argv[1])\n"
        "out = {}\n"
        "for group in groups:\n"
        "    out[group] = {e.name: e.value for e in entry_points().select(group=group)}\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [python_executable, "-c", probe, json.dumps(groups)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        return {
            "outcome": "UNAVAILABLE",
            "reason": f"metadata probe failed in {python_executable}: "
            f"{proc.stderr.strip()[:400]}",
            "groups": None,
        }
    return {
        "outcome": "OK",
        "reason": None,
        "groups": json.loads(proc.stdout),
    }


def entry_point_drift(
    source: dict[str, dict[str, str]], installed: dict[str, dict[str, str]]
) -> dict:
    """Source-declared vs installed-metadata comparison.

    drift: a source-declared entry whose installed target differs or is
    absent — a real install/source disagreement for THIS package.
    installed_only: entries other packages contribute to a group this
    package declares. For plugin groups that is a relevant shared-
    namespace fact; for console_scripts every environment package
    contributes, so foreign commands are not reported at all."""
    drift = []
    installed_only = []
    for group in sorted(source):
        src = source[group]
        inst = installed.get(group, {})
        for name in sorted(src):
            if src[name] != inst.get(name):
                drift.append(
                    {
                        "group": group,
                        "name": name,
                        "source_target": src[name],
                        "installed_target": inst.get(name),
                    }
                )
        if group != "console_scripts":
            for name in sorted(set(inst) - set(src)):
                installed_only.append(
                    {"group": group, "name": name, "installed_target": inst[name]}
                )
    return {"drift": drift, "installed_only": installed_only}


def import_target_exists(target: str) -> bool:
    """True when 'pkg.module:attr' resolves to a file in this repo."""
    module = target.partition(":")[0]
    candidate = REPO / (module.replace(".", "/") + ".py")
    package = REPO / module.replace(".", "/") / "__init__.py"
    return candidate.is_file() or package.is_file()


# ------------------------------------------------------------ verification


def structural_problems(
    tools: dict[str, dict],
    source_eps: dict[str, dict[str, str]],
    declarations: dict,
) -> list[str]:
    """BLOCK-class contradictions. Duplicate ids cannot arise inside one
    parsed dict, so duplicates are checked across groups sharing a
    namespace and between declared/actual files."""
    problems = []
    seen_commands: dict[str, str] = {}
    for group, entries in source_eps.items():
        for name, target in entries.items():
            if group == "console_scripts" and name in seen_commands:
                problems.append(f"duplicate console command {name!r}")
            seen_commands.setdefault(name, group)
            if not import_target_exists(target):
                problems.append(
                    f"entry point {group}/{name} has an invalid import "
                    f"target: {target!r}"
                )
    for filename, decl in declarations.get("tools", {}).items():
        lifecycle = decl.get("lifecycle")
        if lifecycle not in LIFECYCLES:
            problems.append(
                f"declaration for {filename!r} has invalid lifecycle "
                f"{lifecycle!r}"
            )
        if decl.get("mutability") not in MUTABILITIES:
            problems.append(
                f"declaration for {filename!r} has invalid mutability "
                f"{decl.get('mutability')!r}"
            )
        if lifecycle == "supported" and filename not in tools:
            problems.append(
                f"supported declaration {filename!r} references a file "
                "that no longer exists (stale supported entry)"
            )
    return problems


def unclassified_new_executables(
    tools: dict[str, dict], declarations: dict
) -> list[str]:
    """Executable tools that are neither declared nor grandfathered.
    CI fails on these; the grandfather baseline is versioned and only
    ever shrinks."""
    declared = set(declarations.get("tools", {}))
    baseline = set(declarations.get("known_unclassified_baseline", []))
    offenders = []
    for filename, facts in sorted(tools.items()):
        executable = facts["has_main_guard"] or facts["defines_main"]
        if executable and filename not in declared and filename not in baseline:
            offenders.append(filename)
    return offenders


# ------------------------------------------------------------------ build


def build_index(
    *,
    declarations_path: Path,
    runtime_python: str | None,
) -> dict:
    declarations = json.loads(declarations_path.read_text(encoding="utf-8"))
    tools = {
        path.name: discover_tool(path)
        for path in sorted((REPO / "tools").glob("*.py"))
    }
    source_eps = discover_source_entry_points(REPO / "setup.py")

    installed: dict = {"outcome": "UNAVAILABLE", "reason": "no runtime python given", "groups": None}
    if runtime_python:
        installed = discover_installed_entry_points(
            runtime_python, sorted(source_eps)
        )
    comparison = (
        entry_point_drift(source_eps, installed["groups"])
        if installed["outcome"] == "OK"
        else None
    )

    problems = structural_problems(tools, source_eps, declarations)
    unclassified = unclassified_new_executables(tools, declarations)

    merged_tools = {}
    for filename, facts in tools.items():
        decl = declarations.get("tools", {}).get(filename)
        merged_tools[filename] = {
            "discovered": facts,
            "declared": decl if decl else "UNCLASSIFIED",
        }

    plugins = {}
    for group, entries in source_eps.items():
        plugins[group] = {}
        for name, target in entries.items():
            annotation = (
                declarations.get("plugins", {}).get(f"{group}/{name}")
            )
            plugins[group][name] = {
                "discovered": {
                    "source_target": target,
                    "installed_target": (
                        (installed["groups"] or {}).get(group, {}).get(name)
                        if installed["outcome"] == "OK"
                        else None
                    ),
                    "import_target_exists": import_target_exists(target),
                },
                "declared": annotation if annotation else "UNCLASSIFIED",
            }

    if problems:
        outcome = "BLOCK"
    elif installed["outcome"] != "OK":
        outcome = "UNAVAILABLE"
    else:
        outcome = "OK"

    facts = {
        "repo": "agent-multi",
        "tools": merged_tools,
        "plugins": plugins,
        "entry_point_comparison": comparison,
        "structural_problems": problems,
        "unclassified_new_executables": unclassified,
        "installed_metadata": {
            "outcome": installed["outcome"],
            "reason": installed["reason"],
        },
        "declarations_sha256": sha256_file(declarations_path),
    }
    return {
        "schema": GENERATOR_VERSION,
        "typed_outcome": outcome,
        "facts": facts,
        "facts_hash": content_hash(facts),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--declarations",
        default=str(REPO / "tools/TOOL_DECLARATIONS.json"),
    )
    parser.add_argument(
        "--runtime-python",
        default=None,
        help="python executable of the NAMED environment for installed-"
        "metadata comparison; omitting it yields UNAVAILABLE, never a guess",
    )
    parser.add_argument(
        "--output",
        default=str(REPO / "tools/ENGINEERING_SURFACE_INDEX.json"),
    )
    args = parser.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    index = build_index(
        declarations_path=Path(args.declarations),
        runtime_python=args.runtime_python,
    )
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    index["provenance"] = {
        "generator": GENERATOR_VERSION,
        "generator_sha256": sha256_file(Path(__file__)),
        "repo_head": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(dirty),
        "environment_identity": {
            "generator_python": sys.executable,
            "runtime_python": args.runtime_python,
        },
        "arguments": vars(args),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(args.output)
    output.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "typed_outcome": index["typed_outcome"],
                "facts_hash": index["facts_hash"],
                "tools": len(index["facts"]["tools"]),
                "plugin_groups": len(index["facts"]["plugins"]),
                "structural_problems": index["facts"]["structural_problems"],
                "unclassified_new_executables": index["facts"][
                    "unclassified_new_executables"
                ],
                "output": str(output),
            },
            indent=1,
        )
    )
    return {"OK": 0, "UNAVAILABLE": 3, "BLOCK": 2}[index["typed_outcome"]]


if __name__ == "__main__":
    raise SystemExit(main())
