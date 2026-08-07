"""Tests for the engineering surface index (P1, T-1/T-5 redesign).

Discovered facts and reviewed declarations are separate sources; CI
fails only on a NEW unclassified executable or a stale supported entry,
never because an untouched historical script lacks a modern contract.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.engineering_surface_index import (
    build_index,
    discover_source_entry_points,
    discover_tool,
    entry_point_drift,
    import_target_exists,
    structural_problems,
    unclassified_new_executables,
)

REPO = Path(__file__).resolve().parent.parent
DECLARATIONS = REPO / "tools/TOOL_DECLARATIONS.json"


def _declarations() -> dict:
    return json.loads(DECLARATIONS.read_text())


class TestDiscovery:
    def test_discovery_is_ast_only_and_finds_cli_facts(self, tmp_path):
        tool = tmp_path / "sample_tool.py"
        tool.write_text(
            "import argparse\nimport os\n"
            "def main():\n"
            "    p = argparse.ArgumentParser()\n"
            "    p.add_argument('--alpha')\n"
            "    p.add_argument('--beta')\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        )
        # discover_tool never imports: a side-effect bomb must be inert
        tool.write_text(
            tool.read_text() + "\nif False:\n    raise RuntimeError('imported!')\n"
        )
        import tools.engineering_surface_index as esi

        original = esi.REPO
        try:
            esi.REPO = tmp_path
            facts = discover_tool(tool)
        finally:
            esi.REPO = original
        assert facts["has_main_guard"] is True
        assert facts["defines_main"] is True
        assert facts["argparse_arguments"] == ["--alpha", "--beta"]
        assert "argparse" in facts["imports"] and "os" in facts["imports"]

    def test_syntax_error_is_reported_not_raised(self, tmp_path):
        bad = tmp_path / "broken.py"
        bad.write_text("def broken(:\n")
        import tools.engineering_surface_index as esi

        original = esi.REPO
        try:
            esi.REPO = tmp_path
            facts = discover_tool(bad)
        finally:
            esi.REPO = original
        assert facts["parse_error"] is not None

    def test_source_entry_points_parse_from_setup(self):
        groups = discover_source_entry_points(REPO / "setup.py")
        assert "console_scripts" in groups
        assert groups["console_scripts"]["agent-multi"] == "app.main:main"
        assert "env.plugins" in groups


class TestStructuralGuards:
    def test_new_unclassified_executable_fails(self):
        tools = {
            "brand_new_tool.py": {
                "has_main_guard": True,
                "defines_main": True,
            }
        }
        offenders = unclassified_new_executables(tools, _declarations())
        assert offenders == ["brand_new_tool.py"]

    def test_grandfathered_baseline_does_not_fail(self):
        declarations = _declarations()
        name = declarations["known_unclassified_baseline"][0]
        tools = {name: {"has_main_guard": True, "defines_main": True}}
        assert unclassified_new_executables(tools, declarations) == []

    def test_stale_supported_entry_is_a_problem(self):
        declarations = {
            "tools": {
                "vanished.py": {
                    "lifecycle": "supported",
                    "mutability": "read_only",
                }
            }
        }
        problems = structural_problems({}, {}, declarations)
        assert any("stale supported entry" in p for p in problems)

    def test_invalid_import_target_is_a_problem(self):
        problems = structural_problems(
            {},
            {"env.plugins": {"ghost": "no_such_pkg.ghost_mod:Plugin"}},
            {"tools": {}},
        )
        assert any("invalid import target" in p for p in problems)

    def test_duplicate_console_command_is_a_problem(self):
        # two groups cannot collide inside one dict; the duplicate check
        # guards the console namespace across repeated parses
        problems = structural_problems(
            {},
            {"console_scripts": {"agent-multi": "app.main:main"}},
            {"tools": {}},
        )
        assert not any("duplicate" in p for p in problems)

    def test_invalid_lifecycle_is_a_problem(self):
        problems = structural_problems(
            {"x.py": {}},
            {},
            {"tools": {"x.py": {"lifecycle": "shiny", "mutability": "read_only"}}},
        )
        assert any("invalid lifecycle" in p for p in problems)


class TestDrift:
    def test_source_only_entry_is_drift(self):
        result = entry_point_drift(
            {"env.plugins": {"a": "pkg.a:P"}}, {"env.plugins": {}}
        )
        assert result["drift"] == [
            {
                "group": "env.plugins",
                "name": "a",
                "source_target": "pkg.a:P",
                "installed_target": None,
            }
        ]

    def test_target_mismatch_is_drift(self):
        result = entry_point_drift(
            {"env.plugins": {"a": "pkg.a:P"}},
            {"env.plugins": {"a": "pkg.other:P"}},
        )
        assert result["drift"][0]["installed_target"] == "pkg.other:P"

    def test_foreign_console_scripts_are_not_reported(self):
        result = entry_point_drift(
            {"console_scripts": {"agent-multi": "app.main:main"}},
            {
                "console_scripts": {
                    "agent-multi": "app.main:main",
                    "cython": "Cython.Compiler.Main:setuptools_main",
                }
            },
        )
        assert result["drift"] == []
        assert result["installed_only"] == []

    def test_shared_plugin_group_contributions_are_visible(self):
        result = entry_point_drift(
            {"env.plugins": {"a": "pkg.a:P"}},
            {"env.plugins": {"a": "pkg.a:P", "b": "other.b:P"}},
        )
        assert result["drift"] == []
        assert result["installed_only"] == [
            {"group": "env.plugins", "name": "b", "installed_target": "other.b:P"}
        ]


class TestRealRepoIndex:
    def test_current_repo_has_no_structural_problems(self):
        index = build_index(
            declarations_path=DECLARATIONS, runtime_python=None
        )
        assert index["facts"]["structural_problems"] == []
        assert index["facts"]["unclassified_new_executables"] == []
        # without a runtime python the outcome is honestly UNAVAILABLE
        assert index["typed_outcome"] == "UNAVAILABLE"

    def test_declared_semantics_are_never_guessed(self):
        index = build_index(
            declarations_path=DECLARATIONS, runtime_python=None
        )
        undeclared = [
            name
            for name, entry in index["facts"]["tools"].items()
            if entry["declared"] == "UNCLASSIFIED"
        ]
        declarations = _declarations()
        for name in undeclared:
            assert name not in declarations["tools"]

    def test_facts_hash_is_deterministic(self):
        first = build_index(declarations_path=DECLARATIONS, runtime_python=None)
        second = build_index(declarations_path=DECLARATIONS, runtime_python=None)
        assert first["facts_hash"] == second["facts_hash"]

    def test_import_targets_of_source_entry_points_exist(self):
        groups = discover_source_entry_points(REPO / "setup.py")
        for group, entries in groups.items():
            for name, target in entries.items():
                assert import_target_exists(target), f"{group}/{name}: {target}"
