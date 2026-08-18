"""EC-13 regression tests (order 2026-08-18 WP2).

The installed base template carried a v1 contract default and an old
gate, so a unit could be GATED against one contract and RUN another.
These tests fail when ExecStartPre and ExecStart name — or resolve to —
different contract files.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "systemd_unit_coherence", REPO / "tools" / "systemd_unit_coherence.py")
uc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uc)

RUNTIME = "/runtime/agent-multi-p1lr-v4-8758273f"
V2 = "examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v2.json"
V1 = "examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial.json"


def _pre(contract: str) -> str:
    return f"{{ path=/gate.sh ; argv[]=/gate.sh $GATE {contract} ; }}"


def _run(contract: str, seed: int = 101) -> str:
    return (f"{{ path=/py ; argv[]=/py tools/p1_difficulty_lr_factorial.py "
            f"--seed {seed} --mode decision --contract {contract} ; }}")


class TestEC13:
    def test_the_live_shape_is_coherent(self):
        """Absolute gate path + relative runner path, same file."""
        result = uc.check_unit(
            exec_start_pre=_pre(f"{RUNTIME}/{V2}"),
            exec_start=_run(V2), working_directory=RUNTIME)
        assert result["coherent"] is True
        assert result["problems"] == []

    def test_ec13_stale_v1_default_is_caught(self):
        """THE defect: gated on v2, runs v1."""
        result = uc.check_unit(
            exec_start_pre=_pre(f"{RUNTIME}/{V2}"),
            exec_start=_run(V1), working_directory=RUNTIME)
        assert result["coherent"] is False
        assert any("CONTRACT_NAME_MISMATCH" in p
                   for p in result["problems"])

    def test_working_directory_drift_is_caught(self):
        """Same FILENAME on both sides, different resolved file: the
        relative runner path follows WorkingDirectory, so a stale
        working directory silently runs another contract."""
        result = uc.check_unit(
            exec_start_pre=_pre(f"{RUNTIME}/{V2}"),
            exec_start=_run(V2),
            working_directory="/runtime/agent-multi-p1lr-v1-OLD")
        assert result["coherent"] is False
        assert any("CONTRACT_PATH_MISMATCH" in p
                   for p in result["problems"])

    def test_runner_without_a_contract_is_caught(self):
        result = uc.check_unit(
            exec_start_pre=_pre(f"{RUNTIME}/{V2}"),
            exec_start="{ path=/py ; argv[]=/py run.py --mode decision ; }",
            working_directory=RUNTIME)
        assert result["coherent"] is False
        assert any("EXEC_START_NAMES_NO_CONTRACT" in p
                   for p in result["problems"])

    def test_gate_without_a_contract_is_caught(self):
        result = uc.check_unit(
            exec_start_pre="{ path=/gate.sh ; argv[]=/gate.sh $GATE ; }",
            exec_start=_run(V2), working_directory=RUNTIME)
        assert result["coherent"] is False
        assert any("EXEC_START_PRE_NAMES_NO_CONTRACT" in p
                   for p in result["problems"])

    def test_two_different_contracts_on_one_side_are_caught(self):
        """A runner naming TWO contract files cannot be gated by one."""
        two = (f"{{ path=/py ; argv[]=/py run.py --mode decision "
               f"--contract {V2} --fallback-contract {V1} ; }}")
        result = uc.check_unit(
            exec_start_pre=_pre(f"{RUNTIME}/{V2}"), exec_start=two,
            working_directory=RUNTIME)
        assert result["coherent"] is False
        assert any("CONTRACT_NAME_MISMATCH" in p
                   for p in result["problems"])

    def test_digest_branch_fires_when_paths_differ_in_content(
            self, tmp_path):
        """Same resolved set, differing digests -> refused."""
        a, b = tmp_path / "p1_difficulty_lr_factorial_v2.json", \
            tmp_path / "p1_difficulty_lr_factorial_v9.json"
        a.write_text("{}")
        b.write_text("{\"x\": 1}")
        both = (f"{{ argv[]=/py run.py --contract {a} "
                f"--contract {b} ; }}")
        result = uc.check_unit(
            exec_start_pre=both, exec_start=both,
            working_directory=str(tmp_path),
            digest_of=lambda p: uc._sha(p))
        assert result["coherent"] is False
        assert any("CONTRACT_DIGEST_MISMATCH" in p
                   for p in result["problems"])

    @pytest.mark.parametrize("command,expected", [
        (f"--contract {V2}", [V2]),
        ("--contract /abs/p1_difficulty_lr_factorial_v2.json",
         ["/abs/p1_difficulty_lr_factorial_v2.json"]),
        ("--mode decision", []),
    ])
    def test_contract_extraction(self, command, expected):
        assert uc.contracts_in(command) == expected
