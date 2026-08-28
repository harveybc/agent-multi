"""P4 identity tests for the paired SAC dispatch driver — CPU only,
model-free; every GPU path is a refusal by construction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.dispatch_paired_pretrain_comparison import (  # noqa: E402
    DispatchRefused, verify_cell)


@pytest.fixture(scope="module")
def design():
    path = (REPO / "docs/audits/evidence/"
            "PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json")
    return json.loads(path.read_text())


def test_unknown_arm_and_seed_refuse(design):
    with pytest.raises(DispatchRefused, match="unknown arm"):
        verify_cell(design, Path("/nonexistent"), 101, "mystery_arm")
    arm = list(design["arms"])[0]
    with pytest.raises(DispatchRefused, match="not in the design"):
        verify_cell(design, Path("/nonexistent"), 999, arm)


def test_two_arm_design_with_frozen_deferred(design):
    assert set(design["arms"]) == {"control_random_init",
                                   "pretrained_finetuned"}
    assert "pretrained_frozen" in design.get("deferred_arms", {})
    assert len(design["trial_ledger"]) == 8  # 2 arms x 4 seeds
    orders = list(design["arm_order_counterbalanced"].values())
    first_positions = [order[0] for order in orders]
    assert first_positions.count("control_random_init") == 2
    assert first_positions.count("pretrained_finetuned") == 2


def test_driver_gpu_path_is_refusal_by_construction():
    source = (REPO / "tools/dispatch_paired_pretrain_comparison.py"
              ).read_text()
    assert "NOT_LAUNCHED" in source
    assert "deliberately NOT implemented" in source
    assert "gpu-authorized-by-musashi" in source
