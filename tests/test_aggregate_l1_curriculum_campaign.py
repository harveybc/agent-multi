import copy
import json
from pathlib import Path

import pytest

from tools.aggregate_l1_curriculum_campaign import (
    AggregationError,
    _direction,
    aggregate,
)


def test_direction_rule_is_predeclared_shape():
    assert _direction([1, 2, 3, -9]) == "DIRECTIONAL_SIGNAL_FOR"
    assert _direction([-1, -2, -3, 9]) == "DIRECTIONAL_SIGNAL_AGAINST"
    assert _direction([1, -1, 0, 0]) == "INCONCLUSIVE"


def test_real_terminal_evidence_aggregates():
    root = Path("docs/audits/evidence/p1_curriculum_terminal_20260825")
    result = aggregate(root / "reports", root / "manifests",
                       root / "nested_split_contract_v1.json")
    assert result["reports_complete"] == 12
    assert result["sealed_2025_used"] is False
    assert result["promotion_authorized"] is False
    assert result["results"]["EN-W"]["informative_easy_seeds"] == 0
    assert result["results"]["EN-F"]["informative_easy_seeds"] == 0


def test_missing_report_refuses(tmp_path):
    root = Path("docs/audits/evidence/p1_curriculum_terminal_20260825")
    reports = tmp_path / "reports"
    reports.mkdir()
    for source in (root / "reports").glob("*.json"):
        if source.name != "seed404_EN-F_report.json":
            (reports / source.name).write_bytes(source.read_bytes())
    with pytest.raises(AggregationError, match="missing report"):
        aggregate(reports, root / "manifests",
                  root / "nested_split_contract_v1.json")


def test_pair_mutation_refuses(tmp_path):
    root = Path("docs/audits/evidence/p1_curriculum_terminal_20260825")
    reports = tmp_path / "reports"
    reports.mkdir()
    for source in (root / "reports").glob("*.json"):
        (reports / source.name).write_bytes(source.read_bytes())
    target = reports / "seed303_EN-F_report.json"
    record = json.loads(target.read_text())
    record["contracts"]["pair_contract"]["learning_rate"] = 0.9
    target.write_text(json.dumps(record))
    with pytest.raises(AggregationError, match="pair identity"):
        aggregate(reports, root / "manifests",
                  root / "nested_split_contract_v1.json")
