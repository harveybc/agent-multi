"""Executable genome validity (AUD-F1-20260805-113 / decision order §2.1).

The forbid_value rule must be ENFORCED at the decode boundary — the
point every genome origin (fresh, resume, migration, network champion)
passes before environment or GPU construction — and rule schemas must
fail closed on anything unknown or cosmetic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from optimizer_plugins.project3_full_genome_optimizer import Plugin

REPO = Path(__file__).resolve().parent.parent
EN_V2 = (REPO / "examples/config/phase_2_eth_curriculum/optimization/"
         "phase_2_eth_en_v2.json")

RULES = [{"rule": "forbid_value", "gene": "preprocessing_mode",
          "value": "none", "repair": "resample_categorical",
          "reason": "no precomputed causal feature contract"}]
SCHEMA = [{"name": "preprocessing_mode", "kind": "categorical",
           "choices": ["rolling_zscore", "expanding_zscore"],
           "target": "feature_scaling"}]


def _config(rules=RULES, schema=SCHEMA):
    return {"mixed_genome_repair_rules": rules,
            "mixed_genome_schema": schema}


def test_fresh_genomes_never_offer_none():
    config = json.loads(EN_V2.read_text())
    gene = next(g for g in config["optimization"]["mixed_genome_schema"]
                if g["name"] == "preprocessing_mode")
    assert "none" not in gene["choices"]


def test_injected_legacy_genome_is_repaired_before_env():
    """A resume/migration/network genome carrying 'none' is repaired
    deterministically at decode, before any env/GPU work."""
    run_config = {}
    decoded = {"preprocessing_mode": "none"}
    Plugin._apply_repair_rules(run_config, decoded, _config())
    assert decoded["preprocessing_mode"] == "rolling_zscore"
    assert run_config["feature_scaling"] == "rolling_zscore"
    repair = run_config["_genome_repairs"][0]
    assert repair["forbidden"] == "none"
    assert repair["replacement"] == "rolling_zscore"


def test_reject_repair_mode_raises():
    rules = [dict(RULES[0], repair="reject")]
    with pytest.raises(ValueError, match="forbidden value"):
        Plugin._apply_repair_rules(
            {}, {"preprocessing_mode": "none"}, _config(rules))


def test_clean_genome_untouched():
    run_config = {}
    decoded = {"preprocessing_mode": "expanding_zscore"}
    Plugin._apply_repair_rules(run_config, decoded, _config())
    assert decoded["preprocessing_mode"] == "expanding_zscore"
    assert "_genome_repairs" not in run_config


def test_unknown_rule_kind_fails():
    with pytest.raises(ValueError, match="unknown rule kind"):
        Plugin.validate_repair_rules(
            [{"rule": "wish_upon_a_star", "gene": "x"}])


def test_empty_cosmetic_rule_fails():
    with pytest.raises(ValueError, match="non-empty"):
        Plugin.validate_repair_rules([{}])
    with pytest.raises(ValueError, match="requires 'if' and 'set'"):
        Plugin.validate_repair_rules([{"note": "looks official"}])
    with pytest.raises(ValueError, match="non-empty"):
        Plugin.validate_repair_rules([{"if": {}, "set": {}}])


def test_forbid_value_requires_gene_value_and_known_repair():
    with pytest.raises(ValueError, match="requires 'gene'"):
        Plugin.validate_repair_rules([{"rule": "forbid_value"}])
    with pytest.raises(ValueError, match="requires 'value'"):
        Plugin.validate_repair_rules(
            [{"rule": "forbid_value", "gene": "g"}])
    with pytest.raises(ValueError, match="unknown repair"):
        Plugin.validate_repair_rules(
            [{"rule": "forbid_value", "gene": "g", "value": 1,
              "repair": "pray"}])


def test_no_allowed_replacement_fails_closed():
    schema = [{"name": "preprocessing_mode", "kind": "categorical",
               "choices": ["none"], "target": "feature_scaling"}]
    with pytest.raises(ValueError, match="no allowed replacement"):
        Plugin._apply_repair_rules(
            {}, {"preprocessing_mode": "none"},
            _config(schema=schema))


def test_v2_config_rules_validate_and_execute():
    config = json.loads(EN_V2.read_text())
    opt = config["optimization"]
    Plugin.validate_repair_rules(opt["mixed_genome_repair_rules"], opt)
    run_config = {}
    decoded = {"preprocessing_mode": "none"}
    Plugin._apply_repair_rules(run_config, decoded, opt)
    assert decoded["preprocessing_mode"] != "none"
