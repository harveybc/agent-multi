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
           "choices": ["none", "rolling_zscore", "expanding_zscore"],
           "target": "feature_scaling"}]


def _config(rules=RULES, schema=SCHEMA):
    return {"mixed_genome_repair_rules": rules,
            "mixed_genome_schema": schema}


def test_forbidden_value_is_declared_but_never_generable():
    """AUD-F1-20260806-138: the forbidden value must remain a DECLARED
    choice (else the rule is inert), and the executable rule must make
    it non-generable at the decode boundary."""
    config = json.loads(EN_V2.read_text())
    opt = config["optimization"]
    gene = next(g for g in opt["mixed_genome_schema"]
                if g["name"] == "preprocessing_mode")
    assert "none" in gene["choices"], "rule would be inert"
    Plugin.validate_repair_rules(opt["mixed_genome_repair_rules"], opt)
    decoded = {"preprocessing_mode": "none"}
    run_config = {}
    Plugin._apply_repair_rules(run_config, decoded, opt)
    assert decoded["preprocessing_mode"] != "none"
    assert run_config["_genome_repairs"][0]["forbidden"] == "none"


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


def test_rule_for_nonexistent_gene_fails():
    """AUD-F1-20260806-132: a repair rule must bind to a real gene."""
    with pytest.raises(ValueError, match="does not exist"):
        Plugin.validate_repair_rules(
            [{"rule": "forbid_value", "gene": "ghost_gene",
              "value": "x", "repair": "resample_categorical"}],
            _config())


def test_rule_for_non_categorical_gene_fails():
    schema = [{"name": "learning_rate_gene", "kind": "continuous",
               "low": 0.0, "high": 1.0}]
    with pytest.raises(ValueError, match="not categorical"):
        Plugin.validate_repair_rules(
            [{"rule": "forbid_value", "gene": "learning_rate_gene",
              "value": 0.5, "repair": "resample_categorical"}],
            _config(schema=schema))


def test_resample_without_replacement_fails_at_validation():
    schema = [{"name": "preprocessing_mode", "kind": "categorical",
               "choices": ["none"], "target": "feature_scaling"}]
    with pytest.raises(ValueError, match="no allowed replacement"):
        Plugin.validate_repair_rules(RULES, _config(schema=schema))


def test_repair_draw_is_choice_order_invariant():
    """AUD-F1-20260806-132: the seeded draw must not depend on the
    declaration order of choices."""
    decoded_template = {"preprocessing_mode": "none", "other_gene": 7}
    schema_a = [{"name": "preprocessing_mode", "kind": "categorical",
                 "choices": ["none", "rolling_zscore",
                             "expanding_zscore", "minmax"],
                 "target": "feature_scaling"}]
    schema_b = [{"name": "preprocessing_mode", "kind": "categorical",
                 "choices": ["minmax", "expanding_zscore", "none",
                             "rolling_zscore"],
                 "target": "feature_scaling"}]
    picks = []
    for schema in (schema_a, schema_b):
        decoded = dict(decoded_template)
        run_config = {}
        Plugin._apply_repair_rules(run_config, decoded,
                                   _config(schema=schema))
        picks.append(decoded["preprocessing_mode"])
        repair = run_config["_genome_repairs"][0]
        assert repair["allowed_choices"] == sorted(
            ["rolling_zscore", "expanding_zscore", "minmax"])
        assert "seed_derivation" in repair
        assert repair["original_value"] == "none"
    assert picks[0] == picks[1], "draw depended on declaration order"


def test_repair_draw_distribution_sanity():
    """Across many candidate identities the draw must not collapse to
    one choice (the first-allowed bias this replaces)."""
    schema = [{"name": "preprocessing_mode", "kind": "categorical",
               "choices": ["none", "rolling_zscore",
                           "expanding_zscore", "minmax"],
               "target": "feature_scaling"}]
    counts = {}
    for i in range(300):
        decoded = {"preprocessing_mode": "none", "candidate_index": i}
        Plugin._apply_repair_rules({}, decoded, _config(schema=schema))
        counts[decoded["preprocessing_mode"]] = counts.get(
            decoded["preprocessing_mode"], 0) + 1
    assert len(counts) == 3, f"draw collapsed: {counts}"
    assert all(count > 50 for count in counts.values()), counts



def test_missing_typed_schema_is_an_error_not_a_pass():
    """Musashi reproducer `repair_validation_fail_open` (a): a rule
    without a typed schema must FAIL, not be accepted."""
    with pytest.raises(ValueError, match="typed"):
        Plugin.validate_repair_rules(RULES, {})
    with pytest.raises(ValueError, match="typed"):
        Plugin.validate_repair_rules(
            RULES, {"mixed_genome_schema": []})


def test_forbidden_value_outside_domain_is_rejected():
    """Reproducer `repair_validation_fail_open` (b): a typo'd forbidden
    value would be a valid but INERT rule; it must be rejected."""
    rules = [dict(RULES[0], value="nonexistent_mode")]
    with pytest.raises(ValueError, match="not a declared choice"):
        Plugin.validate_repair_rules(rules, _config(rules))


def test_duplicate_choices_are_rejected():
    schema = [{"name": "preprocessing_mode", "kind": "categorical",
               "choices": ["none", "rolling_zscore", "rolling_zscore"],
               "target": "feature_scaling"}]
    with pytest.raises(ValueError, match="duplicate choices"):
        Plugin.validate_repair_rules(RULES, _config(schema=schema))
