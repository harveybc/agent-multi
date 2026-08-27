"""Canonical grouped-architecture materializer (DATA-SOTA-357).

ONE materialization path shared by the SAC construction route
(`sac_agent.py`) and the transfer smoke: the complete effective
grouped-extractor architecture is read from configuration, strictly
merged against the extractor defaults, validated, digest-bound and
frozen BEFORE any module is constructed. No caller may author
state-branch/fusion/branch dictionaries of its own.

Refusals (typed): absent ``feature_extractor_config``; absent
state_branch / fusion / branches / feature_columns / state_keys
declarations (no silent defaults for structural blocks); unknown extra
keys (strict merge); architecture feature_columns differing from the
experiment feature_columns (silent branch-misrouting refusal, order
2026-08-23 §1, moved here from sac_agent); fusion without an explicit
declared output dimension; and a post-construction output dimension or
family order differing from the bound expectation.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from agent_plugins.branch_pretraining import sha256_file, sha256_obj
from pipeline_plugins._observation_contract import feature_columns_sha256


class ArchitectureError(ValueError):
    """Typed refusal: the declared grouped architecture is invalid or
    drifted. Never construct."""


REQUIRED_DECLARATIONS = ("feature_columns", "branches", "state_keys",
                         "state_branch", "fusion")


def materialize_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Materialize the effective architecture from a merged experiment
    config dict (the SAC route input)."""
    architecture = config.get("feature_extractor_config")
    if not isinstance(architecture, dict):
        raise ArchitectureError(
            "feature_extractor_config must be declared as an object "
            "(DATA-SOTA-357: no caller-authored architecture)")
    for key in REQUIRED_DECLARATIONS:
        if not architecture.get(key):
            raise ArchitectureError(
                f"feature_extractor_config.{key} must be EXPLICITLY "
                f"declared — structural blocks have no silent defaults "
                f"(DATA-SOTA-357)")
    from agent_plugins.component_config import deep_merge_strict
    defaults = {
        "schema": "agent_multi.grouped_features.v1",
        "feature_columns": [], "branches": [], "state_keys": [],
        "state_branch": {"plugin": "mlp_branch", "params": None},
        "fusion": {"plugin": "gated_fusion", "params": None},
        "share_features_extractor": False,
    }
    effective = deep_merge_strict(defaults, architecture,
                                  path="feature_extractor_config")
    env_columns = list(config.get("feature_columns") or [])
    arch_columns = list(effective["feature_columns"])
    if env_columns and env_columns != arch_columns:
        raise ArchitectureError(
            "feature_extractor_config.feature_columns must be "
            "IDENTICAL (same names, same order) to the experiment "
            "feature_columns that define the observation emission "
            "order; refusing silent branch misrouting")
    fusion_params = (effective["fusion"] or {}).get("params") or {}
    expected_output_dim = fusion_params.get("output_dim")
    if not isinstance(expected_output_dim, int) \
            or isinstance(expected_output_dim, bool) \
            or expected_output_dim < 1:
        raise ArchitectureError(
            "fusion.params.output_dim must be an explicit positive "
            "integer — the expected output dimension is bound, never "
            "inferred (DATA-SOTA-357)")
    ordered_families = [str(b.get("name") or f"branch_{i}")
                       for i, b in enumerate(effective["branches"])]
    if effective["state_keys"]:
        ordered_families.append("account_state")
    return {
        "schema": "agent_multi.materialized_grouped_architecture.v1",
        "architecture": deepcopy(effective),
        "architecture_digest": sha256_obj(effective),
        "feature_columns_sha256": feature_columns_sha256(arch_columns),
        "branch_plugins": [{"name": b["name"], "plugin": b["plugin"],
                            "params": b.get("params")}
                           for b in effective["branches"]],
        "state_branch": deepcopy(effective["state_branch"]),
        "fusion": deepcopy(effective["fusion"]),
        "ordered_families": ordered_families,
        "state_keys": list(effective["state_keys"]),
        "expected_output_dim": int(expected_output_dim),
    }


def materialize_from_file(config_path: Path) -> dict[str, Any]:
    """The smoke/tooling route: same materialization plus the config
    FILE digest."""
    config = json.loads(Path(config_path).read_text())
    materialized = materialize_from_config(config)
    materialized["config_sha256"] = sha256_file(config_path)
    return materialized


def construct_extractor(materialized: dict[str, Any],
                        observation_space):
    """Construct the grouped extractor ONLY from the materialized
    object, then verify the bound expectations against what was
    actually built."""
    import agent_plugins.grouped_features_extractor as gfe

    Extractor = gfe.build_grouped_extractor_class()
    extractor = Extractor(observation_space,
                          deepcopy(materialized["architecture"]))
    if int(extractor.features_dim) != materialized["expected_output_dim"]:
        raise ArchitectureError(
            f"constructed output dimension {extractor.features_dim} "
            f"differs from the bound expectation "
            f"{materialized['expected_output_dim']}")
    if list(extractor.ordered_families) != \
            list(materialized["ordered_families"]):
        raise ArchitectureError(
            "constructed family order differs from the materialized "
            "ordered_families")
    return extractor


def assert_same_materialization(a: dict[str, Any],
                                b: dict[str, Any]) -> None:
    """DATA-SOTA-357 parity proof helper: two routes must bind the
    SAME effective architecture digest."""
    if a["architecture_digest"] != b["architecture_digest"]:
        raise ArchitectureError(
            f"architecture digest divergence between routes: "
            f"{a['architecture_digest'][:12]} vs "
            f"{b['architecture_digest'][:12]}")
