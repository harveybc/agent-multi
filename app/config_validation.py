"""Shared, pure config validators — one implementation per invariant.

Implements P2 of MUSASHI_DISPOSITION_SATOSHI_III_DETERMINISTIC_TOOLING
_2026_08_06 (T-2 redesign). These functions are the single owners of the
five defect classes behind findings 108/110/113/126/142, which had no
owning module before this file. They are called from three seams:

  1. materialization tests (tests/test_config_validation.py),
  2. the read-only doctor CLI (tools/config_doctor.py),
  3. the campaign-launch preflight (app/campaign_supervisor.py).

Rules that already have an owner are NOT restated here: schema/type and
runtime-key collisions belong to app/canonical_config.resolve_config;
dataset manifest asset/timeframe/sha binding belongs to
CampaignSupervisor._validate_dataset_evidence; plan structure belongs
to CampaignSupervisor._validate_plan; repair-rule executability belongs
to the genome plugin's validate_repair_rules. This module adds only the
previously ownerless cross-section invariants.

Typed outcomes (per the disposition):
  PASS         every aspect of the rule executed and held;
  BLOCK        an executable invariant is contradicted;
  WARNING      a declared, non-safety concern needs review;
  UNAVAILABLE  a required fact could not be established here.

Launch must refuse BLOCK and required-UNAVAILABLE. No per-launch human
sign-off and no conversational override exist: a blocked config is fixed
in a new revision, or the validator is. Warnings may be acknowledged
only through a versioned suppression carrying rule id, reason, owner,
scope and expiry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

SCHEMA = "config_validation_report.v1"

PASS = "PASS"
BLOCK = "BLOCK"
WARNING = "WARNING"
UNAVAILABLE = "UNAVAILABLE"

# Rules whose UNAVAILABLE outcome must refuse a launch. metric_resolvable
# is deliberately required: it can only be evaluated where the pipeline
# package is importable, which forces the authoritative preflight to run
# in the real runtime environment (T-2), never in the tooling venv.
REQUIRED_RULES = frozenset(
    {
        "metric_consistency",
        "metric_resolvable",
        "asset_namespace",
        "genome_choice_repair",
        "pinned_references",
        "dormant_year_fields",
        "split_overlap",
    }
)

# Tokens that identify an asset family in names and artifact paths.
ASSET_TOKENS = (
    "usdcad",
    "eurusd",
    "gbpusd",
    "audusd",
    "usdjpy",
    "usdchf",
    "ethusdt",
    "ethusd",
    "eth",
    "btcusdt",
    "btcusd",
    "btc",
)

_METRIC_KEYS = ("selection_metric", "optimization_metric", "metric")
_YEAR_KEYS = ("train_years", "val_years", "test_years")
_DATE_KEYS = (
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
)


@dataclass
class CheckResult:
    rule_id: str
    outcome: str
    detail: str
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "outcome": self.outcome,
            "detail": self.detail,
            "evidence": self.evidence,
        }


class ConfigPreflightError(ValueError):
    """Raised by preflight_or_raise for BLOCK or required-UNAVAILABLE."""


# ------------------------------------------------------------- traversal


def _walk(node: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    yield path, node
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield from _walk(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")


def _string_leaves(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (path, value)
        for path, value in _walk(config)
        if isinstance(value, str)
    ]


def _asset_tokens_in(text: str) -> set[str]:
    """Asset tokens appearing as DELIMITED segments. 'eth' matches
    'eth_curriculum' and 'phase/eth/x' but never 'method', and never the
    'eth' inside 'ethusdt' — an adjacent alphanumeric disqualifies the
    span, so a shorter token can only match where no longer token does."""
    low = text.lower()
    return {
        token
        for token in ASSET_TOKENS
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", low)
    }


def _asset_family(token: str) -> str:
    for family in ("eth", "btc"):
        if token.startswith(family):
            return family
    return token


# ----------------------------------------------------------------- rules


def check_metric_consistency(config: Mapping[str, Any]) -> CheckResult:
    """Finding-108 class: every declared selection/optimization metric in
    the document must agree; a silent disagreement trains one objective
    while declaring another."""
    declared: dict[str, str] = {}
    for path, node in _walk(config):
        if isinstance(node, Mapping):
            for key in _METRIC_KEYS:
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    declared[f"{path}.{key}" if path else key] = value.strip()
    distinct = sorted(set(declared.values()))
    if len(distinct) > 1:
        return CheckResult(
            "metric_consistency",
            BLOCK,
            "the document declares contradictory selection metrics: "
            + ", ".join(distinct),
            {"declared": declared},
        )
    return CheckResult(
        "metric_consistency",
        PASS,
        "all declared metrics agree" if declared else "no metric declared",
        {"declared": declared},
    )


def check_metric_resolvable(
    config: Mapping[str, Any],
    implemented_metrics: frozenset[str] | None,
) -> CheckResult:
    """Finding-108 class, second half: the declared metric must be one an
    installed pipeline actually implements. The implemented set MUST be
    supplied by the runtime environment (the owning pipeline module
    declares it); this function never guesses it."""
    declared = {
        value.strip()
        for path, node in _walk(config)
        if isinstance(node, Mapping)
        for key in _METRIC_KEYS
        for value in [node.get(key)]
        if isinstance(value, str) and value.strip()
    }
    if implemented_metrics is None:
        return CheckResult(
            "metric_resolvable",
            UNAVAILABLE,
            "the implemented-metric set is only observable in the runtime "
            "environment; this environment cannot import the pipeline",
            {"declared": sorted(declared)},
        )
    unresolved = sorted(declared - implemented_metrics)
    if unresolved:
        return CheckResult(
            "metric_resolvable",
            BLOCK,
            "declared metrics no installed pipeline implements: "
            + ", ".join(unresolved),
            {"unresolved": unresolved, "implemented": sorted(implemented_metrics)},
        )
    return CheckResult(
        "metric_resolvable",
        PASS,
        "every declared metric is implemented",
        {"declared": sorted(declared)},
    )


def check_asset_namespace(config: Mapping[str, Any]) -> CheckResult:
    """Finding-110 class: the experiment name and every artifact path must
    belong to the configured asset's namespace. A foreign asset token in
    an artifact path can overwrite another campaign's history."""
    data = config.get("data") if isinstance(config.get("data"), Mapping) else {}
    configured = str(data.get("asset") or config.get("asset") or "").lower()
    reference = (
        configured
        or str(data.get("input_data_file") or config.get("input_data_file") or "").lower()
    )
    own_families = {_asset_family(token) for token in _asset_tokens_in(reference)}
    if not own_families:
        return CheckResult(
            "asset_namespace",
            WARNING,
            "no configured asset token found; namespace check cannot bind",
            {},
        )
    foreign: list[dict[str, str]] = []
    for path, value in _string_leaves(config):
        if "input_data" in path:
            continue
        for token in _asset_tokens_in(value):
            if _asset_family(token) not in own_families:
                foreign.append({"path": path, "token": token, "value": value[:120]})
    if foreign:
        return CheckResult(
            "asset_namespace",
            BLOCK,
            f"{len(foreign)} references carry a foreign asset token "
            f"(configured families: {sorted(own_families)})",
            {"foreign": foreign[:20]},
        )
    return CheckResult(
        "asset_namespace",
        PASS,
        f"all references stay inside {sorted(own_families)}",
        {},
    )


def check_genome_choice_repair(config: Mapping[str, Any]) -> CheckResult:
    """Finding-113 class: a categorical gene choice that other parts of the
    config forbid (preprocessing_mode='none' with a required feature-aware
    preprocessor) must be covered by a repair rule, otherwise the genome
    deterministically emits candidates that die at decode. Executability
    of the rule itself belongs to the genome plugin's
    validate_repair_rules; presence and coverage are checked here."""
    genes: list[tuple[str, Mapping[str, Any]]] = []
    repair_rules: list[Any] | None = None
    for path, node in _walk(config):
        if isinstance(node, Mapping):
            if (
                node.get("kind") == "categorical"
                and isinstance(node.get("name"), str)
                and isinstance(node.get("choices"), list)
            ):
                genes.append((path, node))
            rules = node.get("mixed_genome_repair_rules")
            if isinstance(rules, list):
                repair_rules = rules
    hazardous = [
        (path, gene)
        for path, gene in genes
        if gene.get("name") == "preprocessing_mode" and "none" in gene["choices"]
    ]
    if not hazardous:
        return CheckResult(
            "genome_choice_repair",
            PASS,
            "no hazardous categorical choice found",
            {"genes": len(genes)},
        )
    covering = [
        rule
        for rule in (repair_rules or [])
        if "preprocessing_mode" in str(rule)
    ]
    if not covering:
        return CheckResult(
            "genome_choice_repair",
            BLOCK,
            "preprocessing_mode offers 'none' but no repair rule covers it; "
            "the genome will emit deterministically invalid candidates",
            {
                "hazardous_genes": [path for path, _ in hazardous],
                "repair_rules_declared": len(repair_rules or []),
            },
        )
    return CheckResult(
        "genome_choice_repair",
        PASS,
        "hazardous choice is covered by a declared repair rule",
        {"covering_rules": len(covering)},
    )


def check_pinned_references(config: Mapping[str, Any]) -> CheckResult:
    """Finding-126 class: a decision-bearing config that names an input
    dataset must bind it by content (sha256 or a dataset manifest);
    an unpinned mutable reference lets semantics drift between hosts."""
    problems = []
    for path, node in _walk(config):
        if not isinstance(node, Mapping):
            continue
        # experiment.legacy_flat is the preserved RAW legacy overlay
        # (app/canonical_config.py owns that contract); runtime values
        # flow from the canonical sections, so the archival echo of a
        # reference that the data section pins is not a live defect.
        # First-cycle false positive, corrected in the validator and
        # disclosed in the confusion matrix — the fixture was not touched.
        if "legacy_flat" in path:
            continue
        if node.get("input_data_file"):
            pinned = bool(
                node.get("input_data_sha256") or node.get("dataset_manifest_file")
            )
            if not pinned:
                problems.append(f"{path}.input_data_file" if path else "input_data_file")
    if problems:
        return CheckResult(
            "pinned_references",
            BLOCK,
            "dataset references without a content binding: "
            + ", ".join(problems),
            {"unpinned": problems},
        )
    return CheckResult(
        "pinned_references", PASS, "every dataset reference is content-bound", {}
    )


def check_dormant_year_fields(config: Mapping[str, Any]) -> CheckResult:
    """Finding-142 class: year-count shorthand surviving beside explicit
    split dates is a dormant contradiction — one of the two governs and
    the other silently lies."""
    for path, node in _walk(config):
        if not isinstance(node, Mapping):
            continue
        years = {key: node[key] for key in _YEAR_KEYS if key in node}
        dates = {key: node[key] for key in _DATE_KEYS if key in node}
        if years and dates:
            return CheckResult(
                "dormant_year_fields",
                BLOCK,
                "year-count shorthand coexists with explicit split dates "
                f"at {path or '<root>'}; the shorthand is dormant and "
                "contradictory",
                {"years": years, "dates": sorted(dates)},
            )
    return CheckResult(
        "dormant_year_fields", PASS, "no dormant year/date contradiction", {}
    )


def check_split_overlap(config: Mapping[str, Any]) -> CheckResult:
    """Train/validation/test windows must be disjoint and ordered.
    ISO-8601 strings of equal shape compare lexicographically."""
    for path, node in _walk(config):
        if not isinstance(node, Mapping):
            continue
        dates = {key: str(node[key]) for key in _DATE_KEYS if node.get(key)}
        pairs = (
            ("train_end", "validation_start"),
            ("validation_end", "test_start"),
        )
        for earlier, later in pairs:
            if earlier in dates and later in dates and dates[earlier] >= dates[later]:
                return CheckResult(
                    "split_overlap",
                    BLOCK,
                    f"{earlier} ({dates[earlier]}) does not precede "
                    f"{later} ({dates[later]}) at {path or '<root>'}",
                    {"dates": dates},
                )
        for start, end in (
            ("train_start", "train_end"),
            ("validation_start", "validation_end"),
            ("test_start", "test_end"),
        ):
            if start in dates and end in dates and dates[start] >= dates[end]:
                return CheckResult(
                    "split_overlap",
                    BLOCK,
                    f"{start} ({dates[start]}) does not precede {end} "
                    f"({dates[end]}) at {path or '<root>'}",
                    {"dates": dates},
                )
    return CheckResult("split_overlap", PASS, "split windows are ordered", {})


# ------------------------------------------------------------ evaluation


def evaluate(
    config: Mapping[str, Any],
    *,
    implemented_metrics: frozenset[str] | None = None,
) -> dict:
    """Run every shared rule over one config document. Pure: no file,
    network or environment access; the runtime-only fact (implemented
    metrics) is injected by the caller or honestly UNAVAILABLE."""
    results = [
        check_metric_consistency(config),
        check_metric_resolvable(config, implemented_metrics),
        check_asset_namespace(config),
        check_genome_choice_repair(config),
        check_pinned_references(config),
        check_dormant_year_fields(config),
        check_split_overlap(config),
    ]
    blocked = [r for r in results if r.outcome == BLOCK]
    unavailable_required = [
        r
        for r in results
        if r.outcome == UNAVAILABLE and r.rule_id in REQUIRED_RULES
    ]
    if blocked:
        overall = BLOCK
    elif unavailable_required:
        overall = UNAVAILABLE
    elif any(r.outcome == WARNING for r in results):
        overall = WARNING
    else:
        overall = PASS
    return {
        "schema": SCHEMA,
        "overall": overall,
        "results": [r.as_dict() for r in results],
        "blocking": [r.rule_id for r in blocked],
        "unavailable_required": [r.rule_id for r in unavailable_required],
    }


def preflight_or_raise(
    config: Mapping[str, Any],
    *,
    implemented_metrics: frozenset[str] | None,
    context: str,
) -> dict:
    """The launch seam. Refuses BLOCK and required-UNAVAILABLE with the
    full report in the exception; returns the report otherwise."""
    report = evaluate(config, implemented_metrics=implemented_metrics)
    if report["overall"] in (BLOCK, UNAVAILABLE):
        raise ConfigPreflightError(
            f"config preflight refused ({context}): overall="
            f"{report['overall']} blocking={report['blocking']} "
            f"unavailable_required={report['unavailable_required']}"
        )
    return report


def runtime_implemented_metrics() -> frozenset[str] | None:
    """Observe the implemented selection metrics from the OWNING pipeline
    module in this environment. Returns None (never a guess) when the
    pipeline is not importable here."""
    try:
        from pipeline_plugins.rl_pipeline_with_validation import (
            IMPLEMENTED_SELECTION_METRICS,
        )
    except Exception:
        return None
    return frozenset(IMPLEMENTED_SELECTION_METRICS)
