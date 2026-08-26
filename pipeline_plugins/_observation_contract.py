"""Fail-closed validation for feature-aware policy observations.

Two layers live here and they are deliberately separate:

``validate_observation_contract``
    the fail-closed runtime validator. It now refuses an undeclared contract;
    legacy raw observations require an explicit, recorded waiver.

``apply_observation_contract``
    AUD-P1LR-20260815-235.  The refusal above was inert for every run
    that mattered, because a candidate config is materialised from a
    *base* config (``examples/results/.../config_out.json``, which
    declares ``include_price_window: true`` and no
    ``require_feature_aware_preprocessor``) and no experiment program
    ever re-declared those two fields on the way to the pipeline.  The
    guard therefore returned on its first line and validated nothing,
    while 64 unnormalized raw-price dimensions killed the actor's first
    layer.  This layer lets an EXPERIMENT CONTRACT declare the
    observation fields its program runs under, applies them to the
    resolved runtime config before any model or env is constructed, and
    records a typed provenance block naming every field it moved.

The declaration is resolved in one of two ways, in order:

1. ``config["observation_contract"]`` — an inline block.  This is the
   preferred path and the one a runner should populate directly
   (``config["observation_contract"] = contract["observation_contract"]``).
2. the experiment contract the config is already CONTENT-ADDRESSED to,
   via ``config["_identity"]["experiment_contract_sha256"]``, located by
   digest under ``examples/config/``.  Content addressing (never a path)
   is the repository's existing identity idiom, so a program cannot
   acquire a different observation contract than the one whose sha is
   sealed into its records.

A config that declares nothing is left byte-identical by the application
layer, then refused by the validator. Already-sealed artifacts are untouched;
only a new attempt must declare its observation contract or waiver.
"""
from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

APPLICATION_SCHEMA = "agent_multi.observation_contract.application.v1"
VALIDATION_SCHEMA = "agent_multi.observation_contract.validation.v1"

#: Typed refusal codes.
UNDECLARED_OBSERVATION_CONTRACT = "UNDECLARED_OBSERVATION_CONTRACT"
UNWAIVED_RAW_OBSERVATION = "UNWAIVED_RAW_OBSERVATION"

#: A waiver has to say something. Long enough that "n/a" cannot buy the
#: raw price window, short enough that it is never ceremony.
MIN_WAIVER_REASON_CHARS = 12

#: Repository root (this file lives in ``<repo>/pipeline_plugins/``).
_REPO = Path(__file__).resolve().parent.parent

#: Where an experiment contract may live.  Bounded on purpose: the
#: resolver hashes every JSON below this root, so it must stay small and
#: must never include run outputs.
_CONTRACT_ROOT = _REPO / "examples" / "config"

#: The observation fields an experiment contract may bind.  Everything
#: here is consumed by ``feature_window_preprocessor`` (gym-fx) or by the
#: validator below; nothing else may be smuggled through this block.
OBSERVATION_RUNTIME_KEYS: Tuple[str, ...] = (
    "require_feature_aware_preprocessor",
    "preprocessor_plugin",
    "feature_scaling",
    "feature_scaling_window",
    "feature_clip",
    "include_price_window",
    "include_agent_state",
    "agent_state_contract",
    "holding_duration_scale_bars",
    "window_size",
    "precomputed_causal_features",
    "precomputed_feature_contract_sha256",
)

_UNSET = "<unset>"

_CONTRACT_INDEX: Dict[str, Dict[str, Any]] | None = None


def feature_columns_sha256(columns: Any) -> str:
    """Content digest of an ORDERED feature-column list.

    Lets a contract pin the exact observation feature set it was written
    against without copying 83 column names into the contract.
    """
    payload = json.dumps([str(name) for name in (columns or [])],
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _contract_index() -> Dict[str, Dict[str, Any]]:
    """``{sha256: parsed contract}`` for every JSON under the config root.

    Built once per process.  72 small files today; the cost is a few
    milliseconds and it happens before any GPU work.
    """
    global _CONTRACT_INDEX
    if _CONTRACT_INDEX is not None:
        return _CONTRACT_INDEX
    index: Dict[str, Dict[str, Any]] = {}
    if _CONTRACT_ROOT.is_dir():
        for path in sorted(_CONTRACT_ROOT.rglob("*.json")):
            try:
                raw = path.read_bytes()
                payload = json.loads(raw)
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            if "observation_contract" not in payload:
                # Only contracts that DECLARE one can bind one; skipping
                # the rest keeps the index tiny and the lookup honest.
                continue
            index.setdefault(hashlib.sha256(raw).hexdigest(),
                             {"path": str(path.relative_to(_REPO)),
                              "contract": payload})
    _CONTRACT_INDEX = index
    return index


def declared_observation_contract(config: Mapping[str, Any]
                                  ) -> Tuple[Dict[str, Any] | None, str]:
    """``(declaration, source)`` for this run, or ``(None, "undeclared")``."""
    inline = config.get("observation_contract")
    if isinstance(inline, Mapping) and inline:
        return dict(inline), "config.observation_contract"
    identity = config.get("_identity")
    sha = None
    if isinstance(identity, Mapping):
        sha = identity.get("experiment_contract_sha256")
    if not sha:
        return None, "undeclared"
    entry = _contract_index().get(str(sha))
    if entry is None:
        return None, "undeclared"
    block = entry["contract"].get("observation_contract")
    if not isinstance(block, Mapping) or not block:
        return None, "undeclared"
    return dict(block), f"experiment_contract:{entry['path']}"


def apply_observation_contract(config: Dict[str, Any]
                               ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Bind the program's declared observation fields onto ``config``.

    Returns ``(config, provenance)``.  When nothing is declared the input
    config object is returned unchanged and the provenance says so, so a
    sealed program's runtime config stays byte-identical.

    Fail-closed extras:

    * an unknown key in the declaration is REFUSED (a contract may bind
      observation fields, not arbitrary runtime config);
    * ``feature_columns_sha256``, when declared, must match the config's
      actual ordered ``feature_columns`` — a contract that was written
      against a different feature set may not silently bind.
    """
    declared, source = declared_observation_contract(config)
    if not declared:
        return config, {
            "schema": APPLICATION_SCHEMA,
            "declared": False,
            "source": source,
            "applied": {},
            "unchanged": [],
            "reason": "no experiment observation contract is declared for "
                      "this run; the runtime observation fields are the "
                      "base config's own",
        }

    unknown = sorted(
        str(key) for key in declared
        if str(key) not in OBSERVATION_RUNTIME_KEYS
        and not str(key).startswith("$")
        and str(key) not in ("feature_columns_sha256",
                             "expected_flattened_dimension"))
    if unknown:
        raise ValueError(
            "observation_contract may only bind observation fields; "
            f"unknown keys {unknown} (allowed: "
            f"{sorted(OBSERVATION_RUNTIME_KEYS)})")

    pinned = declared.get("feature_columns_sha256")
    actual_columns_sha = feature_columns_sha256(config.get("feature_columns"))
    if pinned and str(pinned) != actual_columns_sha:
        raise ValueError(
            "observation_contract pins feature_columns_sha256="
            f"{pinned} but the resolved runtime config carries "
            f"{actual_columns_sha} — the contract was written against a "
            "different feature set")

    updated = dict(config)
    applied: Dict[str, Any] = {}
    unchanged = []
    for key in OBSERVATION_RUNTIME_KEYS:
        if key not in declared:
            continue
        before = config.get(key, _UNSET)
        after = declared[key]
        updated[key] = after
        if before == after and key in config:
            unchanged.append(key)
        else:
            applied[key] = {"from": before, "to": after}

    return updated, {
        "schema": APPLICATION_SCHEMA,
        "declared": True,
        "source": source,
        "applied": applied,
        "unchanged": sorted(unchanged),
        "feature_columns_sha256": actual_columns_sha,
        "feature_columns_pinned": bool(pinned),
        "reason": "the experiment contract's observation fields were bound "
                  "onto the runtime config before any env or model was "
                  "constructed",
    }


def verify_flattened_dimension(config: Mapping[str, Any],
                               observation_space) -> Dict[str, Any]:
    """C6 (finding 322): refuse ANY env whose flattened observation
    dimension differs from the declared contract expectation.

    Runs at every env construction (fit, eval splits, resume) — a
    driver-only check does not protect alternate entry paths. When no
    contract declares ``expected_flattened_dimension`` this is a no-op
    with a typed facts block, so sealed legacy runs stay byte-stable.
    """
    declared, source = declared_observation_contract(config)
    expected = (declared or {}).get("expected_flattened_dimension")
    if expected is None:
        return {"checked": False, "source": source}
    if observation_space is None:
        raise ValueError(
            "observation contract declares a flattened dimension but the "
            "constructed env exposes no observation_space — REFUSED")
    try:
        from gymnasium.spaces.utils import flatdim
        actual = int(flatdim(observation_space))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"cannot derive flattened dimension from the observation "
            f"space for the declared contract check: {exc}")
    if actual != int(expected):
        raise ValueError(
            f"observation contract declares flattened dimension "
            f"{expected} but the constructed env observes {actual} — "
            f"REFUSED before model creation (finding 322)")
    return {"checked": True, "source": source,
            "expected_flattened_dimension": int(expected),
            "actual_flattened_dimension": actual}


def validate_observation_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    """Refuse a run whose observation contract is not DECLARED.

    Reversed default, AUD-P1LR-20260815-235.  Until 2026-08-15 this
    function opened with ``if not config.get(
    "require_feature_aware_preprocessor", False): return`` — the guard
    was opt-in and defaulted OFF.  Every experiment that mattered
    declared neither field, so the validator returned on its first line
    and validated nothing while 64 unnormalized raw-price dimensions
    killed the actor's first layer across three campaigns.  A guard that
    defends only the configs which already asked to be defended is not a
    guard.

    The default is now the opposite: **absence of a declaration is a
    refusal**.  Exactly three outcomes exist, and each is mechanical —
    no approval step, no prompt, no human in the loop:

    ``require_feature_aware_preprocessor`` absent
        ``UNDECLARED_OBSERVATION_CONTRACT``.  The run refuses.  Nothing
        reaches an env or a model.
    ``require_feature_aware_preprocessor: true``
        the feature-aware contract is enforced exactly as before (same
        four checks, none weakened).
    ``require_feature_aware_preprocessor: false``
        an EXPLICIT, RECORDED opt-out.  It is legal — some legacy and
        baseline runs genuinely want the raw window — but it must be
        stated in the config together with a non-empty
        ``observation_contract_waiver_reason``.  A silent omission can
        no longer buy what a written waiver costs.

    Returns a typed facts block describing which outcome fired, so a
    caller can record WHY a run was allowed to build its observation.
    """
    declared = config.get("require_feature_aware_preprocessor")
    if declared is None:
        raise ValueError(
            f"{UNDECLARED_OBSERVATION_CONTRACT}: this run declares no "
            "observation contract. Set require_feature_aware_preprocessor="
            "true (engineered, causally scaled features reach the policy) "
            "or declare the explicit opt-out "
            "require_feature_aware_preprocessor=false together with a "
            "non-empty observation_contract_waiver_reason. Absence is no "
            "longer a pass: an undeclared observation contract is exactly "
            "how 64 unnormalized raw-price dimensions reached the actor "
            "and killed its first layer (AUD-P1LR-20260815-235)."
        )

    if not bool(declared):
        reason = str(
            config.get("observation_contract_waiver_reason") or "").strip()
        if len(reason) < MIN_WAIVER_REASON_CHARS:
            raise ValueError(
                f"{UNWAIVED_RAW_OBSERVATION}: "
                "require_feature_aware_preprocessor=false is an explicit "
                "opt-out from the feature-aware observation contract and "
                "must carry observation_contract_waiver_reason (at least "
                f"{MIN_WAIVER_REASON_CHARS} characters) naming why this "
                "run may feed unscaled observations to the policy; got "
                f"{reason!r}"
            )
        return {
            "schema": VALIDATION_SCHEMA,
            "outcome": "OBSERVATION_CONTRACT_WAIVED",
            "feature_aware": False,
            "include_price_window": bool(
                config.get("include_price_window", True)),
            "waiver_reason": reason,
        }

    plugin = str(config.get("preprocessor_plugin") or "").strip()
    if plugin != "feature_window_preprocessor":
        raise ValueError(
            "feature-aware observation contract requires "
            "preprocessor_plugin='feature_window_preprocessor'; "
            f"got {plugin!r}"
        )

    columns = config.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(
            "feature-aware observation contract requires a non-empty, "
            "materialized feature_columns list"
        )
    if len(columns) != len(set(map(str, columns))):
        raise ValueError("feature_columns contains duplicates")

    scaling = str(config.get("feature_scaling") or "").strip().lower()
    precomputed = bool(config.get("precomputed_causal_features", False))
    precomputed_hash = str(
        config.get("precomputed_feature_contract_sha256") or ""
    ).strip()
    if scaling == "none" and precomputed:
        if (
            len(precomputed_hash) != 64
            or any(character not in string.hexdigits for character in precomputed_hash)
        ):
            raise ValueError(
                "precomputed causal features require a 64-character hexadecimal "
                "precomputed_feature_contract_sha256"
            )
    elif scaling not in {"rolling_zscore", "expanding_zscore"}:
        raise ValueError(
            "feature-aware observation contract requires causal z-score "
            "scaling or a content-hashed precomputed causal feature contract; "
            f"got {scaling!r}"
        )
    if bool(config.get("include_price_window", True)):
        raise ValueError(
            "feature-aware observation contract forbids the unscaled raw "
            "price window; set include_price_window=false"
        )
    return {
        "schema": VALIDATION_SCHEMA,
        "outcome": "FEATURE_AWARE_OBSERVATION_CONTRACT",
        "feature_aware": True,
        "include_price_window": False,
        "preprocessor_plugin": plugin,
        "feature_scaling": scaling,
        "feature_column_count": len(columns),
        "feature_columns_sha256": feature_columns_sha256(columns),
    }
