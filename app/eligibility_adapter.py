"""Locate and consume the ONE eligibility gate.

The gate itself lives in `predictor/eligibility/` and is shared by
every repository in the program. This adapter only finds it and
asks it; it never reimplements a decision, because two
implementations of a permission rule are two different rules.

Resolution order:
  1. `config["eligibility_gate_path"]`
  2. `$CRISPDM_ELIGIBILITY_GATE`
  3. the sibling checkout `../predictor`

If a manifest IS configured and the gate cannot be located, the
caller refuses: proceeding ungated while claiming to be gated is
the failure this adapter exists to prevent.

This file is byte-identical across the consuming repositories; a
test asserts that, so the rule cannot drift between them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

GATE_PACKAGE = "eligibility"
ENV_VAR = "CRISPDM_ELIGIBILITY_GATE"
STATUS_GATED = "ELIGIBILITY_GATED"
STATUS_LEGACY = "LEGACY_NON_AUTHORITATIVE"


class EligibilityUnavailable(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _candidate_roots(config: dict) -> list[Path]:
    out = []
    configured = (config or {}).get("eligibility_gate_path")
    if configured:
        out.append(Path(configured))
    env = os.environ.get(ENV_VAR)
    if env:
        out.append(Path(env))
    # A sibling counts only if it actually CARRIES the gate.
    # Matching on the directory name alone finds look-alikes
    # (e.g. an internal `predictor/` package) and would refuse
    # for the wrong reason.
    here = Path(__file__).resolve()
    for parent in here.parents:
        sibling = parent.parent / "predictor"
        if (sibling / GATE_PACKAGE / "gate.py").is_file():
            out.append(sibling)
            break
    return out


def load_gate(config: dict | None = None):
    """Import the shared gate package, or raise."""
    tried = []
    for root in _candidate_roots(config or {}):
        root = root.resolve()
        pkg = root / GATE_PACKAGE
        if not (pkg / "gate.py").is_file():
            tried.append(str(pkg))
            continue
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import importlib
        mod = importlib.import_module(f"{GATE_PACKAGE}.gate")
        integ = importlib.import_module(
            f"{GATE_PACKAGE}.integration")
        return mod, integ
    raise EligibilityUnavailable(
        "the shared eligibility gate could not be located "
        f"(tried: {tried or 'no candidate roots'}) — set "
        f"config['eligibility_gate_path'] or ${ENV_VAR}")


def gate_subjects(config: dict, *, consumer: str,
                  subject_ids=None, scope: str | None = None,
                  subject_kind: str = "variable") -> dict:
    """Ask the gate for this run, exactly as `predictor` does."""
    if not (config or {}).get("eligibility_manifest"):
        return {"eligibility_status": STATUS_LEGACY,
                "consumer": consumer,
                "reason": "no reviewed eligibility manifest was "
                          "configured for this run; results are "
                          "not gated evidence"}
    _, integ = load_gate(config)
    return integ.gate_subjects(config, scope=scope,
                               subject_ids=subject_ids,
                               subject_kind=subject_kind,
                               consumer=consumer)


def gate_operator(config: dict, *, consumer: str,
                  operator_id: str, version: str,
                  code_digest: str, scope: str | None = None,
                  plugin_name: str | None = None) -> dict:
    if not (config or {}).get("eligibility_manifest"):
        return {"eligibility_status": STATUS_LEGACY,
                "consumer": consumer,
                "operator_id": operator_id,
                "reason": "no reviewed eligibility manifest was "
                          "configured; this transformation is "
                          "experimental, not licensed"}
    _, integ = load_gate(config)
    return integ.gate_operator(config, operator_id=operator_id,
                               version=version,
                               code_digest=code_digest,
                               scope=scope,
                               plugin_name=plugin_name,
                               consumer=consumer)


def describe(stamp: dict) -> str:
    if stamp.get("eligibility_status") == STATUS_GATED:
        return (f"eligibility: GATED by "
                f"{stamp.get('manifest_sha256', '?')[:12]} "
                f"scope={stamp.get('scope')}")
    return ("eligibility: LEGACY_NON_AUTHORITATIVE (no reviewed "
            "manifest configured)")
