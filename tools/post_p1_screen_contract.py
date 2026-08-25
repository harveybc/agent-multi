#!/usr/bin/env python3
"""Post-P1 screen materialization contract (SOTA-R01, SOTA-R02).

Three refusal gates, importable by every screen materializer:

1. causal eligibility — a policy whose fit or selection information
   reaches or passes an origin's score start may NOT enter that origin's
   comparison (C1). Frozen P1 artifacts are admissible only under the
   explicit `diagnostic_2024` label.
2. sealed absence — no development-screen config may materialize,
   reference or score any data at or after the sealed boundary (C2/§23).
3. release singleton — a sealed release packet must carry EXACTLY ONE
   decision-authoritative candidate; reported-only companions may not
   select, retune or trigger fallbacks (C2).

All checks raise ScreenContractViolation; materializers must not catch it.
"""
from dataclasses import dataclass, field

SEALED_START = "2025-01-01"


class ScreenContractViolation(SystemExit):
    pass


@dataclass
class PolicyIdentity:
    name: str
    fit_data_end: str          # ISO date: last bar date usable for fitting
    selection_info_end: str    # ISO date: latest data that informed selection
    labels: tuple = ()


@dataclass
class Origin:
    key: str
    fit_end: str
    score_start: str
    score_end: str


def check_causal_eligibility(policy, origin):
    """Refuse future-trained policies on earlier origins (SOTA-R01)."""
    if "diagnostic_2024" in policy.labels:
        if origin.score_start >= "2024-01-01" and origin.score_end <= "2024-12-31":
            return True
        raise ScreenContractViolation(
            f"REFUSED: policy {policy.name} is diagnostic_2024-labeled and "
            f"only admissible on the 2024 diagnostic, not origin {origin.key}")
    latest = max(policy.fit_data_end, policy.selection_info_end)
    if latest >= origin.score_start:
        raise ScreenContractViolation(
            f"REFUSED: policy {policy.name} fit/selection information "
            f"reaches {latest} >= score start {origin.score_start} of "
            f"origin {origin.key} (temporal lookahead)")
    return True


def check_sealed_absence(config):
    """Refuse any development config touching the sealed period (C2)."""
    def scan(node, path="config"):
        if isinstance(node, dict):
            for k, v in node.items():
                scan(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                scan(v, f"{path}[{i}]")
        elif isinstance(node, str) and len(node) >= 10:
            head = node[:10]
            if head >= SEALED_START and head[:4].isdigit() and head[4] == "-":
                raise ScreenContractViolation(
                    f"REFUSED: sealed-period date '{node}' at {path} in a "
                    f"development screen config")
    scan(config)
    roles = (config.get("roles") or {}) if isinstance(config, dict) else {}
    st = roles.get("sealed_test")
    if st and (st.get("csv") or st.get("materialized")):
        raise ScreenContractViolation(
            "REFUSED: sealed_test role is materialized in a development "
            "screen config")
    return True


def check_release_packet(candidates):
    """Exactly one decision-authoritative finalist (SOTA-R02)."""
    auth = [c for c in candidates if c.get("decision_authoritative")]
    if len(auth) != 1:
        raise ScreenContractViolation(
            f"REFUSED: sealed release packet must carry exactly one "
            f"decision-authoritative candidate, found {len(auth)}")
    for c in candidates:
        if not c.get("decision_authoritative"):
            for k in ("fallback_trigger", "retune_rule", "selection_rule"):
                if c.get(k):
                    raise ScreenContractViolation(
                        f"REFUSED: reported-only candidate "
                        f"{c.get('name', '?')} declares {k}; report-only "
                        f"companions may not select, retune or trigger "
                        f"fallbacks")
    return True
