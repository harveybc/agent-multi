#!/usr/bin/env python3
"""Post-P1 screen materialization contract (SOTA-R01/R02, C03/C04, F1/F3).

Refusal gates, importable by every screen materializer. All checks raise
ScreenContractViolation; materializers must not catch it.

1. causal eligibility — parsed ISO dates; fit information must respect
   the origin's declared fit boundary AND selection information must
   respect the origin's declared pre-score selection boundary. Frozen P1
   artifacts admissible only under the explicit `diagnostic_2024` label.
2. origin set validation — ISO-parsed, internally sane, ordered,
   non-overlapping score windows.
3. sealed absence — no development config may materialize, reference or
   score data at or after the sealed boundary.
4. release singleton — exactly one decision-authoritative finalist WITH
   frozen artifact/config/code/ensemble digests; report-only companions
   restricted to a typed allowlist (no authority smuggling).
5. observation identity — pre-model refusal binding the exact ordered
   feature list, count, digest and flattened shape of the effective
   config against the declared observation contract (SOTA-C01: the P1
   campaign executed 84 features incl. `typical_price` while the system
   contract declared 83).
"""
import hashlib
from dataclasses import dataclass
from datetime import date

SEALED_START = "2025-01-01"

REPORT_ONLY_ALLOWED_KEYS = frozenset(
    {"name", "metrics", "series_sha256", "notes", "decision_authoritative"})
FINALIST_REQUIRED_DIGESTS = ("artifact_sha256", "config_sha256",
                             "code_commit", "ensemble_rule_sha256")


class ScreenContractViolation(SystemExit):
    pass


def _parse_date(value, label):
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ScreenContractViolation(
            f"REFUSED: {label} is not a valid ISO date: {value!r}")


@dataclass
class PolicyIdentity:
    name: str
    fit_data_end: str          # last bar date usable for fitting
    selection_info_end: str    # latest data that informed selection
    labels: tuple = ()


@dataclass
class Origin:
    key: str
    fit_end: str               # declared fit boundary
    selection_boundary: str    # declared pre-score selection boundary
    score_start: str
    score_end: str


def validate_origins(origins):
    """Origins must parse, be internally sane, ordered, non-overlapping."""
    parsed = []
    for o in origins:
        fe = _parse_date(o.fit_end, f"{o.key}.fit_end")
        sb = _parse_date(o.selection_boundary, f"{o.key}.selection_boundary")
        ss = _parse_date(o.score_start, f"{o.key}.score_start")
        se = _parse_date(o.score_end, f"{o.key}.score_end")
        if not (fe <= sb < ss <= se):
            raise ScreenContractViolation(
                f"REFUSED: origin {o.key} boundaries not ordered "
                f"(need fit_end <= selection_boundary < score_start <= "
                f"score_end)")
        parsed.append((o.key, ss, se))
    for (ka, sa, ea), (kb, sb2, eb) in zip(parsed, parsed[1:]):
        if not sa < sb2:
            raise ScreenContractViolation(
                f"REFUSED: origins {ka},{kb} not in ascending score order")
        if sb2 <= ea:
            raise ScreenContractViolation(
                f"REFUSED: origins {ka},{kb} have overlapping score windows")
    return True


def check_causal_eligibility(policy, origin):
    """Refuse fit-boundary or selection-boundary violations (C03)."""
    if "diagnostic_2024" in policy.labels:
        ss = _parse_date(origin.score_start, "score_start")
        se = _parse_date(origin.score_end, "score_end")
        if ss >= date(2024, 1, 1) and se <= date(2024, 12, 31):
            return True
        raise ScreenContractViolation(
            f"REFUSED: policy {policy.name} is diagnostic_2024-labeled and "
            f"only admissible on the 2024 diagnostic, not origin {origin.key}")
    fit = _parse_date(policy.fit_data_end, f"{policy.name}.fit_data_end")
    sel = _parse_date(policy.selection_info_end,
                      f"{policy.name}.selection_info_end")
    fit_end = _parse_date(origin.fit_end, f"{origin.key}.fit_end")
    sel_bound = _parse_date(origin.selection_boundary,
                            f"{origin.key}.selection_boundary")
    score_start = _parse_date(origin.score_start, f"{origin.key}.score_start")
    if fit > fit_end:
        raise ScreenContractViolation(
            f"REFUSED: policy {policy.name} fit data end {fit} exceeds "
            f"origin {origin.key} fit boundary {fit_end}")
    if sel > sel_bound:
        raise ScreenContractViolation(
            f"REFUSED: policy {policy.name} selection information end {sel} "
            f"exceeds origin {origin.key} selection boundary {sel_bound}")
    if max(fit, sel) >= score_start:
        raise ScreenContractViolation(
            f"REFUSED: policy {policy.name} information reaches score start "
            f"of origin {origin.key} (temporal lookahead)")
    return True


def check_sealed_absence(config):
    """Refuse any development config touching the sealed period."""
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
    """Exactly one digest-frozen finalist; typed report-only schema (C04)."""
    auth = [c for c in candidates if c.get("decision_authoritative")]
    if len(auth) != 1:
        raise ScreenContractViolation(
            f"REFUSED: sealed release packet must carry exactly one "
            f"decision-authoritative candidate, found {len(auth)}")
    finalist = auth[0]
    for k in FINALIST_REQUIRED_DIGESTS:
        v = finalist.get(k)
        if not isinstance(v, str) or len(v) < 7:
            raise ScreenContractViolation(
                f"REFUSED: finalist {finalist.get('name', '?')} missing "
                f"frozen digest '{k}'")
    for c in candidates:
        if c.get("decision_authoritative"):
            continue
        extra = set(c) - REPORT_ONLY_ALLOWED_KEYS
        if extra:
            raise ScreenContractViolation(
                f"REFUSED: report-only candidate {c.get('name', '?')} "
                f"carries non-allowlisted keys {sorted(extra)}; report-only "
                f"entries cannot encode selection/fallback authority")
    return True


def feature_list_digest(columns):
    return hashlib.sha256("\n".join(columns).encode()).hexdigest()


def expected_flattened_shape(contract):
    n = len(contract["feature_columns"])
    if contract.get("include_price_window"):
        n += 2
    size = contract["window_size"] * n
    if contract.get("include_agent_state"):
        size += 4
    return size


def check_observation_identity(effective_config, contract):
    """Pre-model refusal on any observation-identity drift (SOTA-C01/F1).

    Binds ordered feature list, count, digest, price-window flag and the
    flattened shape implied by the contract arithmetic.
    """
    exe = list(effective_config.get("feature_columns") or [])
    decl = list(contract.get("feature_columns") or [])
    if len(exe) != len(decl):
        extra = sorted(set(exe) - set(decl))
        missing = sorted(set(decl) - set(exe))
        raise ScreenContractViolation(
            f"REFUSED: executed feature count {len(exe)} != declared "
            f"{len(decl)} (extra={extra}, missing={missing})")
    if exe != decl:
        raise ScreenContractViolation(
            "REFUSED: executed feature ORDER differs from declared contract "
            "(same length); observation tensors are not comparable")
    if feature_list_digest(exe) != feature_list_digest(decl):
        raise ScreenContractViolation("REFUSED: feature digest mismatch")
    for flag in ("include_price_window", "include_agent_state",
                 "window_size"):
        if effective_config.get(flag) != contract.get(flag):
            raise ScreenContractViolation(
                f"REFUSED: observation flag '{flag}' differs "
                f"(executed={effective_config.get(flag)!r}, "
                f"declared={contract.get(flag)!r})")
    return {"feature_count": len(exe),
            "feature_digest": feature_list_digest(exe),
            "flattened_shape": expected_flattened_shape(contract)}


def executed_observation_identity(launch_manifest):
    """Terminal-aggregation labeling helper (F1.4): report the EXECUTED
    identity of a P1 arm honestly — never rewrite artifacts."""
    cfg = launch_manifest["effective_config"]
    cols = list(cfg["feature_columns"])
    n = len(cols)
    flat = cfg.get("window_size", 32) * (
        n + (2 if cfg.get("include_price_window") else 0)) + 4
    return {
        "executed_feature_count": n,
        "executed_feature_digest": feature_list_digest(cols),
        "executed_feature_columns": cols,
        "executed_include_price_window": bool(
            cfg.get("include_price_window")),
        "executed_flattened_shape": flat,
    }
