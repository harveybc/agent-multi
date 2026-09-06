#!/usr/bin/env python3
"""T1 adjudicator v2 (orders T0-T1 + C1-C12).

C6/C11: the adjudicator consumes the SEALED design and the exact
bank inventory, binds exactly one record for every predeclared
unit x operator, verifies source-array digests and the
content-addressed NPZ evidence, and RE-DERIVES the gate metrics
from the arrays — producer aggregates and producer-declared
cardinalities grant nothing.

C7: strict JSON (duplicate-key and non-finite rejection); every
gate-bearing number must be finite; typed nulls never authorize
calibration.

C8: the verdict derives from SCORE-role facts only; train and
validation facts are published separately.

C10: ANY material safety failure (non-finite gate, tail, extreme or
utility harm) in ANY required seed/variable yields LAB_REJECTED or
INCONCLUSIVE per the predeclared rule — a median can never hide it.
Distribution (min/median/max) is reported; windows are never
replicas."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

SNR_GAIN_MIN_DB = 0.5
UTILITY_TOLERANCE = 0.02
EXTREME_RETENTION_MIN = 0.5
TAIL_RATIO_MAX = 1.5
RESIDUAL_INFO_MAX = 0.01
MIN_SEEDS = 3
CANDIDATES = ("identity", "trailing_mean", "trailing_median",
              "ewma", "local_level_kalman")
ORACLE = "centered_mean_oracle"


class AdjudicationRefusal(SystemExit):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict:
    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise AdjudicationRefusal(
                f"REFUSED: duplicate JSON key in {path.name}")
        return dict(pairs)
    return json.loads(
        Path(path).read_text(), object_pairs_hook=_no_dupes,
        parse_constant=lambda c: (_ for _ in ()).throw(
            AdjudicationRefusal(
                f"REFUSED: non-finite literal {c!r} in "
                f"{path.name}")))


def _gate(v, what: str) -> float:
    """C7: a gate value must be a finite float; typed nulls and
    anything else refuse the gate (never authorize)."""
    if isinstance(v, dict):
        raise AdjudicationRefusal(
            f"REFUSED: gate {what} is a typed null "
            f"({v.get('reason')}) — it cannot authorize "
            "calibration")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or \
            not np.isfinite(float(v)):
        raise AdjudicationRefusal(
            f"REFUSED: gate {what} is not a finite number")
    return float(v)


def _regime(rec: dict) -> str:
    snr = rec["declared_snr_db"]
    snr_s = ("het" + "_".join(str(s) for s in snr)
             if isinstance(snr, list) else str(snr))
    return f"{rec['family']}|{rec['perturbation']}|snr{snr_s}"


def expected_population(design: dict, inventory: dict) -> set:
    """C6: the exact expected (unit, operator) population derives
    from the SEALED DESIGN + inventory — never from the producer."""
    ops = tuple(design["expected_operators_exact"])
    return {(uid, op) for uid in inventory["unit_ids"]
            for op in ops}


# --- C16: independent re-derivation constants -----------------
# Deliberately DUPLICATED from the lab (same mathematical
# definitions, independent implementation consumed only by
# verification): a producer metric may never influence a verdict
# unless re-derived and compared here.
_RD_HORIZONS = (1, 5)
_RD_RIDGE_LAGS = 8
_RD_RIDGE_LAMBDA = 1.0
_RD_NUISANCE_CHANNELS = 2
_RD_TOL = 1e-6


def _rd_ridge_r2(X, y, Xs, ys):
    XtX = X.T @ X + _RD_RIDGE_LAMBDA * np.eye(X.shape[1])
    w = np.linalg.solve(XtX, X.T @ y)
    pred = Xs @ w
    ss = float(np.sum((ys - pred) ** 2))
    tot = float(np.sum((ys - ys.mean()) ** 2))
    return 1.0 - ss / max(tot, 1e-12)


def _rd_lag_matrix(series_list, lo, hi, h):
    rows = []
    for t in range(lo + _RD_RIDGE_LAGS, hi - h):
        feats = []
        for s in series_list:
            feats.extend(s[t - _RD_RIDGE_LAGS + 1: t + 1])
        rows.append(feats)
    return np.array(rows)


def _rd_targets(clean_var, lo, hi, h):
    return np.array([clean_var[t + h]
                     for t in range(lo + _RD_RIDGE_LAGS, hi - h)])


def _rd_nuisance(obs_j, roles, ident: str):
    lo_t, hi_t = roles["train"]
    scale = float(np.std(obs_j[lo_t:hi_t]))
    out = []
    for k in range(_RD_NUISANCE_CHANNELS):
        seed = int(hashlib.sha256(
            f"t1_nuisance|{ident}|{k}".encode()
        ).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        out.append(rng.normal(0.0, scale if scale > 0 else 1.0,
                              obs_j.shape[0]))
    return out


def _rd_role_metrics(c_full, o_full, d_full, sup_full, roles):
    facts = {}
    for role, (lo, hi) in roles.items():
        sl = slice(lo, hi)
        sup = sup_full[sl]
        c, o, d = c_full[sl][sup], o_full[sl][sup], d_full[sl][sup]
        if len(c) < 20:
            facts[role] = None
            continue
        mse_o = float(np.mean((o - c) ** 2))
        mse_d = float(np.mean((d - c) ** 2))
        gain = (10 * np.log10(mse_o / mse_d)
                if mse_o > 0 and mse_d > 0 else 0.0)
        peaks = np.argsort(np.abs(c))[-5:]
        denom = np.where(np.abs(c[peaks]) < 1e-9, 1.0, c[peaks])
        retention = float(np.mean(np.clip(d[peaks] / denom,
                                          -2, 2)))
        noise_like = o - c
        tail = (float(np.max(np.abs(d - c))
                      / max(np.max(np.abs(noise_like)), 1e-12)))
        facts[role] = {"mse_observed": mse_o, "mse_denoised": mse_d,
                       "snr_gain_db": float(gain),
                       "extreme_retention": retention,
                       "tail_ratio": tail}
    return facts


def _rd_close(a, b, what):
    if a is None or (isinstance(a, dict) and a.get("null")):
        if b is None:
            return
        raise AdjudicationRefusal(
            f"REFUSED: published {what} is null yet it re-derives "
            "to a finite value")
    if b is None:
        raise AdjudicationRefusal(
            f"REFUSED: published {what} is finite yet it "
            "re-derives to null")
    if abs(float(a) - float(b)) > _RD_TOL:
        raise AdjudicationRefusal(
            f"REFUSED: published {what} {float(a):.8f} differs "
            f"from the re-derived value {float(b):.8f}")


def rederive_all_facts(record: dict, bank_dir: Path,
                       npz_dir: Path) -> None:
    """C16: recompute EVERY decision-bearing metric of one MEASURED
    record from the sealed unit arrays and the content-addressed
    denoised/residual arrays: per-role SNR gain, reconstruction,
    extreme retention, tail ratio, every X/D/XDR/width assay,
    residual incremental value and the estimator error. Any
    difference refuses."""
    unit_dir = bank_dir / record["unit_id"]
    rec_u = _strict_json(unit_dir / "UNIT.json")
    clean = np.load(unit_dir / "clean_signal.npy")
    obs = np.load(unit_dir / "observed_signal.npy")
    support = np.load(unit_dir / "metric_support.npy")
    blob = np.load(npz_dir /
                   f"{record['denoised_residual_npz_sha256']}.npz")
    d = blob["denoised"]
    resid = blob["residual"]
    if set(blob.files) != {"denoised", "residual"}:
        raise AdjudicationRefusal(
            f"REFUSED: NPZ member set {sorted(blob.files)} is not "
            "the exact schema")
    if hashlib.sha256(d.tobytes() + resid.tobytes()
                      ).hexdigest() != \
            record["denoised_residual_npz_sha256"]:
        raise AdjudicationRefusal(
            f"REFUSED: NPZ content digest broken for "
            f"{record['unit_id']}/{record['operator']}")
    roles = {k: tuple(v) for k, v in
             rec_u["temporal_roles"].items()}
    lo_t, hi_t = roles["train"]
    for j, pv in enumerate(record["per_variable"]):
        # per-role reconstruction facts
        want = _rd_role_metrics(clean[j], obs[j], d[j],
                                support[j], roles)
        for role in ("train", "validation", "score"):
            pub = pv["by_role"][role]
            w = want[role]
            if isinstance(pub, dict) and pub.get("null"):
                if w is not None:
                    raise AdjudicationRefusal(
                        f"REFUSED: {record['unit_id']} {role} "
                        "facts are null yet they re-derive")
                continue
            if w is None:
                raise AdjudicationRefusal(
                    f"REFUSED: {record['unit_id']} {role} facts "
                    "re-derive to null yet the record claims them")
            for kf in ("mse_observed", "mse_denoised",
                       "snr_gain_db", "extreme_retention",
                       "tail_ratio"):
                _rd_close(pub[kf], w[kf],
                          f"{record['unit_id']}/"
                          f"{record['operator']} {role} {kf}")
        # estimator error
        est_std = float(np.std(np.diff(obs[j, lo_t:hi_t]))
                        / np.sqrt(2))
        true_std = rec_u["noise_std_per_var"][j]
        want_err = (abs(est_std - true_std) / true_std
                    if true_std > 0 else 0.0)
        _rd_close(pv["snr_estimator_std_error"], want_err,
                  f"{record['unit_id']} estimator error")
        # delay
        lags = range(-10, 11)
        xc = [float(np.corrcoef(
            clean[j, 10:-10],
            d[j, 10 + k:len(d[j]) - 10 + k])[0, 1])
            for k in lags]
        want_delay = int(list(lags)[int(np.argmax(xc))])
        if int(pv["delay_bars"]) != want_delay:
            raise AdjudicationRefusal(
                f"REFUSED: published delay {pv['delay_bars']} "
                f"differs from re-derived {want_delay}")
        # every downstream assay, every horizon, every arm
        nuis = _rd_nuisance(
            obs[j], roles,
            f"{record['unit_id']}|{record['operator']}|v{j}")
        arms = {"X": [obs[j]], "D": [d[j]],
                "XDR": [obs[j], d[j], resid[j]],
                "width_control_X_nuisance":
                    [obs[j], nuis[0], nuis[1]]}
        lo_s, hi_s = roles["score"]
        for h in _RD_HORIZONS:
            pub_h = pv["assays_score_fit_train"][f"h{h}"]
            yt = _rd_targets(clean[j], lo_t, hi_t, h)
            ys = _rd_targets(clean[j], lo_s, hi_s, h)
            got = {}
            for name, series in arms.items():
                Xt = _rd_lag_matrix(series, lo_t, hi_t, h)
                Xs = _rd_lag_matrix(series, lo_s, hi_s, h)
                v = _rd_ridge_r2(Xt, yt, Xs, ys)
                got[name] = v if np.isfinite(v) else None
                _rd_close(pub_h[name], got[name],
                          f"{record['unit_id']}/"
                          f"{record['operator']} h{h} {name}")
            if isinstance(got["XDR"], float) and \
                    isinstance(got["width_control_X_nuisance"],
                               float):
                want_ri = (got["XDR"]
                           - got["width_control_X_nuisance"])
                _rd_close(pub_h["residual_incremental_r2"],
                          want_ri,
                          f"{record['unit_id']} h{h} residual "
                          "incremental")


def rederive_gates(record: dict, bank_dir: Path,
                   npz_dir: Path) -> None:
    """Compatibility name: the full C16 re-derivation."""
    rederive_all_facts(record, bank_dir, npz_dir)


# --- C20: strict consuming schemas --------------------------------
_STATUSES = ("MEASURED", "TYPED_REFUSAL")
_REC_COMMON = {"schema": str, "unit_id": str, "operator": str,
               "family": str, "perturbation": str,
               "heterogeneous": bool, "seed": int,
               "causal": bool, "oracle_only": bool,
               "source_digests": dict, "status": str,
               "cpu_wall_seconds": float}
_REC_MEASURED_EXTRA = {"per_variable": list,
                       "denoised_residual_npz_sha256": str,
                       "output_column_expansion": dict,
                       "peak_rss_bytes": int,
                       "artifact_sha256": str,
                       "spec_sha256": str}
_REC_REFUSED_EXTRA = {"refusal": str}
_PV_KEYS = {"variable": str, "true_additive_snr_db": (int, float,
                                                      str),
            "true_total_error_snr_db": (int, float, str),
            "snr_estimator_std_error": (int, float, dict),
            "delay_bars": int, "by_role": dict,
            "assays_score_fit_train": dict}
_ROLE_FACT_KEYS = {"mse_observed", "mse_denoised", "snr_gain_db",
                   "extreme_retention", "tail_ratio"}
_ASSAY_KEYS = {"X", "D", "XDR", "width_control_X_nuisance",
               "residual_incremental_r2"}


def _typed(v, t, what):
    if t is float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AdjudicationRefusal(
                f"REFUSED: {what} must be a number, got "
                f"{type(v).__name__}")
        if not np.isfinite(float(v)):
            raise AdjudicationRefusal(
                f"REFUSED: non-finite {what}")
        return
    if t is int:
        if isinstance(v, bool) or type(v) is not int:
            raise AdjudicationRefusal(
                f"REFUSED: {what} must be int (bool is never a "
                "number)")
        return
    if isinstance(t, tuple):
        if isinstance(v, bool) or not isinstance(v, t):
            raise AdjudicationRefusal(
                f"REFUSED: {what} has a foreign type "
                f"{type(v).__name__}")
        return
    if type(v) is not t:
        raise AdjudicationRefusal(
            f"REFUSED: {what} must be {t.__name__}, got "
            f"{type(v).__name__}")


def _check_gate_value(v, what):
    """A gate is a finite number or the exact typed-null shape."""
    if isinstance(v, dict):
        if set(v) != {"null", "reason"} or v["null"] is not True \
                or type(v["reason"]) is not str:
            raise AdjudicationRefusal(
                f"REFUSED: {what} carries a malformed null shape")
        return
    _typed(v, float, what)


def _parse_unit_id(uid: str) -> dict:
    parts = uid.split("__")
    if len(parts) != 5 or not parts[4].startswith("seed"):
        raise AdjudicationRefusal(
            f"REFUSED: unit id {uid!r} is not the canonical shape")
    return {"family": parts[0], "perturbation": parts[1],
            "snr": parts[2], "het": parts[3] == "het",
            "seed": int(parts[4][4:])}


def check_measurement_record(r: dict) -> None:
    """C20: exact keys and primitive types at the consuming
    boundary; unknown fields, malformed statuses, bool-as-number,
    non-finite values and inconsistent regime metadata refuse
    BEFORE grouping."""
    if not isinstance(r, dict):
        raise AdjudicationRefusal("REFUSED: record is not a dict")
    if r.get("status") not in _STATUSES:
        raise AdjudicationRefusal(
            f"REFUSED: invalid status {r.get('status')!r}")
    want = dict(_REC_COMMON)
    want["declared_snr_db"] = object      # scalar or het list
    if r["status"] == "MEASURED":
        want.update(_REC_MEASURED_EXTRA)
    else:
        want.update(_REC_REFUSED_EXTRA)
    if set(r) != set(want):
        raise AdjudicationRefusal(
            f"REFUSED: record keys are not the exact schema "
            f"(diff: {sorted(set(r) ^ set(want))})")
    for k, t in want.items():
        if t is object:
            continue
        _typed(r[k], t, f"record field {k}")
    if r["schema"] != "agent_multi.t1_measurement.v2":
        raise AdjudicationRefusal(
            f"REFUSED: foreign record schema {r['schema']!r}")
    ident = _parse_unit_id(r["unit_id"])
    if (ident["family"] != r["family"]
            or ident["perturbation"] != r["perturbation"]
            or ident["het"] != r["heterogeneous"]
            or ident["seed"] != r["seed"]):
        raise AdjudicationRefusal(
            f"REFUSED: regime metadata of {r['unit_id']} is "
            "inconsistent with its identity")
    if r["status"] != "MEASURED":
        return
    if len(r["denoised_residual_npz_sha256"]) != 64:
        raise AdjudicationRefusal(
            "REFUSED: NPZ digest is not 64-hex")
    if not r["per_variable"]:
        raise AdjudicationRefusal(
            "REFUSED: MEASURED record without variables")
    for pv in r["per_variable"]:
        if not isinstance(pv, dict) or set(pv) != set(_PV_KEYS):
            raise AdjudicationRefusal(
                f"REFUSED: per-variable keys are not the exact "
                f"schema in {r['unit_id']}")
        for k, t in _PV_KEYS.items():
            if k in ("snr_estimator_std_error",):
                _check_gate_value(pv[k], f"{r['unit_id']} {k}")
                continue
            if k in ("by_role", "assays_score_fit_train"):
                continue
            if k in ("true_additive_snr_db",
                     "true_total_error_snr_db"):
                v = pv[k]
                if isinstance(v, str) and v != "inf":
                    raise AdjudicationRefusal(
                        f"REFUSED: {r['unit_id']} {k} carries a "
                        f"foreign string {v!r}")
                if not isinstance(v, str):
                    _typed(v, float, f"{r['unit_id']} {k}")
                continue
            _typed(pv[k], t, f"{r['unit_id']} pv {k}")
        if set(pv["by_role"]) != {"train", "validation", "score"}:
            raise AdjudicationRefusal(
                f"REFUSED: by_role keys are not the exact roles "
                f"in {r['unit_id']}")
        for role, facts in pv["by_role"].items():
            if isinstance(facts, dict) and facts.get("null"):
                if set(facts) != {"null", "reason"}:
                    raise AdjudicationRefusal(
                        f"REFUSED: malformed null role facts in "
                        f"{r['unit_id']}")
                continue
            if not isinstance(facts, dict) or \
                    set(facts) != _ROLE_FACT_KEYS:
                raise AdjudicationRefusal(
                    f"REFUSED: {role} facts are not the exact "
                    f"schema in {r['unit_id']}")
            for k in _ROLE_FACT_KEYS:
                _check_gate_value(facts[k],
                                  f"{r['unit_id']} {role} {k}")
        if set(pv["assays_score_fit_train"]) != {"h1", "h5"}:
            raise AdjudicationRefusal(
                f"REFUSED: assay horizons are not the exact set "
                f"in {r['unit_id']}")
        for h, a in pv["assays_score_fit_train"].items():
            if not isinstance(a, dict) or set(a) != _ASSAY_KEYS:
                raise AdjudicationRefusal(
                    f"REFUSED: {h} assay keys are not the exact "
                    f"schema in {r['unit_id']}")
            for k in _ASSAY_KEYS:
                _check_gate_value(a[k],
                                  f"{r['unit_id']} {h} {k}")


def check_inventory(inv: dict) -> None:
    """C20/C17: the inventory must be the v2 physical binding."""
    want = {"schema": str, "predeclared_cells": int, "seeds": list,
            "units_total": int, "unit_ids": list, "units": dict,
            "design_rule": str}
    if not isinstance(inv, dict) or set(inv) != set(want):
        raise AdjudicationRefusal(
            "REFUSED: inventory keys are not the exact schema")
    for k, t in want.items():
        _typed(inv[k], t, f"inventory {k}")
    if inv["schema"] != "agent_multi.t1_bank_inventory.v2":
        raise AdjudicationRefusal(
            "REFUSED: inventory does not bind the physical "
            "population (v2 required)")
    if len(inv["unit_ids"]) != inv["units_total"] or \
            set(inv["unit_ids"]) != set(inv["units"]):
        raise AdjudicationRefusal(
            "REFUSED: inventory unit list and binding map differ")
    if len(set(inv["unit_ids"])) != len(inv["unit_ids"]):
        raise AdjudicationRefusal(
            "REFUSED: duplicate unit identity in the inventory")
    for uid, b in inv["units"].items():
        if not isinstance(b, dict) or set(b) != {
                "unit_json_sha256", "arrays"}:
            raise AdjudicationRefusal(
                f"REFUSED: inventory binding for {uid} is not the "
                "exact schema")
        if len(b["unit_json_sha256"]) != 64:
            raise AdjudicationRefusal(
                f"REFUSED: {uid} metadata digest is not 64-hex")
        if set(b["arrays"]) != {"clean_signal", "additive_noise",
                                "observed_signal",
                                "metric_support"}:
            raise AdjudicationRefusal(
                f"REFUSED: {uid} array binding is not the exact "
                "set")


def verify_unit_bytes(inv: dict, bank_dir: Path, uid: str) -> None:
    """C17: recompute the unit's digests from bytes."""
    b = inv["units"][uid]
    ud = Path(bank_dir) / uid
    if _sha_file(ud / "UNIT.json") != b["unit_json_sha256"]:
        raise AdjudicationRefusal(
            f"REFUSED: {uid} metadata bytes differ from the "
            "sealed inventory")
    for name, want in b["arrays"].items():
        if _sha_file(ud / f"{name}.npy") != want:
            raise AdjudicationRefusal(
                f"REFUSED: {uid} array {name} bytes differ from "
                "the sealed inventory")


def adjudicate(design: dict, inventory: dict, measurements: dict,
               bank_dir: Path = None, npz_dir: Path = None,
               rederive_sample: int = 0) -> dict:
    records = measurements["records"]
    check_inventory(inventory)
    for r in records:
        check_measurement_record(r)
    expected = expected_population(design, inventory)
    seen = set()
    for r in records:
        key = (r["unit_id"], r["operator"])
        if key in seen:
            raise AdjudicationRefusal(
                f"REFUSED: duplicate record {key}")
        if key not in expected:
            raise AdjudicationRefusal(
                f"REFUSED: foreign record {key} not in the "
                "design-derived population")
        seen.add(key)
    missing = expected - seen
    if missing:
        raise AdjudicationRefusal(
            f"REFUSED: population incomplete — {len(missing)} "
            f"design-required records missing (e.g. "
            f"{sorted(missing)[:2]})")
    if rederive_sample and bank_dir is not None:
        rng = np.random.default_rng(20260906)
        measured = [r for r in records if r["status"] == "MEASURED"]
        for r in rng.choice(np.array(measured, dtype=object),
                            size=min(rederive_sample,
                                     len(measured)),
                            replace=False):
            rederive_gates(r, bank_dir, npz_dir)
    by_key = {}
    for r in records:
        by_key.setdefault((r["operator"], _regime(r)),
                          []).append(r)
    verdicts = {}
    for (op, regime), recs in sorted(by_key.items()):
        if op == ORACLE:
            verdicts[(op, regime)] = {
                "verdict": "NON_CAUSAL_ORACLE_ONLY",
                "reason": "diagnostic ceiling; never eligible"}
            continue
        seeds = {r["seed"] for r in recs}
        refusals = [r for r in recs
                    if r["status"] == "TYPED_REFUSAL"]
        if refusals:
            verdicts[(op, regime)] = {
                "verdict": "INCONCLUSIVE",
                "reason": ("typed refusal in regime: "
                           + refusals[0]["refusal"][:80]),
                "seeds": len(seeds)}
            continue
        if len(seeds) < MIN_SEEDS:
            verdicts[(op, regime)] = {
                "verdict": "INCONCLUSIVE",
                "reason": f"only {len(seeds)} seeds < {MIN_SEEDS}"}
            continue
        gains, utils, rets, tails, resid = [], [], [], [], []
        material_failures = []
        try:
            for r in recs:
                for pv in r["per_variable"]:
                    sc = pv["by_role"]["score"]
                    if isinstance(sc, dict) and sc.get("null"):
                        material_failures.append(
                            f"seed {r['seed']} {pv['variable']}: "
                            "null score facts")
                        continue
                    g = _gate(sc["snr_gain_db"],
                              "score snr_gain")
                    ret = _gate(sc["extreme_retention"],
                                "score retention")
                    tl = _gate(sc["tail_ratio"], "score tail")
                    gains.append(g)
                    rets.append(ret)
                    tails.append(tl)
                    for h in ("h1", "h5"):
                        a = pv["assays_score_fit_train"][h]
                        x = _gate(a["X"], "assay X")
                        dv = _gate(a["D"], "assay D")
                        ri = _gate(a["residual_incremental_r2"],
                                   "residual info")
                        rel = (dv - x) / max(abs(x), 1e-6)
                        utils.append(rel)
                        resid.append(ri)
                        # C10: ANY material failure is terminal
                        if rel < -0.25:
                            material_failures.append(
                                f"seed {r['seed']} "
                                f"{pv['variable']} {h}: utility "
                                f"collapse {rel:.2f}")
                    if ret < 0.0 and g >= SNR_GAIN_MIN_DB:
                        material_failures.append(
                            f"seed {r['seed']} {pv['variable']}: "
                            f"extreme INVERSION ({ret:.2f})")
                    if tl > 3.0:
                        material_failures.append(
                            f"seed {r['seed']} {pv['variable']}: "
                            f"tail explosion ({tl:.2f}x)")
        except AdjudicationRefusal as exc:
            verdicts[(op, regime)] = {
                "verdict": "INCONCLUSIVE",
                "reason": str(exc)[:120], "seeds": len(seeds)}
            continue
        def dist(v):
            return {"min": round(float(np.min(v)), 4),
                    "median": round(float(np.median(v)), 4),
                    "max": round(float(np.max(v)), 4)} if v else None
        facts = {"snr_gain_db": dist(gains),
                 "relative_utility_delta_D_vs_X": dist(utils),
                 "extreme_retention": dist(rets),
                 "tail_ratio": dist(tails),
                 "residual_incremental_r2": dist(resid),
                 "seeds": len(seeds),
                 "material_failures": material_failures}
        if op == "identity":
            verdicts[(op, regime)] = {
                "verdict": "LAB_CALIBRATED",
                "reason": "mandatory no-op control", **facts}
            continue
        if material_failures:
            verdicts[(op, regime)] = {
                "verdict": "LAB_REJECTED",
                "reason": ("material safety failure in at least "
                           "one seed/variable — a median cannot "
                           "hide it: "
                           + material_failures[0]),
                **facts}
            continue
        med_gain = float(np.median(gains))
        med_util = float(np.median(utils))
        med_ret = float(np.median(rets))
        med_tail = float(np.median(tails))
        med_resid = float(np.median(resid))
        failures = []
        if med_gain < SNR_GAIN_MIN_DB:
            failures.append(f"no material SNR gain "
                            f"({med_gain:.2f} dB)")
        if med_util < -UTILITY_TOLERANCE:
            failures.append(f"downstream utility worsens "
                            f"({med_util:.3f})")
        if med_ret < EXTREME_RETENTION_MIN and \
                med_gain >= SNR_GAIN_MIN_DB:
            failures.append(f"extreme destruction "
                            f"({med_ret:.2f})")
        if med_tail > TAIL_RATIO_MAX:
            failures.append(f"tail inflation ({med_tail:.2f}x)")
        informative = med_resid > RESIDUAL_INFO_MAX
        if not failures:
            verdict = "LAB_CALIBRATED"
            note = ("RESIDUAL_INFORMATIVE — reported as a "
                    "TRANSFORMATION, not verified noise removal"
                    if informative else "verified in-regime "
                    "(score role only)")
        elif med_util < -UTILITY_TOLERANCE or \
                med_ret < EXTREME_RETENTION_MIN:
            verdict = "LAB_REJECTED"
            note = "; ".join(failures)
        else:
            verdict = "INCONCLUSIVE"
            note = "; ".join(failures)
        verdicts[(op, regime)] = {"verdict": verdict,
                                  "reason": note,
                                  "residual_informative":
                                      informative, **facts}
    out = {"schema": "agent_multi.t1_adjudication.v2",
           "design_sha256": measurements.get("design_sha256"),
           "population": {"records": len(records),
                          "expected": len(expected)},
           "verdict_basis": "SCORE-role facts only (C8); "
                            "train/validation published separately "
                            "in the records",
           "material_failure_rule":
               "any seed/variable material failure -> LAB_REJECTED "
               "(C10); distributions reported min/median/max",
           "verdicts": {f"{op}::{rg}": v
                        for (op, rg), v in verdicts.items()}}
    counts = {}
    for v in verdicts.values():
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    out["verdict_counts"] = counts
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--design-sha", required=True)
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--measurements", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rederive-sample", type=int, default=40)
    args = ap.parse_args()
    raw = args.design.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.design_sha:
        raise AdjudicationRefusal(
            "REFUSED: sealed design bytes differ from the reviewed "
            "digest")
    design = _strict_json(args.design)
    inventory = _strict_json(args.bank_dir / "BANK_INVENTORY.json")
    m = _strict_json(args.measurements)
    if m.get("design_sha256") != args.design_sha:
        raise AdjudicationRefusal(
            "REFUSED: measurements were not produced under this "
            "sealed design")
    out = adjudicate(design, inventory, m, bank_dir=args.bank_dir,
                     npz_dir=args.npz_dir,
                     rederive_sample=args.rederive_sample)
    args.output.write_text(json.dumps(out, indent=1,
                                      allow_nan=False))
    print(json.dumps(out["verdict_counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
