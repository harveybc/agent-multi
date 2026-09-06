#!/usr/bin/env python3
"""T2.1 causal assay harness — ONE shared path for all arms.

Arms: X (identity), D(X) (a T1 LAB_CALIBRATED causal configuration
selected from the sealed T1 v4 record, never from T2 outcomes),
[X, D, X-D], and a matched-capacity width control (train-frozen
independent channels). Assays: seasonal naive, a lagged regularized
linear model, and one small neural model. Splits, fit windows,
transform fitting, model budgets, seed tape and score rows are
identical across paired arms; nothing observes held-out rows or
future timestamps (fits go through the T0 causal operator contract).

Natural data does not reveal clean signal: this harness reports
downstream forecasting utility, calibration, extreme preservation
and cost ONLY. It never emits a noise claim or a true-SNR field,
and it never emits an eligibility field — eligibility belongs to
the sealed confirmatory verifier, which requires the immutable
T2.2 design and a lawful public bank (absent locally: the census
verdict is CONFIRMATORY_BANK_UNAVAILABLE, so confirmatory calls
refuse with PUBLIC_DATA_REQUIRED)."""
import hashlib
import io
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

RIDGE_LAGS = 8
RIDGE_LAMBDA = 1.0
NUISANCE_CHANNELS = 2
SPLITS = (0.6, 0.2, 0.2)          # train / validation / score
SEED_TAPE = (11, 12, 13)
MLP_BUDGET = {"hidden_layer_sizes": (16,), "max_iter": 200,
              "tol": 1e-4}
# Selected FROM the sealed T1 v4 record (ewma is LAB_CALIBRATED in
# the additive regimes relevant to smooth natural series), fixed
# BEFORE any T2 score exists.
T1_OPERATOR = {"kind": "ewma", "params": {"alpha": 0.3}}
# Predeclared seasonal periods per development unit (design-time
# knowledge, never estimated from held-out rows).
SEASONAL_PERIOD = {"sm_co2": 52, "sm_sunspots": 11, "sm_nile": 1}


class HarnessRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def resolve_preprocessor_root() -> Path:
    root = os.environ.get("B4_T1_PREPROCESSOR_ROOT")
    if not root:
        raise HarnessRefusal(
            "set B4_T1_PREPROCESSOR_ROOT to the accepted "
            "preprocessor checkout")
    return Path(root)


def load_co():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "t2_causal_operators",
        resolve_preprocessor_root() / "app/causal_operators.py")
    co = ilu.module_from_spec(spec)
    spec.loader.exec_module(co)
    return co


def load_task_unit(census: dict, unit_id: str) -> dict:
    """T2.0.2: a task is reconstructed FROM BYTES whose digest the
    census records. A dataframe supplied separately from its bytes
    is not evidence and has no entry point here."""
    units = census["development_only_units"]
    if unit_id not in units:
        raise HarnessRefusal(f"unit {unit_id!r} is not in the "
                             "census")
    meta = units[unit_id]
    if meta["status"] != "PRESENT_DEVELOPMENT_ONLY":
        raise HarnessRefusal(f"unit {unit_id!r} is not physically "
                             "present")
    import importlib
    mod = importlib.import_module(
        f"statsmodels.datasets.{meta['module']}")
    csv = Path(mod.__file__).parent / f"{meta['module']}.csv"
    raw = csv.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != meta["bytes_sha256"]:
        raise HarnessRefusal(
            f"unit {unit_id!r} bytes {got[:12]} differ from the "
            "census digest — the frame is decoupled from its "
            "bytes")
    import pandas as pd
    df = pd.read_csv(io.BytesIO(raw))
    col = meta["column"]
    if col not in df.columns:
        raise HarnessRefusal(f"unit {unit_id!r} lacks the declared "
                             f"column {col!r}")
    y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    n_nan = int(np.isnan(y).sum())
    if "forward-fill" in meta["missingness_policy"]:
        s = pd.Series(y).ffill().bfill()
        y = s.to_numpy(float)
    elif n_nan:
        raise HarnessRefusal(
            f"unit {unit_id!r} has {n_nan} NaN and its declared "
            "policy refuses missingness")
    if not np.isfinite(y).all():
        raise HarnessRefusal(f"unit {unit_id!r} not finite after "
                             "declared policy")
    return {"unit_id": unit_id, "y": y,
            "bytes_sha256": meta["bytes_sha256"],
            "family": meta["family"],
            "frequency": meta["frequency"],
            "license_note": meta["license_note"],
            "nan_filled": n_nan,
            "seasonal_period": SEASONAL_PERIOD[unit_id]}


def roles_of(n: int) -> dict:
    a = int(n * SPLITS[0])
    b = a + int(n * SPLITS[1])
    return {"train": (0, a), "validation": (a, b),
            "score": (b, n)}


def causal_denoise(co, y: np.ndarray, roles: dict,
                   unit_id: str) -> dict:
    """D(X) through the accepted T0 contract: fit bound to the
    train interval, batch transform causal over the full series."""
    lo, hi = roles["train"]
    spec = co.validate_spec({
        "schema": co.SCHEMA_VERSION,
        "operator_id": f"t2_{T1_OPERATOR['kind']}",
        "kind": T1_OPERATOR["kind"], "version": "1",
        "params": dict(T1_OPERATOR["params"]), "columns": ["y"],
        "fit_role": "train",
        "lookback": co.derived_lookback(T1_OPERATOR["kind"],
                                        T1_OPERATOR["params"]),
        "availability_rule": "bar_close"})
    stream = f"t2_unit:{unit_id}"
    train_m = y[lo:hi].reshape(-1, 1)
    art = co.fit(spec, train_m, ["y"], "train",
                 co.make_train_contract(train_m, hi - lo,
                                        float(lo), stream))
    tc = co.make_bar_close_contract(len(y), 0.0, stream)
    d = co.transform_batch(art, y.reshape(-1, 1), ["y"], tc)[:, 0]
    return {"d": d, "artifact_sha256": art["artifact_sha256"],
            "spec_sha256": art["spec_sha256"]}


def nuisance(y: np.ndarray, roles: dict, ident: str) -> list:
    lo, hi = roles["train"]
    scale = float(np.std(y[lo:hi]))
    out = []
    for k in range(NUISANCE_CHANNELS):
        seed = int(hashlib.sha256(
            f"t2_nuisance|{ident}|{k}".encode()).hexdigest()[:8],
            16)
        rng = np.random.default_rng(seed)
        out.append(rng.normal(0.0, scale if scale > 0 else 1.0,
                              y.shape[0]))
    return out


def _lag_matrix(series_list, lo, hi, h):
    rows = []
    for t in range(lo + RIDGE_LAGS, hi - h):
        feats = []
        for s in series_list:
            feats.extend(s[t - RIDGE_LAGS + 1: t + 1])
        rows.append(feats)
    return np.array(rows)


def _targets(y, lo, hi, h):
    return np.array([y[t + h]
                     for t in range(lo + RIDGE_LAGS, hi - h)])


def _ridge(Xt, yt, Xs):
    XtX = Xt.T @ Xt + RIDGE_LAMBDA * np.eye(Xt.shape[1])
    w = np.linalg.solve(XtX, Xt.T @ yt)
    return Xs @ w


def _mlp(Xt, yt, Xs, seed):
    from sklearn.neural_network import MLPRegressor
    m = MLPRegressor(random_state=seed, **MLP_BUDGET)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xt, yt)
    return m.predict(Xs)


def _metrics(y_true, y_pred, resid_train_q):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    lo_q, hi_q = resid_train_q
    coverage = float(np.mean((err >= lo_q) & (err <= hi_q)))
    k = max(1, len(y_true) // 10)
    top = np.argsort(np.abs(y_true))[-k:]
    mae_extreme = float(np.mean(np.abs(err[top])))
    return {"mae": mae, "rmse": rmse,
            "interval_coverage_train_q90": coverage,
            "mae_top_decile_abs_target": mae_extreme}


def assay_unit(co, unit: dict, h: int = 1) -> dict:
    """All arms through the identical path; every cost counted."""
    t0 = time.perf_counter()
    y = unit["y"]
    n = len(y)
    if n < 120:
        raise HarnessRefusal(
            f"unit {unit['unit_id']} too short for the lag/split "
            "geometry")
    roles = roles_of(n)
    lo_t, hi_t = roles["train"]
    lo_s, hi_s = roles["score"]
    den = causal_denoise(co, y, roles, unit["unit_id"])
    d = den["d"]
    r = y - d
    nui = nuisance(y, roles, f"{unit['unit_id']}|{T1_OPERATOR['kind']}")
    arms = {"X": [y], "D": [d], "XDR": [y, d, r],
            "width_control": [y, nui[0], nui[1]]}
    yt = _targets(y, lo_t, hi_t, h)
    ys = _targets(y, lo_s, hi_s, h)
    period = unit["seasonal_period"]
    snaive_idx = [t + h - period for t in
                  range(lo_s + RIDGE_LAGS, hi_s - h)]
    if min(snaive_idx) < 0:
        raise HarnessRefusal("seasonal naive would need rows "
                             "before the series start")
    snaive_pred = y[snaive_idx]
    results = {"seasonal_naive": {
        "predictor": "y[t+h-period]",
        "period_source": "predeclared_design_constant",
        "metrics": _metrics(
            ys, snaive_pred,
            _train_resid_q(y, lo_t, hi_t, h, period))}}
    for arm, series in arms.items():
        Xt = _lag_matrix(series, lo_t, hi_t, h)
        Xs = _lag_matrix(series, lo_s, hi_s, h)
        ridge_pred = _ridge(Xt, yt, Xs)
        rq = np.quantile(_ridge(Xt, yt, Xt) - yt, [0.05, 0.95])
        arm_out = {"ridge": _metrics(ys, ridge_pred,
                                     (float(rq[0]),
                                      float(rq[1])))}
        mlp_runs = {}
        for seed in SEED_TAPE:
            pred = _mlp(Xt, yt, Xs, seed)
            tq = np.quantile(_mlp(Xt, yt, Xt, seed) - yt,
                             [0.05, 0.95])
            mlp_runs[f"seed{seed}"] = _metrics(
                ys, pred, (float(tq[0]), float(tq[1])))
        arm_out["mlp_small"] = mlp_runs
        results[arm] = arm_out
    peak_rss = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss * 1024
    rec = {"schema": "agent_multi.t2_assay_record.v1",
           "authority": "DEVELOPMENT_ONLY_ZERO_CONFIRMATORY_"
                        "AUTHORITY",
           "unit_id": unit["unit_id"],
           "family": unit["family"],
           "bytes_sha256": unit["bytes_sha256"],
           "license_note": unit["license_note"],
           "nan_filled": unit["nan_filled"],
           "horizon": h,
           "roles": {k: list(v) for k, v in roles.items()},
           "operator": {**T1_OPERATOR,
                        "selection_source":
                            "T1_v4_record_LAB_CALIBRATED",
                        "artifact_sha256": den["artifact_sha256"],
                        "spec_sha256": den["spec_sha256"]},
           "seed_tape": list(SEED_TAPE),
           "unit_is_the_statistical_unit": True,
           "claim_classes_only": ["utility", "calibration",
                                  "extreme_preservation", "cost"],
           "results": results,
           "cost": {"cpu_wall_seconds":
                    round(time.perf_counter() - t0, 3),
                    "peak_rss_bytes": int(peak_rss),
                    "includes": "fit + transform + all arms + all "
                                "seeds (failed attempts would be "
                                "recorded here too)"}}
    body = {k: rec[k] for k in sorted(rec)}
    rec["record_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return rec


def _train_resid_q(y, lo_t, hi_t, h, period):
    idx = [t + h - period for t in range(lo_t + RIDGE_LAGS,
                                         hi_t - h)]
    idx = [i for i in idx if i >= 0]
    tr_pred = y[idx]
    tr_true = np.array([y[t + h] for t in
                        range(lo_t + RIDGE_LAGS, hi_t - h)
                        ])[-len(idx):] if idx else np.array([0.0])
    q = np.quantile(tr_pred - tr_true, [0.05, 0.95])
    return (float(q[0]), float(q[1]))


def check_record_schema(rec: dict) -> None:
    """T2.4 consuming discipline: exact keys; forbidden claim
    fields refuse (noise/SNR/eligibility are structurally
    impossible claims for T2)."""
    want = {"schema", "authority", "unit_id", "family",
            "bytes_sha256", "license_note", "nan_filled",
            "horizon", "roles", "operator", "seed_tape",
            "unit_is_the_statistical_unit", "claim_classes_only",
            "results", "cost", "record_sha256"}
    if set(rec) != want:
        raise HarnessRefusal(
            f"record keys are not the exact schema (diff: "
            f"{sorted(set(rec) ^ want)})")
    blob = json.dumps(rec).lower()
    for tok in ("true_snr", "noise_removed", "publicly_eligible",
                "eligibility"):
        if tok in blob:
            raise HarnessRefusal(
                f"forbidden claim token {tok!r} inside a T2 "
                "record")
    if "cost" not in rec or "cpu_wall_seconds" not in rec["cost"]:
        raise HarnessRefusal("record omits its cost")
    body = {k: rec[k] for k in sorted(rec)
            if k != "record_sha256"}
    if hashlib.sha256(json.dumps(
            body, sort_keys=True,
            allow_nan=False).encode()).hexdigest() != \
            rec["record_sha256"]:
        raise HarnessRefusal("record digest does not re-derive")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--development-only", action="store_true")
    ap.add_argument("--confirmatory", action="store_true")
    args = ap.parse_args()
    census = json.loads(args.census.read_text())
    if args.confirmatory:
        raise HarnessRefusal(
            "PUBLIC_DATA_REQUIRED: the census verdict is "
            f"{census['verdict']!r} and no sealed T2.2 design "
            "exists — confirmatory scoring is impossible until "
            "the operator supplies the lawful public bank")
    if not args.development_only:
        raise HarnessRefusal(
            "choose --development-only (zero authority) or "
            "--confirmatory (sealed design required)")
    co = load_co()
    # task duplication guard: two census units may never share the
    # same physical bytes
    seen_bytes = {}
    for uid, meta in census["development_only_units"].items():
        b = meta.get("bytes_sha256")
        if b and b in seen_bytes:
            raise HarnessRefusal(
                f"units {seen_bytes[b]!r} and {uid!r} share the "
                "same bytes — task duplication inflates support")
        if b:
            seen_bytes[b] = uid
    records = []
    for uid in sorted(census["development_only_units"]):
        meta = census["development_only_units"][uid]
        if meta["status"] != "PRESENT_DEVELOPMENT_ONLY":
            continue
        unit = load_task_unit(census, uid)
        try:
            rec = assay_unit(co, unit)
        except SystemExit as exc:
            records.append({
                "schema": "agent_multi.t2_assay_refusal.v1",
                "authority": "DEVELOPMENT_ONLY_ZERO_CONFIRMATORY_"
                             "AUTHORITY",
                "unit_id": uid,
                "bytes_sha256": unit["bytes_sha256"],
                "status": "TYPED_REFUSAL",
                "refusal": str(exc)})
            continue
        check_record_schema(rec)
        records.append(rec)
    payload = {"schema": "agent_multi.t2_pilot.v1",
               "authority": "DEVELOPMENT_ONLY_ZERO_CONFIRMATORY_"
                            "AUTHORITY",
               "census_sha256": hashlib.sha256(
                   args.census.read_bytes()).hexdigest(),
               "records": records}
    args.output.write_text(json.dumps(payload, indent=1,
                                      allow_nan=False))
    print(json.dumps({"units": len(records),
                      "authority": payload["authority"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
