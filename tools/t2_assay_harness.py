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
    import t2_bank as bank
    df = pd.read_csv(io.BytesIO(raw))
    col = meta["column"]
    if col not in df.columns:
        raise HarnessRefusal(f"unit {unit_id!r} lacks the declared "
                             f"column {col!r}")
    # C1: strict tokens (malformed refuses), leading-prefix rule,
    # bounded causal forward fill only — bfill does not exist.
    y_raw = bank.parse_strict_numeric(df[col].tolist(),
                                      f"{unit_id}/{col}")
    fixed = bank.apply_causal_missingness(
        y_raw, unit_id,
        max_gap_run=meta.get("max_gap_run",
                             bank.DEFAULT_MAX_GAP_RUN))
    y = fixed["y"]
    # C2: a real time index — parsed or mechanically reconstructed
    # by the unit's declared rule, then checked for duplicates,
    # ordering and spacing.
    tcol = meta.get("time_column")
    if tcol and tcol in df.columns:
        ts_all = bank.parse_strict_numeric(
            df[tcol].tolist(), f"{unit_id}/{tcol}")
        ts = ts_all[fixed["missingness"]["leading_dropped"]:]
        time_provenance = f"parsed_column:{tcol}"
    else:
        ts = np.arange(len(y), dtype=float)
        time_provenance = ("mechanical_row_index (declared: "
                           "regular sampling per census "
                           "frequency)")
    time_facts = bank.check_time_index(ts, unit_id)
    return {"unit_id": unit_id, "y": y,
            "bytes_sha256": meta["bytes_sha256"],
            "dataset": f"statsmodels:{meta['module']}",
            "family": meta["family"],
            "frequency": meta["frequency"],
            "license_note": meta["license_note"],
            "missingness": fixed["missingness"],
            "time_index": time_facts,
            "time_provenance": time_provenance,
            "seasonal_period": SEASONAL_PERIOD[unit_id],
            "seasonal_period_provenance":
                "predeclared_design_constant"}


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


def _train_scaler(Xt):
    """C5: train-only standardization — mean/std from the fit rows
    only, applied unchanged everywhere else."""
    mu = Xt.mean(axis=0)
    sd = Xt.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def _ridge_fit(Xt, yt):
    """C5: intercept + train-only standardization. C70: ONE
    linear solve per origin/arm; the fitted (mu, sd, w) produces
    BOTH the score and the in-sample predictions — numerically
    identical to the retired two-solve path, because both solves
    ran np.linalg.solve on the same matrices."""
    mu, sd = _train_scaler(Xt)
    Zt = (Xt - mu) / sd
    Zt1 = np.hstack([np.ones((len(Zt), 1)), Zt])
    reg = RIDGE_LAMBDA * np.eye(Zt1.shape[1])
    reg[0, 0] = 0.0                    # never penalize the mean
    w = np.linalg.solve(Zt1.T @ Zt1 + reg, Zt1.T @ yt)
    return (mu, sd, w)


def _ridge_predict(rfit, X):
    mu, sd, w = rfit
    Z1 = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    return Z1 @ w


class _RidgeFitClosure:
    """A picklable single ridge solve for the supervised worker."""
    def __init__(self, Xt, yt):
        self.Xt, self.yt = Xt, yt

    def __call__(self):
        return _ridge_fit(self.Xt, self.yt)


def _mlp_select(Xt, yt, seed, val_frac=0.2, guard=None,
                fit_supervisor=None, label=""):
    """C5: same train-only scaling; the VALIDATION role drives the
    epoch rule on the temporally FINAL fraction of the fit rows.
    C70: ONE selection sequence per origin/arm/seed — the four
    candidate fits run once (each under the supervised worker
    when provided) and the SELECTED fitted model produces both
    prediction sets. Numerically identical to the retired
    two-call path: that path ran the same deterministic selection
    twice and predicted from the same chosen state. C52: `guard`
    is checked between epoch candidates."""
    from sklearn.neural_network import MLPRegressor
    mu, sd = _train_scaler(Xt)
    Zt = (Xt - mu) / sd
    n_val = max(8, int(len(Zt) * val_frac))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best, best_val = None, np.inf
        fit_Z, fit_y = Zt[:-n_val], yt[:-n_val]
        val_Z, val_y = Zt[-n_val:], yt[-n_val:]
        for epochs in (40, 80, 120, 200):
            if guard is not None:
                guard(f"{label}:epoch_candidate_{epochs}")
            mm = MLPRegressor(random_state=seed,
                              **{**MLP_BUDGET,
                                 "max_iter": epochs})
            if fit_supervisor is not None:
                mm = fit_supervisor(
                    _FitClosure(mm, fit_Z, fit_y),
                    f"{label}:fit_epochs{epochs}")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mm.fit(fit_Z, fit_y)
            v = float(np.mean(np.abs(mm.predict(val_Z) - val_y)))
            if v < best_val:
                best, best_val = mm, v
    return (best, mu, sd)


def _mlp_predict(mfit, X):
    best, mu, sd = mfit
    return best.predict((X - mu) / sd)


class _FitClosure:
    """A picklable single fit call for the supervised worker: run
    estimator.fit(X, y) and return the fitted estimator."""
    def __init__(self, est, X, y):
        self.est, self.X, self.y = est, X, y

    def __call__(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.est.fit(self.X, self.y)
        return self.est


def _metrics(y_true, y_pred, resid_train_q, mase_denom,
             extreme_mask):
    """C5: MASE is the scale-free primary loss (train-defined
    seasonal-naive denominator); raw MAE/RMSE stay PER-SERIES
    diagnostics and are never pooled across families. C6: the
    extreme set is predeclared from train-scaled INNOVATIONS (not
    raw level), and calibration reports coverage AND width."""
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mase = float(mae / mase_denom) if mase_denom > 0 else None
    lo_q, hi_q = resid_train_q
    coverage = float(np.mean((err >= lo_q) & (err <= hi_q)))
    width = float(hi_q - lo_q)
    out = {"mase_primary": mase,
           "mae_per_series_diagnostic": mae,
           "rmse_per_series_diagnostic": rmse,
           "interval_coverage_train_q90": coverage,
           "interval_width_train_q90": width}
    if extreme_mask is not None and extreme_mask.any():
        out["mase_on_extreme_innovations"] = (
            float(np.mean(np.abs(err[extreme_mask])) / mase_denom)
            if mase_denom > 0 else None)
        out["extreme_support"] = int(extreme_mask.sum())
    else:
        out["mase_on_extreme_innovations"] = None
        out["extreme_support"] = 0
    return out


def train_innovation_extremes(y, lo_t, hi_t, targets_idx,
                              period, q=0.9):
    """C6: an extreme is a large TRAIN-SCALED innovation
    |y[t+h] - y[t+h-period]| exceeding the train-quantile of the
    same statistic — a rising level series does not turn 'late'
    into 'extreme'."""
    train_innov = np.abs(np.diff(y[lo_t:hi_t]))
    if len(train_innov) < 10:
        return None, 0.0
    thresh = float(np.quantile(train_innov, q))
    mask = np.array([abs(y[t] - y[t - 1]) > thresh
                     for t in targets_idx])
    return mask, thresh


# C35: the COMMON two-origin geometry for every panel — 60% initial
# fit fraction, two consecutive score windows over the final 40%.
# No panel gets an exception; feasibility derives from the models'
# real minimums inside the ONE productive rule.
ROLLING_ORIGINS = 2
ORIGIN_BASE_FRAC = 0.6


def unit_origins(n: int, seasonal_period: int = None) -> list:
    """C5/C32: causal rolling origins — DELEGATED to the single
    productive geometry authority (t2_bank.origin_windows_for) so
    the harness, the design generator, the validator and the fresh
    verifier can never drift. Predeclared, never
    outcome-dependent."""
    import t2_bank as _bank
    try:
        wins = _bank.origin_windows_for(
            n, ROLLING_ORIGINS, ORIGIN_BASE_FRAC,
            seasonal_period=seasonal_period)
    except SystemExit as exc:
        raise HarnessRefusal(str(exc))
    return [(wins[f"origin{k}"]["score"][0],
             wins[f"origin{k}"]["score"][1])
            for k in range(ROLLING_ORIGINS)]


def _mase_denominator(y, lo_t, hi_t, period):
    """C5: train-defined seasonal-naive MAE — the score suffix is
    never used for normalization."""
    if hi_t - lo_t <= period + 1:
        return 0.0
    d = np.abs(y[lo_t + period:hi_t] - y[lo_t:hi_t - period])
    return float(np.mean(d))


def assay_unit(co, unit: dict, h: int = 1, sink: dict = None,
               guard=None, fit_supervisor=None) -> dict:
    """All arms on identical rows/budgets per origin; every cost
    phase recorded separately (C7). C44 (executor order): when
    `sink` is a dict, the RAW predictions, observations AND
    in-sample fit rows of every origin/arm/model (and the
    seasonal-naive baseline) are captured into it under
    (origin_key, arm, model) -> (pred, obs, fit_in_sample) so the
    executor can persist them and an independent verifier can
    recompute EVERY consumed metric from persisted arrays — a
    producer summary alone never authorizes. C52
    (execution-custody order): `guard(label)` is an EXECUTING
    bound check invoked before and after every origin, arm, ridge
    fit, MLP seed and epoch candidate; `fit_supervisor` wraps each
    non-interruptible sklearn fit. Both default to None and change
    no number."""
    y = unit["y"]
    n = len(y)
    period = unit["seasonal_period"]

    def _g(label):
        if guard is not None:
            guard(label)
    # C35: feasibility is decided by the ONE geometry authority
    # (real model minimums, period-aware) — the retired fixed 120
    # floor lives only in git history.
    origins = unit_origins(n, seasonal_period=period)
    per_origin = {}
    costs = {}
    for oi, (o_lo, o_hi) in enumerate(origins):
        okey = f"origin{oi}"
        _g(f"{okey}:start")
        ocost = {}
        lo_t, hi_t = 0, o_lo
        t0 = time.perf_counter()
        den = causal_denoise(co, y, {"train": (lo_t, hi_t)},
                             f"{unit['unit_id']}|{okey}")
        ocost["denoise_fit_transform_s"] = round(
            time.perf_counter() - t0, 4)
        d = den["d"]
        r = y - d
        nui = nuisance(y, {"train": (lo_t, hi_t)},
                       f"{unit['unit_id']}|{T1_OPERATOR['kind']}"
                       f"|{okey}")
        arms = {"X": [y], "D": [d], "XDR": [y, d, r],
                "width_control": [y, nui[0], nui[1]]}
        t0 = time.perf_counter()
        yt = _targets(y, lo_t, hi_t, h)
        ys = _targets(y, o_lo, o_hi, h)
        targets_idx = [t + h for t in
                       range(o_lo + RIDGE_LAGS, o_hi - h)]
        ocost["target_construction_s"] = round(
            time.perf_counter() - t0, 4)
        mase_den = _mase_denominator(y, lo_t, hi_t, period)
        ex_mask, ex_thresh = train_innovation_extremes(
            y, lo_t, hi_t, targets_idx, period)
        # seasonal-naive reference on the same score rows
        snv_idx = [t - period for t in targets_idx]
        if min(snv_idx) < 0:
            raise HarnessRefusal(
                f"{unit['unit_id']} {okey}: seasonal naive needs "
                "rows before the series start")
        t0 = time.perf_counter()
        tr_idx = [t + h for t in
                  range(lo_t + RIDGE_LAGS, hi_t - h)]
        tr_snv = y[[t - period for t in tr_idx]] - y[tr_idx]
        q = np.quantile(tr_snv, [0.05, 0.95])
        oout = {"seasonal_naive": {
            "period_source": unit["seasonal_period_provenance"],
            "metrics": _metrics(
                ys, y[snv_idx], (float(q[0]), float(q[1])),
                mase_den, ex_mask)}}
        if sink is not None:
            sink[(okey, "seasonal_naive", "baseline")] = (
                np.asarray(y[snv_idx], dtype=np.float64),
                np.asarray(ys, dtype=np.float64), None)
        ocost["seasonal_naive_s"] = round(
            time.perf_counter() - t0, 4)
        for arm, series in arms.items():
            _g(f"{okey}:{arm}:start")
            acost = {}
            t0 = time.perf_counter()
            Xt = _lag_matrix(series, lo_t, hi_t, h)
            Xs = _lag_matrix(series, o_lo, o_hi, h)
            acost["lag_features_s"] = round(
                time.perf_counter() - t0, 4)
            t0 = time.perf_counter()
            if fit_supervisor is not None:
                rfit = fit_supervisor(
                    _RidgeFitClosure(Xt, yt),
                    f"{okey}:{arm}:ridge_fit")
            else:
                rfit = _ridge_fit(Xt, yt)
            ridge_pred = _ridge_predict(rfit, Xs)
            ridge_in = _ridge_predict(rfit, Xt)
            acost["ridge_fit_forecast_s"] = round(
                time.perf_counter() - t0, 4)
            _g(f"{okey}:{arm}:ridge_done")
            rq = np.quantile(ridge_in - yt, [0.05, 0.95])
            arm_out = {"ridge": _metrics(
                ys, ridge_pred, (float(rq[0]), float(rq[1])),
                mase_den, ex_mask)}
            if sink is not None:
                sink[(okey, arm, "ridge")] = (
                    np.asarray(ridge_pred, dtype=np.float64),
                    np.asarray(ys, dtype=np.float64),
                    np.asarray(ridge_in, dtype=np.float64))
            mlp_runs = {}
            for seed in SEED_TAPE:
                _g(f"{okey}:{arm}:mlp_seed{seed}:start")
                t0 = time.perf_counter()
                lbl = f"{okey}:{arm}:mlp_seed{seed}"
                mfit = _mlp_select(Xt, yt, seed, guard=guard,
                                   fit_supervisor=fit_supervisor,
                                   label=f"{lbl}:select")
                pred = _mlp_predict(mfit, Xs)
                inp = _mlp_predict(mfit, Xt)
                acost[f"mlp_fit_forecast_seed{seed}_s"] = round(
                    time.perf_counter() - t0, 4)
                _g(f"{okey}:{arm}:mlp_seed{seed}:done")
                tq = np.quantile(inp - yt, [0.05, 0.95])
                mlp_runs[f"seed{seed}"] = _metrics(
                    ys, pred, (float(tq[0]), float(tq[1])),
                    mase_den, ex_mask)
                if sink is not None:
                    sink[(okey, arm, f"mlp_seed{seed}")] = (
                        np.asarray(pred, dtype=np.float64),
                        np.asarray(ys, dtype=np.float64),
                        np.asarray(inp, dtype=np.float64))
            arm_out["mlp_small"] = mlp_runs
            oout[arm] = arm_out
            ocost[f"arm_{arm}"] = acost
        per_origin[okey] = {
            "train": [lo_t, hi_t], "score": [o_lo, o_hi],
            "mase_denominator_train_snaive": mase_den,
            "extreme_innovation_threshold_train": ex_thresh,
            "operator_artifact_sha256": den["artifact_sha256"],
            "results": oout}
        costs[okey] = ocost
        _g(f"{okey}:done")
    peak_rss = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss * 1024
    import t2_bank as _bank
    rec = {"schema": "agent_multi.t2_assay_record.v3",
           "authority": "DEVELOPMENT_MECHANICS_ONLY_REQUIRES_"
                        "C1_C8_CORRECTION_CLEARED",
           "unit_id": unit["unit_id"],
           "family": unit["family"],
           # C26: IDENTITY is physical in the record — the panel
           # it came from and the exact numeric digest of the
           # scored series, both under record_sha256.
           "dataset": unit.get("dataset", "development_bank"),
           "series_numeric_sha256":
               _bank.series_numeric_digest(y),
           "bytes_sha256": unit["bytes_sha256"],
           "license_note": unit["license_note"],
           "missingness": unit["missingness"],
           "time_index": unit["time_index"],
           "time_provenance": unit["time_provenance"],
           "horizon": h,
           "seasonal_period": unit["seasonal_period"],
           "seasonal_period_provenance":
               unit["seasonal_period_provenance"],
           "operator": {**T1_OPERATOR,
                        "selection_source":
                            "T1_v4_record_LAB_CALIBRATED"},
           "seed_tape": list(SEED_TAPE),
           "series_is_the_primary_unit": True,
           "origins_and_seeds_are_nested": True,
           "claim_classes_only": ["utility", "calibration",
                                  "extreme_preservation", "cost"],
           "rolling_origins": per_origin,
           "costs_by_phase": costs,
           "peak_rss_bytes": int(peak_rss)}
    body = {k: rec[k] for k in sorted(rec)}
    rec["record_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return rec


def check_record_schema(rec: dict) -> None:
    """T2.4 consuming discipline: exact keys; forbidden claim
    fields refuse (noise/SNR/eligibility are structurally
    impossible claims for T2)."""
    want = {"schema", "authority", "unit_id", "family",
            "dataset", "series_numeric_sha256",
            "bytes_sha256", "license_note", "missingness",
            "time_index", "time_provenance", "horizon",
            "seasonal_period", "seasonal_period_provenance",
            "operator", "seed_tape",
            "series_is_the_primary_unit",
            "origins_and_seeds_are_nested", "claim_classes_only",
            "rolling_origins", "costs_by_phase",
            "peak_rss_bytes", "record_sha256"}
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
    if "costs_by_phase" not in rec or not rec["costs_by_phase"]:
        raise HarnessRefusal("record omits its per-phase costs")
    for okey, oc in rec["costs_by_phase"].items():
        if "denoise_fit_transform_s" not in oc or not any(
                k.startswith("arm_") for k in oc):
            raise HarnessRefusal(
                f"record omits separated costs at {okey}")
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
        # C3: a REAL gate sequence — each missing element refuses
        # with its own typed reason; nothing is unconditional.
        import t2_confirmatory as conf
        state = Path.home() / ".local/share/agent-multi"
        # C39/C53: the public CLI consumes the ONE current
        # SEALED-v6 identity through the PURE gate sequence only —
        # it can never create a ledger or any durable artifact;
        # scoring belongs exclusively to the executor's --execute.
        conf.verify_confirmatory_gates(
            state / "t2_public_data_manifest_20260906.json",
            conf.t2_active_design_path(),
            census_path=state / "t2_bank_census_20260906.json")
        raise HarnessRefusal(
            "T2_SCORING_ONLY_VIA_EXECUTOR: the gates opened but "
            "this CLI computes nothing — confirmatory scoring "
            "runs only through tools/t2_confirmatory_executor.py "
            "--execute")
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
        try:
            unit = load_task_unit(census, uid)
            rec = assay_unit(co, unit)
        except SystemExit as exc:
            records.append({
                "schema": "agent_multi.t2_assay_refusal.v1",
                "authority": "DEVELOPMENT_MECHANICS_ONLY_"
                             "REQUIRES_C1_C8_CORRECTION_CLEARED",
                "unit_id": uid,
                "bytes_sha256":
                    census["development_only_units"][uid].get(
                        "bytes_sha256"),
                "status": "TYPED_REFUSAL",
                "refusal": str(exc)})
            continue
        check_record_schema(rec)
        records.append(rec)
    payload = {"schema": "agent_multi.t2_pilot.v2",
               "authority": "DEVELOPMENT_MECHANICS_ONLY_REQUIRES_"
                            "C1_C8_CORRECTION_CLEARED",
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
