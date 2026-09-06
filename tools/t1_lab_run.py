#!/usr/bin/env python3
"""T1.3 measurement lab (work plan 43 §5.3, order T0-T1).

For every bank unit x candidate operator: fit on the TRAIN role
only, transform, and derive per-observation records — SNR estimation
error/ordering, reconstruction and SNR gain, delay, discontinuity/
extreme/tail preservation, residual whiteness (Ljung-Box as
diagnostic), the residual's incremental train-only utility, frozen
assays (persistence + causal ridge on lags at declared horizons) on
X vs D(X) vs [X,D(X),R] with an explicit capacity control, CPU wall,
peak RSS and typed failures. CPU-only; refuses if CUDA is visible."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
PREP = Path(os.environ.get(
    "T1_PREPROCESSOR_ROOT",
    Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))
sys.path.insert(0, str(PREP))
from app import causal_operators as co  # noqa: E402

HORIZONS = (1, 5)
RIDGE_LAGS = 8
RIDGE_LAMBDA = 1.0
CANDIDATES = ("identity", "trailing_mean", "trailing_median",
              "ewma", "local_level_kalman")
ORACLE = "centered_mean_oracle"
PARAMS = {"trailing_mean": {"window": 5},
          "trailing_median": {"window": 5},
          "ewma": {"alpha": 0.3},
          "identity": {}, "local_level_kalman": {},
          "centered_mean_oracle": {"window": 5}}


def _rss() -> int:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * 4096


def _spec(kind, columns):
    oid = (f"{kind}_t1" if kind != ORACLE
           else "NON_CAUSAL_ORACLE_ONLY_centered_mean")
    lb = PARAMS[kind].get("window", 1)
    return {"schema": co.SCHEMA_VERSION, "operator_id": oid,
            "kind": kind, "version": "1",
            "params": dict(PARAMS[kind]), "columns": list(columns),
            "fit_role": "train", "lookback": lb,
            "availability_rule": "bar_close"}


def ljung_box_p(resid: np.ndarray, lags: int = 10) -> float:
    from scipy import stats
    x = resid - resid.mean()
    n = len(x)
    acf = np.array([np.dot(x[:-k], x[k:]) / np.dot(x, x)
                    for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    return float(1.0 - stats.chi2.cdf(q, lags))


def ridge_r2(X: np.ndarray, y: np.ndarray, Xs: np.ndarray,
             ys: np.ndarray, lam: float = RIDGE_LAMBDA) -> float:
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    w = np.linalg.solve(XtX, X.T @ y)
    pred = Xs @ w
    ss = float(np.sum((ys - pred) ** 2))
    tot = float(np.sum((ys - ys.mean()) ** 2))
    return 1.0 - ss / max(tot, 1e-12)


def _lag_matrix(series_list, lo, hi, h):
    rows = []
    for t in range(lo + RIDGE_LAGS, hi - h):
        feats = []
        for s in series_list:
            feats.extend(s[t - RIDGE_LAGS + 1: t + 1])
        rows.append(feats)
    return np.array(rows)


def _targets(clean_var, lo, hi, h):
    return np.array([clean_var[t + h]
                     for t in range(lo + RIDGE_LAGS, hi - h)])


def assay_arms(clean_j, obs_j, d_j, r_j, roles, h) -> dict:
    lo_t, hi_t = roles["train"]
    lo_s, hi_s = roles["score"]
    arms = {"X": [obs_j], "D": [d_j], "XDR": [obs_j, d_j, r_j],
            "capacity_control_XXX": [obs_j, obs_j, obs_j]}
    out = {}
    yt = _targets(clean_j, lo_t, hi_t, h)
    ys = _targets(clean_j, lo_s, hi_s, h)
    for name, series in arms.items():
        Xt = _lag_matrix(series, lo_t, hi_t, h)
        Xs = _lag_matrix(series, lo_s, hi_s, h)
        out[name] = ridge_r2(Xt, yt, Xs, ys)
    # persistence baselines on the score role
    per = np.array([obs_j[t] for t in
                    range(lo_s + RIDGE_LAGS, hi_s - h)])
    out["persistence_obs_mse"] = float(np.mean((ys - per) ** 2))
    return out


def measure_unit_operator(unit_dir: Path, kind: str) -> dict:
    rec = json.loads((unit_dir / "UNIT.json").read_text())
    clean = np.load(unit_dir / "clean_signal.npy")
    noise = np.load(unit_dir / "realized_noise.npy")
    obs = np.load(unit_dir / "observed_signal.npy")
    for name, arr in (("clean_signal", clean),
                      ("realized_noise", noise),
                      ("observed_signal", obs)):
        if co.hashlib.sha256(np.ascontiguousarray(arr).tobytes()
                             ).hexdigest() != rec["digests"][name]:
            raise SystemExit(f"REFUSED: {name} digest broken in "
                             f"{rec['unit_id']}")
    v = clean.shape[0]
    columns = [f"v{j}" for j in range(v)]
    roles = {k: tuple(x) for k, x in rec["temporal_roles"].items()}
    lo_t, hi_t = roles["train"]
    t0 = time.perf_counter()
    result = {"schema": "agent_multi.t1_measurement.v1",
              "unit_id": rec["unit_id"], "operator": kind,
              "family": rec["family"],
              "perturbation": rec["perturbation"],
              "declared_snr_db": rec["declared_snr_db"],
              "heterogeneous": rec["heterogeneous"],
              "seed": rec["seed"],
              "causal": kind != ORACLE,
              "oracle_only": kind == ORACLE}
    spec = _spec(kind, columns)
    try:
        if rec["missing_mask"]:
            raise co.CausalOperatorError(
                "unit contains licensed missingness; current "
                "candidates declare NO missingness policy — typed "
                "refusal recorded as the missingness behavior")
        art = co.fit(spec, obs[:, lo_t:hi_t].T, columns, "train")
        d = co.transform_batch(art, obs.T, columns).T
    except co.CausalOperatorError as exc:
        result.update({"status": "TYPED_REFUSAL",
                       "refusal": str(exc),
                       "cpu_wall_seconds":
                           round(time.perf_counter() - t0, 3)})
        return result
    r = obs - d
    per_var = []
    for j in range(v):
        mse_obs = float(np.mean((obs[j] - clean[j]) ** 2))
        mse_d = float(np.mean((d[j] - clean[j]) ** 2))
        snr_gain = (float(10 * np.log10(mse_obs / mse_d))
                    if mse_d > 0 and mse_obs > 0 else 0.0)
        # cheap causal SNR estimator: std(diff)/sqrt(2) on train
        est_std = float(np.std(np.diff(obs[j, lo_t:hi_t]))
                        / np.sqrt(2))
        true_std = rec["noise_std_per_var"][j]
        # delay via cross-correlation around 0 (+-10)
        lags = range(-10, 11)
        xc = [float(np.corrcoef(clean[j, 10:-10],
                                d[j, 10 + k:len(d[j]) - 10 + k])[0, 1])
              for k in lags]
        delay = int(list(lags)[int(np.argmax(xc))])
        # extremes: retention at the 5 largest |clean| points
        peaks = np.argsort(np.abs(clean[j]))[-5:]
        denom = np.where(np.abs(clean[j][peaks]) < 1e-9, 1.0,
                         clean[j][peaks])
        retention = float(np.mean(np.clip(
            d[j][peaks] / denom, -2, 2)))
        # tails: max abs residual-vs-clean error relative to noise
        tail_ratio = float(np.max(np.abs(d[j] - clean[j]))
                           / max(np.max(np.abs(noise[j])), 1e-12))
        lb_p = ljung_box_p(r[j, lo_t:hi_t])
        h_facts = {}
        for h in HORIZONS:
            arms = assay_arms(clean[j], obs[j], d[j], r[j], roles, h)
            # residual incremental utility (train-only fit): XDR vs
            # a capacity control of identical dimension
            arms["residual_incremental_r2"] = (
                arms["XDR"] - arms["capacity_control_XXX"])
            h_facts[f"h{h}"] = arms
        per_var.append({
            "variable": columns[j],
            "true_snr_db": rec["true_realized_snr_db"][j],
            "snr_estimator_std_error":
                (abs(est_std - true_std) / true_std
                 if true_std > 0 else 0.0),
            "mse_observed": mse_obs, "mse_denoised": mse_d,
            "snr_gain_db": snr_gain,
            "delay_bars": delay,
            "extreme_retention": retention,
            "tail_ratio": tail_ratio,
            "ljung_box_p_residual_train": lb_p,
            "assays": h_facts})
    # SNR ordering accuracy across variables (heterogeneous only)
    ordering = None
    if v > 1 and rec["heterogeneous"]:
        est = [np.std(np.diff(obs[j, lo_t:hi_t])) for j in range(v)]
        true = rec["noise_std_per_var"]
        from scipy import stats
        ordering = float(stats.spearmanr(est, true).statistic)
    result.update({
        "status": "MEASURED",
        "per_variable": per_var,
        "snr_ordering_spearman": ordering,
        "output_column_expansion": {"X": 1, "D": 1, "XDR": 3},
        "cpu_wall_seconds": round(time.perf_counter() - t0, 3),
        "peak_rss_bytes": _rss(),
        "artifact_sha256": art["artifact_sha256"],
        "spec_sha256": art["spec_sha256"]})
    return result


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import torch
    if torch.cuda.is_available() and os.environ.get(
            "CUDA_VISIBLE_DEVICES", "x") != "":
        raise SystemExit("REFUSED: CUDA visible — T1 is CPU-only")
    inv = json.loads((args.bank_dir / "BANK_INVENTORY.json"
                      ).read_text())
    records = []
    for uid in inv["unit_ids"]:
        for kind in CANDIDATES + (ORACLE,):
            records.append(measure_unit_operator(
                args.bank_dir / uid, kind))
    payload = {"schema": "agent_multi.t1_measurements.v1",
               "code_identity": {
                   "causal_operators_sha256": co.code_identity(),
                   "lab_sha256": hashlib.sha256(
                       Path(__file__).read_bytes()).hexdigest()},
               "expected_units": len(inv["unit_ids"]),
               "expected_operators": len(CANDIDATES) + 1,
               "records": records}
    args.output.write_text(json.dumps(payload, indent=1))
    print(json.dumps({"units": len(inv["unit_ids"]),
                      "records": len(records),
                      "refusals": sum(1 for r in records
                                      if r["status"] ==
                                      "TYPED_REFUSAL")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
