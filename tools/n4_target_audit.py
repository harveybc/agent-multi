#!/usr/bin/env python3
"""N4 target/horizon/data suitability audit (order agent-multi@
13fdf18c §§6-8). NOT another extractor search.

census  (N4.0) one machine-readable census of every target already
        used in N1-N3 and every proposed successor: exact
        definition+units, causal decision-time information, horizon
        and purge, economic interpretation and cost dependence,
        class balance / response distribution per causal DEVELOPMENT
        window, simplest admissible baseline, roles already
        inspected (derived from committed contracts and evidence,
        never producer labels) and remaining untouched confirmation
        roles.
screen  (N4.2) the cheap CPU identifiability screen over the sealed
        N4.1 design: prior/persistence + target-history + simple
        regularized baselines only; deterministic replay; complete
        per-window records; two-hour ceiling; development windows
        only — 2026 confirmation rows are structurally absent
        because only the FROZEN <=2025 CSV is ever loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.experiment_runtime import sha_file, sha_obj  # noqa: E402
from agent_plugins.paired_inference import holm_adjust  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "target_horizon_census_n2",
    REPO / "tools" / "target_horizon_census_n2.py")
tcn2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tcn2)

PREDICTOR = Path.home() / "Documents/GitHub/predictor"
FROZEN_CSV = (PREDICTOR / "examples/data/project3/"
              "ethusdt_4h_tech_stat_full_model_ready.csv")
FROZEN_SHA = ("1b447c66e68495e826c53e2ab2b08ecd3922c8fdc7357476"
              "28f8d0435ebe440f")
N2_NPZ_SHA = ("07c5ff085dfd8bab0dfa33d038005c8fdb2d6c2acff3961d0fe"
              "4b042ef57cca7")
N2_BUNDLE = ("docs/audits/evidence/"
             "TARGET_HORIZON_CENSUS_N2_BUNDLE_2026_09_03.json")
DESIGN = ("docs/audits/evidence/"
          "N4_TARGET_AUDIT_DESIGN_2026_09_04.json")
STRIDE = 4
WINDOW = 64
BOOT_B = 2000
BOOT_SEED = 808
BLOCK_LEN = 6
MARGIN = 0.01
SUPPORT_MIN = 30
ROUND_TRIP_COST = 0.0010   # 10 bp, sealed in the design
EXCEEDANCE_Q = 0.80        # fit-role quantile, sealed

# proposed successors: family -> horizons (<=3 x <=3, order §7)
PROPOSED = {
    "tradeable_move": (6, 12, 24),
    "mfe_mae_logratio": (6, 12),
    "large_move": (6, 12),
}
FAILED_REGRESSION = [f"ret_h{h}" for h in (1, 3, 6, 12)] \
    + [f"vol_h{h}" for h in (3, 6, 12)]
BARRIER_UNCHANGED = ["bar_h6", "bar_h12"]


class N4Refusal(ValueError):
    """Typed refusal."""


# ------------------------------------------------------------------ #
# shared data plane: DEVELOPMENT rows only (frozen <=2025 CSV)       #
# ------------------------------------------------------------------ #

def _load_dev_arrays(run_root: Path):
    """Reproduces the exact N2 sampled development arrays and maps
    them to raw OHLC rows of the FROZEN fit slice. 2026 rows cannot
    appear: only the frozen <=2025 predictor CSV is loaded."""
    import numpy as np
    import pandas as pd
    from agent_plugins.branch_pretraining import (
        build_step_index, load_fit_slice,
        realized_volatility_targets, validate_contract)
    from agent_plugins.pretrained_branch_loader import verify_source
    if sha_file(FROZEN_CSV) != FROZEN_SHA:
        raise N4Refusal("frozen model-ready CSV digest changed")
    npz_path = run_root / "inputs" / "census_inputs.npz"
    if sha_file(npz_path) != N2_NPZ_SHA:
        raise N4Refusal("N2 census npz digest changed")
    data = np.load(npz_path, allow_pickle=False)
    split_contract = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
         "eth_nested_split_contract_o2022_paired_v1.json")
        .read_text())
    data_path = Path(split_contract["source_csv"])
    pretrain_dir = (Path.home() / ".local/share/agent-multi/"
                    "restricted_evidence/"
                    "candidate_full5_pcgrad_o2022_20260828")
    source = verify_source(pretrain_dir, REPO, data_path)
    contract = source["contract"]
    parsed = validate_contract(contract)
    df, ordered, close_col = load_fit_slice(data_path, contract)
    closes = df[close_col].to_numpy()
    ohlc = contract["objectives"]["barrier_hit"]["ohlc_columns"]
    highs = df[ohlc["high"]].to_numpy()
    lows = df[ohlc["low"]].to_numpy()
    warmup = max(int(parsed["warmup_bars"]), WINDOW)
    steps = build_step_index(len(df), warmup, STRIDE, 12, 2200)
    eps = float(contract["objectives"]["volatility"]
                .get("epsilon", 1e-8))
    vol_h6 = realized_volatility_targets(
        closes, steps, [6], eps, None)[:, 0]
    trail_h6 = realized_volatility_targets(
        closes, [max(0, t - 6) for t in steps], [6], eps,
        None)[:, 0]
    keep = np.isfinite(vol_h6) & np.isfinite(trail_h6)
    kept_steps = [s for s, k in zip(steps, keep) if k]
    if len(kept_steps) != len(data["ret"]):
        raise N4Refusal("sampling drift vs the frozen N2 arrays")
    anchors = np.asarray(kept_steps) - 1
    # anchor+24 forward must stay inside the fit slice
    if anchors.max() + 24 >= len(closes):
        usable = anchors + 24 < len(closes)
    else:
        usable = np.ones(len(anchors), dtype=bool)
    geometry = json.loads(
        (REPO / N2_BUNDLE).read_text())["ledger"]["role_geometry"]
    return {"data": data, "anchors": anchors, "closes": closes,
            "highs": highs, "lows": lows, "usable24": usable,
            "geometry": geometry,
            "digests": {"frozen_csv": FROZEN_SHA,
                        "n2_npz": N2_NPZ_SHA,
                        "n2_bundle_geometry": sha_file(
                            REPO / N2_BUNDLE)}}


def build_targets(plane) -> dict:
    """Exact successor-target builders — sealed definitions.

    tradeable_move (ternary): with r = log(close[a+h]/close[a]) and
      round-trip cost c=0.0010: 0 if r > c (long clears cost), 1 if
      r < -c (short clears cost), 2 otherwise (no trade beats cost).
    mfe_mae_logratio (continuous): log((MFE+eps)/(MAE+eps)) with
      MFE = max(high[a+1..a+h])/close[a] - 1 and
      MAE = 1 - min(low[a+1..a+h])/close[a], eps = 1e-6.
    large_move (binary): 1 if |r| >= q_h where q_h is the sealed
      fit-role 80th percentile of |r| per horizon, else 0.
    Every input is available at decision time a (close of bar a);
    every label consumes only bars (a, a+h]."""
    import numpy as np
    a = plane["anchors"]
    closes, highs, lows = (plane["closes"], plane["highs"],
                           plane["lows"])
    out = {}
    n_raw = len(closes)
    for h in sorted({h for hs in PROPOSED.values() for h in hs}):
        valid = a + h < n_raw
        av = a[valid]
        r = np.full(len(a), np.nan)
        r[valid] = np.log(closes[av + h] / closes[av])
        out[f"r_h{h}"] = r
        mfe = np.max(np.stack(
            [highs[av + i] for i in range(1, h + 1)], 1),
            axis=1) / closes[av] - 1.0
        mae = 1.0 - np.min(np.stack(
            [lows[av + i] for i in range(1, h + 1)], 1),
            axis=1) / closes[av]
        ratio = np.full(len(a), np.nan)
        ratio[valid] = np.log(
            (np.clip(mfe, 0, None) + 1e-6)
            / (np.clip(mae, 0, None) + 1e-6))
        out[f"mfemae_h{h}"] = ratio
        tm = np.full(len(a), 2)
        tm[np.nan_to_num(r, nan=0.0) > ROUND_TRIP_COST] = 0
        tm[np.nan_to_num(r, nan=0.0) < -ROUND_TRIP_COST] = 1
        # tail anchors beyond the slice are NaN-masked: they carry
        # sentinel class 2 here but every role purges them anyway
        out[f"tm_h{h}"] = tm
        out[f"valid_h{h}"] = valid
    return out


def _purge_rows(rows, upper, h):
    tail = math.ceil(h / STRIDE)
    return [r for r in rows if r + tail < upper]


def _window_roles(geometry, wk, h):
    """N2 window roles with per-horizon label purge so no label
    crosses a role boundary (sealed in the design)."""
    roles = geometry["windows"][wk]
    fit = list(range(*roles["fit"]))
    cal = list(range(*roles["cal"]))
    sc = list(range(*roles["score"]))
    return (_purge_rows(fit, roles["fit"][1], h),
            _purge_rows(cal, roles["cal"][1], h),
            _purge_rows(sc, roles["score"][1], h))


# ------------------------------------------------------------------ #
# N4.0 census                                                        #
# ------------------------------------------------------------------ #

PRIOR_USE_EVIDENCE = {
    "screen_v2": "docs/audits/evidence/repro_runs — accepted "
                 "screen-v2 report (order @65ee8488): ret/vol "
                 "targets consumed as pretraining objectives and "
                 "branch evaluations on fit/monitor roles",
    "n1": "docs/audits/evidence/TARGET_IDENTIFIABILITY_"
          "PREDECLARATION_N1_2026_09_03.json — vol_h6 inspected "
          "and selected on dev windows w1-w4",
    "n2": "docs/audits/evidence/TARGET_HORIZON_DATA_CENSUS_N2_"
          "PREDECLARATION_2026_09_03.json + bundle — all nine "
          "candidates scored on dev windows w1-w4",
    "n3": "docs/audits/evidence/N3_FRESH_CONFIRMATION_CONTRACT_"
          "2026_09_04.json + v2 bundle — bar_h6/bar_h12 CONFIRMED "
          "(negative) on the 2026 Jan-Aug interval",
    "role_census": "docs/audits/evidence/N3_UNTOUCHED_ROLE_CENSUS_"
                   "2026_09_04.json — every region through 2025 "
                   "consumed or sealed",
}


def _dist(values, rows):
    import numpy as np
    v = np.asarray(values)[rows]
    if v.dtype.kind in "iu" or set(np.unique(v)) <= {0, 1, 2}:
        return {str(c): int((v == c).sum())
                for c in sorted(set(int(x) for x in np.unique(v)))}
    return {"mean": round(float(v.mean()), 6),
            "std": round(float(v.std()), 6),
            "q10": round(float(np.quantile(v, 0.1)), 6),
            "q50": round(float(np.quantile(v, 0.5)), 6),
            "q90": round(float(np.quantile(v, 0.9)), 6)}


def census(run_root: Path, out_path: Path) -> dict:
    import numpy as np
    plane = _load_dev_arrays(run_root)
    data = plane["data"]
    geometry = plane["geometry"]
    wks = sorted(geometry["windows"])
    succ = build_targets(plane)
    entries = {}

    def add(key, definition, units, decision_info, horizon,
            purge, econ, baseline, values, roles_used, untouched):
        entries[key] = {
            "definition": definition, "units": units,
            "causal_decision_time_information": decision_info,
            "horizon_bars": horizon,
            "overlap_purge_requirement": purge,
            "economic_interpretation_and_cost_dependence": econ,
            "simplest_admissible_baseline": baseline,
            "distribution_by_dev_window": {
                wk: _dist(values, list(range(
                    *geometry["windows"][wk]["score"])))
                for wk in wks} if values is not None else
                "not recomputed (committed in the cited evidence)",
            "roles_already_inspected_selected_confirmed":
                roles_used,
            "remaining_untouched_confirmation_roles": untouched}

    untouched_none = ("NONE on this data contract: dev windows and "
                      "fit/cal consumed (N1/N2), 2026 Jan-Aug "
                      "consumed as N3 confirmation, later rows "
                      "absent — derived from " +
                      PRIOR_USE_EVIDENCE["role_census"] + " and "
                      + PRIOR_USE_EVIDENCE["n3"])
    for h in (1, 3, 6, 12):
        add(f"ret_h{h}",
            f"log(close[a+{h}]/close[a])", "log return",
            "close of bar a", h, f"ceil({h}/4) sampled rows",
            "direct PnL proxy before costs; cost enters only via "
            "thresholding (none here) — FAILED its calibrated "
            "baselines in N2",
            "zero return / fit mean",
            data["ret"][:, (1, 3, 6, 12).index(h)],
            [PRIOR_USE_EVIDENCE["screen_v2"],
             PRIOR_USE_EVIDENCE["n2"]], untouched_none)
    for h in (3, 6, 12):
        add(f"vol_h{h}",
            f"log(sqrt(mean(r^2 over (a,a+{h}])) + 1e-8)",
            "log volatility", "close of bar a", h,
            f"ceil({h}/4) sampled rows",
            "risk sizing input; no direct cost dependence — FAILED "
            "calibrated AR1 in N1/N2",
            "literal trailing vol / calibrated AR1",
            data["vol"][:, (3, 6, 12).index(h)],
            ([PRIOR_USE_EVIDENCE["n1"]] if h == 6 else [])
            + [PRIOR_USE_EVIDENCE["screen_v2"],
               PRIOR_USE_EVIDENCE["n2"]], untouched_none)
    for h in (6, 12):
        add(f"bar_h{h}",
            "first intrabar touch of +/-2*trailing-vol barriers "
            f"within {h} bars (0 upper, 1 lower, 2 censored)",
            "class", "close+trailing vol at bar a", h,
            f"ceil({h}/4) sampled rows",
            "bracket-order outcome; barrier width scales with vol "
            "-> the N2/N3 chain proved the gain was the width "
            "scalar and did NOT replicate on fresh 2026",
            "fit+cal class prior",
            data["bar"][:, (6, 12).index(h)],
            [PRIOR_USE_EVIDENCE["n2"], PRIOR_USE_EVIDENCE["n3"]],
            untouched_none)
    for h in PROPOSED["tradeable_move"]:
        add(f"tm_h{h}",
            f"0 if log-move > +{ROUND_TRIP_COST} (long clears "
            f"round-trip cost), 1 if < -{ROUND_TRIP_COST}, else 2",
            "class", "close of bar a; sealed cost 10bp", h,
            f"ceil({h}/4) sampled rows",
            "does ANY trade beat round-trip cost within h bars — "
            "the minimal economically actionable question; cost "
            "enters the DEFINITION (distinct from raw return "
            "regression and from intrabar path-dependent barriers)",
            "fit+cal class prior", succ[f"tm_h{h}"],
            ["NEVER inspected as a target (derived: absent from "
             "every committed predeclaration/bundle above)"],
            untouched_none)
    for h in PROPOSED["mfe_mae_logratio"]:
        add(f"mfemae_h{h}",
            "log((MFE+1e-6)/(MAE+1e-6)); MFE/MAE = max favorable/"
            f"adverse excursion vs close[a] over (a,a+{h}]",
            "log ratio", "close of bar a", h,
            f"ceil({h}/4) sampled rows",
            "reward/risk of entering now: sets brackets and sizing; "
            "costs shift the usable threshold but not the object "
            "(distinct: continuous path-asymmetry, not first-touch "
            "class, not close-to-close return)",
            "fit-role median (constant)", succ[f"mfemae_h{h}"],
            ["NEVER inspected as a target (derived as above)"],
            untouched_none)
    for h in PROPOSED["large_move"]:
        q = None
        add(f"lm_h{h}",
            f"1 if |log-move over (a,a+{h}]| >= fit-role "
            f"{int(EXCEEDANCE_Q*100)}th percentile else 0",
            "binary", "close of bar a; threshold sealed on fit "
            "role only", h, f"ceil({h}/4) sampled rows",
            "activation/sizing gate: is a move worth acting on "
            "coming — decision-oriented exceedance, distinct from "
            "vol-level regression",
            "fit-role base rate (prior)", None,
            ["NEVER inspected as a target (derived as above)"],
            untouched_none)
    out = {"schema": "agent_multi.n4_target_census.v1",
           "order": "agent-multi@13fdf18c §6",
           "data_plane_digests": plane["digests"],
           "dev_windows": {wk: geometry["windows"][wk]
                           for wk in wks},
           "prior_use_derivation_note":
               "role usage derived from the committed contracts "
               "and evidence cited per entry — never from "
               "producer-supplied labels",
           "targets": entries}
    out_path.write_text(json.dumps(out, indent=1, default=float)
                        + "\n")
    return out


# ------------------------------------------------------------------ #
# N4.2 screen                                                        #
# ------------------------------------------------------------------ #

def _logloss_vec(probs, y):
    import numpy as np
    return -np.log(np.clip(
        probs[np.arange(len(y)), y.astype(int)], 1e-12, None))


def _screen_candidate(plane, targets, key, h, kind, design):
    """One candidate x four dev windows: prior/constant,
    target-history and causal-linear arms; per-window per-obs
    primary losses. Deterministic."""
    import numpy as np
    data = plane["data"]
    geometry = plane["geometry"]
    y_full = np.asarray(targets[key])
    summary = data["summary"]
    records = {}
    for wk in sorted(geometry["windows"]):
        fit, cal, sc = _window_roles(geometry, wk, h)
        if h == 24:
            usable = plane["usable24"]
            fit = [r for r in fit if usable[r]]
            cal = [r for r in cal if usable[r]]
            sc = [r for r in sc if usable[r]]
        yf, yc, ys = y_full[fit], y_full[cal], y_full[sc]
        rec = {"n_score": len(sc), "window": wk}
        if kind in ("class3", "class2"):
            n_classes = 3 if kind == "class3" else 2
            counts = np.bincount(
                np.concatenate([yf, yc]).astype(int),
                minlength=n_classes)
            prior = np.clip(counts / counts.sum(), 1e-12, None)
            prior = prior / prior.sum()
            rec["class_support_score"] = {
                str(c): int((ys == c).sum())
                for c in range(n_classes)}
            support_classes = [0, 1] if kind == "class3" \
                else [0, 1]
            rec["licensed"] = all(
                (ys == c).sum() >= SUPPORT_MIN
                for c in support_classes)
            pad = np.zeros((len(ys), 3))
            pad[:, :n_classes] = prior
            base = _logloss_vec(
                np.clip(pad, 1e-12, None)
                / np.clip(pad, 1e-12, None).sum(
                    axis=1, keepdims=True), ys)
            # target-history: trailing SAME-target frequencies are
            # nonstationary summaries; use trailing realized vol
            # lags (the only causal scalar history shared by all
            # class targets) via multinomial logistic
            hist_x = data["barscale"]
            hp, _ = tcn2._logistic(hist_x[fit], yf, hist_x[cal],
                                   yc, hist_x[sc])
            sp, _ = tcn2._logistic(summary[fit], yf, summary[cal],
                                   yc, summary[sc])
            if hp is None or sp is None:
                rec["licensed"] = False
            else:
                rec["losses"] = {
                    "prior": [round(float(v), 8) for v in base],
                    "target_history": [
                        round(float(v), 8)
                        for v in _logloss_vec(hp, ys)],
                    "causal_linear": [
                        round(float(v), 8)
                        for v in _logloss_vec(sp, ys)]}
        else:  # continuous: squared error on the log-ratio
            med = float(np.median(np.concatenate([yf, yc])))
            base = (ys - med) ** 2
            hist_x = data["barscale"]
            hpred, _ = tcn2._ridge(hist_x[fit], yf, hist_x[cal],
                                   yc, hist_x[sc], tcn2._se)
            spred, _ = tcn2._ridge(summary[fit], yf,
                                   summary[cal], yc, summary[sc],
                                   tcn2._se)
            rec["licensed"] = bool(np.isfinite(base).all())
            rec["response_var_score"] = round(float(ys.var()), 6)
            rec["losses"] = {
                "prior": [round(float(v), 8) for v in base],
                "target_history": [
                    round(float(v), 8)
                    for v in (ys - hpred) ** 2],
                "causal_linear": [
                    round(float(v), 8)
                    for v in (ys - spred) ** 2]}
        records[wk] = rec
    return records


def screen(run_root: Path, out_path: Path) -> dict:
    import numpy as np
    started = time.time()
    design = json.loads((REPO / DESIGN).read_bytes())
    plane = _load_dev_arrays(run_root)
    targets = build_targets(plane)
    # large_move thresholds: FIT role of w1 prefix only (sealed)
    geometry = plane["geometry"]
    fit_all = list(range(*geometry["windows"]["w1"]["fit"]))
    for h in PROPOSED["large_move"]:
        q = float(np.quantile(
            np.abs(np.asarray(targets[f"r_h{h}"])[fit_all]),
            EXCEEDANCE_Q))
        targets[f"lm_h{h}"] = (
            np.abs(targets[f"r_h{h}"]) >= q).astype(int)
        targets[f"lm_q_h{h}"] = q
    candidates = {
        **{f"tm_h{h}": ("tm_h%d" % h, h, "class3")
           for h in PROPOSED["tradeable_move"]},
        **{f"mfemae_h{h}": ("mfemae_h%d" % h, h, "cont")
           for h in PROPOSED["mfe_mae_logratio"]},
        **{f"lm_h{h}": ("lm_h%d" % h, h, "class2")
           for h in PROPOSED["large_move"]},
    }
    results = {}
    total = len(candidates)
    done = 0
    for ckey, (tkey, h, kind) in candidates.items():
        results[ckey] = _screen_candidate(plane, targets, tkey, h,
                                          kind, design)
        done += 1
        print(json.dumps({"progress": f"{done}/{total}",
                          "candidate": ckey,
                          "elapsed_s": round(
                              time.time() - started, 1)}),
              flush=True)
        if time.time() - started > 7200:
            raise N4Refusal("two-hour wall ceiling reached")
    # ---- decision: skill vs strongest baseline-arm pair ----
    pvals, assessment = {}, {}
    for ckey, per_w in results.items():
        licensed = all(r.get("licensed") for r in per_w.values())
        entry = {"licensed": licensed, "windows": {}}
        if licensed:
            best_model = None
            # model chosen by pooled CALIBRATION? screen uses the
            # sealed rule: better pooled SCORE loss is NOT allowed
            # to choose — both models evaluated, candidate passes
            # if EITHER passes with Holm over ALL (candidate,
            # model) contrasts
            for model in ("target_history", "causal_linear"):
                diffs = []
                skills = {}
                for wk, r in per_w.items():
                    b = np.asarray(r["losses"]["prior"])
                    m = np.asarray(r["losses"][model])
                    skills[wk] = round(
                        1.0 - float(m.sum() / b.sum()), 6)
                    diffs.append(b - m)
                pooled = round(1.0 - float(
                    sum(np.asarray(r["losses"][model]).sum()
                        for r in per_w.values())
                    / sum(np.asarray(r["losses"]["prior"]).sum()
                          for r in per_w.values())), 6)
                rng = np.random.default_rng(BOOT_SEED)
                n_low = 0
                for _ in range(BOOT_B):
                    parts = []
                    for d in diffs:
                        n = len(d)
                        nb = math.ceil(n / BLOCK_LEN)
                        starts = rng.integers(0, n, size=nb)
                        idx = (starts[:, None] + np.arange(
                            BLOCK_LEN)[None, :]).reshape(-1) % n
                        parts.append(d[idx[:n]])
                    if float(np.concatenate(parts).mean()) <= 0:
                        n_low += 1
                p = (1 + n_low) / (BOOT_B + 1)
                pvals[f"{ckey}:{model}"] = min(1.0, p)
                entry["windows"][model] = {
                    "per_window_skill": skills,
                    "pooled_skill": pooled,
                    "all_windows_positive": all(
                        v > 0 for v in skills.values()),
                    "bootstrap_p": ("<= 1/2001"
                                    if p <= 1 / (BOOT_B + 1)
                                    + 1e-12 else round(p, 6))}
        assessment[ckey] = entry
    holm = holm_adjust(pvals) if pvals else {}
    passers = []
    for ckey, entry in assessment.items():
        if not entry["licensed"]:
            entry["outcome"] = "UNLICENSED"
            continue
        entry["outcome"] = "FAILS"
        for model, s in entry["windows"].items():
            s["holm_p"] = round(holm[f"{ckey}:{model}"], 6)
            if s["all_windows_positive"] \
                    and s["pooled_skill"] >= MARGIN \
                    and s["holm_p"] < 0.05:
                entry["outcome"] = "PASSES"
                passers.append((ckey, model))
    if passers:
        verdict = ("TARGET_FORMULATION_CANDIDATE_FOR_FUTURE_"
                   "CONFIRMATION")
    else:
        verdict = "TARGET_FORMULATION_NOT_IDENTIFIED"
    out = {"schema": "agent_multi.n4_screen_result.v1",
           "order": "agent-multi@13fdf18c §8",
           "classification": "EXPLORATORY — development windows "
                             "only; consumed fit/cal roles; no "
                             "confirmation claim is possible from "
                             "this screen",
           "design": DESIGN,
           "design_sha256": sha_file(REPO / DESIGN),
           "data_plane_digests": plane["digests"],
           "decision_constants": {
               "margin": MARGIN, "boot_b": BOOT_B,
               "boot_seed": BOOT_SEED, "block_len": BLOCK_LEN,
               "support_min": SUPPORT_MIN,
               "round_trip_cost": ROUND_TRIP_COST,
               "exceedance_q": EXCEEDANCE_Q,
               "lm_thresholds": {
                   f"h{h}": round(float(
                       targets[f"lm_q_h{h}"]), 8)
                   for h in PROPOSED["large_move"]}},
           "per_candidate": assessment,
           "per_window_records": results,
           "passers": [f"{c}:{m}" for c, m in passers],
           "verdict": verdict,
           "elapsed_s": round(time.time() - started, 1),
           "gpu_neural_gate": "CLOSED — this screen cannot open "
                              "it; a passing candidate authorizes "
                              "only a later confirmation design"}
    out_path.write_text(json.dumps(out, indent=1, default=float)
                        + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census")
    c.add_argument("--run-root", required=True)
    c.add_argument("--out", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--run-root", required=True)
    s.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.cmd == "census":
            out = census(Path(args.run_root), Path(args.out))
            print(json.dumps({"targets": len(out["targets"])}))
        else:
            out = screen(Path(args.run_root), Path(args.out))
            print(json.dumps({"verdict": out["verdict"],
                              "passers": out["passers"],
                              "elapsed_s": out["elapsed_s"]}))
        return 0
    except N4Refusal as refusal:
        print(json.dumps({"refusal": str(refusal)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
