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

DESIGN_V2 = ("docs/audits/evidence/"
             "N4_TARGET_AUDIT_DESIGN_V2_2026_09_04.json")
# proposed successors: family -> horizons (<=3 x <=3, order §7).
# N4-C3 (order @9fd016b0): 'terminal_return_class' names what the
# label IS — the TERMINAL close-to-close h-bar return thresholded at
# the sealed round-trip cost. It does NOT ask whether any trade wins
# WITHIN h bars.
PROPOSED = {
    "terminal_return_class": (6, 12, 24),
    "mfe_mae_logratio": (6, 12),
    "large_move": (6, 12),
}
# N4-C1: licensing derives its required classes from the target
# contract — never a hard-coded subset
REQUIRED_CLASSES = {"class3": (0, 1, 2), "class2": (0, 1)}
CANDIDATE_TABLE = {
    "tm_h6": ("tm_h6", 6, "class3"),
    "tm_h12": ("tm_h12", 12, "class3"),
    "tm_h24": ("tm_h24", 24, "class3"),
    "mfemae_h6": ("mfemae_h6", 6, "cont"),
    "mfemae_h12": ("mfemae_h12", 12, "cont"),
    "lm_h6": ("lm_h6", 6, "class2"),
    "lm_h12": ("lm_h12", 12, "class2"),
}
FITTED_ARMS = ("volatility_history", "causal_linear")
# legacy, NON-semantic alias: only for reading the superseded v1
# records, whose fitted arm was misnamed (it always fitted the
# trailing-volatility lags)
LEGACY_ARM_ALIAS = {"target_history": "volatility_history"}
FAMILY_SLOTS = tuple(f"{ck}:{arm}" for ck in CANDIDATE_TABLE
                     for arm in FITTED_ARMS)  # exactly 14, ordered
VERDICT_LABELS = (
    "TARGET_FORMULATION_CANDIDATE_FOR_FUTURE_CONFIRMATION",
    "TARGET_FORMULATION_NOT_IDENTIFIED",
    "NO_UNTOUCHED_CONFIRMATION_ROLE_AVAILABLE")
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

    terminal_return_class (ternary): with the TERMINAL h-bar return
      r = log(close[a+h]/close[a]) and round-trip cost c=0.0010:
      0 if r > c, 1 if r < -c, 2 otherwise. This is a property of
      the terminal close only — NOT of any intrahorizon path.
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
    for h in PROPOSED["terminal_return_class"]:
        add(f"tm_h{h}",
            f"terminal {h}-bar log return r: 0 if r > "
            f"+{ROUND_TRIP_COST}, 1 if r < -{ROUND_TRIP_COST}, "
            "else 2 — a TERMINAL close-to-close class",
            "class", "close of bar a; sealed cost 10bp", h,
            f"ceil({h}/4) sampled rows",
            "does the TERMINAL h-bar move clear round-trip cost "
            "— class 2 is the no-trade answer; cost enters the "
            "DEFINITION (distinct from raw return regression and "
            "from intrabar path-dependent barriers); it does NOT "
            "measure intrahorizon trade opportunities",
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
# N4-C4: design bound to execution; N4-C5: pure adjudication         #
# ------------------------------------------------------------------ #

def executable_binding() -> dict:
    """The EXACT machine configuration this module executes. The
    sealed design must carry an identical copy; validate_design
    refuses any divergence field by field."""
    return {
        "families": {k: list(v) for k, v in PROPOSED.items()},
        "candidate_table": {k: list(v)
                            for k, v in CANDIDATE_TABLE.items()},
        "fitted_arms": list(FITTED_ARMS),
        "required_classes": {k: list(v)
                             for k, v in
                             REQUIRED_CLASSES.items()},
        "family_slots_ordered": list(FAMILY_SLOTS),
        "round_trip_cost": ROUND_TRIP_COST,
        "exceedance_q": EXCEEDANCE_Q,
        "margin": MARGIN, "support_min": SUPPORT_MIN,
        "boot_b": BOOT_B, "boot_seed": BOOT_SEED,
        "block_len": BLOCK_LEN, "stride": STRIDE,
        "purge": "per role, exclude anchors with fewer than "
                 "ceil(h/stride) sampled rows before the role end",
        "wall_ceiling_s": 7200,
        "verdict_labels": list(VERDICT_LABELS)}


def validate_design(expected_sha256: str) -> dict:
    """N4-C4: exact schema and field-by-field equality between the
    sealed design's executable_binding and this module's executing
    configuration, plus the pre-execution digest identity supplied
    by the reviewed invocation. No self-review authority — this is
    configuration binding, not publication approval."""
    path = Path(DESIGN_V2)
    if not path.is_absolute():
        path = REPO / path
    actual = sha_file(path)
    if actual != expected_sha256:
        raise N4Refusal(
            f"design digest {actual[:12]} differs from the "
            f"expected pre-execution identity "
            f"{str(expected_sha256)[:12]}")
    design = json.loads(path.read_bytes())
    binding = design.get("executable_binding")
    if not isinstance(binding, dict):
        raise N4Refusal("design lacks executable_binding")
    expected = executable_binding()
    unknown = set(binding) - set(expected)
    missing = set(expected) - set(binding)
    if unknown or missing:
        raise N4Refusal(
            f"design binding schema violation: unknown="
            f"{sorted(unknown)} missing={sorted(missing)}")
    for key, val in expected.items():
        if binding[key] != val:
            raise N4Refusal(
                f"design/execution mismatch at {key}: design="
                f"{binding[key]!r} != executable={val!r}")
    return design


def _boot_p_from_diffs(diffs):
    import numpy as np
    rng = np.random.default_rng(BOOT_SEED)
    n_low = 0
    for _ in range(BOOT_B):
        parts = []
        for d in diffs:
            n = len(d)
            nb = math.ceil(n / BLOCK_LEN)
            starts = rng.integers(0, n, size=nb)
            idx = (starts[:, None]
                   + np.arange(BLOCK_LEN)[None, :]).reshape(-1) % n
            parts.append(d[idx[:n]])
        if float(np.concatenate(parts).mean()) <= 0:
            n_low += 1
    return (1 + n_low) / (BOOT_B + 1)


def adjudicate(records: dict) -> dict:
    """N4-C5: pure, strict adjudication from complete per-window
    records only. Re-derives licensing (required classes from the
    candidate contract), skills, raw p-values, the COMPLETE ordered
    14-slot Holm family (unlicensed slots get non-rejecting
    placeholders), passers and the verdict. Refuses unknown or
    missing candidates/windows, duplicate slots, non-finite values,
    unequal loss-vector lengths and any producer-supplied verdict
    or license flag."""
    import numpy as np
    if set(records) != set(CANDIDATE_TABLE):
        raise N4Refusal(
            f"candidate set mismatch: "
            f"{sorted(set(records) ^ set(CANDIDATE_TABLE))[:4]}")
    wks = ("w1", "w2", "w3", "w4")
    assessment = {}
    for ck, per_w in records.items():
        if set(per_w) != set(wks):
            raise N4Refusal(f"{ck}: window set mismatch")
        kind = CANDIDATE_TABLE[ck][2]
        licensed = True
        reasons = []
        for wk in wks:
            rec = per_w[wk]
            allowed = {"window", "n_score", "losses",
                       "class_support_score",
                       "response_var_score"}
            forbidden = set(rec) - allowed
            if forbidden & {"licensed", "verdict", "outcome",
                            "holm_p"}:
                raise N4Refusal(
                    f"{ck}/{wk}: producer-supplied adjudication "
                    f"field refused: {sorted(forbidden)[:3]}")
            if forbidden:
                raise N4Refusal(
                    f"{ck}/{wk}: unknown fields "
                    f"{sorted(forbidden)[:3]}")
            losses = rec.get("losses")
            if losses is None:
                licensed = False
                reasons.append(f"{wk}: degenerate fit (no losses)")
                continue
            if set(losses) != {"prior", *FITTED_ARMS}:
                raise N4Refusal(f"{ck}/{wk}: arm set mismatch "
                                f"{sorted(losses)}")
            lengths = {len(v) for v in losses.values()}
            if len(lengths) != 1:
                raise N4Refusal(
                    f"{ck}/{wk}: unequal loss-vector lengths")
            for arm, v in losses.items():
                arr = np.asarray(v, dtype="float64")
                if not np.isfinite(arr).all():
                    raise N4Refusal(
                        f"{ck}/{wk}/{arm}: non-finite loss")
            if kind in REQUIRED_CLASSES:
                support = rec.get("class_support_score")
                if not isinstance(support, dict):
                    licensed = False
                    reasons.append(f"{wk}: missing class support")
                    continue
                for c in REQUIRED_CLASSES[kind]:
                    got = support.get(str(c))
                    if not isinstance(got, int) or got < SUPPORT_MIN:
                        licensed = False
                        reasons.append(
                            f"{wk}: required class {c} support "
                            f"{got} < {SUPPORT_MIN}")
            else:
                rv = rec.get("response_var_score")
                if not isinstance(rv, (int, float)) or rv <= 0:
                    licensed = False
                    reasons.append(f"{wk}: non-positive response "
                                   "variance")
        assessment[ck] = {"kind": kind, "licensed": licensed,
                          "license_reasons": reasons,
                          "models": {}}
    # complete ordered 14-slot family — materialized BEFORE Holm
    family = []
    pvals = {}
    for slot in FAMILY_SLOTS:
        ck, arm = slot.split(":")
        entry = assessment[ck]
        if not entry["licensed"]:
            family.append({"slot": slot,
                           "status": "UNLICENSED_PLACEHOLDER",
                           "raw_p": 1.0})
            pvals[slot] = 1.0
            continue
        per_w = records[ck]
        diffs, skills = [], {}
        for wk in wks:
            b = np.asarray(per_w[wk]["losses"]["prior"])
            m = np.asarray(per_w[wk]["losses"][arm])
            skills[wk] = round(1.0 - float(m.sum() / b.sum()), 6)
            diffs.append(b - m)
        pooled = round(1.0 - float(
            sum(np.asarray(per_w[wk]["losses"][arm]).sum()
                for wk in wks)
            / sum(np.asarray(per_w[wk]["losses"]["prior"]).sum()
                  for wk in wks)), 6)
        p = _boot_p_from_diffs(diffs)
        pvals[slot] = min(1.0, p)
        family.append({"slot": slot, "status": "TESTED",
                       "raw_p": ("<= 1/2001"
                                 if p <= 1 / (BOOT_B + 1) + 1e-12
                                 else round(p, 6)),
                       "per_window_skill": skills,
                       "pooled_skill": pooled,
                       "all_windows_positive": all(
                           v > 0 for v in skills.values())})
        entry["models"][arm] = family[-1]
    if len(family) != 14 or len(pvals) != 14             or [f["slot"] for f in family] != list(FAMILY_SLOTS):
        raise N4Refusal(
            f"family cardinality/order violated: {len(family)}")
    holm = holm_adjust(pvals)
    passers = []
    for f in family:
        f["holm_p"] = round(holm[f["slot"]], 6)
        if f["status"] == "TESTED"                 and f["all_windows_positive"]                 and f["pooled_skill"] >= MARGIN                 and f["holm_p"] < 0.05:
            f["passes"] = True
            passers.append(f["slot"])
        else:
            f["passes"] = False
    for ck, entry in assessment.items():
        if not entry["licensed"]:
            entry["outcome"] = "UNLICENSED"
        elif any(s.startswith(ck + ":") for s in passers):
            entry["outcome"] = "PASSES"
        else:
            entry["outcome"] = "FAILS"
    verdict = (VERDICT_LABELS[0] if passers
               else VERDICT_LABELS[1])
    return {"family_cardinality_proven": len(family),
            "family": family, "per_candidate": assessment,
            "passers": passers, "verdict": verdict}


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
            # licensing is NOT the producer's call: the pure
            # adjudicator derives it from these supports and the
            # candidate contract (N4-C1/C5)


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
                rec["losses"] = None
            else:
                rec["losses"] = {
                    "prior": [round(float(v), 8) for v in base],
                    "volatility_history": [
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
            rec["response_var_score"] = round(float(ys.var()), 6)
            rec["losses"] = {
                "prior": [round(float(v), 8) for v in base],
                "volatility_history": [
                    round(float(v), 8)
                    for v in (ys - hpred) ** 2],
                "causal_linear": [
                    round(float(v), 8)
                    for v in (ys - spred) ** 2]}
        records[wk] = rec
    return records


def screen(run_root: Path, out_path: Path,
           expected_design_sha: str) -> dict:
    import numpy as np
    started = time.time()
    design = validate_design(expected_design_sha)
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
           for h in PROPOSED["terminal_return_class"]},
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
    adjudication = adjudicate(results)
    out = {"schema": "agent_multi.n4_screen_result.v2",
           "order": "agent-multi@13fdf18c §8 + @9fd016b0",
           "classification": "EXPLORATORY — development windows "
                             "only; consumed fit/cal roles; no "
                             "confirmation claim is possible from "
                             "this screen",
           "design": DESIGN_V2,
           "design_sha256": sha_file(REPO / DESIGN_V2),
           "data_plane_digests": plane["digests"],
           "decision_constants": executable_binding(),
           "lm_thresholds": {
               f"h{h}": round(float(targets[f"lm_q_h{h}"]), 8)
               for h in PROPOSED["large_move"]},
           "per_window_records": results,
           **adjudication,
           "scope_note": "conclusions apply ONLY to the frozen "
                         "ETH H4 tech_stat data contract, the "
                         "N1-N4 target families actually "
                         "evaluated, the declared simple "
                         "baselines and the development windows "
                         "(order @9fd016b0 N4-C6)",
           "elapsed_s": round(time.time() - started, 1),
           "gpu_neural_gate": "CLOSED — this screen cannot open "
                              "it"}
    out_path.write_text(json.dumps(out, indent=1, default=float)
                        + "\n")
    return out


def readjudicate(v1_result: Path, out_path: Path) -> dict:
    """N4-C5 preferred path: re-adjudicate the EXISTING frozen v1
    records — no data plane, no fitting, no torch. Maps the legacy
    misnamed arm through the explicit non-semantic alias, strips
    producer adjudication fields, runs the pure adjudicator under
    the v2 design, and publishes a successor artifact with an
    explicit supersession relation. Loss vectors are copied
    verbatim and their equality with v1 is proven by digest."""
    v1_raw = v1_result.read_bytes()
    v1 = json.loads(v1_raw)
    records = {}
    for ck, per_w in v1["per_window_records"].items():
        records[ck] = {}
        for wk, rec in per_w.items():
            new = {"window": rec["window"],
                   "n_score": rec["n_score"]}
            if "class_support_score" in rec:
                new["class_support_score"] =                     rec["class_support_score"]
            if "response_var_score" in rec:
                new["response_var_score"] =                     rec["response_var_score"]
            losses = rec.get("losses")
            if losses is None:
                new["losses"] = None
            else:
                new["losses"] = {
                    LEGACY_ARM_ALIAS.get(arm, arm): v
                    for arm, v in losses.items()}
            records[ck][wk] = new
    def loss_digest(recs):
        return sha_obj([[ck, wk,
                         recs[ck][wk].get("losses")]
                        for ck in sorted(recs)
                        for wk in sorted(recs[ck])])
    v1_losses_digest = sha_obj(
        [[ck, wk,
          None if v1["per_window_records"][ck][wk].get("losses")
          is None else {
              LEGACY_ARM_ALIAS.get(a, a): v
              for a, v in v1["per_window_records"][ck][wk]
              ["losses"].items()}]
         for ck in sorted(v1["per_window_records"])
         for wk in sorted(v1["per_window_records"][ck])])
    if loss_digest(records) != v1_losses_digest:
        raise N4Refusal("loss vectors drifted during "
                        "re-adjudication")
    adjudication = adjudicate(records)
    out = {"schema": "agent_multi.n4_screen_result.v2",
           "order": "agent-multi@9fd016b0 N4-C5",
           "classification": v1["classification"],
           "design": DESIGN_V2,
           "design_sha256": sha_file(REPO / DESIGN_V2),
           "data_plane_digests": v1["data_plane_digests"],
           "decision_constants": executable_binding(),
           "lm_thresholds": v1["decision_constants"][
               "lm_thresholds"],
           "per_window_records": records,
           **adjudication,
           "supersession": {
               "v1_result_sha256": hashlib.sha256(
                   v1_raw).hexdigest(),
               "v1_design": v1["design"],
               "v1_design_sha256": v1["design_sha256"],
               "v1_status": "PRESERVED UNCHANGED, SUPERSEDED — "
                            "its observed losses remain evidence; "
                            "its licensing and report are "
                            "superseded by this adjudication",
               "loss_vectors_identical_to_v1": True,
               "legacy_arm_alias_applied": dict(
                   LEGACY_ARM_ALIAS)},
           "scope_note": "conclusions apply ONLY to the frozen "
                         "ETH H4 tech_stat data contract, the "
                         "N1-N4 target families actually "
                         "evaluated, the declared simple "
                         "baselines and the development windows; "
                         "no claim about all possible forecasting "
                         "targets, data sources or pretraining "
                         "formulations; no untouched confirmation "
                         "role remains in this dataset — a new "
                         "program requires new data or a "
                         "separately motivated scientific design "
                         "(order @9fd016b0 N4-C6)",
           "gpu_neural_gate": "CLOSED"}
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
    s.add_argument("--design-sha", required=True)
    r = sub.add_parser("readjudicate")
    r.add_argument("--v1-result", required=True)
    r.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.cmd == "census":
            out = census(Path(args.run_root), Path(args.out))
            print(json.dumps({"targets": len(out["targets"])}))
        elif args.cmd == "screen":
            out = screen(Path(args.run_root), Path(args.out),
                         args.design_sha)
            print(json.dumps({"verdict": out["verdict"],
                              "passers": out["passers"],
                              "elapsed_s": out["elapsed_s"]}))
        else:
            out = readjudicate(Path(args.v1_result),
                               Path(args.out))
            print(json.dumps({"verdict": out["verdict"],
                              "passers": out["passers"],
                              "family": out[
                                  "family_cardinality_proven"]}))
        return 0
    except N4Refusal as refusal:
        print(json.dumps({"refusal": str(refusal)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
