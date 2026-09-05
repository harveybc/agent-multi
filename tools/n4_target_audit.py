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
# C17 (order @af1ca667): one strict JSON and schema boundary         #
# ------------------------------------------------------------------ #

import re as _re
HEX64 = _re.compile(r"^[0-9a-f]{64}$")
DESIGN_V3 = ("docs/audits/evidence/"
             "N4_TARGET_AUDIT_DESIGN_V3_2026_09_04.json")
# C23 (order @0b4d2748): the executing correction path CARRIES the
# reviewed identities. A caller-supplied value that is not exactly
# the carried constant refuses BEFORE any file is opened — the
# candidate can never choose the trust root.
REVIEWED_V1_RESULT_SHA = ("d696886c4e0d8f59378e29505eaea509ffeef8"
                          "0b6fa8b1c5c9517587333d7400")
REVIEWED_V1_DESIGN_SHA = ("ae05f1878305cc3aee9003849d4f147f2685a1"
                          "59ed3afbdc3870ec7e8c58f4ef")
REVIEWED_DESIGN_V3_SHA = ("c5ccb0eb88113d29761e98bee44fbcb92a0877"
                          "277abc0690456e4ffc92001ad7")


def _reject_const(name):
    raise N4Refusal(f"non-finite JSON constant {name} refused")


def _no_dup_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise N4Refusal(f"duplicate JSON key {k!r} refused")
        d[k] = v
    return d


def strict_json_bytes(raw: bytes):
    """Rejects duplicate keys and non-finite constants BEFORE an
    object exists. Callers hash exactly these bytes and parse
    exactly these bytes, once."""
    return json.loads(raw, parse_constant=_reject_const,
                      object_pairs_hook=_no_dup_pairs)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_real(x) -> bool:
    return (_is_int(x) or isinstance(x, float)) \
        and math.isfinite(x)


def _typed_equal(a, b, path="binding") -> None:
    """Exact primitive types before comparison: booleans are never
    integers/reals here, 30.0 never equals 30."""
    if isinstance(b, bool) or isinstance(a, bool):
        if type(a) is not bool or type(b) is not bool or a != b:
            raise N4Refusal(f"{path}: boolean mismatch")
        return
    if isinstance(b, int):
        if not _is_int(a) or a != b:
            raise N4Refusal(
                f"{path}: expected integer {b}, got {a!r} "
                f"({type(a).__name__})")
        return
    if isinstance(b, float):
        if type(a) is not float or a != b:
            raise N4Refusal(
                f"{path}: expected real {b}, got {a!r} "
                f"({type(a).__name__})")
        return
    if isinstance(b, str):
        if type(a) is not str or a != b:
            raise N4Refusal(f"{path}: string mismatch")
        return
    if isinstance(b, list):
        if type(a) is not list or len(a) != len(b):
            raise N4Refusal(f"{path}: list shape mismatch")
        for i, (x, y) in enumerate(zip(a, b)):
            _typed_equal(x, y, f"{path}[{i}]")
        return
    if isinstance(b, dict):
        if type(a) is not dict:
            raise N4Refusal(f"{path}: expected object")
        unknown = set(a) - set(b)
        missing = set(b) - set(a)
        if unknown or missing:
            raise N4Refusal(
                f"{path}: schema violation unknown="
                f"{sorted(unknown)[:3]} missing="
                f"{sorted(missing)[:3]}")
        for k in b:
            _typed_equal(a[k], b[k], f"{path}.{k}")
        return
    raise N4Refusal(f"{path}: unsupported type")


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


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise N4Refusal(msg)


def _check_schema(value, spec, path="design"):
    """C24.1 (order @0b4d2748): every consumed design field has an
    exact schema and exact primitive type; unknown, missing or
    type-changed values refuse RECURSIVELY."""
    if spec == "str":
        _need(type(value) is str, f"{path}: expected string")
        return
    if spec == "int":
        _need(_is_int(value), f"{path}: expected integer "
              "(booleans and floats refuse)")
        return
    if spec == "real":
        _need(_is_real(value) and type(value) is not bool,
              f"{path}: expected finite real")
        return
    if spec == "bool":
        _need(type(value) is bool, f"{path}: expected boolean")
        return
    if isinstance(spec, tuple) and spec[0] == "list":
        _need(type(value) is list, f"{path}: expected list")
        for i, item in enumerate(value):
            _check_schema(item, spec[1], f"{path}[{i}]")
        return
    if isinstance(spec, tuple) and spec[0] == "map":
        _need(type(value) is dict, f"{path}: expected object")
        for k, v in value.items():
            _need(type(k) is str, f"{path}: non-string key")
            _check_schema(v, spec[1], f"{path}.{k}")
        return
    if isinstance(spec, dict):
        _need(type(value) is dict, f"{path}: expected object")
        unknown = set(value) - set(spec)
        missing = set(spec) - set(value)
        _need(not unknown and not missing,
              f"{path}: schema violation unknown="
              f"{sorted(unknown)[:3]} missing="
              f"{sorted(missing)[:3]}")
        for k, sub in spec.items():
            _check_schema(value[k], sub, f"{path}.{k}")
        return
    raise N4Refusal(f"{path}: unsupported schema spec")


_FAMILY_SPEC = {"horizons": ("list", "int"), "definition": "str",
                "online_computable": "str",
                "economic_interpretation": "str",
                "distinct_from_failures": "str"}
DESIGN_V3_SCHEMA = {
    "schema": "str", "experiment": "str", "order": "str",
    "chronology": {"v1": "str", "v2": "str", "v3": "str"},
    "classification": "str",
    "candidate_families_max3_horizons_max3": ("map", _FAMILY_SPEC),
    "arms_per_candidate": ("list", "str"),
    "primary_losses": ("map", "str"),
    "causal_design": ("map", "str"),
    "decision_rule": {
        "candidate_passes_iff": "str",
        "verdicts": {
            "TARGET_FORMULATION_CANDIDATE_FOR_FUTURE_"
            "CONFIRMATION": "str",
            "TARGET_FORMULATION_NOT_IDENTIFIED": "str",
            "NO_UNTOUCHED_CONFIRMATION_ROLE_AVAILABLE": "str"}},
    "execution_bounds": {
        "device": "str", "wall_ceiling_s": "int",
        "progress": "str", "determinism": "str",
        "prohibited": "str"},
    "supersedes": {
        "v1_design_sha256": "str", "v2_design_sha256": "str",
        "statuses": "str", "corrections_v3": ("list", "str")},
    "executable_binding": ("map", None),  # typed separately
}


OWNER_ACT_PATH = ("docs/audits/evidence/OWNER_RATIFICATION_"
                  "OBSERVATION_V2_AND_MT5_BUILD_6140_2026_09_04"
                  ".json")
OWNER_ACT_SHA = ("399483a14ab4821a49155afd72d153e870e2f9c0519458"
                 "75ca7fdfb5a5726186")
RATIFIED_FEATURE_COLUMNS_SHA = (
    "c4697681c1323245691b8e577905894b96bed81738411b439995e2c2d4b4"
    "4e4d")
RATIFIED_AGENT_STATE_SHA = (
    "b5beeb97e2031b8b696fad452cf42d1781d87848ce753855413f0f46eef9"
    "f160")
RATIFIED_PROPOSED_CONTRACT_SHA = (
    "0ecc3d004b26ef4d913fd06ab585f9ce0885011a4cf4d1cc88d0a743b3e9"
    "81a7")


def verify_owner_act() -> dict:
    """C24.2 (order @0b4d2748): the owner decision record is
    consumed EXECUTABLY — complete SHA-256 equality (prefix
    comparisons forbidden), strict parsing, exact schema and exact
    agreement on authority, decisions, parent order, resolved
    items, proposed contract identity, every observation term with
    its FULL digest, the build-6140 scope and the remaining gates.
    Every B4 materializer must call this before accepting
    OWNER_RATIFIED — a status string alone grants nothing."""
    path = Path(OWNER_ACT_PATH)
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        raise N4Refusal("owner act absent from the executing "
                        "branch — OWNER_RATIFIED cannot be "
                        "accepted")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != OWNER_ACT_SHA:
        raise N4Refusal(
            "owner act bytes differ from the recorded identity "
            "— refused (full-digest equality, no prefixes)")
    act = strict_json_bytes(raw)
    _need(act.get("authority") == "project_owner",
          "owner act: authority is not the project owner")
    _need(act.get("resolves", {}).get("order")
          == "agent-multi@af1ca667"
          and act["resolves"].get("section") == "13"
          and act["resolves"].get("items") == [1, 2],
          "owner act: foreign parent order or resolved items")
    resp = act.get("owner_response", {})
    _need(resp.get("observation_contract_v2") == "RATIFIED"
          and resp.get("mt5_terminal_build_6140") == "ACCEPTED",
          "owner act: decision differs")
    obs = act.get("observation_contract_v2", {})
    _need(obs.get("decision") == "OWNER_RATIFIED"
          and obs.get("contract_file_sha256")
          == RATIFIED_PROPOSED_CONTRACT_SHA
          and obs.get("feature_count") == 83
          and obs.get("feature_columns_sha256")
          == RATIFIED_FEATURE_COLUMNS_SHA
          and obs.get("excluded_feature") == "typical_price"
          and obs.get("include_price_window") is False
          and obs.get("include_agent_state") is True
          and obs.get("agent_state_fields")
          == ["position", "equity_norm", "unrealized_pnl_norm",
              "holding_duration_norm"]
          and obs.get("agent_state_fields_sha256")
          == RATIFIED_AGENT_STATE_SHA
          and obs.get("window_size") == 32
          and obs.get("flattened_shape") == [2660],
          "owner act: an observation term differs from the "
          "ratified tuple")
    build = act.get("mt5_collector_build", {})
    _need(build.get("decision")
          == "ACCEPTED_AS_CURRENT_EXPECTED_BUILD"
          and build.get("expected_terminal_build") == 6140
          and type(build.get("expected_terminal_build")) is int
          and build.get("supersedes_expected_terminal_build")
          == 6090,
          "owner act: build decision differs")
    gates = act.get("remaining_owner_dependent_gates")
    _need(isinstance(gates, list) and len(gates) == 3,
          "owner act: remaining gates differ")
    # the live contract must agree with the act
    contract = strict_json_bytes(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/"
         "systems/ethusdt_4h_l1_system_v2.json").read_bytes())
    _need(contract.get("status") == "OWNER_RATIFIED",
          "contract status is not OWNER_RATIFIED")
    cols = contract["observation"]["feature_columns"]
    cols_sha = hashlib.sha256(json.dumps(
        cols, separators=(",", ":")).encode()).hexdigest()
    state_sha = hashlib.sha256(json.dumps(
        contract["observation"]["agent_state_fields"],
        separators=(",", ":")).encode()).hexdigest()
    _need(cols_sha == RATIFIED_FEATURE_COLUMNS_SHA
          and state_sha == RATIFIED_AGENT_STATE_SHA
          and len(cols) == 83,
          "live contract terms differ from the ratified digests")
    _need("pending" not in contract.get("$doc", "").lower()
          and "AWAITING" not in contract.get("$doc", ""),
          "contract $doc still claims a pending ratification")
    return {"owner_act": "VERIFIED",
            "act_sha256": actual,
            "observation_identity": "OWNER_RATIFIED",
            "expected_terminal_build": 6140}


DESIGN_V3_TOP_KEYS = {
    "schema", "experiment", "order", "chronology",
    "classification", "candidate_families_max3_horizons_max3",
    "arms_per_candidate", "primary_losses", "causal_design",
    "decision_rule", "execution_bounds", "executable_binding",
    "supersedes"}


def validate_design(expected_sha256: str) -> dict:
    """C17 (order @af1ca667): one strict boundary. Reads ONE byte
    stream, hashes those bytes, parses those bytes with duplicate-
    key and non-finite rejection; validates the exact top-level
    schema and the executable_binding with exact primitive types
    (booleans excluded from integer/real fields; 30.0 != 30);
    requires the pre-execution digest supplied by the reviewed
    invocation. Configuration binding, never publication
    approval."""
    if not isinstance(expected_sha256, str) \
            or not HEX64.match(expected_sha256):
        raise N4Refusal("expected design digest must be a "
                        "canonical lowercase sha256")
    path = Path(DESIGN_V3)
    if not path.is_absolute():
        path = REPO / path
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise N4Refusal(
            f"design digest {actual[:12]} differs from the "
            f"expected pre-execution identity "
            f"{expected_sha256[:12]}")
    design = strict_json_bytes(raw)
    # C24.1: the WHOLE document is schema-checked recursively with
    # exact primitive types; executable_binding is then compared
    # field-by-field against the executing module
    schema = dict(DESIGN_V3_SCHEMA)
    schema["executable_binding"] = ("map", None)
    for key, spec in schema.items():
        _need(key in design, f"design: missing {key}")
    unknown = set(design) - set(schema)
    _need(not unknown, f"design top-level schema violation: "
          f"unknown={sorted(unknown)[:3]}")
    for key, spec in schema.items():
        if key == "executable_binding":
            continue
        _check_schema(design[key], spec, f"design.{key}")
    fams = design["candidate_families_max3_horizons_max3"]
    _need(len(fams) <= 3 and all(
        len(f["horizons"]) <= 3 for f in fams.values()),
        "design: family/horizon cardinality violated")
    _typed_equal(design["executable_binding"],
                 executable_binding())
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
    """C18/C20 (order @af1ca667): pure adjudication from
    EVIDENCE-COMPLETE records. Each (candidate, window) record must
    carry the per-observation facts that DECIDE licensing —
    ordered class labels (classification) or target values
    (continuous), the ordered-anchor digest and cardinality, and
    the three ordered loss vectors. n_score, class supports and
    response variance are DERIVED here; any published diagnostics
    are compared, never trusted. Typed refusals fire BEFORE any
    NumPy coercion. No producer verdict, license, support or
    variance can authorize anything."""
    import numpy as np
    if set(records) != set(CANDIDATE_TABLE):
        raise N4Refusal(
            f"candidate set mismatch: "
            f"{sorted(set(records) ^ set(CANDIDATE_TABLE))[:4]}")
    wks = ("w1", "w2", "w3", "w4")
    assessment = {}
    derived = {}
    for ck, per_w in records.items():
        if set(per_w) != set(wks):
            raise N4Refusal(f"{ck}: window set mismatch")
        kind = CANDIDATE_TABLE[ck][2]
        licensed = True
        reasons = []
        for wk in wks:
            rec = per_w[wk]
            required = {"window", "n_score", "anchors_sha256",
                        "losses"}
            required |= ({"labels"} if kind in REQUIRED_CLASSES
                         else {"target_values"})
            forbidden = set(rec) - required
            if forbidden & {"licensed", "verdict", "outcome",
                            "holm_p", "class_support_score",
                            "response_var_score"}:
                raise N4Refusal(
                    f"{ck}/{wk}: producer-supplied adjudication "
                    f"field refused: {sorted(forbidden)[:3]}")
            if forbidden or (required - set(rec)):
                raise N4Refusal(
                    f"{ck}/{wk}: record schema violation "
                    f"unknown={sorted(forbidden)[:3]} missing="
                    f"{sorted(required - set(rec))[:3]}")
            if rec["window"] != wk:
                raise N4Refusal(
                    f"{ck}/{wk}: declared window "
                    f"{rec['window']!r} differs from its key")
            if not (isinstance(rec["anchors_sha256"], str)
                    and HEX64.match(rec["anchors_sha256"])):
                raise N4Refusal(
                    f"{ck}/{wk}: anchors_sha256 not canonical")
            n = rec["n_score"]
            if not _is_int(n) or n <= 0:
                raise N4Refusal(
                    f"{ck}/{wk}: n_score must be a positive "
                    "integer")
            losses = rec["losses"]
            if losses is None:
                licensed = False
                reasons.append(f"{wk}: degenerate fit (no losses)")
                continue
            if not isinstance(losses, dict) or \
                    set(losses) != {"prior", *FITTED_ARMS}:
                raise N4Refusal(
                    f"{ck}/{wk}: arm set mismatch")
            for arm, vec in losses.items():
                if not isinstance(vec, list) or len(vec) != n:
                    raise N4Refusal(
                        f"{ck}/{wk}/{arm}: loss cardinality != "
                        "n_score")
                for v in vec:
                    if not _is_real(v):
                        raise N4Refusal(
                            f"{ck}/{wk}/{arm}: loss value must be "
                            "a finite JSON number (booleans, "
                            "strings, null, NaN, infinity "
                            "refuse)")
            prior_sum = sum(losses["prior"])
            if not (prior_sum > 0 and math.isfinite(prior_sum)):
                raise N4Refusal(
                    f"{ck}/{wk}: non-positive or non-finite "
                    "baseline loss denominator")
            if kind in REQUIRED_CLASSES:
                labels = rec["labels"]
                if not isinstance(labels, list) \
                        or len(labels) != n:
                    raise N4Refusal(
                        f"{ck}/{wk}: label cardinality != "
                        "n_score")
                allowed = set(REQUIRED_CLASSES[kind])
                if kind == "class3":
                    allowed = {0, 1, 2}
                for v in labels:
                    if not _is_int(v) or v not in allowed:
                        raise N4Refusal(
                            f"{ck}/{wk}: class label outside the "
                            f"target contract: {v!r}")
                support = {c: sum(1 for v in labels if v == c)
                           for c in sorted(allowed)}
                derived[(ck, wk)] = {"support": support}
                for c in REQUIRED_CLASSES[kind]:
                    if support.get(c, 0) < SUPPORT_MIN:
                        licensed = False
                        reasons.append(
                            f"{wk}: required class {c} derived "
                            f"support {support.get(c, 0)} < "
                            f"{SUPPORT_MIN}")
            else:
                values = rec["target_values"]
                if not isinstance(values, list) \
                        or len(values) != n:
                    raise N4Refusal(
                        f"{ck}/{wk}: target cardinality != "
                        "n_score")
                for v in values:
                    if not _is_real(v):
                        raise N4Refusal(
                            f"{ck}/{wk}: target value must be a "
                            "finite JSON number")
                arr = np.asarray(values, dtype="float64")
                var = float(arr.var())
                derived[(ck, wk)] = {"response_var": var}
                if not (var > 0):
                    licensed = False
                    reasons.append(f"{wk}: non-positive DERIVED "
                                   "response variance")
        assessment[ck] = {"kind": kind, "licensed": licensed,
                          "license_reasons": reasons,
                          "models": {}}
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
        import numpy as np
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
        pv = _boot_p_from_diffs(diffs)
        pvals[slot] = min(1.0, pv)
        family.append({"slot": slot, "status": "TESTED",
                       "raw_p": ("<= 1/2001"
                                 if pv <= 1 / (BOOT_B + 1) + 1e-12
                                 else round(pv, 6)),
                       "per_window_skill": skills,
                       "pooled_skill": pooled,
                       "all_windows_positive": all(
                           v > 0 for v in skills.values())})
        entry["models"][arm] = family[-1]
    if len(family) != 14 or len(pvals) != 14 \
            or [f["slot"] for f in family] != list(FAMILY_SLOTS):
        raise N4Refusal(
            f"family cardinality/order violated: {len(family)}")
    holm = holm_adjust(pvals)
    passers = []
    for f in family:
        f["holm_p"] = round(holm[f["slot"]], 6)
        if f["status"] == "TESTED" \
                and f["all_windows_positive"] \
                and f["pooled_skill"] >= MARGIN \
                and f["holm_p"] < 0.05:
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
            "derived_licensing_facts": {
                f"{ck}:{wk}": v
                for (ck, wk), v in sorted(derived.items())},
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
           "design": DESIGN_V3,
           "design_sha256": sha_file(REPO / DESIGN_V3),
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


def rebind(v1_result: Path, expected_v1_sha: str,
           expected_v1_design_sha: str, design_sha: str,
           run_root: Path, out_path: Path) -> dict:
    """C19/C21 (order @af1ca667): evidence-complete corrective
    re-adjudication, BOUND to the independently reviewed source
    identities before parsing. Reconstructs the missing target and
    anchor evidence from the already frozen local data plane (same
    windows, purges, thresholds — nothing selected or refitted),
    copies the v1 loss vectors with digest proof, and PROVES
    alignment by recomputing every prior-baseline loss vector
    exactly from the reconstructed observations. Authorized bounded
    CPU reconstruction; no new hypothesis, threshold, target,
    model, neural run or GPU."""
    import numpy as np
    # C23: supplied identities must EQUAL the carried reviewed
    # constants — checked before any file is opened
    for supplied, carried, label in (
            (expected_v1_sha, REVIEWED_V1_RESULT_SHA,
             "v1 result"),
            (expected_v1_design_sha, REVIEWED_V1_DESIGN_SHA,
             "v1 design"),
            (design_sha, REVIEWED_DESIGN_V3_SHA,
             "corrected design")):
        if supplied != carried:
            raise N4Refusal(
                f"supplied {label} identity is not the carried "
                "reviewed constant — the caller cannot choose the "
                "trust root; refused before opening any file")
    raw = v1_result.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != REVIEWED_V1_RESULT_SHA:
        raise N4Refusal(
            f"source v1 result {actual[:12]} differs from the "
            "carried reviewed identity — refused before parsing")
    v1_design_path = REPO / ("docs/audits/evidence/"
                             "N4_TARGET_AUDIT_DESIGN_2026_09_04"
                             ".json")
    v1_design_actual = sha_file(v1_design_path)
    if v1_design_actual != REVIEWED_V1_DESIGN_SHA:
        raise N4Refusal("original v1 design bytes differ from the "
                        "reviewed identity")
    design = validate_design(design_sha)
    v1 = strict_json_bytes(raw)
    dp = v1.get("data_plane_digests", {})
    if dp.get("frozen_csv") != FROZEN_SHA \
            or dp.get("n2_npz") != N2_NPZ_SHA:
        raise N4Refusal(
            "source data-plane identities differ from the frozen "
            "contract — a copied digest is not source authority")
    plane = _load_dev_arrays(run_root)
    if dp.get("n2_bundle_geometry") != plane["digests"][
            "n2_bundle_geometry"]:
        raise N4Refusal("source geometry identity differs from "
                        "the committed N2 bundle")
    targets = build_targets(plane)
    geometry = plane["geometry"]
    fit_all = list(range(*geometry["windows"]["w1"]["fit"]))
    lm_thresholds = {}
    for h in PROPOSED["large_move"]:
        q = float(np.quantile(
            np.abs(np.asarray(targets[f"r_h{h}"])[fit_all]),
            EXCEEDANCE_Q))
        targets[f"lm_h{h}"] = (
            np.abs(targets[f"r_h{h}"]) >= q).astype(int)
        lm_thresholds[f"h{h}"] = round(q, 8)
    v1_thresholds = v1["decision_constants"]["lm_thresholds"]
    if {k: round(float(v), 8)
            for k, v in v1_thresholds.items()} != lm_thresholds:
        raise N4Refusal("reconstructed lm thresholds differ from "
                        "the frozen v1 thresholds")
    wks = ("w1", "w2", "w3", "w4")
    records = {}
    alignment = {}
    for ck, (tkey, h, kind) in CANDIDATE_TABLE.items():
        per_w_v1 = v1["per_window_records"][ck]
        y_full = np.asarray(targets[tkey])
        records[ck] = {}
        for wk in wks:
            fit, cal, sc = _window_roles(geometry, wk, h)
            if h == 24:
                usable = plane["usable24"]
                fit = [r for r in fit if usable[r]]
                cal = [r for r in cal if usable[r]]
                sc = [r for r in sc if usable[r]]
            rec_v1 = per_w_v1[wk]
            losses_v1 = rec_v1.get("losses")
            losses = (None if losses_v1 is None else {
                LEGACY_ARM_ALIAS.get(a, a): list(v)
                for a, v in losses_v1.items()})
            ys = y_full[sc]
            yfc = np.concatenate([y_full[fit], y_full[cal]])
            if losses is not None:
                if len(losses["prior"]) != len(sc):
                    raise N4Refusal(
                        f"{ck}/{wk}: v1 loss cardinality differs "
                        "from the reconstructed anchor set")
                # alignment proof: recompute the PRIOR losses
                # exactly from reconstructed observations
                if kind in REQUIRED_CLASSES:
                    n_classes = 3 if kind == "class3" else 2
                    counts = np.bincount(yfc.astype(int),
                                         minlength=n_classes)
                    prior = np.clip(counts / counts.sum(),
                                    1e-12, None)
                    prior = prior / prior.sum()
                    recomputed = [round(float(
                        -np.log(max(prior[int(v)], 1e-12))), 8)
                        for v in ys]
                else:
                    med = float(np.median(yfc))
                    recomputed = [round(float((v - med) ** 2), 8)
                                  for v in ys]
                if recomputed != losses["prior"]:
                    raise N4Refusal(
                        f"{ck}/{wk}: recomputed prior losses do "
                        "not equal the frozen v1 vector — "
                        "observation alignment unproven")
                alignment[f"{ck}:{wk}"] = "prior_vector_exact"
            rec = {"window": wk, "n_score": len(sc),
                   "anchors_sha256": sha_obj(
                       [int(r) for r in sc]),
                   "losses": losses}
            if kind in REQUIRED_CLASSES:
                rec["labels"] = [int(v) for v in ys]
            else:
                rec["target_values"] = [
                    round(float(v), 8) for v in ys]
            records[ck][wk] = rec

    def loss_digest(source):
        return sha_obj([[ck, wk,
                         None if source[ck][wk].get("losses")
                         is None else source[ck][wk]["losses"]]
                        for ck in sorted(source)
                        for wk in sorted(source[ck])])
    v1_mapped = {ck: {wk: {"losses": None
                           if per_w[wk].get("losses") is None
                           else {LEGACY_ARM_ALIAS.get(a, a): v
                                 for a, v in per_w[wk]["losses"]
                                 .items()}}
                      for wk in per_w}
                 for ck, per_w in
                 v1["per_window_records"].items()}
    if loss_digest(records) != loss_digest(v1_mapped):
        raise N4Refusal("copied loss vectors drifted")
    adjudication = adjudicate(records)
    out = {"schema": "agent_multi.n4_screen_result.v3",
           "order": "agent-multi@af1ca667 C17-C21",
           "chronology": {
               "v1": "original design sealed BEFORE scores; "
                     "licensing defective (hard-coded class "
                     "subset)",
               "v2": "auditor-prescribed post-result correction "
                     "over the frozen v1 losses; NOT a "
                     "predeclaration; evidence boundary was "
                     "incomplete (producer-declared supports)",
               "v3": "AUDITOR_PRESCRIBED_CORRECTIVE_ADJUDICATION_"
                     "NO_NEW_HYPOTHESIS — evidence-complete: "
                     "per-observation labels/targets and anchors "
                     "reconstructed from the frozen data plane, "
                     "prior vectors re-derived exactly, licensing "
                     "DERIVED, nothing predeclared and nothing "
                     "refitted"},
           "classification": v1["classification"],
           "design": DESIGN_V3,
           "design_sha256": design_sha,
           "data_plane_digests": plane["digests"],
           "decision_constants": executable_binding(),
           "lm_thresholds": lm_thresholds,
           "per_window_records": records,
           **adjudication,
           "alignment_proofs": alignment,
           "supersession": {
               "v1_result_sha256": expected_v1_sha,
               "v1_design_sha256": expected_v1_design_sha,
               "v2_result": "N4_SCREEN_RESULT_V2_2026_09_04.json",
               "v2_status": "PRESERVED UNCHANGED, SUPERSEDED "
                            "(incomplete evidence boundary "
                            "disclosed)",
               "v1_status": "PRESERVED UNCHANGED, SUPERSEDED",
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
                         "separately motivated scientific design",
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
    r = sub.add_parser("rebind")
    r.add_argument("--v1-result", required=True)
    r.add_argument("--expected-v1-sha", required=True)
    r.add_argument("--expected-v1-design-sha", required=True)
    r.add_argument("--design-sha", required=True)
    r.add_argument("--run-root", required=True)
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
            out = rebind(Path(args.v1_result),
                         args.expected_v1_sha,
                         args.expected_v1_design_sha,
                         args.design_sha, Path(args.run_root),
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
