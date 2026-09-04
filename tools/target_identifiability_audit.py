#!/usr/bin/env python3
"""TARGET_REPRESENTATION_IDENTIFIABILITY_AUDIT — N1 full execution
(order agent-multi@89d099aa; predeclared in
TARGET_IDENTIFIABILITY_PREDECLARATION_N1_2026_09_03.json, which
supersedes the N0 predeclaration without rewriting it).

Question: is useful out-of-sample signal for realized volatility h6
detectable from the available causal inputs before compression, or is
predictability beyond persistence not demonstrated?

Units are CAUSAL SCORE WINDOWS (four, all ending strictly before the
first consumed screen-v2 monitor row). Arms are paired treatments
within each unit: literal persistence (no fitted coefficient),
calibrated one-variable autoregression, direct ridge on the causal
raw inputs, and a direct end-to-end GRU (no candidate extractor).
Frozen screen-v2 branch/fusion evidence is historical context only.

Subcommands: materialize / worker / supervise / aggregate. The
supervisor is BOUNDED to this ledger; no retry after a terminal
scientific result; infrastructure interruptions resume only through
the same ledgered identity; stop-file honored with terminate+reap."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.experiment_runtime import (  # noqa: E402
    RunDirectory, RuntimePreflightError, atomic_write_json,
    aggregate as runtime_aggregate, run_one_unit, sha_file, sha_obj,
    unit_id)
from agent_plugins.paired_inference import (  # noqa: E402
    holm_adjust, paired_t)

PREDECLARATION = (
    "docs/audits/evidence/"
    "TARGET_IDENTIFIABILITY_PREDECLARATION_N1_2026_09_03.json")
EXPERIMENT = "target_identifiability_n1"
WINDOW = 64
HORIZON = 6
STRIDE = 4
EMBARGO = math.ceil(HORIZON / STRIDE)
SEEDS = (101, 202, 303, 404)
MAX_UPDATES = 5000
MARGIN = 0.02
CPU_ARMS = ("literal_persistence", "calibrated_ar1", "direct_linear")
T_CRIT_DF3 = 3.182  # two-sided 95%, df = 3


def code_digest() -> str:
    files = [REPO / "tools/target_identifiability_audit.py",
             REPO / "agent_plugins/experiment_runtime.py",
             REPO / "agent_plugins/branch_pretraining.py",
             REPO / "agent_plugins/temporal_information.py"]
    return sha_obj({str(f.relative_to(REPO)): sha_file(f)
                    for f in files})


def role_geometry(n: int) -> dict:
    """Exact predeclared derivation. Returns window index ranges or
    the typed insufficient-units refusal facts."""
    origin0_limit = int(n * 0.85)
    frontier = int(origin0_limit * 0.82)
    cal_len = int(0.08 * n)
    length = (frontier - int(0.30 * n) - 3 * EMBARGO) // 4
    windows = {}
    end = frontier
    for k in range(4, 0, -1):
        start = end - length
        cal_lo = start - cal_len - EMBARGO
        if cal_lo <= 0 or length < 30:
            return {"sufficient": False, "n": n,
                    "reason": f"window {k} infeasible "
                              f"(cal_lo={cal_lo}, L={length})"}
        windows[f"w{k}"] = {
            "fit": [0, cal_lo], "cal": [cal_lo, start - EMBARGO],
            "score": [start, end]}
        end = start - EMBARGO
    # invariants: disjoint score rows, cal precedes score w/ embargo
    spans = sorted(v["score"] for v in windows.values())
    for a, b in zip(spans, spans[1:]):
        if b[0] - a[1] < EMBARGO:
            return {"sufficient": False, "n": n,
                    "reason": "score windows closer than the embargo"}
    return {"sufficient": True, "n": n, "frontier": frontier,
            "embargo_rows": EMBARGO, "cal_len": cal_len,
            "score_len": length, "windows": windows}


def _identity(arm: str, window_key: str, seed: int) -> dict:
    return {"experiment": EXPERIMENT, "family": arm,
            "window": WINDOW, "latent": 0, "budget": MAX_UPDATES,
            "seed": seed, "origin": window_key, "treatment": arm}


def materialize_inputs(root: Path, pretrain_dir: Path, *,
                       max_windows: int, stride: int) -> dict:
    import numpy as np
    from agent_plugins.branch_pretraining import (
        build_step_index, collect_preprocessed_windows,
        load_fit_slice, realized_volatility_targets,
        validate_contract)
    from agent_plugins.pretrained_branch_loader import verify_source
    split_contract = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
         "eth_nested_split_contract_o2022_paired_v1.json").read_text())
    data_path = Path(split_contract["source_csv"])
    source = verify_source(pretrain_dir, REPO, data_path)
    contract = source["contract"]
    parsed = validate_contract(contract)
    df, ordered, close_col = load_fit_slice(data_path, contract)
    closes = df[close_col].to_numpy()
    vol_spec = contract["objectives"]["volatility"]
    annualization = vol_spec.get("annualization")
    periods = (None if annualization in (None, "none")
               else annualization["periods_per_year"])
    env_source = contract["observation_pipeline"]["source_config"]
    env_config = json.loads(
        (Path(env_source) if Path(env_source).is_absolute()
         else REPO / env_source).read_text())
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    path = root / "inputs" / f"windows_w{WINDOW}.npz"
    if not path.exists():
        warmup = max(int(parsed["warmup_bars"]), WINDOW)
        steps = build_step_index(len(df), warmup, max(1, stride),
                                 12, max_windows)
        contract_w = {**contract, "window_size": WINDOW}
        env_w = {**env_config, "window_size": WINDOW}
        windows = collect_preprocessed_windows(df, contract_w, env_w,
                                               steps)
        target = realized_volatility_targets(
            closes, steps, [HORIZON],
            float(vol_spec.get("epsilon", 1e-8)), periods)[:, 0]
        trailing = realized_volatility_targets(
            closes, [max(0, t - HORIZON) for t in steps], [HORIZON],
            float(vol_spec.get("epsilon", 1e-8)), periods)[:, 0]
        keep = np.isfinite(target) & np.isfinite(trailing)
        np.savez_compressed(
            path, windows=windows[keep].astype("float32"),
            target=np.asarray(target)[keep].astype("float64"),
            trailing=np.asarray(trailing)[keep].astype("float64"))
    return {f"input_w{WINDOW}": sha_file(path),
            "data_csv": sha_file(data_path),
            "pretrain_generation": sha_file(
                Path(pretrain_dir) / "generation.json"),
            "code": code_digest(),
            "config": sha_file(REPO / PREDECLARATION)}


def materialize(root: Path, pretrain_dir: Path, *,
                max_windows: int, stride: int) -> dict:
    import numpy as np
    digests = materialize_inputs(root, pretrain_dir,
                                 max_windows=max_windows,
                                 stride=stride)
    data = np.load(root / "inputs" / f"windows_w{WINDOW}.npz",
                   allow_pickle=False)
    geometry = role_geometry(len(data["target"]))
    if not geometry["sufficient"]:
        run = RunDirectory(root / "diagnostic")
        atomic_write_json(run.root / "INSUFFICIENT_UNITS.json", {
            "schema": ("agent_multi.target_identifiability_"
                       "insufficient.v1"),
            "verdict": "INCONCLUSIVE_INSUFFICIENT_UNITS",
            "geometry": geometry})
        return {"units": 0,
                "verdict": "INCONCLUSIVE_INSUFFICIENT_UNITS",
                "geometry": geometry}
    units = []
    for wk in sorted(geometry["windows"]):
        for arm in CPU_ARMS:
            units.append(_identity(arm, wk, 0))
        for seed in SEEDS:
            units.append(_identity("direct_temporal", wk, seed))
    run = RunDirectory(root / "diagnostic")
    digest = run.write_ledger({
        "schema": "agent_multi.target_identifiability_ledger.v2",
        "experiment": EXPERIMENT,
        "units": [{"unit_id": unit_id(u), "identity": u}
                  for u in units],
        "digests": digests,
        "campaign_wall_ceiling_s": 21600.0,
        "unit_timeout_s": 3600.0,
        "predeclaration": PREDECLARATION,
        "role_geometry": geometry,
        "frozen_arms_note": (
            "frozen best branch and frozen fusion are historical "
            "context from the accepted screen-v2 report — never "
            "executed, never pooled with these units")})
    return {"units": len(units), "ledger_digest": digest,
            "geometry": geometry}


def _r2(pred, true):
    residual = float(((true - pred) ** 2).sum())
    total = float(((true - true.mean()) ** 2).sum()) or 1.0
    return 1.0 - residual / total


def execute_unit(identity: dict, root: Path,
                 log_path: Path) -> dict:
    import numpy as np
    from agent_plugins.temporal_information import ridge_fit_cal_score
    run = RunDirectory(root / "diagnostic")
    geometry = run.ledger()["role_geometry"]
    roles = geometry["windows"][identity["origin"]]
    data = np.load(root / "inputs" / f"windows_w{WINDOW}.npz",
                   allow_pickle=False)
    windows, target = data["windows"], data["target"]
    trailing = data["trailing"]
    fit_i = np.arange(*roles["fit"])
    cal_i = np.arange(*roles["cal"])
    sc_i = np.arange(*roles["score"])
    arm = identity["treatment"]
    started = time.perf_counter()
    if arm == "literal_persistence":
        # NO fitted coefficient: the trailing value IS the prediction
        result = {"score_r2": round(_r2(trailing[sc_i],
                                        target[sc_i]), 4)}
    elif arm == "calibrated_ar1":
        score = ridge_fit_cal_score(
            trailing[fit_i].reshape(-1, 1), target[fit_i],
            trailing[cal_i].reshape(-1, 1), target[cal_i],
            trailing[sc_i].reshape(-1, 1), target[sc_i])["score"]
        result = {"score_r2": round(float(score), 4)}
    elif arm == "direct_linear":
        flat = windows.reshape(len(windows), -1)
        score = ridge_fit_cal_score(
            flat[fit_i], target[fit_i], flat[cal_i], target[cal_i],
            flat[sc_i], target[sc_i])["score"]
        result = {"score_r2": round(float(score), 4)}
    elif arm == "direct_temporal":
        import torch
        if os.environ.get("SCREEN_V2_DEVICE") != "cuda" or \
                not torch.cuda.is_available():
            raise RuntimeError(
                "TEMPORAL_ARM_REQUIRES_BOUND_CUDA: no silent CPU "
                "fallback (order @89d099aa §5)")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "exactly ONE CUDA device must be bound")
        from agent_plugins.component_config import deep_merge_strict
        from app.plugin_loader import load_plugin
        plugin_class, _ = load_plugin("feature_branch.plugins",
                                      "gru_branch")
        params = deep_merge_strict(
            plugin_class.plugin_params,
            {"hidden_size": 64, "num_layers": 1, "dropout": 0.0,
             "bidirectional": False}, path="n1")
        torch.manual_seed(identity["seed"])
        module, dim = plugin_class.build(windows.shape[2], WINDOW,
                                         params)
        dev = torch.device("cuda")
        module = module.to(dev)
        head = torch.nn.Linear(dim, 1).to(dev)
        optimizer = torch.optim.Adam(
            list(module.parameters()) + list(head.parameters()),
            lr=1e-3)
        x = torch.tensor(windows, dtype=torch.float32, device=dev)
        y = torch.tensor(target, dtype=torch.float32,
                         device=dev).unsqueeze(-1)
        fit_t = torch.tensor(fit_i)
        generator = torch.Generator().manual_seed(identity["seed"])
        best_cal, best_state = -1e18, None
        for update in range(int(identity["budget"])):
            batch = fit_t[torch.randint(
                0, len(fit_t), (min(256, len(fit_t)),),
                generator=generator)].to(dev)
            optimizer.zero_grad()
            out = head(module(x[batch]))
            loss = torch.nn.functional.mse_loss(out, y[batch])
            loss.backward()
            optimizer.step()
            if (update + 1) % 100 == 0 or \
                    update + 1 == int(identity["budget"]):
                module.eval()
                with torch.no_grad():
                    cal_pred = head(module(x[cal_i])).squeeze(
                        -1).cpu().numpy()
                module.train()
                cal_score = _r2(cal_pred, target[cal_i])
                if cal_score > best_cal:
                    best_cal = cal_score
                    best_state = (
                        {k: v.detach().clone() for k, v in
                         module.state_dict().items()},
                        {k: v.detach().clone() for k, v in
                         head.state_dict().items()})
        if best_state is not None:
            module.load_state_dict(best_state[0])
            head.load_state_dict(best_state[1])
        module.eval()
        with torch.no_grad():
            sc_pred = head(module(x[sc_i])).squeeze(-1).cpu().numpy()
        result = {"score_r2": round(_r2(sc_pred, target[sc_i]), 4),
                  "calibration_r2": round(float(best_cal), 4),
                  "device": "cuda",
                  "updates": int(identity["budget"])}
    else:
        raise RuntimeError(
            f"unknown arm {arm} — frozen arms are citations")
    result["wall_s"] = round(time.perf_counter() - started, 1)
    result["score_rows"] = roles["score"]
    log_path.write_text(json.dumps({"unit": identity, **result},
                                   default=float))
    return result


def worker_main(args) -> int:
    root = Path(args.run_root)
    run = RunDirectory(root / "diagnostic")
    ledger = run.ledger()
    identity = None
    for unit in ledger["units"]:
        if unit["unit_id"] == args.unit:
            identity = unit["identity"]
    if identity is None:
        raise RuntimePreflightError(f"unit {args.unit} not in ledger")
    expected = {"code": code_digest(),
                "config": sha_file(REPO / PREDECLARATION),
                f"input_w{WINDOW}": sha_file(
                    root / "inputs" / f"windows_w{WINDOW}.npz")}
    outcome = run_one_unit(
        run, args.unit,
        lambda ident, log: execute_unit(ident, root, log),
        expected_digests=expected,
        timeout_s=float(args.timeout or 3600))
    print(json.dumps({"unit": args.unit,
                      "state": outcome["state"]}))
    return 0 if outcome["state"] == "COMPLETED" else 1


# ------------------------------------------------------------------ #
# aggregate: recomputed from terminal unit records, never declared   #
# ------------------------------------------------------------------ #

def aggregate_final(root: Path) -> dict:
    from statistics import mean, stdev
    run = RunDirectory(root / "diagnostic")
    ledger = run.ledger()
    states = run.states()
    expected = [u["unit_id"] for u in ledger["units"]]
    by_unit = {u["unit_id"]: u["identity"] for u in ledger["units"]}
    problems = []
    for uid in expected:
        state = states.get(uid)
        if state is None:
            problems.append({"unit": uid, "why": "missing state"})
        elif state["state"] != "COMPLETED":
            problems.append({"unit": uid,
                             "why": state["state"],
                             "arm": by_unit[uid]["treatment"],
                             "window": by_unit[uid]["origin"]})
    trace = {"schema": ("agent_multi.target_identifiability_"
                        "interpretation.v1"),
             "predeclaration": PREDECLARATION,
             "problems_preserved": problems}
    if problems:
        trace["verdict"] = "INCONCLUSIVE_INFRASTRUCTURE"
        trace["cause"] = ("failed/timed-out/missing units preserved "
                          "in the verdict — never dropped")
        return trace
    results = runtime_aggregate(run, expected)
    window_keys = sorted(ledger["role_geometry"]["windows"])
    if len(window_keys) < 4:
        trace["verdict"] = "INCONCLUSIVE_INSUFFICIENT_UNITS"
        return trace
    # exact arm pairing per score window + all four temporal seeds
    per_window: dict = {}
    for uid, res in results.items():
        ident = by_unit[uid]
        wk, arm = ident["origin"], ident["treatment"]
        per_window.setdefault(wk, {"temporal_seeds": {}})
        if arm == "direct_temporal":
            per_window[wk]["temporal_seeds"][ident["seed"]] = \
                res["score_r2"]
        else:
            per_window[wk][arm] = res["score_r2"]
    for wk in window_keys:
        entry = per_window.get(wk, {})
        for arm in CPU_ARMS:
            if arm not in entry:
                trace["verdict"] = "INCONCLUSIVE_INFRASTRUCTURE"
                trace["cause"] = f"{wk} lacks arm {arm} — pairing " \
                                 "broken"
                return trace
        if set(entry["temporal_seeds"]) != set(SEEDS):
            trace["verdict"] = "INCONCLUSIVE_INFRASTRUCTURE"
            trace["cause"] = (f"{wk} temporal seeds "
                              f"{sorted(entry['temporal_seeds'])} != "
                              f"{list(SEEDS)}")
            return trace
        entry["direct_temporal"] = round(mean(
            entry["temporal_seeds"].values()), 4)
        entry["temporal_seed_dispersion"] = round(stdev(
            entry["temporal_seeds"].values()), 4)

    def paired(arm: str) -> dict:
        diffs = [per_window[wk][arm]
                 - per_window[wk]["literal_persistence"]
                 for wk in window_keys]
        # repaired reusable helper (order @8fce8da0 §4 R3):
        # rejects non-finite scores; zero-variance differences
        # yield predeclared FINITE outcomes instead of the retired
        # inf -> linspace -> NaN path
        stats = paired_t(diffs, t_crit=T_CRIT_DF3)
        all_positive = all(per_window[wk][arm] > 0
                           for wk in window_keys)
        return {"per_window": {wk: per_window[wk][arm]
                               for wk in window_keys},
                "diffs_vs_literal": [round(d, 4) for d in diffs],
                "mean_diff": round(stats["mean"], 4),
                "ci95": [round(stats["ci95"][0], 4),
                         round(stats["ci95"][1], 4)],
                "t_stat": (round(stats["t_stat"], 3)
                           if stats["t_stat"] is not None else None),
                "p_one_sided": round(stats["p_one_sided"], 4),
                "zero_variance": stats["zero_variance"],
                "all_windows_positive": all_positive}

    analysis = {arm: paired(arm)
                for arm in ("calibrated_ar1", "direct_linear",
                            "direct_temporal")}

    direct_arms = ["direct_linear", "direct_temporal"]
    pvals = {arm: analysis[arm]["p_one_sided"]
             for arm in direct_arms}
    # step-down Holm WITH the cumulative maximum (monotone
    # non-decreasing adjusted p-values) — repaired per R3
    holm = holm_adjust(pvals)
    advancing = []
    hinting = []
    for arm in direct_arms:
        a = analysis[arm]
        advances = (a["all_windows_positive"]
                    and a["mean_diff"] >= MARGIN
                    and a["ci95"][0] > 0
                    and holm[arm] < 0.05)
        analysis[arm]["holm_p"] = round(holm[arm], 4)
        analysis[arm]["advances"] = advances
        if advances:
            advancing.append(arm)
        elif a["all_windows_positive"]:
            hinting.append(arm)
    if advancing:
        verdict = "REPRESENTATION_BOTTLENECK_DEMONSTRATED"
    elif hinting:
        verdict = "INCONCLUSIVE_DISCORDANT"
    else:
        verdict = "PREDICTABILITY_NOT_DEMONSTRATED"
    trace.update({
        "verdict": verdict,
        "per_window": per_window,
        "paired_analysis": analysis,
        "holm_pvalues": {k: round(v, 4) for k, v in holm.items()},
        "calibrated_ar1_note": ("a stronger baseline reported "
                                "SEPARATELY — never a replacement "
                                "for literal persistence"),
        "frozen_historical_context": {
            "best_branch": "all 25 screen-v2 survivors "
                           "NOT_DEMONSTRATED (accepted report)",
            "fusion": "all 5 variants DOES_NOT_ADVANCE"},
        "decision_trace": {
            "advancing_direct_arms": advancing,
            "all_window_positive_without_license": hinting,
            "margin": MARGIN, "t_crit_df3": T_CRIT_DF3}})
    return trace


# ------------------------------------------------------------------ #
# bounded supervisor                                                 #
# ------------------------------------------------------------------ #

def supervise(args) -> int:
    root = Path(args.run_root)
    pretrain_dir = Path(args.pretrain_dir)
    summary = materialize(root, pretrain_dir,
                          max_windows=args.max_windows,
                          stride=args.stride)
    print(json.dumps({"materialized": {
        k: summary[k] for k in ("units",)
        if k in summary}}), flush=True)
    if summary.get("verdict") == "INCONCLUSIVE_INSUFFICIENT_UNITS":
        print(json.dumps(summary["geometry"]), flush=True)
        return 0
    # device + plugin + immutable-input preflights BEFORE any unit
    import torch
    cuda_ok = torch.cuda.is_available() and \
        torch.cuda.device_count() == 1
    from agent_plugins.dispatch_authorization import (
        resolve_required_entry_points)
    entry_points = resolve_required_entry_points(REPO)
    run = RunDirectory(root / "diagnostic")
    ledger = run.ledger()
    stop_file = root / "STOP"
    children: dict = {}
    started = time.time()
    last_beat = 0.0

    def spawnable(uid, state):
        ident = {u["unit_id"]: u["identity"]
                 for u in ledger["units"]}[uid]
        if state["state"] == "PENDING":
            return True
        # infra states resume through the SAME identity; FAILED is a
        # terminal scientific result — never retried automatically
        return state["state"] in ("INTERRUPTED", "TIMED_OUT")

    while True:
        states = run.states()
        pend = []
        for uid, st in states.items():
            ident = {u["unit_id"]: u["identity"]
                     for u in ledger["units"]}[uid]
            temporal = ident["treatment"] == "direct_temporal"
            if temporal and not cuda_ok:
                continue  # stays PENDING with the typed reason below
            if spawnable(uid, st):
                pend.append((uid, temporal))
        running = [u for u, s in states.items()
                   if s["state"] == "RUNNING"]
        if stop_file.exists():
            for pid, proc in children.items():
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            print("[stop-file] workers terminated and reaped; "
                  "durable states preserved", flush=True)
            break
        if not pend and not running:
            break
        children = {pid: pr for pid, pr in children.items()
                    if pr.poll() is None}
        while pend and len(children) < args.workers:
            uid, temporal = pend.pop(0)
            env = dict(os.environ)
            if temporal:
                env["SCREEN_V2_DEVICE"] = "cuda"
            else:
                env.pop("SCREEN_V2_DEVICE", None)
                env["CUDA_VISIBLE_DEVICES"] = ""
            cmd = [sys.executable, str(REPO / "tools" /
                                       "target_identifiability_audit.py"),
                   "worker", "--run-root", str(root),
                   "--unit", uid, "--timeout", "3600"]
            log = open(run.root / "logs" / f"worker_{uid}.out", "ab")
            proc = subprocess.Popen(cmd, stdout=log,
                                    stderr=subprocess.STDOUT,
                                    env=env)
            children[proc.pid] = proc
            states = run.states()  # avoid double-spawn same loop
        if time.time() - last_beat >= 60.0:
            def kill_child(pid):
                proc = children.get(pid)
                if proc is None:
                    return not Path(f"/proc/{pid}").exists()
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=30)
                return proc.poll() is not None
            run.heartbeat(
                current_unit=(running[0] if running else None),
                workers=args.workers,
                device_class="cuda+cpu" if cuda_ok else "cpu",
                extra={"phase": "diagnostic",
                       "elapsed_s": round(time.time() - started, 1),
                       "temporal_pending_reason": (
                           None if cuda_ok else
                           "NO_REVIEWED_CUDA_SLOT_FREE — temporal "
                           "units stay PENDING (no CPU fallback)"),
                       "entry_points_verified":
                           bool(entry_points)})
            for alert in run.watchdog(kill_child=kill_child):
                print(f"[watchdog] {json.dumps(alert)}", flush=True)
            last_beat = time.time()
        if time.time() - started > 21600:
            print("[ceiling] six-hour campaign wall — stopping",
                  flush=True)
            for proc in children.values():
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)
            break
        time.sleep(2.0)
    for proc in children.values():
        if proc.poll() is None:
            proc.wait(timeout=120)
    trace = aggregate_final(root)
    atomic_write_json(root / "INTERPRETATION_TRACE.json", trace)
    print(json.dumps({"verdict": trace["verdict"]}, indent=1),
          flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("materialize", "supervise"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-root", required=True)
        sp.add_argument("--pretrain-dir", required=True)
        sp.add_argument("--max-windows", type=int, default=2200)
        sp.add_argument("--stride", type=int, default=4)
        if name == "supervise":
            sp.add_argument("--workers", type=int, default=3)
    w = sub.add_parser("worker")
    w.add_argument("--run-root", required=True)
    w.add_argument("--unit", required=True)
    w.add_argument("--timeout", type=float, default=None)
    a = sub.add_parser("aggregate")
    a.add_argument("--run-root", required=True)
    a.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.cmd == "materialize":
        print(json.dumps(materialize(
            Path(args.run_root), Path(args.pretrain_dir),
            max_windows=args.max_windows, stride=args.stride),
            indent=1, default=str))
        return 0
    if args.cmd == "worker":
        return worker_main(args)
    if args.cmd == "supervise":
        return supervise(args)
    trace = aggregate_final(Path(args.run_root))
    if args.output:
        atomic_write_json(Path(args.output), trace)
    print(json.dumps(trace, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
