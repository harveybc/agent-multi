#!/usr/bin/env python3
"""TARGET_REPRESENTATION_IDENTIFIABILITY_AUDIT runner (order
@1649e7c0 §7 N0; predeclared in
TARGET_REPRESENTATION_IDENTIFIABILITY_PREDECLARATION_2026_09_03.json).

Question: did the candidate fail because the representation discarded
useful signal, or because realized-volatility h6 is not predictable
beyond persistence on the available causal data?

Arms per the predeclaration; selection stays INSIDE fit/calibration
causal folds; the consumed screen-v2 monitors and every intact
confirmation role are untouched. Atomic units on the C1-corrected
observable runtime.

This tool implements ``materialize``, ``worker`` and ``preflight``.
The §7 authorization covers implementation and the bounded preflight
(one fast + one heavy unit, <= 5000 updates, <= one hour); the FULL
diagnostic requires a separate order and is structurally refused
here (no supervise subcommand exists)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.experiment_runtime import (  # noqa: E402
    RunDirectory, RuntimePreflightError, atomic_write_json,
    run_one_unit, sha_file, sha_obj, unit_id)

PREDECLARATION = (
    "docs/audits/evidence/TARGET_REPRESENTATION_IDENTIFIABILITY_"
    "PREDECLARATION_2026_09_03.json")
EXPERIMENT = "target_identifiability_audit"
FOLDS = {
    "A": {"fit": (0.00, 0.43), "cal": (0.43, 0.53),
          "score": (0.55, 0.65)},
    "B": {"fit": (0.00, 0.53), "cal": (0.53, 0.63),
          "score": (0.65, 0.75)},
}
PURGE = 12
SEEDS = (101, 202, 303, 404)
MAX_UPDATES = 5000
WINDOW = 64


def code_digest() -> str:
    files = [REPO / "tools/target_identifiability_audit.py",
             REPO / "agent_plugins/experiment_runtime.py",
             REPO / "agent_plugins/branch_pretraining.py",
             REPO / "agent_plugins/temporal_information.py"]
    return sha_obj({str(f.relative_to(REPO)): sha_file(f)
                    for f in files})


def _identity(arm: str, fold: str, seed: int) -> dict:
    return {"experiment": EXPERIMENT, "family": arm,
            "window": WINDOW, "latent": 0, "budget": MAX_UPDATES,
            "seed": seed, "origin": fold, "treatment": arm}


def _fold_indices(n: int, fold: str):
    import numpy as np
    spec = FOLDS[fold]
    fit_lo, fit_hi = (int(n * spec["fit"][0]),
                      int(n * spec["fit"][1]))
    cal_lo, cal_hi = (int(n * spec["cal"][0]),
                      int(n * spec["cal"][1]))
    sc_lo, sc_hi = (int(n * spec["score"][0]),
                    int(n * spec["score"][1]))
    fit = np.arange(fit_lo, max(fit_lo, fit_hi - PURGE))
    cal = np.arange(cal_lo, max(cal_lo, cal_hi - PURGE))
    score = np.arange(sc_lo, sc_hi)
    return fit, cal, score


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
            closes, steps, [6], float(vol_spec.get("epsilon", 1e-8)),
            periods)[:, 0]
        trailing = realized_volatility_targets(
            closes, [max(0, t - 6) for t in steps], [6],
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
    digests = materialize_inputs(root, pretrain_dir,
                                 max_windows=max_windows,
                                 stride=stride)
    units = []
    for fold in FOLDS:
        units.append(_identity("persistence", fold, 0))
        units.append(_identity("direct_linear", fold, 0))
        for seed in SEEDS:
            units.append(_identity("direct_temporal", fold, seed))
    run = RunDirectory(root / "diagnostic")
    digest = run.write_ledger({
        "schema": "agent_multi.target_identifiability_ledger.v1",
        "experiment": EXPERIMENT,
        "units": [{"unit_id": unit_id(u), "identity": u}
                  for u in units],
        "digests": digests,
        "campaign_wall_ceiling_s": 21600.0,
        "unit_timeout_s": 3600.0,
        "predeclaration": PREDECLARATION,
        "frozen_arms_note": (
            "arms 4 (best frozen branch) and 5 (candidate fusion) "
            "are FROZEN citations from the accepted screen-v2 "
            "report — never re-executed here")})
    return {"units": len(units), "ledger_digest": digest}


def execute_unit(identity: dict, root: Path,
                 log_path: Path) -> dict:
    import numpy as np
    from agent_plugins.temporal_information import ridge_fit_cal_score
    data = np.load(root / "inputs" / f"windows_w{WINDOW}.npz",
                   allow_pickle=False)
    windows, target = data["windows"], data["target"]
    trailing = data["trailing"].reshape(-1, 1)
    fit_i, cal_i, sc_i = _fold_indices(len(windows),
                                       identity["origin"])
    arm = identity["treatment"]
    started = time.perf_counter()
    if arm == "persistence":
        score = ridge_fit_cal_score(
            trailing[fit_i], target[fit_i], trailing[cal_i],
            target[cal_i], trailing[sc_i], target[sc_i])["score"]
        result = {"score_r2": round(float(score), 4)}
    elif arm == "direct_linear":
        flat = windows.reshape(len(windows), -1)
        score = ridge_fit_cal_score(
            flat[fit_i], target[fit_i], flat[cal_i], target[cal_i],
            flat[sc_i], target[sc_i])["score"]
        result = {"score_r2": round(float(score), 4)}
    elif arm == "direct_temporal":
        import torch
        from agent_plugins.component_config import deep_merge_strict
        from app.plugin_loader import load_plugin
        plugin_class, _ = load_plugin("feature_branch.plugins",
                                      "gru_branch")
        params = deep_merge_strict(
            plugin_class.plugin_params,
            {"hidden_size": 64, "num_layers": 1, "dropout": 0.0,
             "bidirectional": False}, path="n0")
        device = os.environ.get("SCREEN_V2_DEVICE", "cpu")
        torch.manual_seed(identity["seed"])
        module, dim = plugin_class.build(windows.shape[2], WINDOW,
                                         params)
        dev = torch.device(device)
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

        def r2(pred, true):
            residual = float(((true - pred) ** 2).sum())
            total = float(((true - true.mean()) ** 2).sum()) or 1.0
            return 1.0 - residual / total

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
                cal_score = r2(cal_pred, target[cal_i])
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
        result = {"score_r2": round(float(r2(sc_pred,
                                             target[sc_i])), 4),
                  "calibration_r2": round(float(best_cal), 4),
                  "device": device,
                  "updates": int(identity["budget"])}
    else:
        raise RuntimeError(f"unknown arm {arm} — arms 4/5 are "
                           "frozen citations, never executed")
    result["wall_s"] = round(time.perf_counter() - started, 1)
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


def preflight(args) -> int:
    """§7 bounded preflight: EXACTLY one fast and one heavy unit,
    mechanics only, <= one hour total. The full diagnostic has no
    entry point in this tool."""
    root = Path(args.run_root)
    summary = materialize(root, Path(args.pretrain_dir),
                          max_windows=args.max_windows,
                          stride=args.stride)
    print(json.dumps({"materialized": summary}))
    run = RunDirectory(root / "diagnostic")
    fast = unit_id(_identity("direct_linear", "A", 0))
    heavy = unit_id(_identity("direct_temporal", "A", 101))
    started = time.time()
    outcomes = {}
    for label, uid in (("fast", fast), ("heavy", heavy)):
        if time.time() - started > 3600:
            print(json.dumps({"refusal": "one-hour preflight "
                              "ceiling reached"}))
            return 1
        expected = {"code": code_digest(),
                    "config": sha_file(REPO / PREDECLARATION),
                    f"input_w{WINDOW}": sha_file(
                        root / "inputs" /
                        f"windows_w{WINDOW}.npz")}
        outcome = run_one_unit(
            run, uid,
            lambda ident, log: execute_unit(ident, root, log),
            expected_digests=expected, timeout_s=1800.0)
        outcomes[label] = {"unit": uid,
                           "state": outcome["state"],
                           "result": outcome.get("result")}
    report = {"schema": ("agent_multi.target_identifiability_"
                         "preflight.v1"),
              "classification": "MECHANICS_ONLY_PREFLIGHT",
              "predeclaration": PREDECLARATION,
              "wall_s": round(time.time() - started, 1),
              "outcomes": outcomes,
              "note": ("mechanics only — NO scientific claim; the "
                       "full diagnostic requires a separate order")}
    atomic_write_json(root / "PREFLIGHT_REPORT.json", report)
    print(json.dumps(report, indent=1, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("materialize")
    for sp in (m,):
        sp.add_argument("--run-root", required=True)
        sp.add_argument("--pretrain-dir", required=True)
        sp.add_argument("--max-windows", type=int, default=2200)
        sp.add_argument("--stride", type=int, default=4)
    w = sub.add_parser("worker")
    w.add_argument("--run-root", required=True)
    w.add_argument("--unit", required=True)
    w.add_argument("--timeout", type=float, default=None)
    pf = sub.add_parser("preflight")
    pf.add_argument("--run-root", required=True)
    pf.add_argument("--pretrain-dir", required=True)
    pf.add_argument("--max-windows", type=int, default=2200)
    pf.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    if args.cmd == "materialize":
        print(json.dumps(materialize(
            Path(args.run_root), Path(args.pretrain_dir),
            max_windows=args.max_windows, stride=args.stride)))
        return 0
    if args.cmd == "worker":
        return worker_main(args)
    return preflight(args)


if __name__ == "__main__":
    raise SystemExit(main())
