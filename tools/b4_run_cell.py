#!/usr/bin/env python3
"""The ONE B4 runner (order @61622469, B4-E5).

Every B4 execution — the CPU mechanics replay now, the later bounded
GPU preflight, any future B4 cell — goes through THIS runner. The
launch interface selects only a reviewed cell id, the materialization
root, an output root and the device. Every scientific and budget
value comes from the materialized cell; overriding any of them is
impossible here because no such interface exists. Before any model or
environment construction the full authority chain is established at
point of use: sealed design + append-only amendment chain + final
code pins, live gym-fx lineage, owner-ratified observation v2, the
Musashi-reviewed fixed Alpaca cost model, data and causal split
identities, and the evidence-complete comparator population. All
artifacts are NON-PROMOTABLE."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import b4_authority as b4a  # noqa: E402


class RunnerRefusal(SystemExit):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rss_bytes() -> int:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * 4096


def _thermal_celsius() -> float | None:
    zones = sorted(Path("/sys/class/thermal").glob(
        "thermal_zone*/temp"))
    temps = []
    for z in zones:
        try:
            temps.append(int(z.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(temps) if temps else None


def make_guard_callback(peak: dict, rss_cap: int, thermal_cap: float):
    from stable_baselines3.common.callbacks import BaseCallback

    class ResourceGuardCallback(BaseCallback):
        def _on_step(self) -> bool:
            rss = _rss_bytes()
            peak["peak_rss_bytes"] = max(peak["peak_rss_bytes"], rss)
            if rss > rss_cap:
                peak["stop"] = f"RSS cap {rss_cap} exceeded at {rss}"
                return False
            t = _thermal_celsius()
            if t is not None:
                peak["peak_thermal_celsius"] = max(
                    peak.get("peak_thermal_celsius") or 0.0, t)
                if t > thermal_cap:
                    peak["stop"] = (f"thermal cap {thermal_cap}C "
                                    f"exceeded at {t:.1f}C")
                    return False
            return True

    return ResourceGuardCallback()


def load_cell(mat_root: Path, cell_id: str) -> dict:
    cells = json.loads((mat_root / "B4_CELL_CONFIGS.json").read_text())
    if cell_id not in cells:
        raise RunnerRefusal(
            f"REFUSED: cell {cell_id!r} is not a reviewed "
            "materialized cell")
    cell = cells[cell_id]
    cfg = cell["effective_config"]
    recomputed = hashlib.sha256(json.dumps(
        cfg, sort_keys=True, default=str).encode()).hexdigest()
    if recomputed != cell["config_sha256"]:
        raise RunnerRefusal("REFUSED: cell config digest mismatch")
    b4a.verify_cell_complete(cfg)
    binding = json.loads(
        (mat_root / "genesis" / "GENESIS_BINDING.json").read_text())
    if binding["binding"].get(cell_id) != cell["config_sha256"]:
        raise RunnerRefusal(
            "REFUSED: genesis binding does not carry this cell's "
            "final config digest")
    return cell


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=("B4 runner: cell id, materialization root, "
                     "output root and device ONLY — every scientific "
                     "and budget value comes from the reviewed cell"))
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--materialization-root", type=Path,
                    required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--device", required=True,
                    choices=["cpu", "cuda:0", "cuda:1"])
    args = ap.parse_args(argv)
    mat, out = args.materialization_root, args.output_root
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.device != "cpu":
        raise RunnerRefusal(
            "REFUSED: gpu_economic requires the separate explicit "
            "Musashi GPU authorization artifact; none exists — only "
            "the bounded CPU mechanics replay is authorized "
            "(order @61622469 §11)")

    # 1. Full authority chain at point of use (E3 items 1-8).
    packet = json.loads((mat / "B4_MATERIALIZATION.json").read_text())
    comparator_dir = packet.get("comparator_dir")
    if not comparator_dir:
        raise RunnerRefusal(
            "REFUSED: materialization carries no comparator anchor")
    authority = b4a.verify_full_authority_chain(Path(comparator_dir))
    lineage = authority["comparator"]["lineage"]

    # 2. The cell: complete, digest-bound, genesis-bound.
    cell = load_cell(mat, args.cell_id)
    cfg = dict(cell["effective_config"])
    if (cfg["gymfx_lineage_manifest_sha256"]
            != lineage["manifest_sha256"]):
        raise RunnerRefusal(
            "REFUSED: cell lineage differs from the live point-of-use "
            "gym-fx manifest")
    gmeta = packet["genesis"]["cells"].get(args.cell_id)
    if not gmeta or gmeta.get("n_updates") != 0:
        raise RunnerRefusal(
            "REFUSED: genesis metadata not zero-update for this cell")
    year = None
    for tok in args.cell_id.split("_"):
        if tok.startswith("o") and tok[1:].isdigit():
            year = int(tok[1:])
    seed = int(cfg["seed"])
    gzip = (mat / "genesis" / f"o{year}" / f"seed{seed}" /
            f"zero_update_genesis_seed{seed}.zip")
    if _sha_file(gzip) != gmeta["container_sha256"]:
        raise RunnerRefusal(
            "REFUSED: genesis container digest mismatch — foreign or "
            "altered genesis")

    mode = cfg["execution_modes"]["cpu_mechanics_replay"]

    # 3. Data: the cell's declared calibration-year role only —
    #    zero scored-year rows may reach a gradient.
    sb_spec = importlib.util.spec_from_file_location(
        "sbb_runner", REPO / "tools/screen_b_baselines.py")
    sb = importlib.util.module_from_spec(sb_spec)
    sb_spec.loader.exec_module(sb)
    df = sb.load_source()
    src_sha = hashlib.sha256(Path(sb.DATA).read_bytes()).hexdigest()
    if src_sha != cfg["source_data_sha256"]:
        raise RunnerRefusal(
            "REFUSED: source dataset differs from the cell's "
            "declared identity")
    train_year = int(mode["train_year"])
    origin = sb.materialize_origin(df, train_year, out / "origins")
    slice_df = None
    import pandas as pd
    slice_df = pd.read_csv(origin["csv"], parse_dates=["DATE_TIME"])
    years = sorted(slice_df["DATE_TIME"].dt.year.unique().tolist())
    if max(years) > train_year or year in years or 2025 in years:
        raise RunnerRefusal(
            f"REFUSED: training slice years {years} leak beyond the "
            f"calibration role (origin {year} sealed away)")

    run_cfg = dict(cfg)
    run_cfg["input_data_file"] = origin["csv"]
    run_cfg["quiet_mode"] = True
    run_cfg["device"] = "cpu"
    run_cfg["budget_max_env_steps"] = int(
        mode["budget_max_env_steps"])
    run_cfg["budget_max_updates"] = int(mode["budget_max_updates"])
    run_cfg["budget_max_wall_seconds"] = float(
        mode["budget_max_wall_seconds"])
    run_cfg["budget_stop_file"] = str(out / "STOP")
    for k in b4a.FORBIDDEN_CELL_KEYS:
        if run_cfg.get(k) not in (None, "", False):
            raise RunnerRefusal(
                f"REFUSED: hidden pretrained/replay/resume input {k}")

    rl = importlib.import_module(
        "pipeline_plugins.rl_pipeline_with_validation")
    env = rl._load_env_plugin(
        run_cfg["env_plugin"], run_cfg).make_env(run_cfg)
    obs_contract = importlib.import_module(
        "pipeline_plugins._observation_contract")
    obs_facts = obs_contract.verify_flattened_dimension(
        run_cfg, getattr(env, "observation_space", None))

    # 4. SAC from the CELL only; cold same-seed construction must
    #    reproduce the materialized genesis tensor identity.
    from agent_plugins.sac_agent import (Plugin as SacPlugin,
                                         _policy_tensor_hash)
    plugin = SacPlugin()
    env = plugin.wrap_env(env, run_cfg)
    build_cfg = dict(run_cfg)
    build_cfg["buffer_size"] = min(
        int(cfg["buffer_size"]), int(mode["replay_buffer_cap"]))
    build_cfg["net_arch"] = tuple(cfg["net_arch"])
    model = plugin.build(env, build_cfg)
    if (int(getattr(model, "_n_updates", -1)) != 0
            or int(getattr(model, "num_timesteps", -1)) != 0):
        raise RunnerRefusal(
            "REFUSED: constructed genesis is not zero-update")
    genesis_tensor_sha = _policy_tensor_hash(model.policy)
    if genesis_tensor_sha != gmeta["policy_tensor_sha256"]:
        raise RunnerRefusal(
            "REFUSED: same-seed construction does not reproduce the "
            "materialized genesis tensor identity")

    # 5. Bounded learning — F9.2 inside EVERY segment; RSS, thermal,
    #    stop-file and wall enforced during the segment.
    from stable_baselines3.common.callbacks import CallbackList
    peak = {"peak_rss_bytes": _rss_bytes(), "stop": None,
            "peak_thermal_celsius": _thermal_celsius()}
    thermal_available = peak["peak_thermal_celsius"] is not None
    segments = []
    for seg_steps in mode["learn_segments"]:
        try:
            rl._check_executing_budget(
                run_cfg, model, started_wall=t0,
                next_segment_timesteps=seg_steps)
        except rl.ExecutingBudgetExceeded as exc:
            segments.append({"requested_timesteps": seg_steps,
                             "pre_segment_refusal": str(exc)})
            break
        budget_cb = rl.make_executing_budget_callback(run_cfg, t0)
        guard_cb = make_guard_callback(
            peak, int(mode["rss_cap_bytes"]),
            float(mode["thermal_cap_celsius"]))
        before = int(model.num_timesteps)
        model.learn(total_timesteps=seg_steps,
                    callback=CallbackList([budget_cb, guard_cb]),
                    reset_num_timesteps=False, progress_bar=False)
        segments.append({
            "requested_timesteps": seg_steps,
            "real_timesteps": int(model.num_timesteps) - before,
            "cumulative_env_steps": int(model.num_timesteps),
            "cumulative_updates": int(model._n_updates),
            "budget_stop": budget_cb.budget_stop,
            "resource_stop": peak["stop"],
            "wall_seconds": round(time.time() - t0, 1)})
        try:
            rl._check_executing_budget(run_cfg, model,
                                       started_wall=t0)
            segments[-1]["post_segment_check"] = "within budget"
        except rl.ExecutingBudgetExceeded as exc:
            segments[-1]["post_segment_check"] = f"typed stop: {exc}"

    steps = int(model.num_timesteps)
    updates = int(model._n_updates)
    if steps > run_cfg["budget_max_env_steps"] or \
            updates > run_cfg["budget_max_updates"]:
        raise RunnerRefusal(
            f"CAP VIOLATED: {steps} steps / {updates} updates")

    # 6. Exact-update-stop and stop-file proofs (typed).
    try:
        rl._check_executing_budget(run_cfg, model, started_wall=t0,
                                   next_segment_timesteps=50)
        if updates >= run_cfg["budget_max_updates"]:
            raise RunnerRefusal(
                "PROOF FAILED: exhausted update budget did not "
                "refuse")
        exact_stop = "budgets not yet exhausted pre-segment"
    except rl.ExecutingBudgetExceeded as exc:
        exact_stop = f"typed refusal: {exc}"
    (out / "STOP").write_text("stop-file proof")
    try:
        rl._check_executing_budget(run_cfg, model, started_wall=t0)
        raise RunnerRefusal("PROOF FAILED: stop-file ignored")
    except rl.ExecutingBudgetExceeded as exc:
        stop_file_proof = f"typed refusal: {exc}"
    (out / "STOP").unlink()

    # 7. Finiteness + save/load roundtrip.
    import torch
    n_params = 0
    for p_t in model.policy.parameters():
        n_params += int(p_t.numel())
        if not torch.isfinite(p_t).all():
            raise RunnerRefusal(
                "PROOF FAILED: non-finite policy parameter")
    final_zip = out / f"mechanics_final_{args.cell_id}.zip"
    plugin.save(model, str(final_zip))
    trained_sha = _policy_tensor_hash(model.policy)
    reloaded = plugin.load(str(final_zip), env)
    if _policy_tensor_hash(reloaded.policy) != trained_sha or \
            int(reloaded._n_updates) != updates:
        raise RunnerRefusal("PROOF FAILED: save/load roundtrip")
    if updates > 0 and trained_sha == genesis_tensor_sha:
        raise RunnerRefusal(
            "PROOF FAILED: tensors unchanged after nonzero updates")

    record = {
        "schema": "agent_multi.b4_mechanics_cell_record.v2",
        "status": "MECHANICS_PROVEN_NON_PROMOTABLE",
        "g1_eligible": False,
        "runner": "tools/b4_run_cell.py",
        "cell": args.cell_id,
        "cell_config_sha256": cell["config_sha256"],
        "authority_chain": {
            "design_sha256": authority["chain"]["design_sha256"],
            "amendment_shas": authority["chain"]["amendment_shas"],
            "final_code_pins": authority["chain"]["final_code_pins"],
            "comparator_n_results":
                authority["comparator"]["n_results"],
            "comparator_n_ledger":
                authority["comparator"]["n_ledger"]},
        "train_slice": {"year": train_year, "years_present": years,
                        "csv_sha256": origin["csv_sha256"]},
        "genesis": {"container_sha256": gmeta["container_sha256"],
                    "policy_tensor_sha256": genesis_tensor_sha,
                    "zero_update_verified": True},
        "gymfx_lineage_manifest_sha256": lineage["manifest_sha256"],
        "gymfx_commit": lineage["commit"],
        "observation_dimension_facts": obs_facts,
        "caps": {k: mode[k] for k in (
            "budget_max_env_steps", "budget_max_updates",
            "budget_max_wall_seconds", "rss_cap_bytes",
            "thermal_cap_celsius")},
        "segments": segments,
        "observed": {"env_steps": steps,
                     "optimizer_updates": updates,
                     "wall_seconds": round(time.time() - t0, 1),
                     "peak_rss_bytes": peak["peak_rss_bytes"],
                     "peak_thermal_celsius":
                         peak.get("peak_thermal_celsius"),
                     "thermal_sensors_available": thermal_available,
                     "resource_stop": peak["stop"]},
        "proofs": {"exact_update_stop": exact_stop,
                   "stop_file": stop_file_proof,
                   "finite": (f"all {n_params} policy parameters "
                              f"finite after {updates} real updates"),
                   "trained_tensor_sha256": trained_sha},
        "sealed_2025_used": False,
    }
    b4a.verify_language(record, "mechanics record")
    (out / "B4_MECHANICS_CELL_RECORD.json").write_text(
        json.dumps(record, indent=1))
    print(json.dumps({"status": record["status"], "steps": steps,
                      "updates": updates,
                      "wall_s": record["observed"]["wall_seconds"],
                      "peak_rss_mib": round(
                          peak["peak_rss_bytes"] / 2 ** 20),
                      "peak_thermal_c":
                          peak.get("peak_thermal_celsius")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
