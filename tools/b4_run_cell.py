#!/usr/bin/env python3
"""The ONE B4 runner (orders @61622469 B4-E5, @9fb017e3 B4-P1..P5).

Every B4 execution goes through THIS runner. The launch interface
selects only a reviewed cell id, the materialization root, an output
root and the device. Every scientific and budget value comes from the
materialized cell; the GPU-preflight limits come ONLY from the exact
owner authorization record (carried path + digest — never CLI,
environment, materialization root or output root). Before any CUDA
initialization, model construction or output creation the full
authority chain is established at point of use, the owner-approved
artifact identities are compared byte-for-byte, and the comparator's
complete-envelope digests are RE-DERIVED from frozen geometry and
fixed cost bytes. All artifacts are NON-PROMOTABLE."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import b4_authority as b4a  # noqa: E402

GPU_PREFLIGHT_LABEL = "B4_GPU_PREFLIGHT_MECHANICS_AND_THROUGHPUT_ONLY"
GPU_BLOCKED_LABEL = "B4_GPU_PREFLIGHT_RESOURCE_BLOCKED"
GPU_FAILED_LABEL = "B4_GPU_PREFLIGHT_FAILED_TYPED"
SUBSTANTIAL_COMPUTE_MIB = 1024
# E11 (order @e8bb500f): GPU telemetry sampling is TIME-based
# and bounded — one probe per cadence tick, overhead measured
# at callback construction and reported in the record.
TEMP_SAMPLE_SECONDS = 5.0


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


# ------------------------- GPU telemetry --------------------------
def _nvidia_query(query: str, device: str = None) -> list:
    cmd = ["nvidia-smi", f"--query-gpu={query}",
           "--format=csv,noheader,nounits"]
    if device is not None:
        cmd += ["-i", str(device)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines()
            if line.strip()]


def gpu_inventory(device: str) -> dict:
    """The physical identity and telemetry of the ONE bound device.
    Missing or ambiguous telemetry refuses (CPU thermal zones are
    not GPU telemetry)."""
    rows = _nvidia_query(
        "uuid,name,temperature.gpu,memory.used,memory.total",
        device)
    if len(rows) != 1:
        raise RunnerRefusal(
            f"REFUSED: GPU telemetry for device {device!r} is "
            f"missing or ambiguous ({len(rows)} rows)")
    uuid, name, temp, used, total = [x.strip() for x in
                                     rows[0].split(",")]
    try:
        return {"uuid": uuid, "name": name,
                "temperature_celsius": float(temp),
                "memory_used_mib": float(used),
                "memory_total_mib": float(total)}
    except ValueError as exc:
        raise RunnerRefusal(
            f"REFUSED: unparseable GPU telemetry: {exc}")


def gpu_compute_apps(device: str) -> list:
    cmd = ["nvidia-smi", "-i", str(device),
           "--query-compute-apps=pid,used_memory",
           "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    apps = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, mem = [x.strip() for x in line.split(",")]
        apps.append({"pid": int(pid), "used_memory_mib": float(mem)})
    return apps


def _gpu_temp(device: str) -> float | None:
    rows = _nvidia_query("temperature.gpu", device)
    if len(rows) != 1:
        return None
    try:
        return float(rows[0])
    except ValueError:
        return None


def make_guard_callback(peak: dict, rss_cap: int, thermal_cap: float,
                        cuda_cap: int = None, gpu_device: str = None,
                        heartbeat_seconds: int = None,
                        heartbeats: list = None):
    from stable_baselines3.common.callbacks import BaseCallback

    class ResourceGuardCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._last_hb = time.time()
            self._t0 = time.time()
            self._last_temp = 0.0
            if gpu_device is not None:
                probe_t0 = time.perf_counter()
                _gpu_temp(gpu_device)
                peak["temp_probe_seconds_per_call"] = round(
                    time.perf_counter() - probe_t0, 4)
                peak["temp_probe_calls"] = 0

        def _on_step(self) -> bool:
            rss = _rss_bytes()
            peak["peak_rss_bytes"] = max(peak["peak_rss_bytes"], rss)
            if rss > rss_cap:
                peak["stop"] = f"RSS cap {rss_cap} exceeded at {rss}"
                return False
            if cuda_cap is not None:
                import torch
                alloc = int(torch.cuda.max_memory_allocated())
                peak["peak_cuda_bytes"] = max(
                    peak.get("peak_cuda_bytes", 0), alloc)
                if alloc > cuda_cap:
                    peak["stop"] = (f"CUDA allocation cap {cuda_cap} "
                                    f"exceeded at {alloc}")
                    return False
            if gpu_device is not None:
                if time.time() - self._last_temp >= \
                        TEMP_SAMPLE_SECONDS:
                    self._last_temp = time.time()
                    peak["temp_probe_calls"] += 1
                    t = _gpu_temp(gpu_device)
                    if t is None:
                        peak["stop"] = ("GPU temperature telemetry "
                                        "lost mid-segment — refusing "
                                        "to run unguarded")
                        return False
                    peak.setdefault("gpu_temp_series", []).append(
                        [int(self.num_timesteps), t])
                    peak["peak_gpu_temp"] = max(
                        peak.get("peak_gpu_temp") or 0.0, t)
                    if t > thermal_cap:
                        peak["stop"] = (f"GPU thermal cap "
                                        f"{thermal_cap}C exceeded at "
                                        f"{t:.1f}C")
                        return False
            else:
                t = _thermal_celsius()
                if t is not None:
                    peak["peak_thermal_celsius"] = max(
                        peak.get("peak_thermal_celsius") or 0.0, t)
                    if t > thermal_cap:
                        peak["stop"] = (f"thermal cap {thermal_cap}C "
                                        f"exceeded at {t:.1f}C")
                        return False
            if heartbeat_seconds is not None and \
                    time.time() - self._last_hb >= heartbeat_seconds:
                self._last_hb = time.time()
                fact = {"heartbeat_at_seconds":
                        round(time.time() - self._t0, 1),
                        "env_steps": int(self.num_timesteps),
                        "updates": int(getattr(self.model,
                                               "_n_updates", 0)),
                        "rss_bytes": rss,
                        "cuda_bytes": peak.get("peak_cuda_bytes"),
                        "gpu_temp": peak.get("peak_gpu_temp")}
                if heartbeats is not None:
                    heartbeats.append(fact)
                print("HEARTBEAT " + json.dumps(fact), flush=True)
            return True

    return ResourceGuardCallback()


def gpu_mode_from_record(rec: dict, cfg: dict) -> dict:
    """B4-P4: the GPU preflight runtime mode derives ONLY from the
    owner record — never from the gpu_economic campaign budget or
    any CLI/ambient value; the cell contributes only scientific
    values (here: the replay buffer size it materialized)."""
    lim = rec["preflight_limits"]
    mode = {
        "budget_max_env_steps": lim["environment_steps_max"],
        "budget_max_updates": lim["optimizer_updates_max"],
        "budget_max_wall_seconds": float(lim["wall_seconds_max"]),
        "rss_cap_bytes": lim["host_rss_bytes_max"],
        "cuda_cap_bytes": lim["cuda_allocated_bytes_max"],
        "thermal_cap_celsius": lim["gpu_temperature_celsius_max"],
        "learn_segments": list(lim["learning_segments"]),
        "heartbeat_seconds": lim["heartbeat_seconds_max"],
        "train_year": int(rec["execution_contract"]
                          ["training_year"]),
        "replay_buffer_cap": int(cfg["buffer_size"]),
    }
    if mode["train_year"] != int(
            cfg["execution_modes"]["cpu_mechanics_replay"]
            ["train_year"]):
        raise RunnerRefusal(
            "REFUSED: owner training year differs from the "
            "cell's calibration role")
    return mode


def cpu_mode_from_cell(cfg: dict) -> dict:
    mode = dict(cfg["execution_modes"]["cpu_mechanics_replay"])
    mode["heartbeat_seconds"] = None
    mode["cuda_cap_bytes"] = None
    return mode


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


def _write_terminal(out: Path, label: str, detail: dict) -> None:
    rec = {"schema": "agent_multi.b4_gpu_preflight_terminal.v1",
           "status": label, "g1_eligible": False,
           "checkpoint_promotable": False}
    rec.update(detail)
    (out / "B4_GPU_PREFLIGHT_TERMINAL.json").write_text(
        json.dumps(rec, indent=1))
    print(json.dumps({"status": label}, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=("B4 runner: cell id, materialization root, "
                     "output root and device ONLY — every scientific "
                     "and budget value comes from the reviewed cell "
                     "and the owner authorization record"))
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--materialization-root", type=Path,
                    required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--device", required=True,
                    choices=["cpu", "cuda:0"])
    args = ap.parse_args(argv)
    mat, out = args.materialization_root, args.output_root
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gpu = args.device != "cpu"

    # 1. The exact owner authorization is consumed for EVERY run —
    #    the record also externally pins the approved materialization
    #    tree, so a self-rebound replacement refuses on any path.
    rec = b4a.verify_gpu_preflight_authorization()
    if gpu and args.cell_id != rec["approved_cell"]["cell_id"]:
        raise RunnerRefusal(
            f"REFUSED: the owner approved exactly ONE GPU cell "
            f"({rec['approved_cell']['cell_id']}); "
            f"{args.cell_id!r} is not it")
    b4a.verify_approved_materialization(mat, rec)

    # 2. Full authority chain at point of use (E3 items 1-8 + P3
    #    factual envelope re-derivation).
    packet = json.loads((mat / "B4_MATERIALIZATION.json").read_text())
    comparator_dir = packet.get("comparator_dir")
    if not comparator_dir:
        raise RunnerRefusal(
            "REFUSED: materialization carries no comparator anchor")
    b4a.verify_approved_comparator(Path(comparator_dir), rec)
    authority = b4a.verify_full_authority_chain(Path(comparator_dir))
    lineage = authority["comparator"]["lineage"]

    # 3. The cell: complete, digest-bound, genesis-bound; its
    #    complete-envelope digest must equal the comparator-derived
    #    digest for its origin (B4-P3).
    cell = load_cell(mat, args.cell_id)
    cfg = dict(cell["effective_config"])
    if (cfg["gymfx_lineage_manifest_sha256"]
            != lineage["manifest_sha256"]):
        raise RunnerRefusal(
            "REFUSED: cell lineage differs from the live point-of-use "
            "gym-fx manifest")
    year = None
    for tok in args.cell_id.split("_"):
        if tok.startswith("o") and tok[1:].isdigit():
            year = int(tok[1:])
    derived = authority["comparator"][
        "complete_envelope_digest_by_origin"].get(year)
    if derived != cfg["complete_envelope_digest"]:
        raise RunnerRefusal(
            "REFUSED: cell complete-envelope digest differs from the "
            "comparator-derived digest for its origin (B4-P3)")
    gmeta = packet["genesis"]["cells"].get(args.cell_id)
    if not gmeta or gmeta.get("n_updates") != 0:
        raise RunnerRefusal(
            "REFUSED: genesis metadata not zero-update for this cell")
    seed = int(cfg["seed"])
    gzip = (mat / "genesis" / f"o{year}" / f"seed{seed}" /
            f"zero_update_genesis_seed{seed}.zip")
    if _sha_file(gzip) != gmeta["container_sha256"]:
        raise RunnerRefusal(
            "REFUSED: genesis container digest mismatch — foreign or "
            "altered genesis")

    # 4. Runtime mode: CPU mechanics from the cell; GPU preflight
    #    ONLY from the owner record — never the gpu_economic
    #    campaign budget.
    mode = (gpu_mode_from_record(rec, cfg) if gpu
            else cpu_mode_from_cell(cfg))

    # 5. GPU pre-dispatch inventory + single-device binding.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    inventory_pre = None
    if gpu:
        if not cvd or "," in cvd:
            raise RunnerRefusal(
                "REFUSED: exactly one explicit CUDA_VISIBLE_DEVICES "
                "binding is required for the preflight")
        try:
            inventory_pre = gpu_inventory(cvd)
            apps = gpu_compute_apps(cvd)
        except RunnerRefusal as exc:
            _write_terminal(out, GPU_BLOCKED_LABEL,
                            {"reason": str(exc),
                             "attempt_consumed": False})
            return 0
        blocked = None
        if apps is None:
            blocked = "compute-app telemetry unavailable"
        elif any(x["used_memory_mib"] > SUBSTANTIAL_COMPUTE_MIB
                 for x in apps):
            blocked = (f"another substantial CUDA compute workload "
                       f"is active: {apps}")
        elif inventory_pre["temperature_celsius"] > \
                mode["thermal_cap_celsius"]:
            blocked = (f"device already above the thermal limit at "
                       f"{inventory_pre['temperature_celsius']}C")
        elif (inventory_pre["memory_total_mib"]
              - inventory_pre["memory_used_mib"]) * 2 ** 20 < \
                mode["cuda_cap_bytes"]:
            blocked = "required free CUDA memory is unavailable"
        import torch
        if blocked is None and not torch.cuda.is_available():
            blocked = ("cuda requested but PyTorch reports no CUDA "
                       "device — silent CPU fallback refused")
        if blocked:
            _write_terminal(out, GPU_BLOCKED_LABEL,
                            {"reason": blocked,
                             "inventory": inventory_pre,
                             "attempt_consumed": False})
            return 0
        # the ONE attempt: consumed at CUDA/model construction.
        if b4a.GPU_ATTEMPT_LEDGER.exists():
            raise RunnerRefusal(
                "REFUSED: the single authorized GPU preflight "
                "attempt is already consumed "
                f"({b4a.GPU_ATTEMPT_LEDGER})")
        b4a.GPU_ATTEMPT_LEDGER.parent.mkdir(parents=True,
                                            exist_ok=True)
        b4a.GPU_ATTEMPT_LEDGER.write_text(json.dumps(
            {"authorization_sha256": b4a.OWNER_GPU_AUTH_SHA,
             "cell": args.cell_id, "output_root": str(out),
             "device_uuid": inventory_pre["uuid"],
             "consumed_wall": time.time()}, indent=1))

    # Once CUDA/model construction begins the ONE attempt is
    # consumed: any later crash, typed stop or failed invariant
    # is a RESULT, never a retry.
    try:
        # 6. Data: the cell's declared calibration-year role only.
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
        run_cfg["device"] = "cuda:0" if gpu else "cpu"
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

        # 7. Genesis: the CPU cold build PROVES the approved zero-update
        #    identity; the GPU model is then loaded from those exact
        #    tensors — approved genesis only, no other initialization.
        from agent_plugins.sac_agent import (Plugin as SacPlugin,
                                             _policy_tensor_hash)
        plugin = SacPlugin()
        env = plugin.wrap_env(env, run_cfg)
        build_cfg = dict(run_cfg)
        build_cfg["device"] = "cpu"
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
        effective_device = "cpu"
        if gpu:
            import torch
            tmp = out / "genesis_proof.zip"
            plugin.save(model, str(tmp))
            del model
            from stable_baselines3 import SAC
            model = SAC.load(str(tmp), env=env, device="cuda:0")
            tmp.unlink()
            if _policy_tensor_hash(model.policy) != genesis_tensor_sha:
                raise RunnerRefusal(
                    "REFUSED: CUDA-loaded tensors differ from the "
                    "approved genesis identity")
            effective_device = str(model.device)
            if not effective_device.startswith("cuda"):
                raise RunnerRefusal(
                    "REFUSED: cuda requested but the effective model "
                    "device is CPU — silent fallback refused")
            torch.cuda.reset_peak_memory_stats()

        # 8. Bounded learning — F9.2 + RSS/CUDA/thermal/stop-file/wall
        #    enforced inside EVERY segment; heartbeats on the GPU path.
        from stable_baselines3.common.callbacks import CallbackList
        peak = {"peak_rss_bytes": _rss_bytes(), "stop": None}
        if not gpu:
            peak["peak_thermal_celsius"] = _thermal_celsius()
        heartbeats = []
        segments = []
        for seg_steps in mode["learn_segments"]:
            try:
                rl._check_executing_budget(
                    run_cfg, model, started_wall=t0,
                    next_segment_timesteps=0)
            except rl.ExecutingBudgetExceeded as exc:
                segments.append({"requested_timesteps": seg_steps,
                                 "pre_segment_refusal": str(exc)})
                break
            budget_cb = rl.make_executing_budget_callback(run_cfg, t0)
            guard_cb = make_guard_callback(
                peak, int(mode["rss_cap_bytes"]),
                float(mode["thermal_cap_celsius"]),
                cuda_cap=(int(mode["cuda_cap_bytes"])
                          if gpu else None),
                gpu_device=(cvd if gpu else None),
                heartbeat_seconds=mode.get("heartbeat_seconds"),
                heartbeats=heartbeats)
            before = int(model.num_timesteps)
            seg_t0 = time.time()
            model.learn(total_timesteps=seg_steps,
                        callback=CallbackList([budget_cb, guard_cb]),
                        reset_num_timesteps=False, progress_bar=False)
            segments.append({
                "requested_timesteps": seg_steps,
                "real_timesteps": int(model.num_timesteps) - before,
                "cumulative_env_steps": int(model.num_timesteps),
                "cumulative_updates": int(model._n_updates),
                "segment_seconds": round(time.time() - seg_t0, 1),
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

        # 9. Stop-file proof (typed) — the external stop outranks all.
        (out / "STOP").write_text("stop-file proof")
        try:
            rl._check_executing_budget(run_cfg, model, started_wall=t0)
            raise RunnerRefusal("PROOF FAILED: stop-file ignored")
        except rl.ExecutingBudgetExceeded as exc:
            stop_file_proof = f"typed refusal: {exc}"
        (out / "STOP").unlink()

        # 10. Finiteness + save/load roundtrip.
        import torch
        n_params = 0
        for p_t in model.policy.parameters():
            n_params += int(p_t.numel())
            if not torch.isfinite(p_t).all():
                raise RunnerRefusal(
                    "PROOF FAILED: non-finite policy parameter")
        final_zip = out / f"final_{args.cell_id}.zip"
        plugin.save(model, str(final_zip))
        trained_sha = _policy_tensor_hash(model.policy)
        reloaded = plugin.load(str(final_zip), env)
        if _policy_tensor_hash(reloaded.policy) != trained_sha or \
                int(reloaded._n_updates) != updates:
            raise RunnerRefusal("PROOF FAILED: save/load roundtrip")
        if updates > 0 and trained_sha == genesis_tensor_sha:
            raise RunnerRefusal(
                "PROOF FAILED: tensors unchanged after nonzero updates")

        wall = round(time.time() - t0, 1)
        inventory_post = gpu_inventory(cvd) if gpu else None
        label = GPU_PREFLIGHT_LABEL if gpu else \
            "MECHANICS_PROVEN_NON_PROMOTABLE"
        record = {
            "schema": "agent_multi.b4_preflight_record.v1" if gpu else
                      "agent_multi.b4_mechanics_cell_record.v2",
            "status": label,
            "g1_eligible": False,
            "checkpoint_promotable": False,
            "economic_conclusion_allowed": False,
            "runner": "tools/b4_run_cell.py",
            "cell": args.cell_id,
            "cell_config_sha256": cell["config_sha256"],
            "owner_authorization_sha256":
                b4a.OWNER_GPU_AUTH_SHA if gpu else None,
            "authority_chain": {
                "design_sha256": authority["chain"]["design_sha256"],
                "amendment_shas": authority["chain"]["amendment_shas"],
                "final_code_pins": authority["chain"]["final_code_pins"],
                "comparator_n_results":
                    authority["comparator"]["n_results"],
                "comparator_n_ledger":
                    authority["comparator"]["n_ledger"]},
            "device": {"requested": args.device,
                       "effective": effective_device,
                       "cuda_visible_devices": cvd or None,
                       "inventory_pre_dispatch": inventory_pre,
                       "inventory_terminal": inventory_post},
            "train_slice": {"year": train_year, "years_present": years,
                            "csv_sha256": origin["csv_sha256"],
                            "scored_year_rows_in_gradients": 0,
                            "sealed_2025_rows_read": 0},
            "genesis": {"container_sha256": gmeta["container_sha256"],
                        "policy_tensor_sha256": genesis_tensor_sha,
                        "zero_update_verified": True},
            "gymfx_lineage_manifest_sha256": lineage["manifest_sha256"],
            "gymfx_commit": lineage["commit"],
            "observation_dimension_facts": obs_facts,
            "limits_requested": {k: mode[k] for k in (
                "budget_max_env_steps", "budget_max_updates",
                "budget_max_wall_seconds", "rss_cap_bytes",
                "thermal_cap_celsius") if k in mode},
            "limits_cuda_cap_bytes": mode.get("cuda_cap_bytes"),
            "segments": segments,
            "heartbeats": heartbeats,
            "observed": {"env_steps": steps,
                         "optimizer_updates": updates,
                         "wall_seconds": wall,
                         "steps_per_second": round(steps / wall, 2)
                         if wall else None,
                         "updates_per_second": round(updates / wall, 2)
                         if wall else None,
                         "peak_rss_bytes": peak["peak_rss_bytes"],
                         "peak_cuda_bytes": peak.get("peak_cuda_bytes"),
                         "peak_gpu_temp": peak.get("peak_gpu_temp"),
                         "gpu_temp_series": peak.get("gpu_temp_series"),
                         "temp_probe_seconds_per_call":
                             peak.get("temp_probe_seconds_per_call"),
                         "temp_probe_calls":
                             peak.get("temp_probe_calls"),
                         "peak_thermal_celsius":
                             peak.get("peak_thermal_celsius"),
                         "resource_stop": peak["stop"]},
            "proofs": {"stop_file": stop_file_proof,
                       "finite": (f"all {n_params} policy parameters "
                                  f"finite after {updates} real updates"),
                       "trained_tensor_sha256": trained_sha},
            "sealed_2025_used": False,
        }
        b4a.verify_language(record, "preflight record")
        name = ("B4_GPU_PREFLIGHT_RECORD.json" if gpu else
                "B4_MECHANICS_CELL_RECORD.json")
        (out / name).write_text(json.dumps(record, indent=1))
        print(json.dumps({"status": label, "steps": steps,
                          "updates": updates, "wall_s": wall,
                          "steps_per_s": record["observed"]
                          ["steps_per_second"],
                          "peak_rss_mib": round(
                              peak["peak_rss_bytes"] / 2 ** 20),
                          "peak_cuda_mib": round(
                              (peak.get("peak_cuda_bytes") or 0)
                              / 2 ** 20),
                          "peak_gpu_temp": peak.get("peak_gpu_temp")},
                         indent=1))
        return 0
    except BaseException as exc:
        if gpu:
            _write_terminal(
                out, GPU_FAILED_LABEL,
                {"reason": f"{type(exc).__name__}: {exc}",
                 "attempt_consumed": True})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
