#!/usr/bin/env python3
"""B4 mechanics cell (order @0b4d2748 B4-D3): execute EXACTLY ONE
bounded CPU cell — origin 2024, seed 101 — to prove training
MECHANICS, never science. Caps: 2000 env steps, 1000 optimizer
updates, 30 min wall, 2 GiB RSS. The F9 executing budget guard rides
INSIDE every learning segment (exact-update stop + stop-file). The
run proves finite parameters after real gradient updates and a
save/load tensor-identity roundtrip. Artifacts are NON-PROMOTABLE and
excluded from G1; training data is the origin's CALIBRATION year
(2023) so no scored-year bar enters even a mechanics gradient."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

CELL_KEY = "o2024_seed101"
SEED = 101
ORIGIN_YEAR = 2024
TRAIN_YEAR = 2023          # calibration slice — never the score year
CAPS = {"budget_max_env_steps": 2000,
        "budget_max_updates": 1000,
        "budget_max_wall_seconds": 1800.0}
RSS_CAP_BYTES = 2 * 1024 ** 3
SEGMENT_TIMESTEPS = (1200, 800)   # two segments: the guard rides BOTH


class MechanicsRefusal(SystemExit):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rss_bytes() -> int:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * 4096


def make_rss_callback(peak: dict):
    from stable_baselines3.common.callbacks import BaseCallback

    class RssCapCallback(BaseCallback):
        def _on_step(self) -> bool:
            rss = _rss_bytes()
            peak["peak_rss_bytes"] = max(peak["peak_rss_bytes"], rss)
            if rss > RSS_CAP_BYTES:
                peak["rss_stop"] = (f"RSS cap {RSS_CAP_BYTES} exceeded "
                                    f"at {rss}")
                return False
            return True

    return RssCapCallback()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialization-dir", type=Path, required=True)
    ap.add_argument("--baselines-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    mat, base_dir, out = (args.materialization_dir,
                          args.baselines_dir, args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    b4m = _load("b4mat", "tools/materialize_b4_causal_sac.py")
    sb = _load("sbb", "tools/screen_b_baselines.py")
    import importlib
    rl = importlib.import_module(
        "pipeline_plugins.rl_pipeline_with_validation")

    # 1. The cell identity, its binding and its genesis — verified,
    #    never trusted.
    cells = json.loads((mat / "B4_CELL_CONFIGS.json").read_text())
    if CELL_KEY not in cells:
        raise MechanicsRefusal(f"REFUSED: no cell {CELL_KEY}")
    cell = cells[CELL_KEY]
    cfg_cell = cell["effective_config"]
    recomputed = hashlib.sha256(json.dumps(
        cfg_cell, sort_keys=True, default=str).encode()).hexdigest()
    if recomputed != cell["config_sha256"]:
        raise MechanicsRefusal("REFUSED: cell config digest mismatch")
    b4m.validate_cell_config(cfg_cell)
    binding = json.loads(
        (mat / "genesis" / "GENESIS_BINDING.json").read_text())
    if binding["binding"].get(CELL_KEY) != cell["config_sha256"]:
        raise MechanicsRefusal(
            "REFUSED: genesis binding does not carry this cell's "
            "final config digest")
    packet = json.loads((mat / "B4_MATERIALIZATION.json").read_text())
    gmeta = packet["genesis"]["cells"].get(CELL_KEY)
    if not gmeta or gmeta.get("n_updates") != 0:
        raise MechanicsRefusal("REFUSED: genesis metadata not "
                               "zero-update for this cell")
    gzip = (mat / "genesis" / f"o{ORIGIN_YEAR}" / f"seed{SEED}" /
            f"zero_update_genesis_seed{SEED}.zip")
    if _sha_file(gzip) != gmeta["container_sha256"]:
        raise MechanicsRefusal("REFUSED: genesis container digest "
                               "mismatch — foreign or altered genesis")

    # 2. Point-of-use lineage AT EXECUTION + comparator lineage.
    lineage = b4m.gymfx_lineage_manifest()
    if (lineage["manifest_sha256"]
            != cfg_cell["gymfx_lineage_manifest_sha256"]):
        raise MechanicsRefusal("REFUSED: live gym-fx lineage differs "
                               "from the materialized cell")
    comparator = json.loads(
        (base_dir / "SCREEN_B_RESULTS.json").read_text())
    b4m.check_lineage_match(cfg_cell, comparator)

    # 3. Runnable config: comparator base recipe + THIS cell's sealed
    #    identity on top (the cell wins every shared key), trained on
    #    the CALIBRATION-year slice.
    df = sb.load_source()
    origin = sb.materialize_origin(df, TRAIN_YEAR, out / "origins")
    cost_sets, _cost_sha = sb.load_cost_sets()
    alp = cost_sets["alpaca_ethusd"]["binding"]
    run_cfg = sb.base_config(origin, alp, dict(
        cfg_cell["execution_envelope"]))
    headroom_divergence = {
        "baselines_entry_cost_headroom":
            run_cfg["execution_envelope"]["entry_cost_headroom"],
        "b4_cell_entry_cost_headroom":
            cfg_cell["execution_envelope"]["entry_cost_headroom"],
        "note": ("OBSERVED_DIVERGENCE_FOR_REVIEW: base_config margin "
                 "+0.006 vs cell margin +0.001 over 2x per-side — "
                 "both sealed; neither altered here; the CELL value "
                 "governs this mechanics run")}
    run_cfg.update(cfg_cell)
    run_cfg["input_data_file"] = origin["csv"]
    run_cfg.update(CAPS)
    run_cfg["budget_stop_file"] = str(out / "STOP")
    if run_cfg.get("session_exposure_enabled") is not False:
        raise MechanicsRefusal("REFUSED: session exposure must stay "
                               "explicitly False")

    env = rl._load_env_plugin("gym_fx_env", run_cfg).make_env(run_cfg)
    # The runner's observation seam, unchanged: the declared flattened
    # dimension is enforced at env construction, THEN the agent
    # plugin's own wrapper flattens the Dict space (C6/finding 322).
    import importlib as _il
    obs_contract = _il.import_module(
        "pipeline_plugins._observation_contract")
    obs_facts = obs_contract.verify_flattened_dimension(
        run_cfg, getattr(env, "observation_space", None))

    # 4. Genesis load + zero-update proof through the real plugin.
    from agent_plugins.sac_agent import (Plugin as SacPlugin,
                                         _policy_tensor_hash)
    plugin = SacPlugin()
    env = plugin.wrap_env(env, run_cfg)
    # The ACCEPTED trainer seam builds COLD against the real env and
    # the genesis zip is an IDENTITY artifact: same-seed construction
    # must hash to the materialized zero-update tensors (the
    # determinism fact build_seed_genesis proved). Loading the zip is
    # not the runner's path — equality of tensors is.
    g = _il.import_module("tools.p1lr_genesis_artifacts")
    contract = g.load_v2_contract(g.p1.CONTRACT_PATH_V2)
    bindings = g.p1.load_bindings()
    gfacts = g.resolve_observation_dimension(contract, bindings)
    build_cfg = dict(run_cfg)
    build_cfg.update({"device": "cpu", "train_seed": SEED,
                      "net_arch": tuple(gfacts["net_arch"]),
                      "ent_coef": gfacts["ent_coef"],
                      "buffer_size": 5000, "use_sde": False})
    model = plugin.build(env, build_cfg)
    if (int(getattr(model, "_n_updates", -1)) != 0
            or int(getattr(model, "num_timesteps", -1)) != 0):
        raise MechanicsRefusal("REFUSED: constructed genesis is not "
                               "zero-update")
    genesis_tensor_sha = _policy_tensor_hash(model.policy)
    if genesis_tensor_sha != gmeta["policy_tensor_sha256"]:
        raise MechanicsRefusal(
            "REFUSED: same-seed construction does not reproduce the "
            "materialized genesis tensor identity "
            f"({genesis_tensor_sha[:16]} != "
            f"{gmeta['policy_tensor_sha256'][:16]})")

    # 5. Bounded learning — the F9 guard checked BEFORE, INSIDE (via
    #    the executing callback) and AFTER every segment.
    from stable_baselines3.common.callbacks import CallbackList
    peak = {"peak_rss_bytes": _rss_bytes(), "rss_stop": None}
    segments = []
    for seg_steps in SEGMENT_TIMESTEPS:
        try:
            rl._check_executing_budget(run_cfg, model,
                                       started_wall=t0,
                                       next_segment_timesteps=seg_steps)
        except rl.ExecutingBudgetExceeded as exc:
            # A refused NEXT segment is the guard doing its job: the
            # budget is exhausted, learning ENDS here — recorded, not
            # crashed.
            segments.append({"requested_timesteps": seg_steps,
                             "pre_segment_refusal": str(exc)})
            break
        budget_cb = rl.make_executing_budget_callback(run_cfg, t0)
        model.learn(total_timesteps=seg_steps,
                    callback=CallbackList(
                        [budget_cb, make_rss_callback(peak)]),
                    reset_num_timesteps=False,
                    progress_bar=False)
        segments.append({
            "requested_timesteps": seg_steps,
            "cumulative_env_steps": int(model.num_timesteps),
            "cumulative_updates": int(model._n_updates),
            "budget_stop": budget_cb.budget_stop,
            "wall_seconds": round(time.time() - t0, 1)})
        try:
            rl._check_executing_budget(run_cfg, model,
                                       started_wall=t0)
            post_check = "within budget"
        except rl.ExecutingBudgetExceeded as exc:
            post_check = f"typed stop: {exc}"
        segments[-1]["post_segment_check"] = post_check

    steps = int(model.num_timesteps)
    updates = int(model._n_updates)
    if steps > CAPS["budget_max_env_steps"]:
        raise MechanicsRefusal(f"CAP VIOLATED: {steps} env steps")
    if updates > CAPS["budget_max_updates"]:
        raise MechanicsRefusal(f"CAP VIOLATED: {updates} updates")

    # 6. Exact-update-stop proof: with the update budget consumed (or
    #    the step budget exhausted) another segment must REFUSE with
    #    the typed guard, never run.
    exact_stop = None
    try:
        rl._check_executing_budget(run_cfg, model, started_wall=t0,
                                   next_segment_timesteps=50)
        if updates >= CAPS["budget_max_updates"]:
            raise MechanicsRefusal(
                "PROOF FAILED: exhausted update budget did not refuse")
        exact_stop = ("not demonstrable pre-segment: budgets not yet "
                      "exhausted")
    except rl.ExecutingBudgetExceeded as exc:
        exact_stop = f"typed refusal: {exc}"

    # 7. Stop-file proof: the external stop outranks everything.
    (out / "STOP").write_text("mechanics stop-file proof")
    try:
        rl._check_executing_budget(run_cfg, model, started_wall=t0)
        raise MechanicsRefusal("PROOF FAILED: stop-file ignored")
    except rl.ExecutingBudgetExceeded as exc:
        stop_file_proof = f"typed refusal: {exc}"
    (out / "STOP").unlink()

    # 8. Finiteness proof after real gradient updates.
    import torch
    n_params = 0
    for p in model.policy.parameters():
        n_params += int(p.numel())
        if not torch.isfinite(p).all():
            raise MechanicsRefusal("PROOF FAILED: non-finite policy "
                                   "parameter after training")
    finite_proof = (f"all {n_params} policy parameters finite after "
                    f"{updates} real optimizer updates")

    # 9. Save/load tensor-identity roundtrip.
    final_zip = out / f"mechanics_final_{CELL_KEY}.zip"
    plugin.save(model, str(final_zip))
    trained_sha = _policy_tensor_hash(model.policy)
    reloaded = plugin.load(str(final_zip), env)
    reload_sha = _policy_tensor_hash(reloaded.policy)
    if reload_sha != trained_sha:
        raise MechanicsRefusal("PROOF FAILED: save/load tensor "
                               "identity mismatch")
    if int(reloaded._n_updates) != updates:
        raise MechanicsRefusal("PROOF FAILED: update counter lost in "
                               "the roundtrip")
    if updates > 0 and trained_sha == genesis_tensor_sha:
        raise MechanicsRefusal("PROOF FAILED: tensors unchanged after "
                               "nonzero updates")

    record = {
        "schema": "agent_multi.b4_mechanics_cell_record.v1",
        "status": "MECHANICS_PROVEN_NON_PROMOTABLE",
        "g1_eligible": False,
        "cell": CELL_KEY,
        "cell_config_sha256": cell["config_sha256"],
        "train_year_role": {"year": TRAIN_YEAR,
                            "rule": "calibration slice only — no "
                                    "scored-year or 2025 bar"},
        "train_csv_sha256": origin["csv_sha256"],
        "genesis": {"container_sha256": gmeta["container_sha256"],
                    "policy_tensor_sha256": genesis_tensor_sha,
                    "zero_update_verified": True},
        "gymfx_lineage_manifest_sha256": lineage["manifest_sha256"],
        "gymfx_commit": lineage["commit"],
        "observation_dimension_facts": obs_facts,
        "caps": {**CAPS, "rss_cap_bytes": RSS_CAP_BYTES},
        "segments": segments,
        "observed": {"env_steps": steps, "optimizer_updates": updates,
                     "wall_seconds": round(time.time() - t0, 1),
                     "peak_rss_bytes": peak["peak_rss_bytes"],
                     "rss_stop": peak["rss_stop"]},
        "proofs": {"exact_update_stop": exact_stop,
                   "stop_file": stop_file_proof,
                   "finite": finite_proof,
                   "save_load_roundtrip":
                       f"tensor sha {trained_sha[:16]}… and update "
                       f"counter {updates} identical after reload",
                   "trained_tensor_sha256": trained_sha},
        "headroom_divergence": headroom_divergence,
        "sealed_2025_used": False,
    }
    (out / "B4_MECHANICS_CELL_RECORD.json").write_text(
        json.dumps(record, indent=1))
    print(json.dumps({"status": record["status"], "steps": steps,
                      "updates": updates,
                      "wall_s": record["observed"]["wall_seconds"],
                      "peak_rss_mib": round(
                          peak["peak_rss_bytes"] / 2 ** 20)},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
