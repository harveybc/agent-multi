#!/usr/bin/env python3
"""B4 campaign cell executor (order @e8bb500f, E8).

Maps ONE materialized B4 cell into the REAL
rl_pipeline_with_validation lifecycle — epoch loop, causal validation,
patience, checkpoint selection — and then scores the FROZEN selected
checkpoint on the cell's declared outer origin, emitting per-bar
gross/cost/net returns pairable exactly to every comparator arm.

Nothing scientific arrives by CLI or ambient default: the cell and its
bound materialization determine everything. Execution of a
gpu_economic cell additionally requires the FUTURE owner campaign
authorization artifact — it does not exist yet, so execute refuses;
dry_run validates the complete path offline without constructing any
model. Terminal records are immutable; a written terminal state is
never overwritten. Checkpoints stay non-promotable; sealed-2025 stays
unread; the consumed preflight is mechanics evidence only and can
never enter the twelve scientific results."""
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

# The owner's CAMPAIGN authorization does not exist yet. When it does,
# Musashi pins its digest here by order; until then the economic
# execution path refuses. The preflight authorization does NOT
# authorize campaign cells.
CAMPAIGN_AUTH_PATH = (b4a.EVIDENCE /
                      "OWNER_AUTHORIZATION_B4_12_CELL_CAMPAIGN.json")
CAMPAIGN_AUTH_SHA = None

TERMINAL_CLASSES = ("COMPLETED", "FAILED", "TIMED_OUT",
                    "THERMAL_STOP", "RESOURCE_STOP",
                    "EXTERNALLY_STOPPED")


class ExecutorRefusal(SystemExit):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "b4run_exec", REPO / "tools/b4_run_cell.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_economic_config(cell_id: str, mat_root: Path,
                          out_root: Path, device: str) -> dict:
    """The complete pipeline config for one gpu_economic cell — cell
    terms only, contract path digest-verified, budgets from the
    cell's materialized gpu_economic mode, F9 keys included. No
    parameter enters from anywhere else."""
    runner = _load_runner()
    # E9: the campaign tree binds to the amendment-6 proposed
    # population identities (the owner's future campaign record
    # confirms them); the preflight record binds only the v2
    # preflight tree.
    b4a.verify_campaign_materialization(mat_root)
    cell = runner.load_cell(mat_root, cell_id)
    cfg = dict(cell["effective_config"])
    year = None
    for tok in cell_id.split("_"):
        if tok.startswith("o") and tok[1:].isdigit():
            year = int(tok[1:])
    contract = (Path(mat_root) / "contracts" /
                f"b4_causal_origin_{year}_contract.json")
    if not contract.is_file():
        raise ExecutorRefusal(
            f"REFUSED: origin contract absent for {cell_id}")
    if _sha_file(contract) != cfg["nested_split_contract_sha256"]:
        raise ExecutorRefusal(
            "REFUSED: origin contract bytes differ from the cell's "
            "bound identity")
    econ = cfg["execution_modes"]["gpu_economic"]
    cell_dir = Path(out_root) / cell_id
    cfg.update({
        "nested_split_contract": str(contract),
        "output_dir": str(cell_dir),
        "save_config": str(cell_dir / "cell_effective_config.json"),
        "return_trace_dir": str(cell_dir / "return_traces"),
        "device": device,
        "quiet_mode": True,
        "evaluate_test_split": False,   # sealed-2025 stays unread
        "budget_max_env_steps": int(econ["budget_max_env_steps"]),
        "budget_max_updates": int(econ["budget_max_updates"]),
        "budget_max_wall_seconds": float(
            econ["budget_max_wall_seconds"]),
        "budget_stop_file": str(cell_dir / "STOP"),
    })
    for k in b4a.FORBIDDEN_CELL_KEYS:
        if cfg.get(k) not in (None, "", False):
            raise ExecutorRefusal(
                f"REFUSED: hidden pretrained/replay/resume input {k}")
    return {"cell": cell, "config": cfg, "year": year,
            "contract": str(contract)}


def dry_run_cell(cell_id: str, mat_root: Path,
                 out_root: Path) -> dict:
    """Validate the COMPLETE execution path offline — authority
    chain, cell, contract, observation binding, role materialization,
    sealed absence, genesis — without constructing any model or env.
    Typed findings, never silent gaps."""
    findings = []
    t0 = time.time()
    built = build_economic_config(cell_id, mat_root, out_root, "cpu")
    cfg, year = dict(built["config"]), built["year"]
    packet = json.loads(
        (Path(mat_root) / "B4_MATERIALIZATION.json").read_text())
    comparator_dir = packet.get("comparator_dir")
    authority = b4a.verify_full_authority_chain(Path(comparator_dir))
    if (cfg["gymfx_lineage_manifest_sha256"] !=
            authority["comparator"]["lineage"]["manifest_sha256"]):
        findings.append("cell lineage differs from live gym-fx")
    derived = authority["comparator"][
        "complete_envelope_digest_by_origin"][year]
    if derived != cfg["complete_envelope_digest"]:
        findings.append("complete-envelope digest mismatch vs "
                        "comparator derivation")
    pipe = importlib.import_module(
        "pipeline_plugins.rl_pipeline_with_validation")
    obs_mod = importlib.import_module(
        "pipeline_plugins._observation_contract")
    try:
        cfg2, _application = obs_mod.apply_observation_contract(
            dict(cfg))
        obs_mod.validate_observation_contract(cfg2)
    except BaseException as exc:
        findings.append(f"observation contract binding: "
                        f"{type(exc).__name__}: {exc}")
        cfg2 = dict(cfg)
    plugin_cls = pipe.PipelinePlugin
    pipeline = plugin_cls(cfg2) if _accepts_config(plugin_cls) \
        else plugin_cls()
    try:
        pipeline._assert_episodic_contract(cfg2)
    except BaseException as exc:
        findings.append(f"episodic contract: "
                        f"{type(exc).__name__}: {exc}")
    roles = {}
    try:
        paths = pipeline._split_csv(cfg2)
        for role, csvp in sorted(paths.items()):
            f = Path(csvp)
            roles[role] = {"csv": str(f),
                           "sha256": _sha_file(f) if f.is_file()
                           else "ABSENT"}
        sealed = [r for r in roles if "sealed" in r]
        if sealed:
            findings.append(f"sealed role materialized: {sealed}")
    except BaseException as exc:
        findings.append(f"role materialization: "
                        f"{type(exc).__name__}: {exc}")
    gmeta = packet["genesis"]["cells"].get(cell_id)
    seed = int(cfg["seed"])
    gzip = (Path(mat_root) / "genesis" / f"o{year}" /
            f"seed{seed}" / f"zero_update_genesis_seed{seed}.zip")
    if not gmeta or gmeta.get("n_updates") != 0:
        findings.append("genesis metadata not zero-update")
    elif not gzip.is_file() or _sha_file(gzip) != \
            gmeta["container_sha256"]:
        findings.append("genesis container absent or digest-broken")
    status = ("DRY_RUN_READY" if not findings
              else "DRY_RUN_FINDINGS")
    return {"schema": "agent_multi.b4_campaign_dry_run.v1",
            "cell": cell_id, "status": status,
            "cell_config_sha256": built["cell"]["config_sha256"],
            "contract": built["contract"],
            "roles_materialized": roles,
            "findings": findings,
            "wall_seconds": round(time.time() - t0, 1)}


def _accepts_config(cls) -> bool:
    import inspect
    try:
        sig = inspect.signature(cls.__init__)
        return len(sig.parameters) > 1
    except (TypeError, ValueError):
        return False


def _terminal_path(out_root: Path, cell_id: str) -> Path:
    return Path(out_root) / cell_id / "B4_CELL_TERMINAL.json"


def write_terminal(out_root: Path, cell_id: str, terminal: str,
                   detail: dict) -> Path:
    """Immutable: a written terminal state is NEVER overwritten."""
    if terminal not in TERMINAL_CLASSES:
        raise ExecutorRefusal(
            f"REFUSED: unknown terminal class {terminal!r}")
    p = _terminal_path(out_root, cell_id)
    if p.exists():
        raise ExecutorRefusal(
            f"REFUSED: terminal state already written for {cell_id} "
            "— terminal records are immutable")
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "agent_multi.b4_cell_terminal.v1",
           "cell": cell_id, "terminal": terminal,
           "g1_eligible": False,
           "checkpoint_promotable": False}
    rec.update(detail)
    b4a.verify_language(rec, "cell terminal record")
    p.write_text(json.dumps(rec, indent=1))
    return p


def classify_stop(budget_stop: str, resource_stop: str,
                  timed_out: bool) -> str:
    if resource_stop and "thermal" in resource_stop.lower():
        return "THERMAL_STOP"
    if resource_stop:
        return "RESOURCE_STOP"
    if budget_stop and "external stop request" in budget_stop:
        return "EXTERNALLY_STOPPED"
    if timed_out or (budget_stop and "wall budget" in budget_stop):
        return "TIMED_OUT"
    return "COMPLETED"


def score_frozen_checkpoint(cfg: dict, checkpoint_zip: Path,
                            outer_csv: Path, scored_start: int,
                            out_csv: Path) -> dict:
    """Frozen-checkpoint evaluation on the declared outer origin:
    deterministic policy actions through the SAME shared execution
    envelope and cost contract as every comparator arm; per-bar
    gross return, cost components and net return on the identical
    scored index."""
    import numpy as np
    import pandas as pd
    rl = importlib.import_module(
        "pipeline_plugins.rl_pipeline_with_validation")
    from agent_plugins.sac_agent import Plugin as SacPlugin
    eval_cfg = dict(cfg)
    eval_cfg["input_data_file"] = str(outer_csv)
    eval_cfg["env_mode"] = "inference"
    env = rl._load_env_plugin(
        eval_cfg["env_plugin"], eval_cfg).make_env(eval_cfg)
    plugin = SacPlugin()
    env = plugin.wrap_env(env, eval_cfg)
    model = plugin.load(str(checkpoint_zip), env)
    obs, _ = env.reset(seed=0)
    inner = env
    while not hasattr(inner, "bridge") and hasattr(inner, "env"):
        inner = inner.env
    df = pd.read_csv(outer_csv)
    n = len(df)
    rows = []
    equity_prev = None
    for t in range(n):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        econ = float(info.get("economic_equity", np.nan))
        if t >= scored_start:
            gross = float(info.get("gross_equity", econ))
            ret = (0.0 if equity_prev in (None, 0.0)
                   else econ / equity_prev - 1.0)
            rows.append({
                "bar_index": t,
                "economic_equity": econ,
                "net_return": ret,
                "commission_paid":
                    float(info.get("commission_paid", 0.0)),
                "slippage_paid":
                    float(info.get("slippage_paid", 0.0)),
                "gross_equity": gross})
        if t >= scored_start or equity_prev is None:
            equity_prev = econ
        if term or trunc:
            break
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return {"per_bar_csv": str(out_csv),
            "per_bar_sha256": _sha_file(out_csv),
            "scored_bars": len(rows)}


def execute_cell(cell_id: str, mat_root: Path, out_root: Path,
                 device: str) -> int:
    """The full scientific path — gated on the FUTURE owner campaign
    authorization. Until Musashi pins that artifact, this refuses
    before any model, env or CUDA construction."""
    if CAMPAIGN_AUTH_SHA is None or not CAMPAIGN_AUTH_PATH.is_file():
        raise ExecutorRefusal(
            "REFUSED: no owner campaign authorization exists — the "
            "twelve-cell campaign is a SEPARATE future owner "
            "decision; the consumed preflight authorization grants "
            "nothing here")
    raw = CAMPAIGN_AUTH_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CAMPAIGN_AUTH_SHA:
        raise ExecutorRefusal(
            "REFUSED: campaign authorization bytes differ from the "
            "carried reviewed digest")
    terminal_p = _terminal_path(out_root, cell_id)
    if terminal_p.exists():
        raise ExecutorRefusal(
            f"REFUSED: {cell_id} already holds an immutable terminal "
            "state — attempts are never reused")
    built = build_economic_config(cell_id, mat_root, out_root,
                                  device)
    cfg = built["config"]
    t0 = time.time()
    from app.plugin_loader import load_plugin
    agent_cls, _ = load_plugin("agent.plugins", cfg["agent_plugin"])
    pipeline_cls, _ = load_plugin("pipeline.plugins",
                                  cfg["pipeline_plugin"])
    agent_plugin = agent_cls(cfg)
    pipeline = pipeline_cls(cfg)
    try:
        final = pipeline.run_pipeline(config=cfg, env_plugin=None,
                                      agent_plugin=agent_plugin,
                                      mode="train")
        terminal = "COMPLETED"
        detail = {"pipeline_summary_keys": sorted(final)
                  if isinstance(final, dict) else str(type(final))}
    except BaseException as exc:
        terminal = classify_stop(str(exc), None, False)
        if terminal == "COMPLETED":
            terminal = "FAILED"
        write_terminal(out_root, cell_id, terminal,
                       {"reason": f"{type(exc).__name__}: {exc}",
                        "wall_seconds": round(time.time() - t0, 1)})
        raise
    write_terminal(out_root, cell_id, terminal, {
        "cell_config_sha256": built["cell"]["config_sha256"],
        "wall_seconds": round(time.time() - t0, 1),
        "detail": detail})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=("B4 campaign executor: cell id, materialization "
                     "root, output root, device and action ONLY"))
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--materialization-root", type=Path,
                    required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--device", default="cpu",
                    choices=["cpu", "cuda:0"])
    ap.add_argument("--action", required=True,
                    choices=["dry-run", "execute"])
    args = ap.parse_args(argv)
    if args.action == "dry-run":
        report = dry_run_cell(args.cell_id,
                              args.materialization_root,
                              args.output_root)
        outp = (Path(args.output_root) / args.cell_id /
                "DRY_RUN_REPORT.json")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=1))
        print(json.dumps({"cell": report["cell"],
                          "status": report["status"],
                          "findings": report["findings"]}, indent=1))
        return 0
    return execute_cell(args.cell_id, args.materialization_root,
                        args.output_root, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
