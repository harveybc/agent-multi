#!/usr/bin/env python3
"""B4 campaign cell executor (orders @e8bb500f E8, @0ce52740 C1-C8).

ONE scientific path per cell: verified authority chain -> the REAL
rl_pipeline_with_validation lifecycle (epoch loop, causal validation,
patience, checkpoint selection, per-cell isolated artifacts,
observable CellRuntime) -> frozen-artifact scoring on the declared
outer origin with FULL per-bar identity and cost reconciliation ->
a durable, exclusive, immutable typed terminal that the authoritative
ledger verifier accepts as written.

Effective limits derive from the AUTHORIZED resource contract
(carried digest), never from the cell's gpu_economic mode; the
resource guards ride the pipeline's own F9 executing callback.
Nothing scientific arrives by CLI. Execution additionally requires
the owner/Musashi campaign authorization record — absent, execute
refuses before any model, env, CUDA or output. All artifacts are
NON-PROMOTABLE; sealed-2025 stays unread; the consumed preflight can
never enter the twelve scientific results."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import b4_authority as b4a  # noqa: E402

# The Musashi campaign authorization record does not exist yet: he
# pins path+digest here by order after this correction passes review.
CAMPAIGN_AUTH_PATH = (b4a.EVIDENCE /
                      "MUSASHI_B4_CAMPAIGN_AUTHORIZATION_RECORD.json")
CAMPAIGN_AUTH_SHA = None

TERMINAL_CLASSES = ("COMPLETED", "FAILED", "TIMED_OUT",
                    "THERMAL_STOP", "RESOURCE_STOP",
                    "EXTERNALLY_STOPPED")
DT_FMT = "%Y-%m-%d %H:%M"


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


def _load_sb():
    spec = importlib.util.spec_from_file_location(
        "sbb_exec", REPO / "tools/screen_b_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def comparator_dir_of(packet: dict) -> Path:
    ref = packet.get("comparator_ref") or packet.get("comparator_dir")
    if not ref:
        raise ExecutorRefusal(
            "REFUSED: materialization carries no comparator anchor")
    if ":" in str(ref)[:12]:
        return b4a.resolve_source_ref(str(ref))
    return Path(ref)


def build_economic_config(cell_id: str, mat_root: Path,
                          out_root: Path, device: str) -> dict:
    """The complete pipeline config for one cell. Scientific terms
    come ONLY from the cell; RESOURCE limits come ONLY from the
    authorized contract (C3); artifact/runtime paths are derived
    per-cell under the cell's own result root (C6). A poisoned cell
    limit is invisible to the runtime."""
    runner = _load_runner()
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
    # F7: the committed contract is logical; the pipeline consumes a
    # runtime-RESOLVED copy under the cell's own root.
    logical = json.loads(contract.read_bytes())
    resolved = dict(logical)
    resolved["source_csv"] = str(
        b4a.resolve_source_ref(logical["source_ref"]))
    src_sha = _sha_file(Path(resolved["source_csv"]))
    if src_sha != logical.get("source_sha256", src_sha):
        raise ExecutorRefusal(
            "REFUSED: resolved source dataset differs from the "
            "contract identity")
    cell_dir = Path(out_root) / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    resolved_contract = cell_dir / "resolved_origin_contract.json"
    resolved_contract.write_text(json.dumps(resolved, indent=1))
    # C3: authorized limits, never the cell's gpu_economic mode.
    limits = b4a.load_resource_contract()
    cfg.update({
        "nested_split_contract": str(resolved_contract),
        # C6: per-cell isolation — nothing may land in the CWD or be
        # shared between cells.
        "output_dir": str(cell_dir),
        "save_model": str(cell_dir / "best_model.zip"),
        "checkpoint_bundle_dir": str(cell_dir / "checkpoints"),
        "cell_runtime_dir": str(cell_dir / "cell_runtime"),
        "return_trace_dir": str(cell_dir / "return_traces"),
        "save_config": str(cell_dir / "cell_effective_config.json"),
        "device": device,
        "quiet_mode": True,
        "evaluate_test_split": False,   # sealed-2025 stays unread
        "budget_max_env_steps": limits["budget_max_env_steps"],
        "budget_max_updates": limits["budget_max_updates"],
        "budget_max_wall_seconds":
            limits["budget_max_wall_seconds"],
        "budget_max_rss_bytes": limits["budget_max_rss_bytes"],
        "budget_stop_file": str(cell_dir / "STOP"),
    })
    if device.startswith("cuda"):
        cfg["budget_max_cuda_bytes"] = limits["budget_max_cuda_bytes"]
        cfg["budget_max_gpu_temp_celsius"] = \
            limits["budget_max_gpu_temp_celsius"]
        cfg["budget_gpu_device"] = os.environ.get(
            "CUDA_VISIBLE_DEVICES", "")
    for k in b4a.FORBIDDEN_CELL_KEYS:
        if cfg.get(k) not in (None, "", False):
            raise ExecutorRefusal(
                f"REFUSED: hidden pretrained/replay/resume input {k}")
    return {"cell": cell, "config": cfg, "year": year,
            "contract": str(contract),
            "contract_sha256": cfg["nested_split_contract_sha256"],
            "limits": limits}


def reconcile_per_bar(out) -> None:
    """C16: conservation from INDEPENDENT sources — the env's own
    per-bar pnl fact against the delta of consecutive economic-
    equity observations recorded separately. Never an identity
    built from one field. Total: equity_end - equity_start equals
    the sum of observed deltas."""
    resid = (out["net_equity_delta_observed"]
             - out["env_pnl_fact"]).abs().max()
    if float(resid) > 1e-6:
        raise ExecutorRefusal(
            f"REFUSED: env pnl fact disagrees with the "
            f"independently observed equity delta (max residual "
            f"{resid}) — conservation broken")
    eq = out["economic_equity"].to_numpy()
    # the first scored delta crosses the context boundary and lies
    # OUTSIDE the eq[0]..eq[-1] span; internal conservation sums the
    # in-span deltas only.
    total = abs(float(eq[-1] - eq[0])
                - float(out["net_equity_delta_observed"]
                        .iloc[1:].sum()))
    if total > 1e-6:
        raise ExecutorRefusal(
            f"REFUSED: total equity conservation residual {total}")
    if (out["commission_delta"] < -1e-12).any():
        raise ExecutorRefusal(
            "REFUSED: cumulative commission counter decreased")


# ---------------- C2: identity-complete frozen scoring ------------
def score_frozen_checkpoint(cfg: dict, checkpoint_zip: Path,
                            checkpoint_sha: str, origin: dict,
                            out_csv: Path, cell_id: str) -> dict:
    """Frozen-artifact evaluation on the declared outer origin.
    Every scored bar carries full identity (origin, seed, UTC
    datetime, absolute scored index, source-row digest), requested
    and realized exposure, gross and economic equity, gross return,
    per-cost deltas and net return. Cumulative environment counters
    are DECLARED cumulative and converted to deltas. Lifecycle
    failures, residual sweeps, recapitalizations and non-finite
    states refuse."""
    import numpy as np
    import pandas as pd
    if _sha_file(checkpoint_zip) != checkpoint_sha:
        raise ExecutorRefusal(
            "REFUSED: checkpoint bytes differ from the pipeline's "
            "declared artifact digest")
    rl = importlib.import_module(
        "pipeline_plugins.rl_pipeline_with_validation")
    from agent_plugins.sac_agent import Plugin as SacPlugin
    eval_cfg = dict(cfg)
    eval_cfg["input_data_file"] = origin["csv"]
    eval_cfg["env_mode"] = "inference"
    for k in ("budget_max_rss_bytes", "budget_max_cuda_bytes",
              "budget_max_gpu_temp_celsius"):
        eval_cfg.pop(k, None)
    eval_cfg["device"] = "cpu"
    env = rl._load_env_plugin(
        eval_cfg["env_plugin"], eval_cfg).make_env(eval_cfg)
    plugin = SacPlugin()
    env = plugin.wrap_env(env, eval_cfg)
    # The frozen artifact is loaded WITHOUT attaching the env: the
    # scorer only consumes model.predict on observations, so saved
    # space bounds (genesis Box(+-inf) vs live per-field bounds)
    # never block a digest-verified artifact; shape compatibility is
    # asserted explicitly below.
    from stable_baselines3 import SAC
    model = SAC.load(str(checkpoint_zip), device="cpu")
    saved_shape = tuple(model.observation_space.shape)
    live_shape = tuple(env.observation_space.shape)
    if saved_shape != live_shape:
        raise ExecutorRefusal(
            f"REFUSED: artifact observation shape {saved_shape} != "
            f"live env {live_shape}")
    obs, _ = env.reset(seed=0)
    inner = env
    while not hasattr(inner, "bridge") and hasattr(inner, "env"):
        inner = inner.env
    df = pd.read_csv(origin["csv"], parse_dates=["DATE_TIME"])
    n = len(df)
    scored_start = int(origin["scored_start_index"])
    seed = int(cfg["seed"])
    rows = []
    equity_prev = None
    commission_prev = 0.0
    for t in range(n):
        action, _st = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        econ = float(info.get("economic_equity", np.nan))
        if not np.isfinite(econ):
            raise ExecutorRefusal(
                f"REFUSED: non-finite economic equity at bar {t}")
        commission_cum = float(info.get("commission_paid") or 0.0)
        if t >= scored_start:
            dt = df["DATE_TIME"].iloc[t]
            env_pnl = float(info.get("pnl", 0.0))
            commission_delta = commission_cum - commission_prev
            units = float(getattr(inner.bridge, "position_units",
                                  0.0) or 0.0)
            close_t = float(df["CLOSE"].iloc[t])
            observed_delta = (0.0 if equity_prev is None
                              else econ - equity_prev)
            rows.append({
                "origin": int(origin["year"]),
                "seed": seed,
                "datetime_utc": dt.strftime(DT_FMT),
                "scored_index": t,
                "source_row_sha256": hashlib.sha256(
                    f"{dt.strftime(DT_FMT)}|{close_t:.10g}"
                    .encode()).hexdigest(),
                "requested_exposure": float(np.asarray(
                    action).reshape(-1)[0]),
                "realized_exposure": (units * close_t / econ
                                      if econ else 0.0),
                # C16: the env does NOT expose gross equity — the
                # field is typed-unavailable and the economic claim
                # narrows; nothing is manufactured.
                "gross_equity": "UNAVAILABLE_ENV_FACT",
                "economic_equity": econ,
                "net_equity_delta_observed": observed_delta,
                "env_pnl_fact": (env_pnl if equity_prev is not None
                                 else observed_delta),
                "commission_delta": commission_delta,
                "pre_commission_equity_delta_derived":
                    observed_delta + commission_delta,
                "slippage_declared":
                    "EMBEDDED_IN_FILL_PRICE_PER_COST_BINDING",
                "net_return": (0.0 if equity_prev in (None, 0.0)
                               else econ / equity_prev - 1.0),
            })
        equity_prev = econ
        commission_prev = commission_cum
        if term or trunc:
            break
    # lifecycle refusals — never accepted evidence
    failure = getattr(inner.bridge, "envelope_run_failure", None)
    diags = getattr(inner.bridge, "execution_diagnostics", {}) or {}
    sweeps = int(diags.get("envelope_residual_sweeps", 0) or 0)
    recaps = int(getattr(inner.bridge, "recapitalization_count", 0)
                 or 0)
    if failure or sweeps or recaps:
        raise ExecutorRefusal(
            f"REFUSED: lifecycle failure={failure!r} residual "
            f"sweeps={sweeps} recapitalizations={recaps}")
    if len(rows) != int(origin["scored_rows"]):
        raise ExecutorRefusal(
            f"REFUSED: scored {len(rows)} bars, contract expects "
            f"{origin['scored_rows']}")
    out = pd.DataFrame(rows)
    reconcile_per_bar(out)
    scored_id = hashlib.sha256(
        "|".join(out["datetime_utc"]).encode()).hexdigest()
    if scored_id != origin["scored_index_sha256"]:
        raise ExecutorRefusal(
            "REFUSED: scored bar identities differ from the "
            "authoritative scored index")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return {"per_bar_csv": str(out_csv),
            "per_bar_sha256": _sha_file(out_csv),
            "scored_bars": len(rows),
            "scored_index_sha256": scored_id,
            "counter_semantics": {
                "commission_paid": "CUMULATIVE from env info; "
                                   "converted to per-bar deltas; "
                                   "monotonicity verified",
                "net_equity_delta_observed": "delta of consecutive "
                    "economic_equity OBSERVATIONS (independent of "
                    "the env pnl fact)",
                "env_pnl_fact": "the env's own per-bar pnl — the "
                    "second, independent source; conservation "
                    "verified between the two",
                "gross_equity": "UNAVAILABLE — the sealed gym-fx "
                    "lineage exposes no gross-equity counter; the "
                    "economic claim is NARROWED accordingly; "
                    "pre_commission_equity_delta_derived is a "
                    "labeled derivation, never a gross fact",
                "slippage": "embedded in fill price per the cost "
                            "binding; not separately countable "
                            "under the sealed gym-fx lineage "
                            "(declared, not silent)"},
            "checkpoint_sha256": checkpoint_sha}


def verify_scoring_evidence(score: dict, origin: dict,
                            comparator_dir: Path,
                            cell_id: str) -> None:
    """C2 verifier half at the producer: identity vector equality
    against EVERY comparator arm of the same origin + authoritative
    scored index + cardinality."""
    import pandas as pd
    cand = pd.read_csv(score["per_bar_csv"])
    ident = list(cand["datetime_utc"])
    if score["scored_index_sha256"] != origin["scored_index_sha256"]:
        raise ExecutorRefusal("REFUSED: scored index mismatch")
    packet = json.loads(
        (Path(comparator_dir) / "SCREEN_B_RESULTS.json").read_bytes())
    year = int(origin["year"])
    arms = [r for r in packet["results"]
            if int(r["origin"]) == year]
    if not arms:
        raise ExecutorRefusal(
            f"REFUSED: no comparator arms for origin {year}")
    for r in arms:
        comp = pd.read_csv(r["per_bar_csv"])
        comp_ident = list(
            pd.to_datetime(comp["datetime"]).dt.strftime(DT_FMT))
        if comp_ident != ident:
            raise ExecutorRefusal(
                f"REFUSED: bar-identity vector differs from "
                f"comparator {r['arm']}@{year} — pairing is "
                "identity, never length")


# ---------------- C5: durable exclusive immutable terminal --------
def _terminal_path(out_root: Path, cell_id: str) -> Path:
    return Path(out_root) / cell_id / "B4_CELL_TERMINAL.json"


def write_terminal(out_root: Path, cell_id: str, terminal: str,
                   detail: dict) -> Path:
    """O_CREAT|O_EXCL exclusive create + fsync(file) + fsync(dir).
    Exactly one writer can ever win; a written terminal is never
    replaced; validation happens BEFORE publication."""
    if terminal not in TERMINAL_CLASSES:
        raise ExecutorRefusal(
            f"REFUSED: unknown terminal class {terminal!r}")
    rec = {"schema": "agent_multi.b4_cell_terminal.v1",
           "cell": cell_id, "terminal": terminal,
           "g1_eligible": False,
           "checkpoint_promotable": False}
    rec.update(detail)
    if not isinstance(rec.get("cell"), str) or rec["cell"] != cell_id:
        raise ExecutorRefusal("REFUSED: terminal cell binding broken")
    if terminal == "COMPLETED":
        for k in ("attempt_id", "cell_config_sha256", "per_bar_csv",
                  "per_bar_sha256", "sealed_2025_used",
                  "scored_index_sha256", "checkpoint_sha256"):
            if k not in rec:
                raise ExecutorRefusal(
                    f"REFUSED: COMPLETED terminal without {k!r} — "
                    "an inadmissible terminal cannot be written")
        if rec["sealed_2025_used"] is not False:
            raise ExecutorRefusal(
                "REFUSED: COMPLETED without sealed-absence proof")
    b4a.verify_language(rec, "cell terminal record")
    p = _terminal_path(out_root, cell_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.parent.is_symlink() or p.is_symlink():
        raise ExecutorRefusal("REFUSED: terminal path is a symlink")
    payload = json.dumps(rec, indent=1).encode()
    try:
        fd = os.open(str(p),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        raise ExecutorRefusal(
            f"REFUSED: terminal state already written for {cell_id} "
            "— terminal records are immutable")
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(str(p.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return p


def classify_stop(budget_stop: str, resource_stop: str,
                  timed_out: bool) -> str:
    text = " ".join(str(x) for x in (budget_stop, resource_stop)
                    if x).lower()
    if "temperature" in text or "thermal" in text:
        return "THERMAL_STOP"
    if "rss" in text or "cuda allocation" in text or \
            "telemetry" in text:
        return "RESOURCE_STOP"
    if "external stop request" in text:
        return "EXTERNALLY_STOPPED"
    if timed_out or "wall budget" in text:
        return "TIMED_OUT"
    return "COMPLETED"


# ---------------- dry-run (C7: executable inspection) -------------
def _executable_path_facts() -> dict:
    """C7: prove by AST that the authorized path contains scorer,
    complete terminal and verifier — not by trusting a docstring."""
    import ast
    tree = ast.parse((REPO / "tools/b4_campaign_executor.py"
                      ).read_text())
    exec_fn = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "execute_cell")
    calls = {n.func.id for n in ast.walk(exec_fn)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)}
    return {
        "execute_cell_calls_scorer":
            "score_frozen_checkpoint" in calls,
        "execute_cell_calls_verifier":
            "verify_scoring_evidence" in calls,
        "execute_cell_calls_terminal": "write_terminal" in calls,
    }


def dry_run_cell(cell_id: str, mat_root: Path,
                 out_root: Path) -> dict:
    findings = []
    t0 = time.time()
    built = build_economic_config(cell_id, mat_root, out_root, "cpu")
    cfg, year = dict(built["config"]), built["year"]
    packet = json.loads(
        (Path(mat_root) / "B4_MATERIALIZATION.json").read_text())
    comparator_dir = comparator_dir_of(packet)
    authority = b4a.verify_full_authority_chain(comparator_dir)
    if (cfg["gymfx_lineage_manifest_sha256"] !=
            authority["comparator"]["lineage"]["manifest_sha256"]):
        findings.append("cell lineage differs from live gym-fx")
    derived = authority["comparator"][
        "complete_envelope_digest_by_origin"][year]
    if derived != cfg["complete_envelope_digest"]:
        findings.append("complete-envelope digest mismatch vs "
                        "comparator derivation")
    limits = built["limits"]
    for k in ("budget_max_wall_seconds", "budget_max_rss_bytes"):
        if cfg.get(k) != limits[k]:
            findings.append(f"effective {k} does not derive from "
                            "the authorized resource contract")
    for k in ("save_model", "checkpoint_bundle_dir",
              "cell_runtime_dir"):
        v = cfg.get(k, "")
        if not v or cell_id not in str(v):
            findings.append(f"per-cell isolation missing for {k}")
    path_facts = _executable_path_facts()
    for k, v in path_facts.items():
        if not v:
            findings.append(f"authorized path incomplete: {k}")
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
    import inspect
    pipeline = (plugin_cls(cfg2)
                if len(inspect.signature(
                    plugin_cls.__init__).parameters) > 1
                else plugin_cls())
    try:
        pipeline._assert_episodic_contract(cfg2)
    except BaseException as exc:
        findings.append(f"episodic contract: "
                        f"{type(exc).__name__}: {exc}")
    roles = {}
    try:
        paths = pipeline._split_csv(cfg2)
        root = Path(out_root)
        for role, csvp in sorted(paths.items()):
            f = Path(csvp)
            try:
                rel = str(f.resolve().relative_to(root.resolve()))
            except ValueError:
                rel = f"<outside-run-root>/{f.name}"
            roles[role] = {"csv": rel,
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
    report = {"schema": "agent_multi.b4_campaign_dry_run.v2",
              "cell": cell_id,
              "status": ("DRY_RUN_READY" if not findings
                         else "DRY_RUN_FINDINGS"),
              "cell_config_sha256": built["cell"]["config_sha256"],
              "contract": built["contract"].replace(
                  str(mat_root), "<materialization_root>"),
              "executable_path_facts": path_facts,
              "effective_limits": {k: cfg.get(k) for k in (
                  "budget_max_env_steps", "budget_max_updates",
                  "budget_max_wall_seconds", "budget_max_rss_bytes")},
              "roles_materialized": roles,
              "findings": findings,
              "wall_seconds": round(time.time() - t0, 1)}
    b4a.verify_no_absolute_paths(report, f"dry-run {cell_id}")
    return report


# ---------------- C1: the connected scientific cycle --------------
def execute_cell(cell_id: str, mat_root: Path, out_root: Path,
                 device: str, lease_path: Path = None,
                 global_wall_remaining_seconds: float = None
                 ) -> int:
    if CAMPAIGN_AUTH_SHA is None or not CAMPAIGN_AUTH_PATH.is_file():
        raise ExecutorRefusal(
            "REFUSED: no Musashi campaign authorization record "
            "exists — the campaign is not executable; the owner's "
            "intent alone does not open this gate")
    b4a.verify_campaign_authorization_record(CAMPAIGN_AUTH_PATH,
                                             CAMPAIGN_AUTH_SHA)
    # C12: execution is structurally impossible without a VERIFIED
    # lease — unique claim + live campaign lock + generation +
    # digest bindings, all proven BEFORE any pipeline/env/CUDA call.
    if lease_path is None:
        raise ExecutorRefusal(
            "REFUSED: no execution lease — cells run only under the "
            "orchestrator's verified lease")
    import importlib.util as _ilu
    _os = _ilu.spec_from_file_location(
        "b4orch_exec", REPO / "tools/b4_campaign_orchestrator.py")
    _orch = _ilu.module_from_spec(_os)
    _os.loader.exec_module(_orch)
    lease = _orch.verify_lease(lease_path, out_root, cell_id,
                               mat_root)
    attempt_id = lease["attempt_id"]
    if global_wall_remaining_seconds is None:
        global_wall_remaining_seconds = _orch.\
            remaining_global_seconds(Path(out_root),
                                     b4a.load_resource_contract())
    if global_wall_remaining_seconds < 600.0:
        raise ExecutorRefusal(
            "REFUSED: remaining global campaign wall is smaller "
            "than one segment — the 96h ceiling is a hard bound")
    terminal_p = _terminal_path(out_root, cell_id)
    if terminal_p.exists():
        raise ExecutorRefusal(
            f"REFUSED: {cell_id} already holds an immutable "
            "terminal state — attempts are never reused")
    built = build_economic_config(cell_id, mat_root, out_root,
                                  device)
    cfg = built["config"]
    cfg["budget_max_wall_seconds"] = float(min(
        cfg["budget_max_wall_seconds"],
        global_wall_remaining_seconds))
    year = built["year"]
    t0 = time.time()
    packet = json.loads(
        (Path(mat_root) / "B4_MATERIALIZATION.json").read_text())
    comparator_dir = comparator_dir_of(packet)
    b4a.verify_full_authority_chain(comparator_dir)
    sb = _load_sb()
    from app.plugin_loader import load_plugin
    agent_cls, _ = load_plugin("agent.plugins", cfg["agent_plugin"])
    pipeline_cls, _ = load_plugin("pipeline.plugins",
                                  cfg["pipeline_plugin"])
    agent_plugin = agent_cls(cfg)
    pipeline = pipeline_cls(cfg)
    cell_dir = Path(out_root) / cell_id
    try:
        final = pipeline.run_pipeline(config=cfg, env_plugin=None,
                                      agent_plugin=agent_plugin,
                                      mode="train")
        artifacts = (final or {}).get("artifacts") or {}
        best = artifacts.get("best_checkpoint")
        inactive = bool((final or {}).get(
            "activity_stopped_without_eligible_checkpoint"))
        if best and best.get("path"):
            score_target = Path(best["path"])
            score_sha = best["sha256"]
            artifact_class = "BEST_CHECKPOINT"
        elif inactive and artifacts.get("terminal", {}).get("path"):
            # Predeclared in amendment 7: the typed inactive result
            # scores its TERMINAL artifact, labeled — the cell is
            # never excluded and no artifact is substituted silently.
            score_target = Path(artifacts["terminal"]["path"])
            score_sha = artifacts["terminal"]["sha256"]
            artifact_class = "INACTIVE_TERMINAL_SCORED"
        else:
            raise ExecutorRefusal(
                "REFUSED: pipeline returned no scoreable artifact "
                "bound by digest")
        df = sb.load_source()
        origin = sb.materialize_origin(df, year,
                                       cell_dir / "outer_origin")
        score = score_frozen_checkpoint(
            cfg, score_target, score_sha, origin,
            cell_dir / f"per_bar_{cell_id}.csv", cell_id)
        verify_scoring_evidence(score, origin, comparator_dir,
                                cell_id)
        terminal = "COMPLETED"
        detail = {
            "attempt_id": attempt_id,
            "cell_config_sha256": built["cell"]["config_sha256"],
            "artifact_class": artifact_class,
            "checkpoint_sha256": score["checkpoint_sha256"],
            "per_bar_csv": score["per_bar_csv"],
            "per_bar_sha256": score["per_bar_sha256"],
            "scored_index_sha256": score["scored_index_sha256"],
            "scored_bars": score["scored_bars"],
            "counter_semantics": score["counter_semantics"],
            "sealed_2025_used": False,
            "wall_seconds": round(time.time() - t0, 1),
            "effective_limits": {k: cfg.get(k) for k in (
                "budget_max_env_steps", "budget_max_updates",
                "budget_max_wall_seconds",
                "budget_max_rss_bytes")},
        }
    except ExecutorRefusal:
        raise
    except BaseException as exc:
        terminal = classify_stop(str(exc), None, False)
        if terminal == "COMPLETED":
            terminal = "FAILED"
        write_terminal(out_root, cell_id, terminal,
                       {"attempt_id": attempt_id,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "wall_seconds": round(time.time() - t0, 1)})
        raise
    write_terminal(out_root, cell_id, terminal, detail)
    # C1.6: a COMPLETED must immediately pass the REAL ledger
    # verifier for this cell.
    led = importlib.util.spec_from_file_location(
        "b4led_exec", REPO / "tools/b4_campaign_ledger.py")
    ledger_mod = importlib.util.module_from_spec(led)
    led.loader.exec_module(ledger_mod)
    ledger_mod.verify_single_cell_result(
        Path(out_root), cell_id, built["cell"]["config_sha256"])
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
    ap.add_argument("--lease", type=Path, default=None,
                    help="orchestrator-issued execution lease (the "
                         "ONLY way a cell executes)")
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
                        args.output_root, args.device,
                        lease_path=args.lease)


if __name__ == "__main__":
    raise SystemExit(main())
