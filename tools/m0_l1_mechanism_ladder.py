#!/usr/bin/env python3
"""Bounded M0-to-L1 mechanism ladder runner (finding 220, order WP3).

One arm per invocation, one GPU per arm, one diagnostic identity for
the whole ladder. The four training arms D0/D2/D3/D4 bind identical
immutable common facts (data revision, the L1 nested NON-TEST split
union, seed 101, the M0/D1 anchor, the M0 gradient-update budget,
normal-realistic evaluation with native SL/TP) and differ ONLY by the
contract's declared delta blocks:

  D0_M0_EXACT        the M0 E1_N1_LR03 recipe exactly, including the
                     as-run v3 phase-1 handoff semantics (epoch-0 /
                     anchor handoff eligible) — the positive control;
  D2_BOUNDARY_ONLY   + only the L1 boundary/reload mechanism (trained-
                     epoch handoff, epoch-0 structurally ineligible);
  D3_COST_PROTECTION + only the L1 v3 normal cost/protected-entry
                     contract;
  D4_FULL_L1         + the L1 activity floor, patience and stopping
                     rule.

This is DIAGNOSIS, not model selection: no new learning-rate levels,
seeds, topologies, features or order-type genes; sealed 2025 is never
touched; nothing here is promoted to live authority.

Runtime discipline reuses the corrected L1 machinery: fail-closed
contract/materialization refusals before any model construction,
content-addressed attempt directories, atomic (fsync+replace) records,
flock-backed per-arm exclusive claims, and a v2 heartbeat carrying
assigned vs bound vs observed CUDA facts — CUDA_VISIBLE_DEVICES must
EQUAL the arm's assigned GPU UUID or the arm refuses (WP13).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as sysid  # noqa: E402
from tools.l1_factorial_screen import (  # noqa: E402
    _terminal_tensor_sha,
    atomic_write_json,
)
from tools.l1_fleet_launcher import (  # noqa: E402
    ExclusiveClaim,
    _atomic_json,
    _pid_start_identity,
    visible_gpu_uuids,
)

CONTRACT_PATH = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                 "m0_l1_mechanism_ladder_v1.json")
CONTRACT_SCHEMA = "agent_multi.m0_l1_mechanism_ladder.v1"
RECORD_SCHEMA = "agent_multi.m0_l1_ladder_arm_record.v1"
HEARTBEAT_SCHEMA = "agent_multi.mechanism_ladder_heartbeat.v2"
GYM_FX_ROOT = Path("/home/harveybc/Documents/GitHub/gym-fx")
ARM_ORDER = ("D0_M0_EXACT", "D2_BOUNDARY_ONLY", "D3_COST_PROTECTION",
             "D4_FULL_L1")
HEARTBEAT_INTERVAL_S = 60
LADDER_SEED = 101

# Exit classes are the same contract the L1 launcher/systemd share.
EXIT_CLASS = {
    "ARM_COMPLETE": 0,
    "ALREADY_COMPLETE": 0,
    "ALREADY_RUNNING": 3,
    "REFUSED_WRONG_HOST": 4,
    "REFUSED_GPU_UNBOUND": 4,
    "REFUSED_BAD_CONTRACT": 4,
    "ARM_FAILED": 1,
}

# Per-split facts the five-row contrast table needs (order §3.5).
CONTRAST_METRICS = (
    "trades_total", "trades_won", "trades_lost",
    "action_raw_mean", "action_raw_std", "action_raw_min",
    "action_raw_max", "action_non_hold_rate", "action_dominant_rate",
    "action_deadband_rate", "action_diagnostics",
    "execution_diagnostics", "execution_entry_actions_seen",
    "execution_entry_orders_submitted", "no_trade_diagnosis",
    "total_return", "mean_weekly_return", "sharpe_ratio",
    "final_equity", "max_drawdown_pct", "episode_length",
    "termination_cause", "would_margin_call_count",
    "recapitalization_count", "recapitalization_debt", "solvency_mode",
)
_FORBIDDEN_SPLIT_METRICS = ("trades_total", "total_return",
                            "final_equity", "sharpe_ratio")


def _sha_file(path: Path) -> str:
    return sysid.sha_file(path)


def _agent_plugin(name: str):
    from importlib.metadata import entry_points

    ep = next(e for e in entry_points().select(group="agent.plugins")
              if e.name == name)
    return ep.load()()


# ---------------------------------------------------------------------------
# contract loading — typed refusals BEFORE any model construction
# ---------------------------------------------------------------------------

def load_contract(path: Path = CONTRACT_PATH) -> dict:
    contract = json.loads(Path(path).read_text())
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(
            f"unknown mechanism-ladder contract schema "
            f"{contract.get('schema')!r}")
    if int(contract.get("seed", -1)) != LADDER_SEED:
        raise ValueError(
            "the bounded diagnostic runs seed 101 ONLY (order §3.2); "
            f"contract declares {contract.get('seed')!r}")
    anchor = (contract.get("common") or {}).get("anchor") or {}
    for key in ("path", "sha256", "policy_tensor_sha256"):
        value = anchor.get(key)
        if not value or not isinstance(value, str):
            raise ValueError(f"common.anchor.{key} missing — an unpinned "
                             "anchor cannot anchor a diagnostic")
    for key in ("sha256", "policy_tensor_sha256"):
        if len(anchor[key]) != 64:
            raise ValueError(f"common.anchor.{key} is not a sha256 hex "
                             "digest")
    arms = contract.get("arms") or {}
    if tuple(sorted(arms, key=lambda a: arms[a]["order"])) != ARM_ORDER:
        raise ValueError(
            f"arms must be exactly {ARM_ORDER} in ladder order; got "
            f"{sorted(arms)}")
    groups = contract.get("delta_groups") or {}
    previous = None
    for name in ARM_ORDER:
        spec = arms[name]
        if spec.get("baseline") != previous:
            raise ValueError(
                f"{name}: baseline must be {previous!r}, got "
                f"{spec.get('baseline')!r} — the ladder is a chain of "
                "adjacent single-delta comparisons")
        delta = spec.get("delta")
        if not isinstance(delta, dict):
            raise ValueError(f"{name}: delta block missing")
        group = spec.get("delta_group")
        if previous is None:
            if delta or group is not None:
                raise ValueError(
                    "D0_M0_EXACT must carry an EMPTY delta — it is the "
                    "positive control, not a treatment")
        else:
            allowed = set(groups.get(group) or ())
            if not allowed:
                raise ValueError(
                    f"{name}: delta_group {group!r} is not declared in "
                    "delta_groups")
            stray = set(delta) - allowed
            if stray:
                raise ValueError(
                    f"{name}: delta touches fields outside its declared "
                    f"group {group!r}: {sorted(stray)}")
            if not delta:
                raise ValueError(f"{name}: a treatment arm cannot have "
                                 "an empty delta")
        assignment = (contract.get("assignments") or {}).get(name) or {}
        if not assignment.get("hostname") or not assignment.get(
                "gpu_uuid"):
            raise ValueError(
                f"{name}: worker assignment must pin hostname AND GPU "
                "UUID")
        previous = name
    contract["_contract_sha256"] = _sha_file(Path(path))
    contract["_contract_path"] = str(path)
    return contract


# ---------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------

def source_identities() -> dict:
    return {
        "agent-multi": sysid.source_tree_identity(
            sysid.resolve_repo_root(Path(__file__))),
        "gym-fx": sysid.source_tree_identity(GYM_FX_ROOT),
    }


def diagnostic_identity(contract: dict,
                        sources: dict | None = None) -> str:
    """sha256(contract sha + splits sha + anchor sha + code identities
    + 'diagnostic')[:16] — ONE identity for the whole ladder; reuses no
    L1 decision identity."""
    sources = sources or source_identities()
    nested_rel = contract["common"]["nested_split_contract"]["path"]
    payload = {
        "contract": contract["_contract_sha256"],
        "nested_split_contract": _sha_file(REPO / nested_rel),
        "anchor": contract["common"]["anchor"]["sha256"],
        "code": {name: {"commit": s["commit"],
                        "dirty_untracked_digest":
                            s["dirty_untracked_digest"]}
                 for name, s in sorted(sources.items())},
        "profile": "diagnostic",
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True).encode()).hexdigest()[:16]


def cumulative_delta(contract: dict, arm: str) -> dict:
    """Deltas of every arm at or below this arm's ladder position,
    applied in order — each arm IS its baseline plus its own block."""
    arms = contract["arms"]
    resolved: dict = {}
    for name in ARM_ORDER:
        if arms[name]["order"] > arms[arm]["order"]:
            break
        resolved.update(arms[name].get("delta") or {})
    return resolved


def arm_identity(diag_id: str, arm: str, contract: dict) -> str:
    payload = {
        "diagnostic_id": diag_id,
        "arm": arm,
        "seed": LADDER_SEED,
        "delta": cumulative_delta(contract, arm),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# immutable common facts — verified fail-closed
# ---------------------------------------------------------------------------

_UNION_CACHE: dict = {}


def verify_nested_nontest_union(contract: dict) -> dict:
    """Prove the M0 executable split windows are EXACTLY the union of
    the L1 nested non-test roles on the pinned CSV: train = fit_train +
    inner_validation, validation = outer_validation, sealed 2025 rows
    excluded. Any mismatch refuses the ladder before model construction.
    """
    import pandas as pd

    common = contract["common"]
    data_path = Path(common["data"]["path"])
    data_sha = _sha_file(data_path)
    if data_sha != common["data"]["sha256"]:
        raise RuntimeError("input CSV drifted from the ladder contract")
    cache_key = data_sha
    if cache_key in _UNION_CACHE:
        return _UNION_CACHE[cache_key]

    nested_path = REPO / common["nested_split_contract"]["path"]
    if _sha_file(nested_path) != common["nested_split_contract"][
            "sha256"]:
        raise RuntimeError(
            "nested split contract drifted from the ladder contract")
    nested = json.loads(nested_path.read_text())
    if nested.get("source_sha256") != data_sha:
        raise RuntimeError(
            "nested split contract pins a different CSV than the "
            "ladder data binding")

    date_col = common["data"].get("date_column", "DATE_TIME")
    frame = pd.read_csv(data_path, usecols=[date_col])
    stamps = pd.to_datetime(frame[date_col], errors="coerce").dropna()
    if len(stamps) != int(common["data"]["rows"]):
        raise RuntimeError(
            f"CSV has {len(stamps)} parseable rows, contract pins "
            f"{common['data']['rows']}")

    def rows(start: str, end: str) -> int:
        return int(((stamps >= pd.Timestamp(start))
                    & (stamps < pd.Timestamp(end))).sum())

    roles = nested["roles"]
    expected = nested["expected_rows"]
    observed_roles = {name: rows(spec["start"], spec["end"])
                      for name, spec in roles.items()}
    for name, count in observed_roles.items():
        if count != int(expected[name]):
            raise RuntimeError(
                f"nested role {name} has {count} rows, contract "
                f"expects {expected[name]}")

    splits = common["splits"]
    m0_train = rows(splits["train_start"], splits["train_end"])
    m0_val = rows(splits["validation_start"], splits["validation_end"])
    union = common["nested_split_contract"]["non_test_union"]
    facts = {
        "m0_train_rows": m0_train,
        "m0_validation_rows": m0_val,
        "fit_train_plus_inner_validation": (
            int(expected["fit_train"]) + int(expected["inner_validation"])),
        "outer_validation": int(expected["outer_validation"]),
        "sealed_test_rows_excluded": int(expected["sealed_test"]),
    }
    if m0_train != facts["fit_train_plus_inner_validation"] \
            or m0_train != int(union["train_rows"]):
        raise RuntimeError(
            "M0 train window is NOT the fit_train+inner_validation "
            f"union: {facts}")
    if m0_val != facts["outer_validation"] \
            or m0_val != int(union["validation_rows"]):
        raise RuntimeError(
            "M0 validation window is NOT the outer_validation role: "
            f"{facts}")
    sealed_start = pd.Timestamp(roles["sealed_test"]["start"])
    if pd.Timestamp(splits["train_end"]) > sealed_start or \
            pd.Timestamp(splits["validation_end"]) > sealed_start:
        raise RuntimeError("a training-arm window reaches into sealed "
                           "2025 — refused")
    _UNION_CACHE[cache_key] = facts
    return facts


# ---------------------------------------------------------------------------
# materialization — the ONLY path from contract to a runnable config
# ---------------------------------------------------------------------------

def materialize_arm_config(contract: dict, arm: str,
                           out_dir: Path) -> dict:
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown ladder arm {arm!r}")
    common = contract["common"]

    base_path = REPO / common["base_config"]["path"]
    actual_base_sha = _sha_file(base_path)
    if actual_base_sha != common["base_config"]["sha256"]:
        raise RuntimeError(
            "base config drifted from the ladder contract binding")

    # The D3 delta must equal the LIVE frozen v3 manifest cost contract
    # — an embedded copy that drifted from its source is a lie.
    manifest_path = REPO / contract["l1_reference"][
        "system_manifest_path"]
    if _sha_file(manifest_path) != contract["l1_reference"][
            "system_manifest_sha256"]:
        raise RuntimeError(
            "L1 system manifest drifted from the ladder contract")
    manifest = json.loads(manifest_path.read_text())
    v3_costs = manifest["costs"]["config_bindings"]
    d3_delta = contract["arms"]["D3_COST_PROTECTION"]["delta"]
    if d3_delta != v3_costs:
        raise RuntimeError(
            "D3 cost delta does not equal the frozen v3 manifest cost "
            "bindings — the execution-realism treatment is unbound")
    sysid.validate_normal_contract(d3_delta)

    union_facts = verify_nested_nontest_union(contract)

    # --- the M0 E1_N1_LR03 recipe, exactly as the screen ran it -------
    config = json.loads(base_path.read_text())
    for field in ("train_years", "val_years", "test_years"):
        config.pop(field, None)
    config["split_contract_note"] = (
        "explicit split dates govern; year-count shorthand removed")
    splits = dict(common["splits"])
    evaluate_test = bool(splits.pop("evaluate_test_split", False))
    if evaluate_test:
        raise RuntimeError("ladder contract enables the sealed test "
                           "split — refused")
    config.update(splits)
    config["input_data_file"] = str(Path(common["data"]["path"]))
    config["env_mode"] = "training"
    config["eval_seed"] = LADDER_SEED
    config["train_seed"] = LADDER_SEED
    config["ga_seed"] = LADDER_SEED
    config["epoch_timesteps"] = int(common["budget"]["epoch_timesteps"])
    config["l1_min_checkpoint_timesteps"] = 1
    config["evaluate_test_split"] = False
    # Evidence plumbing, UNIFORM across every arm (never an arm delta):
    # a fully-inactive arm is a MEASURED diagnostic outcome and must
    # land a typed record — the legacy no-eligible-checkpoint raise
    # crashed D2 without evidence (observed 2026-08-11).
    config["inactive_terminal_is_typed_result"] = True
    config["selection_metric"] = str(common["selection_metric"])
    config["selection_min_trades"] = int(common["selection_min_trades"])
    config["quiet_mode"] = True
    config["save_model"] = str(Path(out_dir) / "model.zip")
    config["return_trace_dir"] = str(Path(out_dir) / "return_traces")
    config["learning_rate"] = float(common["learning_rates"]["normal"])
    config["easy_learning_rate"] = float(common["learning_rates"]["easy"])
    anchor_path = Path(common["anchor"]["path"]).expanduser()
    config["warm_start_model"] = str(anchor_path)
    config["warm_start_model_sha256"] = common["anchor"]["sha256"]
    config["l1_patience"] = 10_000          # M0: no early stopping
    config["easy_patience"] = 10_000
    config["max_epochs"] = int(common["budget"]["normal_epochs"])
    config["easy_max_epochs"] = int(common["budget"]["easy_epochs"])
    config["execution_cost_curriculum_epochs"] = max(
        2, int(common["budget"]["normal_epochs"]))
    # Finding 191 discipline: the plugin that EXECUTES is the plugin the
    # contract names. The M0 screen executed sac_agent (proven by the
    # warm-start transfer evidence only sac_agent emits), so the config
    # names it too instead of the base config's dormant value.
    config["agent_plugin"] = str(common["plugins"]["agent_plugin"])
    for key in ("env_plugin", "strategy_plugin"):
        bound = common["plugins"].get(key)
        if bound and config.get(key) != bound:
            raise RuntimeError(
                f"plugin drift: config[{key!r}]={config.get(key)!r} != "
                f"ladder binding {bound!r}")
    # The as-run M0 boundary (post_easy.v3 semantics) is the D0 base;
    # D2 flips exactly this field to the corrected L1 boundary.
    config["phase1_handoff_semantics"] = "m0_epoch0_eligible_v3"

    # --- the arm's cumulative declared delta, and NOTHING else --------
    config.update(cumulative_delta(contract, arm))

    config["_identity"] = {
        "diagnostic_contract_sha256": contract["_contract_sha256"],
        "base_config_sha256": actual_base_sha,
        "data_sha256": common["data"]["sha256"],
        "nested_split_contract_sha256": common["nested_split_contract"][
            "sha256"],
        "nested_nontest_union": union_facts,
        "l1_system_manifest_sha256": contract["l1_reference"][
            "system_manifest_sha256"],
        "m0_contract_sha256": contract["m0_reference"]["contract_sha256"],
        "anchor_sha256": common["anchor"]["sha256"],
        "anchor_policy_tensor_sha256": common["anchor"][
            "policy_tensor_sha256"],
        "arm": arm,
        "delta_applied": cumulative_delta(contract, arm),
        "seed": LADDER_SEED,
        "plugins": dict(common["plugins"]),
    }
    return config


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def record_is_complete(record_path: Path, expected_arm_id: str) -> bool:
    try:
        record = json.loads(record_path.read_text())
    except Exception:
        return False
    if record.get("schema") != RECORD_SCHEMA:
        return False
    if record.get("arm_identity") != expected_arm_id:
        return False
    for key in ("terminal_model_path", "terminal_model_sha256"):
        if not record.get(key):
            return False
    terminal = Path(record["terminal_model_path"])
    if not terminal.is_file():
        return False
    return _sha_file(terminal) == record["terminal_model_sha256"]


def _next_attempt_dir(arm_dir: Path, arm_id: str) -> Path:
    n = 0
    while True:
        n += 1
        attempt = arm_dir / f"attempt-{arm_id}-{n:02d}"
        if not attempt.exists():
            attempt.mkdir(parents=True)
            return attempt


def _splits_raw(result: dict) -> dict:
    """Contrast-table facts per split; the sealed test split must be a
    pure skip marker."""
    splits = result.get("splits") or {}
    out = {}
    for name, summary in splits.items():
        if not isinstance(summary, dict):
            continue
        if name == "test":
            if summary.get("evaluation_skipped") is not True:
                raise AssertionError(
                    "test split was evaluated — contract violated")
            if any(k in summary for k in _FORBIDDEN_SPLIT_METRICS):
                raise AssertionError(
                    "test split carries metric data — contract violated")
            continue
        if name not in ("train", "train_tail", "validation"):
            raise AssertionError(f"forbidden split {name!r} in result")
        out[name] = {key: summary.get(key) for key in CONTRAST_METRICS
                     if summary.get(key) is not None}
    return out


# ---------------------------------------------------------------------------
# GPU binding (WP13): CUDA_VISIBLE_DEVICES must EQUAL the assignment
# ---------------------------------------------------------------------------

def check_gpu_binding(contract: dict, arm: str, *,
                      hostname: str | None = None,
                      cuda_env: str | None = "<from-environment>",
                      observed_uuids: list | None = None,
                      enforce: bool = True) -> dict | None:
    assignment = (contract.get("assignments") or {}).get(arm) or {}
    hostname = hostname or socket.gethostname()
    if cuda_env == "<from-environment>":
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assignment.get("hostname") != hostname:
        return {"outcome": "REFUSED_WRONG_HOST",
                "reason": (f"arm {arm} is assigned to "
                           f"{assignment.get('hostname')!r}, this is "
                           f"{hostname!r}")}
    if not enforce:
        return None
    assigned = assignment.get("gpu_uuid")
    observed = (visible_gpu_uuids() if observed_uuids is None
                else observed_uuids)
    if assigned not in observed:
        return {"outcome": "REFUSED_GPU_UNBOUND",
                "reason": f"assigned GPU {assigned} not visible on "
                          f"{hostname}"}
    if cuda_env is None:
        return {"outcome": "REFUSED_GPU_UNBOUND",
                "reason": ("CUDA_VISIBLE_DEVICES is unset; visibility "
                           "is not an execution binding (WP13)")}
    if cuda_env != assigned:
        return {"outcome": "REFUSED_GPU_UNBOUND",
                "reason": (f"CUDA_VISIBLE_DEVICES={cuda_env!r} does not "
                           f"equal the assignment {assigned!r}")}
    return None


def gpu_telemetry(gpu_uuid: str | None) -> dict:
    """Fresh temperature/utilization facts for the assigned GPU
    (order §3.5.3). Absent facts stay None — never zero."""
    telemetry = {"gpu_temperature_c": None, "gpu_utilization_pct": None}
    if not gpu_uuid:
        return telemetry
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=uuid,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3 and parts[0] == gpu_uuid:
                telemetry["gpu_temperature_c"] = int(parts[1])
                telemetry["gpu_utilization_pct"] = int(parts[2])
                break
    except Exception:
        pass
    return telemetry


class ArmHeartbeat:
    """Atomic heartbeat v2 with assigned/bound/observed CUDA facts."""

    def __init__(self, path: Path, *, contract: dict, arm: str,
                 diag_id: str, arm_id: str):
        self.path = path
        self.contract = contract
        self.arm = arm
        self.diag_id = diag_id
        self.arm_id = arm_id
        self._state: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self, **update) -> None:
        self._state.update(update)
        pid = os.getpid()
        assignment = (self.contract.get("assignments") or {}).get(
            self.arm) or {}
        self._state.update({
            "schema": HEARTBEAT_SCHEMA,
            "arm": self.arm,
            "diagnostic_identity": self.diag_id,
            "arm_identity": self.arm_id,
            "seed": LADDER_SEED,
            "pid": pid,
            "pid_start_identity": _pid_start_identity(pid),
            "hostname": socket.gethostname(),
            "assigned_gpu_uuid": assignment.get("gpu_uuid"),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"),
            "observed_gpu_uuids": visible_gpu_uuids(),
            **gpu_telemetry(assignment.get("gpu_uuid")),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.path, self._state)

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.wait(HEARTBEAT_INTERVAL_S):
                self.beat()
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# the arm run
# ---------------------------------------------------------------------------

def run_arm(arm: str, *, contract: dict,
            enforce_gpu: bool = True) -> dict:
    from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
        PipelinePlugin as CurriculumPipeline)
    from pipeline_plugins.rl_pipeline_with_validation import (
        PipelinePlugin as ValidationPipeline)

    sources_before = source_identities()
    diag_id = diagnostic_identity(contract, sources=sources_before)
    arm_id = arm_identity(diag_id, arm, contract)
    out_root = Path(contract["output_root"]).expanduser()
    ladder_dir = out_root / diag_id
    arm_dir = ladder_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = ArmHeartbeat(arm_dir / "heartbeat.json",
                             contract=contract, arm=arm,
                             diag_id=diag_id, arm_id=arm_id)

    refusal = check_gpu_binding(contract, arm, enforce=enforce_gpu)
    if refusal:
        heartbeat.beat(terminal_state=refusal["outcome"],
                       error=refusal.get("reason"))
        refusal["diagnostic_identity"] = diag_id
        return refusal

    record_path = arm_dir / "ladder_arm_record.json"
    if record_path.exists():
        if record_is_complete(record_path, arm_id):
            record = json.loads(record_path.read_text())
            record["_reuse"] = "ALREADY_COMPLETE"
            heartbeat.beat(terminal_state="ALREADY_COMPLETE")
            return record
        raise RuntimeError(
            f"arm {arm}: existing record fails validation; refusing to "
            "overwrite — recover it explicitly")

    claim = ExclusiveClaim(
        ladder_dir / "locks" / f"exclusive_claim.{arm}.lock")
    if not claim.acquire():
        return {"outcome": "ALREADY_RUNNING",
                "diagnostic_identity": diag_id, "arm": arm,
                "holder": claim.holder()}
    heartbeat.start()
    try:
        anchor_path = Path(
            contract["common"]["anchor"]["path"]).expanduser()
        if not anchor_path.is_file():
            raise RuntimeError(f"anchor missing: {anchor_path}")
        actual_anchor = _sha_file(anchor_path)
        if actual_anchor != contract["common"]["anchor"]["sha256"]:
            raise RuntimeError(
                "anchor hash mismatch — the mature source anchor is not "
                "the bound artifact")

        out_dir = _next_attempt_dir(arm_dir, arm_id)
        heartbeat.beat(terminal_state="RUNNING", attempt=str(out_dir),
                       progress="materializing")
        config = materialize_arm_config(contract, arm, out_dir)
        identity = config.pop("_identity")
        resolved_sha = sysid.resolved_config_sha256(config)
        agent_name = identity["plugins"]["agent_plugin"]
        agent = _agent_plugin(agent_name)

        started = datetime.now(timezone.utc)
        heartbeat.beat(progress="training")
        pipeline = CurriculumPipeline(config)
        result = pipeline.run_pipeline(config=config, env_plugin=None,
                                       agent_plugin=agent, mode="train")
        heartbeat.beat(progress="terminal-evaluation")

        terminal_path = result.get("terminal_model_path")
        if not terminal_path or not Path(terminal_path).is_file():
            raise RuntimeError(
                "arm finished without a terminal artifact — invalid, "
                "record refused")
        terminal_sha = _sha_file(Path(terminal_path))
        terminal_tensor = _terminal_tensor_sha(terminal_path)
        best_path = result.get("best_model_path")
        best_sha = (_sha_file(Path(best_path))
                    if best_path and Path(best_path).is_file() else None)

        # Terminal weights evaluated RAW under normal_realistic with
        # native SL/TP — first under the arm's own cost contract (M0
        # parity), then under the L1 v3 cost contract as the common
        # measuring stick (identical for D3/D4, so evaluated once).
        def _terminal_eval(eval_costs: dict | None, trace_suffix: str):
            eval_config = dict(config)
            if eval_costs:
                eval_config.update(eval_costs)
            eval_config["load_model"] = str(terminal_path)
            eval_config["solvency_mode"] = "normal_realistic"
            eval_config["return_trace_dir"] = str(
                out_dir / f"return_traces_terminal_{trace_suffix}")
            eval_result = ValidationPipeline(eval_config).run_pipeline(
                config=eval_config, env_plugin=None, agent_plugin=agent,
                mode="inference")
            return _splits_raw(eval_result)

        as_run_splits = _terminal_eval(None, "as_run")
        v3_costs = contract["arms"]["D3_COST_PROTECTION"]["delta"]
        costs_already_l1 = all(
            config.get(key) == value for key, value in v3_costs.items())
        if costs_already_l1:
            l1_cost_splits = as_run_splits
        else:
            l1_cost_splits = _terminal_eval(v3_costs, "l1_costs")
        finished = datetime.now(timezone.utc)

        sources_after = source_identities()
        for name in sources_before:
            sysid.assert_source_identity_unmoved(
                sources_before[name], sources_after[name])

        history = result.get("history") or []
        last = history[-1] if history else {}
        post_easy_meta = (result.get("curriculum") or {}).get(
            "post_easy") or {}
        assignment = (contract.get("assignments") or {}).get(arm) or {}
        gpu_binding = {
            "assigned_hostname": assignment.get("hostname"),
            "observed_hostname": socket.gethostname(),
            "assigned_gpu_uuid": assignment.get("gpu_uuid"),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"),
            "observed_gpu_uuids": visible_gpu_uuids(),
        }
        try:
            import torch
            gpu_binding["torch_cuda_available"] = \
                torch.cuda.is_available()
            gpu_binding["torch_cuda_device_count"] = \
                torch.cuda.device_count()
            if torch.cuda.is_available():
                gpu_binding["torch_cuda_device_name"] = \
                    torch.cuda.get_device_name(0)
        except Exception:
            gpu_binding["torch_cuda_available"] = None

        as_run_val = as_run_splits.get("validation") or {}
        terminal_val_trades = as_run_val.get("trades_total")
        updates = last.get("gradient_updates_total")
        m0_activity = {
            "activity_survived_normal": (
                None if terminal_val_trades is None
                else bool(terminal_val_trades
                          and int(terminal_val_trades) > 0)),
            "weights_changed_from_anchor": (
                terminal_tensor != identity[
                    "anchor_policy_tensor_sha256"]),
            "normal_updates_applied": updates,
            "terminal_usable": bool(
                terminal_val_trades and int(terminal_val_trades) > 0
                and terminal_tensor != identity[
                    "anchor_policy_tensor_sha256"]
                and updates and int(updates) > 0),
        }

        record = {
            "schema": RECORD_SCHEMA,
            "evidence_class": "mechanism_diagnostic",
            "decision_eligible": False,
            "performance_aggregate_eligible": False,
            "live_promotion_eligible": False,
            "diagnostic_identity": diag_id,
            "arm_identity": arm_id,
            "arm": arm,
            "ladder_order": contract["arms"][arm]["order"],
            "baseline_arm": contract["arms"][arm]["baseline"],
            "delta_group": contract["arms"][arm].get("delta_group"),
            "delta_own": contract["arms"][arm].get("delta") or {},
            "delta_cumulative": cumulative_delta(contract, arm),
            "seed": LADDER_SEED,
            "attempt_dir": str(out_dir),
            "contract_sha256": contract["_contract_sha256"],
            "resolved_config_sha256": resolved_sha,
            "identity": identity,
            "gpu_binding": gpu_binding,
            "subject_code_identity": sources_before,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
            "phase1_handoff_semantics": config.get(
                "phase1_handoff_semantics"),
            "phase1_selection_basis": post_easy_meta.get(
                "selection_basis"),
            "phase1_best_easy_epoch": post_easy_meta.get(
                "best_easy_epoch"),
            "phase1_gradient_updates": post_easy_meta.get(
                "phase1_gradient_updates"),
            "phase1_artifact_sha256": post_easy_meta.get(
                "artifact_sha256"),
            "phase1_terminal_policy_tensor_sha256": post_easy_meta.get(
                "phase1_terminal_policy_tensor_sha256"),
            "gradient_updates_total": updates,
            "epoch_history": history,
            "stop_reason": result.get("stop_reason"),
            "termination_cause": result.get("termination_cause"),
            "activity_stopped_without_eligible_checkpoint": bool(
                result.get(
                    "activity_stopped_without_eligible_checkpoint",
                    False)),
            "boundary_transfer_evidence": result.get(
                "warm_start_transfer_evidence"),
            "anchor_sha256": actual_anchor,
            "anchor_policy_tensor_sha256": identity[
                "anchor_policy_tensor_sha256"],
            "best_model_path": best_path,
            "best_model_sha256": best_sha,
            "terminal_model_path": str(Path(terminal_path).resolve()),
            "terminal_model_sha256": terminal_sha,
            "terminal_policy_tensor_sha256": terminal_tensor,
            "terminal_evaluation_as_run": {
                "cost_contract": "arm_resolved_config",
                "splits_raw": as_run_splits,
            },
            "terminal_evaluation_l1_costs": {
                "cost_contract": "l1_v3_manifest_bindings",
                "identical_to_as_run": costs_already_l1,
                "splits_raw": l1_cost_splits,
            },
            "m0_activity_facts": m0_activity,
            "curriculum": result.get("curriculum"),
        }
        atomic_write_json(record_path, record)
        heartbeat.beat(terminal_state="ARM_COMPLETE",
                       last_artifact=record["terminal_model_path"],
                       progress="complete")
        return record
    except Exception as exc:
        heartbeat.beat(terminal_state="ARM_FAILED",
                       error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        heartbeat.stop()
        claim.release()


def contrast_row(record: dict) -> dict:
    """One row of the five-row contrast table (order §3.5)."""
    val = ((record.get("terminal_evaluation_as_run") or {})
           .get("splits_raw") or {}).get("validation") or {}
    exec_diag = val.get("execution_diagnostics") or {}
    protected = sum(int(exec_diag.get(key, 0) or 0) for key in (
        "protected_market_entries", "protected_limit_entries",
        "protected_stop_entries"))
    trades = {name: (split or {}).get("trades_total")
              for name, split in ((record.get(
                  "terminal_evaluation_as_run") or {}).get(
                  "splits_raw") or {}).items()}
    return {
        "arm": record.get("arm"),
        "raw_action_mean": val.get("action_raw_mean"),
        "raw_action_std": val.get("action_raw_std"),
        "non_hold_rate": val.get("action_non_hold_rate"),
        "protected_entries_submitted": protected,
        "trades_by_split": trades,
        "gradient_updates_total": record.get("gradient_updates_total"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "terminal_model_sha256": record.get("terminal_model_sha256"),
        "terminal_policy_tensor_sha256": record.get(
            "terminal_policy_tensor_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARM_ORDER)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--no-gpu-check", action="store_true",
                        help="skip GPU-UUID binding enforcement "
                             "(socket-free tests only; a fleet launch "
                             "MUST enforce)")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    try:
        record = run_arm(args.arm, contract=contract,
                         enforce_gpu=not args.no_gpu_check)
    except Exception as exc:                       # noqa: BLE001
        print(json.dumps({"outcome": "ARM_FAILED", "arm": args.arm,
                          "error": f"{type(exc).__name__}: {exc}"},
                         default=str), flush=True)
        return EXIT_CLASS["ARM_FAILED"]
    if record.get("outcome") in EXIT_CLASS:
        print(json.dumps(record, default=str), flush=True)
        return EXIT_CLASS[record["outcome"]]
    outcome = ("ALREADY_COMPLETE"
               if record.get("_reuse") == "ALREADY_COMPLETE"
               else "ARM_COMPLETE")
    print(json.dumps({"outcome": outcome,
                      "diagnostic_identity": record.get(
                          "diagnostic_identity"),
                      "arm_identity": record.get("arm_identity"),
                      "contrast_row": contrast_row(record)},
                     default=str), flush=True)
    return EXIT_CLASS[outcome]


if __name__ == "__main__":
    sys.exit(main())
