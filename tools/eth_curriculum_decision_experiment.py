#!/usr/bin/env python3
"""Paired ETH/SAC curriculum decision experiment (decision order WP-C).

LOCAL-ONLY: writes no DOIN blocks, populations, champion archives or
succession artifacts. One seed per invocation (one per GPU); each host
runs ALL arms for its seed so host/GPU is a blocking factor, never an
arm confound.

Arms (equal-compute primary comparison):
  N14    — 14 normal epochs;
  EN4_10 — 4 easy epochs then 10 normal epochs (fresh replay buffer at
           the dynamics boundary via the WP-D pipeline);
  E4     — 4 easy epochs, inference-only normal evaluation (diagnostic).

Fixed contracts: frozen ETHUSD 4h dataset (sha asserted), 83-feature
causal observation (window 32, rolling 256), one anchor initialization
shared by every arm of the seed, 20,000 timesteps/epoch, NO early
stopping, 2024-validation-only selection metrics, disclosed 2025
disabled and absent from every payload.

Finding 114 evidence per arm: fully resolved config + sha256, anchor
and artifact hashes, return traces (train/train_tail/validation) with
sha256, per-epoch learning/activity trace, lineage, raw same-scale
metrics. Artifacts are replicated to a second host by the wrapper
script; this tool records the primary paths and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from importlib.metadata import entry_points  # noqa: E402

DATA_FILE = ("/home/harveybc/Documents/GitHub/predictor/examples/data/"
             "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
DATA_SHA256 = ("1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435"
               "ebe440f")
ETH_BASE = (REPO / "examples/results/"
            "project3_ethusdt_4h_sac_train_val_test_v2/config_out.json")
OBSERVATION_MANIFEST = (
    REPO / "docs/work_plan/34_ETH_DATA_OBSERVATION_MANIFEST.md")
# AUD-F1-20260806-126: the base contract is PINNED. An unpinned base
# would let the A/B silently change semantics between seeds/hosts.
ETH_BASE_SHA256 = ("5df24c0d78a89613041406213692af5f180f4af3c220ac69792a"
                   "60db8ca70e18")
SPLITS = {
    "train_start": "2017-09-28T04:00:00",
    "train_end": "2023-12-31T23:59:59",
    "validation_start": "2024-01-01T00:00:00",
    "validation_end": "2024-12-31T23:59:59",
    "test_start": "2025-01-01T00:00:00",
    "test_end": "2025-12-31T23:59:59",
}
EPOCH_TIMESTEPS = 20_000
ARMS = ("N14", "EN4_10", "E4")
ALLOWED_SPLITS = ("train", "train_tail", "validation")
RAW_METRICS = (
    "initial_cash", "final_equity", "total_return", "mean_weekly_return",
    "annualized_return", "max_drawdown_fraction", "max_drawdown_pct",
    "trades_total", "trades_won", "trades_lost", "episode_length",
    "would_margin_call_count", "termination_cause",
    "recapitalization_count", "recapitalization_debt",
    "action_counts", "sharpe_ratio",
)


RUNNER_VERSION = "eth_curriculum_decision_experiment.v4"
ARM_RECORD_SCHEMA = "agent_multi.arm_record.v5"
LOCAL_HOST = os.uname().nodename


def _execution_id(arm: str, seed: int, anchor: Path, *,
                  epoch_timesteps: int) -> str:
    """Content address of one arm execution (AUD-F1-20260806-130):
    arm, seed, shared anchor bytes, dataset hash, pinned base contract,
    split contract, budget, metric schema, code lineage and runner
    version. Any change produces a different id."""
    identity = {
        "arm": arm,
        "seed": seed,
        "anchor_sha256": _sha(anchor) if anchor.is_file() else None,
        "data_sha256": DATA_SHA256,
        "base_contract_sha256": ETH_BASE_SHA256,
        "splits": SPLITS,
        "epoch_timesteps": epoch_timesteps,
        "arm_budgets": {"N14": 14, "EN4_10": [4, 10], "E4": 4},
        "metric_schema": "lexicographic_weekly_v1",
        "observation_manifest_sha256": (
            hashlib.sha256(OBSERVATION_MANIFEST.read_bytes()).hexdigest()
            if OBSERVATION_MANIFEST.exists() else "unavailable"),
        "runner_version": RUNNER_VERSION,
        "lineage": {repo: _git_rev(repo) for repo in
                    ("agent-multi", "gym-fx")},
    }
    return hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()


REQUIRED_FINITE_METRICS = ("mean_weekly_return", "total_return",
                           "max_drawdown_fraction")


def validate_arm_record(record: dict, arm: str) -> list:
    """One complete-record validator shared by reuse and aggregation
    (AUD-F1-20260806-136/137). Verifies structure AND the filesystem:
    every declared artifact must exist with its recorded hash."""
    problems = []
    if record.get("schema") != ARM_RECORD_SCHEMA:
        problems.append(
            f"record schema {record.get('schema')!r} !="
            f" {ARM_RECORD_SCHEMA!r}")
    for key in ("execution_id", "resolved_config_sha256",
                "return_trace_sha256", "margin_telemetry",
                "code_revisions_before", "code_revisions_after"):
        if not record.get(key):
            problems.append(f"missing {key}")
    if record.get("code_revisions_before") != record.get(
            "code_revisions_after"):
        problems.append(
            "code revisions changed DURING the arm:"
            f" {record.get('code_revisions_before')} ->"
            f" {record.get('code_revisions_after')}")
    validation = (record.get("splits_raw") or {}).get("validation") or {}
    for key in REQUIRED_FINITE_METRICS:
        if not _finite_number(validation.get(key)):
            problems.append(f"validation {key} missing/non-finite")
    artifacts = record.get("artifacts") or {}
    expected_labels = {"best_checkpoint", "terminal"} if arm != "E4" \
        else {"post_easy"}
    for label in expected_labels:
        ref = artifacts.get(label)
        if not isinstance(ref, dict):
            problems.append(f"artifact {label} missing")
            continue
        for field in ("path", "sha256", "replica_path",
                      "load_proven"):
            if ref.get(field) in (None, ""):
                problems.append(f"artifact {label} missing {field}")
        path = Path(str(ref.get("path") or ""))
        if not path.is_file():
            problems.append(f"artifact {label} not retrievable: {path}")
        elif _sha(path) != ref.get("sha256"):
            problems.append(f"artifact {label} hash mismatch on disk")
        else:
            # AUD-F1-20260806-144: never trust a packet-supplied
            # `load_proven` boolean — LOAD the artifact here.
            try:
                from stable_baselines3 import SAC
                SAC.load(str(path), device="cpu")
            except Exception as exc:
                problems.append(
                    f"artifact {label} failed to load: "
                    f"{type(exc).__name__}: {str(exc)[:120]}")
        replica = Path(str(ref.get("replica_path") or ""))
        if not replica.is_file():
            problems.append(
                f"artifact {label} replica missing: {replica}")
        elif _sha(replica) != ref.get("sha256"):
            problems.append(f"artifact {label} replica hash mismatch")
        else:
            authority = ref.get("replica_authority")
            if not authority or authority == LOCAL_HOST:
                problems.append(
                    f"artifact {label} replica authority"
                    f" {authority!r} is not a SECOND host/storage"
                    " authority; a sibling folder is not a replica")
    if arm != "E4":
        terminal = ((record.get("best_checkpoint_vs_terminal") or {})
                    .get("terminal_evaluation") or {})
        terminal_val = (terminal.get("splits_raw") or {}).get(
            "validation") or {}
        if not terminal.get("artifact_sha256"):
            problems.append("terminal evaluation has no artifact hash")
        if not terminal.get("artifact_path"):
            problems.append("terminal evaluation has no retrieval path")
        terminal_ref = (record.get("artifacts") or {}).get("terminal") \
            or {}
        if terminal.get("artifact_sha256") != terminal_ref.get(
                "sha256"):
            problems.append(
                "terminal evaluation hash is not the artifacts.terminal"
                " hash (cross-binding broken)")
        if str(terminal.get("artifact_path")) != str(
                terminal_ref.get("path")):
            problems.append(
                "terminal evaluation path is not the artifacts.terminal"
                " path (cross-binding broken)")
        for key in REQUIRED_FINITE_METRICS:
            if not _finite_number(terminal_val.get(key)):
                problems.append(f"terminal {key} missing/non-finite")
    return problems


def _publish_artifact(path: Path, out_dir: Path, label: str) -> dict:
    """Typed, hashed, retrievable, replicated and load-proven (136)."""
    from stable_baselines3 import SAC
    durable = out_dir / f"{label}.zip"
    if path.resolve() != durable.resolve():
        shutil.copy2(path, durable)
    # AUD-F1-20260806-144: a sibling folder is NOT a replica. The
    # replica must live on a declared second host/storage authority and
    # its hash must be observed there.
    replica_root = Path(os.environ.get(
        "AGENT_MULTI_REPLICA_ROOT",
        str(Path.home() / ".local/share/agent-multi/replica")))
    authority = os.environ.get("AGENT_MULTI_REPLICA_AUTHORITY", "")
    replica_dir = replica_root / out_dir.name
    replica_dir.mkdir(parents=True, exist_ok=True)
    replica = replica_dir / f"{label}.zip"
    shutil.copy2(durable, replica)
    digest = _sha(durable)
    SAC.load(str(durable), device="cpu")          # load proof, primary
    SAC.load(str(replica), device="cpu")          # load proof, replica
    return {
        "path": str(durable.resolve()),
        "replica_path": str(replica.resolve()),
        "replica_authority": authority or LOCAL_HOST,
        "sha256": digest,
        "replica_sha256": _sha(replica),
        "load_proven": True,
    }


def _finite_number(value) -> bool:
    try:
        import math
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _agent_plugin(name: str):
    ep = next(e for e in entry_points().select(group="agent.plugins")
              if e.name == name)
    return ep.load()()


def _git_rev(repo: str) -> str:
    return subprocess.run(
        ["git", "-C", f"/home/harveybc/Documents/GitHub/{repo}",
         "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()


def _base_config(out_dir: Path, arm: str, seed: int, *,
                 epoch_timesteps: int) -> dict:
    config = json.loads(ETH_BASE.read_text())
    # AUD-F1-20260806-142: the dormant year shorthand contradicts the
    # explicit dates (train_years=4 vs ~6.25 actual years) and must not
    # survive in the EXECUTABLE decision config.
    for field in ("train_years", "val_years", "test_years"):
        config.pop(field, None)
    config["split_contract_note"] = (
        "explicit train/validation/test dates govern; year-count"
        " shorthand removed")
    config.update(SPLITS)
    config["input_data_file"] = DATA_FILE
    config["env_mode"] = "training"
    config["eval_seed"] = seed
    config["train_seed"] = seed
    config["ga_seed"] = seed
    config["epoch_timesteps"] = epoch_timesteps
    config["l1_min_checkpoint_timesteps"] = 1
    config["evaluate_test_split"] = False        # disclosed 2025 disabled
    config["selection_metric"] = "lexicographic_weekly_v1"
    config["selection_min_trades"] = 0           # mechanism packet
    config["quiet_mode"] = True
    config["save_model"] = str(out_dir / "model.zip")
    config["return_trace_dir"] = str(out_dir / "return_traces")
    return config


def _raw(summary: dict) -> dict:
    return {key: summary.get(key) for key in RAW_METRICS
            if summary.get(key) is not None}


def _splits_raw(result: dict) -> dict:
    splits = result.get("splits") or {}
    for name, summary in splits.items():
        if name in ALLOWED_SPLITS or name == "train_tail":
            continue
        if name == "test" and isinstance(summary, dict):
            # A pure skip marker proves the split was DISABLED; any
            # metric content is a contract violation.
            assert summary.get("evaluation_skipped") is True, (
                "test split was evaluated — contract violated")
            assert not any(k in summary for k in RAW_METRICS), (
                "test split carries metric data — contract violated")
            continue
        raise AssertionError(f"forbidden split {name!r} in result")
    return {name: _raw(summary) for name, summary in splits.items()
            if isinstance(summary, dict) and name in ALLOWED_SPLITS}


def _hash_traces(trace_dir: Path) -> dict:
    out = {}
    if trace_dir.exists():
        for path in sorted(trace_dir.rglob("*")):
            if path.is_file():
                out[str(path.relative_to(trace_dir))] = _sha(path)
    return out


def _make_anchor(out_dir: Path, seed: int, agent_name: str) -> Path:
    """One anchor initialization shared by every arm of this seed."""
    from stable_baselines3 import SAC
    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)

    anchor = out_dir / f"anchor_seed{seed}.zip"
    if anchor.exists():
        return anchor
    config = _base_config(out_dir, "anchor", seed,
                          epoch_timesteps=EPOCH_TIMESTEPS)
    config["solvency_mode"] = "normal_realistic"
    env_plugin = _load_env_plugin(
        str(config.get("env_plugin", "gym_fx_env")), config)
    env = env_plugin.make_env(config)
    agent = _agent_plugin(agent_name)
    wrap = getattr(agent, "wrap_env", None)
    if callable(wrap):
        env = wrap(env, config)
    model = SAC("MlpPolicy", env, seed=seed, device="cuda",
                policy_kwargs={"net_arch": [256, 256]})
    model.save(str(anchor))
    env.close()
    return anchor


def run_arm(arm: str, seed: int, out_root: Path, *, agent_name: str,
            epoch_timesteps: int, anchor: Path) -> dict:
    from pipeline_plugins.rl_pipeline_with_validation import (
        PipelinePlugin as ValidationPipeline)
    from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
        PipelinePlugin as CurriculumPipeline)

    out_dir = out_root / f"seed{seed}" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    # AUD-F1-20260806-130: idempotence requires a CONTENT-ADDRESSED
    # execution identity, not a filename plus seed. Reuse is legal only
    # for a byte-compatible complete record; any mismatch fails
    # explicitly so stale evidence can never masquerade as current.
    execution_id = _execution_id(arm, seed, anchor,
                                 epoch_timesteps=epoch_timesteps)
    existing = out_dir / "arm_record.json"
    if existing.exists():
        try:
            record = json.loads(existing.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"arm {arm} seed {seed}: existing record is corrupt"
                f" ({exc}); refusing to overwrite or reuse — move it"
                " aside explicitly")
        if record.get("execution_id") == execution_id:
            # AUD-F1-20260806-136: a matching id is NOT completeness.
            # The same validator the aggregator uses must pass, and
            # every referenced artifact must exist with its recorded
            # hash, before any reuse.
            problems = validate_arm_record(record, arm)
            if problems:
                raise RuntimeError(
                    f"arm {arm} seed {seed}: record matches the"
                    f" execution id but is INCOMPLETE: {problems};"
                    " refusing reuse — archive it and rerun")
            print(f"[decision] seed={seed} arm={arm} already complete"
                  f" under execution_id {execution_id[:16]} — reusing",
                  flush=True)
            return record
        raise RuntimeError(
            f"arm {arm} seed {seed}: existing record has execution_id"
            f" {str(record.get('execution_id'))[:16]} != current"
            f" {execution_id[:16]} (contract, budget, anchor, data or"
            " code changed); refusing stale reuse — archive the old"
            " directory before rerunning")
    config = _base_config(out_dir, arm, seed,
                          epoch_timesteps=epoch_timesteps)
    config["warm_start_model"] = str(anchor)
    config["warm_start_model_sha256"] = _sha(anchor)
    agent = _agent_plugin(agent_name)
    # AUD-F1-20260806-137: a long sequential run can cross a code
    # revision; capture it on BOTH sides of the arm and fail if it moved.
    code_revisions_before = {repo: _git_rev(repo) for repo in
                             ("agent-multi", "gym-fx", "doin-plugins")}
    started = datetime.now(timezone.utc)

    if arm == "N14":
        config["max_epochs"] = 14
        config["l1_patience"] = 10_000       # no early stopping
        config["execution_cost_curriculum_epochs"] = 14
        config["solvency_mode"] = "normal_realistic"
        pipeline = ValidationPipeline(config)
        result = pipeline.run_pipeline(
            config=config, env_plugin=None, agent_plugin=agent,
            mode="train")
    elif arm == "EN4_10":
        config["max_epochs"] = 10
        config["l1_patience"] = 10_000
        config["execution_cost_curriculum_epochs"] = 10
        config["easy_max_epochs"] = 4
        config["easy_patience"] = 10_000     # budget-only easy phase
        pipeline = CurriculumPipeline(config)
        result = pipeline.run_pipeline(
            config=config, env_plugin=None, agent_plugin=agent,
            mode="train")
    elif arm == "E4":
        config["easy_max_epochs"] = 4
        config["easy_patience"] = 10_000
        pipeline = CurriculumPipeline(config)
        post_easy = pipeline._train_easy_phase(
            config=config, env_plugin=None, agent_plugin=agent)
        eval_config = dict(config)
        eval_config["solvency_mode"] = "normal_realistic"
        eval_config["load_model"] = post_easy["path"]
        result = pipeline.run_pipeline(
            config=eval_config, env_plugin=None, agent_plugin=agent,
            mode="inference")
        result["post_easy"] = post_easy["meta"]
    else:
        raise SystemExit(f"unknown arm {arm!r}")

    finished = datetime.now(timezone.utc)

    # AUD-F1-20260806-129: both weight sets are REQUIRED evidence for a
    # training arm. The pipeline now emits typed artifact references
    # (best_checkpoint + terminal, each hashed); a missing or unloadable
    # artifact FAILS the arm — no note is ever written that facts do not
    # prove.
    terminal_eval = None
    published: dict = {}
    if arm != "E4":                      # E4 is inference-only diagnostic
        artifacts = result.get("artifacts") or {}
        terminal_ref = artifacts.get("terminal") or {}
        best_ref = artifacts.get("best_checkpoint") or {}
        terminal_path = Path(terminal_ref.get("path") or "")
        best_path = Path(best_ref.get("path") or "")
        if not terminal_path.is_file() or not best_path.is_file():
            raise RuntimeError(
                f"arm {arm} seed {seed}: pipeline did not preserve both"
                f" artifacts (best={best_path}, terminal="
                f"{terminal_path}); the arm FAILS")
        for label, path, ref in (("best", best_path, best_ref),
                                 ("terminal", terminal_path,
                                  terminal_ref)):
            actual = _sha(path)
            if actual != ref.get("sha256"):
                raise RuntimeError(
                    f"arm {arm} seed {seed}: {label} artifact hash"
                    f" {actual} != recorded {ref.get('sha256')}")
        eval_config = dict(config)
        eval_config["load_model"] = str(terminal_path)
        eval_config["solvency_mode"] = "normal_realistic"
        eval_config["return_trace_dir"] = str(
            out_dir / "return_traces_terminal")
        terminal_result = ValidationPipeline(eval_config).run_pipeline(
            config=eval_config, env_plugin=None,
            agent_plugin=_agent_plugin(agent_name), mode="inference")
        terminal_splits = _splits_raw(terminal_result)
        validation = terminal_splits.get("validation") or {}
        if not validation or any(
            not _finite_number(validation.get(key))
            for key in ("total_return", "mean_weekly_return")
        ):
            raise RuntimeError(
                f"arm {arm} seed {seed}: terminal evaluation produced"
                " no finite validation metrics; the arm FAILS")
        published = {
            "best_checkpoint": _publish_artifact(
                best_path, out_dir, "best_checkpoint"),
            "terminal": _publish_artifact(
                terminal_path, out_dir, "terminal"),
        }
        published["terminal"]["num_timesteps"] = terminal_ref.get(
            "num_timesteps")
        terminal_eval = {
            "artifact_sha256": published["terminal"]["sha256"],
            "artifact_path": published["terminal"]["path"],
            "artifact_replica_path": published["terminal"][
                "replica_path"],
            "num_timesteps": terminal_ref.get("num_timesteps"),
            "splits_raw": terminal_splits,
            "selection_contract": (
                (terminal_result.get("splits") or {}).get(
                    "validation", {}).get("selection_contract")),
        }
    resolved_path = out_dir / "resolved_config.json"
    resolved_text = json.dumps(config, indent=1, sort_keys=True,
                               default=str)
    resolved_path.write_text(resolved_text, encoding="utf-8")
    weights = {}
    if arm == "E4":
        post_easy = out_dir / "model.post_easy.zip"
        if not post_easy.is_file():
            raise RuntimeError(
                f"arm E4 seed {seed}: post_easy artifact missing")
        weights["post_easy"] = _publish_artifact(
            post_easy, out_dir, "post_easy")
    else:
        weights = published
    margin_telemetry = {}
    for split, summary in (result.get("splits") or {}).items():
        if not isinstance(summary, dict) or split not in ALLOWED_SPLITS:
            continue
        # AUD-F1-20260806-125: a missing counter is UNAVAILABLE, never
        # "zero margin events". The distinction decides whether an
        # observed EN/N difference can be attributed to solvency
        # relaxation at all.
        margin_telemetry[split] = {
            key: ("unavailable" if summary.get(key) is None
                  else summary.get(key))
            for key in ("would_margin_call_count", "termination_cause",
                        "recapitalization_count",
                        "recapitalization_debt", "solvency_mode")
        }

    code_revisions_after = {repo: _git_rev(repo) for repo in
                            ("agent-multi", "gym-fx", "doin-plugins")}
    record = {
        "schema": ARM_RECORD_SCHEMA,
        "execution_id": execution_id,
        "code_revisions_before": code_revisions_before,
        "code_revisions_after": code_revisions_after,
        "runner_version": RUNNER_VERSION,
        "arm": arm,
        "seed": seed,
        "best_checkpoint_vs_terminal": {
            "terminal_evaluation": terminal_eval,
            "note": (
                "both weight sets evaluated raw under identical"
                " realistic-normal validation; neither is a selection"
                " change" if terminal_eval else
                "inference-only diagnostic arm: no terminal training"
                " weights exist by design"),
        },
        "margin_telemetry": margin_telemetry,
        "wall_time_seconds": (finished - started).total_seconds(),
        "resolved_config_sha256": hashlib.sha256(
            resolved_text.encode()).hexdigest(),
        "anchor_sha256": config["warm_start_model_sha256"],
        "artifacts": weights,
        "splits_raw": _splits_raw(result),
        "selection_contract": (result.get("splits") or {}).get(
            "validation", {}).get("selection_contract"),
        "curriculum": result.get("curriculum"),
        "epoch_history": result.get("history") or result.get(
            "epoch_history"),
        "return_trace_sha256": _hash_traces(
            Path(config["return_trace_dir"])),
    }
    problems = validate_arm_record(record, arm)
    if problems:
        raise RuntimeError(
            f"arm {arm} seed {seed}: produced an INCOMPLETE record:"
            f" {problems}")
    (out_dir / "arm_record.json").write_text(
        json.dumps(record, indent=1, sort_keys=True, default=str),
        encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epoch-timesteps", type=int,
                        default=EPOCH_TIMESTEPS)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--agent",
                        default="project3_sac_actor_critic_agent")
    args = parser.parse_args()

    assert hashlib.sha256(
        Path(DATA_FILE).read_bytes()).hexdigest() == DATA_SHA256, (
        "dataset sha mismatch — frozen contract violated")
    base_sha = hashlib.sha256(ETH_BASE.read_bytes()).hexdigest()
    assert base_sha == ETH_BASE_SHA256, (
        f"base contract sha {base_sha} != pinned {ETH_BASE_SHA256}")

    out_root = args.output_root
    seed_dir = out_root / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    anchor = _make_anchor(seed_dir, args.seed, args.agent)
    print(json.dumps({"anchor": str(anchor), "sha256": _sha(anchor)}),
          flush=True)

    packet = {
        "schema": "agent_multi.eth_curriculum_decision.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "epoch_timesteps": args.epoch_timesteps,
        "data_sha256": DATA_SHA256,
        "base_contract_sha256": ETH_BASE_SHA256,
        "lineage": {repo: _git_rev(repo) for repo in
                    ("agent-multi", "gym-fx", "doin-node",
                     "doin-plugins", "trading-contracts")},
        "packet_identity_note": (
            "each arm record carries code_revisions_before/after; the"
            " aggregator binds this packet lineage to every arm"),
        "note": ("LOCAL-ONLY paired decision packet; order key is"
                 " transport evidence only; 2025 disclosed period"
                 " disabled and absent"),
        "arms": {},
    }
    for arm in args.arms.split(","):
        arm = arm.strip()
        print(f"[decision] seed={args.seed} arm={arm} starting",
              flush=True)
        packet["arms"][arm] = run_arm(
            arm, args.seed, out_root, agent_name=args.agent,
            epoch_timesteps=args.epoch_timesteps, anchor=anchor)
        (seed_dir / "seed_packet.json").write_text(
            json.dumps(packet, indent=1, sort_keys=True, default=str),
            encoding="utf-8")
        print(f"[decision] seed={args.seed} arm={arm} done", flush=True)
    print(json.dumps({"seed_packet": str(seed_dir / "seed_packet.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
