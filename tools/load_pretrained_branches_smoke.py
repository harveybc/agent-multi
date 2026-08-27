#!/usr/bin/env python3
"""Transfer-loader CPU smoke v2 (DATA-SOTA-357/358 corrected).

Execution is separated from presentation:

* ``--render <evidence.json>`` reads completed evidence and re-prints
  the summary — freely rerunnable, NEVER constructs any model;
* execution mode is guarded by the durable single-use dispatch ledger
  (`agent_plugins.dispatch_custody`): the run is RESERVED atomically
  before the model exists, evidence goes to a UNIQUE non-clobbering
  path derived from the run identity, and a completed or uncertain
  prior attempt refuses another execution.

The effective grouped architecture comes ONLY from the canonical
materializer (`agent_plugins.grouped_architecture`) over the supplied
effective config — the same route SAC construction uses. No
state-branch/fusion/branch dictionary is authored here
(DATA-SOTA-357), the sealed contract must match the materialized
architecture exactly, and all load counts are DERIVED from the loader
accounting with an asserted conservation invariant.

The result remains MECHANICS_ONLY_NOT_ECONOMICALLY_ELIGIBLE: no GPU,
no economics, no promotion, no collector activation.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.dispatch_custody import (  # noqa: E402
    DispatchLedger, ExecutionCustodyError, dispatch_key)
from agent_plugins.grouped_architecture import (  # noqa: E402
    ArchitectureError, assert_same_materialization, construct_extractor,
    materialize_from_file)
from agent_plugins.pretrained_branch_loader import (  # noqa: E402
    TRANSFER_STATUS, TransferLoadError, check_finite_forward,
    load_family_encoders, verify_architecture_matches_contract,
    verify_source)
from tools.pretrain_branches import (  # noqa: E402
    logical_interpreter, repo_relative, resolve_data_path)

DISPATCH_ID = ("MUSASHI_TO_GENERAL_SATOSHI_TRANSFER_LOADER_CPU_SMOKE_"
               "DISPATCH (replacement smoke per 357-358 order)")


def render(evidence_path: Path) -> int:
    """Presentation only: NEVER constructs a model."""
    packet = json.loads(evidence_path.read_text())
    summary = {key: packet.get(key) for key in (
        "schema", "status", "run_id", "tensors_loaded_total_derived",
        "accounting", "forward_output_shape", "forward_output_finite",
        "wall_seconds", "peak_host_memory_mb", "family_digest",
        "architecture_digest")}
    print(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path, default=None,
                        help="read-only presentation of completed "
                             "evidence; no model execution")
    parser.add_argument("--pretrain-dir")
    parser.add_argument("--arch-config",
                        help="EFFECTIVE grouped config; its "
                             "feature_extractor_config is the only "
                             "architecture authority")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ledger-root", default=None,
                        help="override ledger root (tests only)")
    args = parser.parse_args()
    if args.render is not None:
        return render(args.render)
    if not args.pretrain_dir or not args.arch_config:
        raise SystemExit("REFUSED: --pretrain-dir and --arch-config "
                         "are required for execution mode")

    import numpy as np
    import torch

    from pipeline_plugins.rl_pipeline_with_validation import (
        _load_env_plugin)

    t0 = time.perf_counter()
    data_path, data_logical = resolve_data_path()
    pretrain_dir = Path(args.pretrain_dir)
    source = verify_source(pretrain_dir, REPO, data_path)
    contract = source["contract"]
    manifest = source["manifest"]

    # DATA-SOTA-357: canonical materialization; verify twice around the
    # reservation so config mutation after verification refuses
    arch_config_path = Path(args.arch_config)
    materialized = materialize_from_file(arch_config_path)
    verify_architecture_matches_contract(materialized, contract)

    seal = json.loads((pretrain_dir / "generation.json").read_text())
    key = dispatch_key(
        dispatch_id=DISPATCH_ID,
        generation_digest=seal["manifest_sha256"],
        architecture_digest=materialized["architecture_digest"],
        data_digest=source["manifest"]["identity"]["data_sha256"],
        code_identity=source["code_identity_report"])
    run_id = key[:16]
    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO / "docs/audits/evidence")
    out_path = out_dir / f"TRANSFER_LOADER_SMOKE_{run_id}.json"
    ledger = DispatchLedger(Path(args.ledger_root)
                            if args.ledger_root else None)
    forward_started = False
    ledger.reserve(key, identity={
        "dispatch_id": DISPATCH_ID, "run_id": run_id,
        "architecture_digest": materialized["architecture_digest"],
        "generation_digest": seal["manifest_sha256"]},
        output_path=out_path)
    try:
        ledger.transition(key, "running")
        # re-materialize: a config mutated after verification refuses
        assert_same_materialization(materialized,
                                    materialize_from_file(
                                        arch_config_path))

        cfg = json.loads(arch_config_path.read_text())
        sliced = (Path(os.environ.get("TMPDIR", "/tmp"))
                  / f"loader_eth_{run_id}.csv")
        with data_path.open() as src, sliced.open("w") as dst:
            for i, line in enumerate(src):
                if i > 700:
                    break
                dst.write(line)
        cfg["input_data_file"] = str(sliced)
        cfg["max_steps"] = 460
        env = _load_env_plugin("gym_fx_env", cfg).make_env(cfg)

        torch.manual_seed(0)
        extractor = construct_extractor(materialized,
                                        env.observation_space)
        result = load_family_encoders(pretrain_dir, manifest, contract,
                                      extractor)
        families = result["families"]
        accounting = result["accounting"]

        obs, _ = env.reset(seed=7)
        batch = {k: torch.tensor(
            np.repeat(np.asarray(v, dtype=np.float32)[None, ...], 3,
                      axis=0)) for k, v in obs.items()}
        extractor.eval()
        forward_started = True  # from here a failure is SPENT, not retryable
        with torch.no_grad():
            out = check_finite_forward(extractor(batch))
            out_repeat = check_finite_forward(extractor(batch))
        deterministic_repeat = bool(torch.equal(out, out_repeat))
        wall = time.perf_counter() - t0
        peak_host_mb = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024.0

        packet = {
            "schema": "agent_multi.transfer_loader_smoke.v2",
            "dispatch": DISPATCH_ID,
            "run_id": run_id,
            "status": TRANSFER_STATUS,
            "pretrain_dir": f"external:{pretrain_dir.name}",
            "arch_config": repo_relative(arch_config_path),
            "arch_config_sha256": materialized["config_sha256"],
            "architecture_digest": materialized["architecture_digest"],
            "expected_output_dim": materialized["expected_output_dim"],
            "data_source": data_logical,
            "interpreter": logical_interpreter(),
            "device": "cpu (CUDA_VISIBLE_DEVICES empty)",
            "code_identity": source["code_identity_report"],
            "families": families,
            "accounting": accounting,
            "tensors_loaded_total_derived":
                accounting["loaded_tensors"],
            "state_branch_and_fusion": ("from the MATERIALIZED "
                                        "effective config; random-"
                                        "init, DECLARED untransferred"),
            "ordered_families": extractor.ordered_families,
            "family_digest": extractor.fusion.family_digest,
            "observation_shapes": {k: list(v.shape)
                                   for k, v in batch.items()},
            "forward_output_shape": list(out.shape),
            "forward_output_finite": True,
            "deterministic_repeat_forward_equal": deterministic_repeat,
            "wall_seconds": round(wall, 3),
            "peak_host_memory_mb": round(peak_host_mb, 1),
            "gpu": "NONE", "economics": "NONE", "promotion": "NONE",
            "collector_activation": "NONE",
        }
        payload = json.dumps(packet, indent=1).encode()
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY  # no-clobber
        fd = os.open(out_path, flags, 0o644)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        ledger.transition(key, "completed",
                          {"completed_at": "sealed-with-evidence",
                           "evidence": f"external:{out_path.name}"})
        print(json.dumps(packet, indent=1))
        return 0
    except (TransferLoadError, ArchitectureError):
        ledger.transition(key, "interrupted" if forward_started
                          else "failed_before_forward")
        raise
    except KeyboardInterrupt:
        ledger.transition(key, "interrupted")
        raise
    except Exception:
        ledger.transition(key, "interrupted")
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TransferLoadError, ArchitectureError,
            ExecutionCustodyError) as exc:
        print(f"TRANSFER REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
