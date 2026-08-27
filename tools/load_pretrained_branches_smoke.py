#!/usr/bin/env python3
"""Transfer-loader CPU smoke v3 (DATA-SOTA-357..360 corrected).

Execution and presentation are fully separated:

* ``--render <ledger-key>`` presents ONLY the evidence named by a
  COMPLETED ledger record, after the custody layer verifies evidence
  digest, schema, run id, dispatch id and bound identities — model-free
  and freely repeatable (DATA-SOTA-360);
* execution mode reads the effective config bytes EXACTLY ONCE into an
  immutable snapshot (DATA-SOTA-359): file digest, parsed config,
  canonical architecture materialization and env config all derive
  from that single read, the snapshot digest is bound into the dispatch
  key and ledger identity, and no post-reservation config-path read
  exists. The durable enforced state machine guards single use;
  completion happens only after the evidence file is durably written
  and digest-bound — a failed completion write leaves the run SPENT.

The architecture comes ONLY from the canonical materializer inside the
snapshot — the same route SAC construction uses. The sealed contract
must match it exactly; all load counts are DERIVED with an asserted
conservation invariant.

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
    DispatchLedger, ExecutionCustodyError, dispatch_key,
    durable_write_bytes)
from agent_plugins.grouped_architecture import (  # noqa: E402
    ArchitectureError, construct_extractor, snapshot_effective_config)
from agent_plugins.pretrained_branch_loader import (  # noqa: E402
    TRANSFER_STATUS, TransferLoadError, check_finite_forward,
    load_family_encoders, verify_architecture_matches_contract,
    verify_source)
from tools.pretrain_branches import (  # noqa: E402
    logical_interpreter, repo_relative, resolve_data_path)

DISPATCH_ID = ("MUSASHI_TO_GENERAL_SATOSHI_TRANSFER_LOADER_CPU_SMOKE_"
               "DISPATCH (replacement smoke per 359-360 order)")
EVIDENCE_SCHEMA = "agent_multi.transfer_loader_smoke.v3"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", default=None, metavar="LEDGER_KEY",
                        help="verified model-free presentation of the "
                             "completed evidence for this ledger key")
    parser.add_argument("--pretrain-dir")
    parser.add_argument("--arch-config",
                        help="EFFECTIVE grouped config; snapshotted "
                             "in ONE read (DATA-SOTA-359)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ledger-root", default=None,
                        help="override ledger root (tests only)")
    args = parser.parse_args()
    ledger = DispatchLedger(Path(args.ledger_root)
                            if args.ledger_root else None)
    if args.render is not None:
        packet = ledger.verified_render(args.render)
        summary = {key: packet.get(key) for key in (
            "schema", "status", "run_id",
            "tensors_loaded_total_derived", "accounting",
            "forward_output_shape", "forward_output_finite",
            "wall_seconds", "peak_host_memory_mb", "family_digest",
            "architecture_digest", "config_snapshot_digest")}
        print(json.dumps(summary, indent=1))
        return 0
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

    # DATA-SOTA-359: ONE read; everything downstream uses the snapshot
    snapshot = snapshot_effective_config(Path(args.arch_config))
    materialized = snapshot["materialized"]
    verify_architecture_matches_contract(materialized, contract)

    seal = json.loads((pretrain_dir / "generation.json").read_text())
    key = dispatch_key(
        dispatch_id=DISPATCH_ID,
        generation_digest=seal["manifest_sha256"],
        architecture_digest=materialized["architecture_digest"],
        config_snapshot_digest=snapshot["config_sha256"],
        data_digest=source["manifest"]["identity"]["data_sha256"],
        code_identity=source["code_identity_report"])
    run_id = key[:16]
    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO / "docs/audits/evidence")
    out_path = out_dir / f"TRANSFER_LOADER_SMOKE_{run_id}.json"
    ledger.reserve(key, identity={
        "dispatch_id": DISPATCH_ID, "run_id": run_id,
        "architecture_digest": materialized["architecture_digest"],
        "config_snapshot_digest": snapshot["config_sha256"],
        "generation_digest": seal["manifest_sha256"]},
        output_path=out_path)
    try:
        ledger.transition(key, "running")

        cfg = dict(snapshot["env_config"])  # snapshot-derived, no read
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

        obs, _ = env.reset(seed=7)
        batch = {k: torch.tensor(
            np.repeat(np.asarray(v, dtype=np.float32)[None, ...], 3,
                      axis=0)) for k, v in obs.items()}
        extractor.eval()
        ledger.mark_forward_started(key)  # durable BEFORE the forward
        with torch.no_grad():
            out = check_finite_forward(extractor(batch))
            out_repeat = check_finite_forward(extractor(batch))
        deterministic_repeat = bool(torch.equal(out, out_repeat))
        wall = time.perf_counter() - t0
        peak_host_mb = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024.0

        # operator-visibility only: NEVER consumed by execution
        source_unchanged = (snapshot_effective_config(
            Path(args.arch_config))["config_sha256"]
            == snapshot["config_sha256"])

        packet = {
            "schema": EVIDENCE_SCHEMA,
            "dispatch": DISPATCH_ID,
            "run_id": run_id,
            "status": TRANSFER_STATUS,
            "pretrain_dir": f"external:{pretrain_dir.name}",
            "arch_config": repo_relative(Path(args.arch_config)),
            "config_snapshot_digest": snapshot["config_sha256"],
            "source_path_unchanged_at_completion": source_unchanged,
            "architecture_digest": materialized["architecture_digest"],
            "expected_output_dim": materialized["expected_output_dim"],
            "data_source": data_logical,
            "interpreter": logical_interpreter(),
            "device": "cpu (CUDA_VISIBLE_DEVICES empty)",
            "code_identity": source["code_identity_report"],
            "families": result["families"],
            "accounting": result["accounting"],
            "tensors_loaded_total_derived":
                result["accounting"]["loaded_tensors"],
            "state_branch_and_fusion": ("from the SNAPSHOT-materialized "
                                        "effective config; random-init, "
                                        "DECLARED untransferred"),
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
        durable_write_bytes(out_path,
                            json.dumps(packet, indent=1).encode(),
                            exclusive=True)
        try:
            ledger.complete(key, out_path,
                            expected_schema=EVIDENCE_SCHEMA,
                            run_id=run_id, dispatch_id=DISPATCH_ID)
        except Exception:
            # DATA-SOTA-360: a failed completion write leaves the run
            # SPENT — never rerunnable, never acknowledged
            ledger.transition(key, "spent")
            raise
        print(json.dumps(packet, indent=1))
        return 0
    except (TransferLoadError, ArchitectureError):
        record = ledger.read(key)
        if record and record.get("state") == "running":
            ledger.transition(key,
                              "spent" if record.get("forward_started")
                              else "failed_before_forward")
        raise
    except KeyboardInterrupt:
        record = ledger.read(key)
        if record and record.get("state") == "running":
            ledger.transition(key, "interrupted")
        raise
    except Exception:
        record = ledger.read(key)
        if record and record.get("state") == "running":
            ledger.transition(key, "interrupted")
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TransferLoadError, ArchitectureError,
            ExecutionCustodyError) as exc:
        print(f"TRANSFER REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
