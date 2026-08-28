#!/usr/bin/env python3
"""Paired SAC comparison dispatch driver (P4, final probe order
2026-08-28). EVERY GPU COMMAND IS NOT_LAUNCHED: this driver exists so
the identities and per-cell configs are executable and testable on CPU
BEFORE Musashi dispatches; actually starting SAC training requires
`--gpu-authorized-by-musashi <dispatch-doc-path>` naming his written
dispatch, and even then refuses without CUDA visibility explicitly
granted by the operator.

Modes:
* ``--dry-run`` (CPU, default-safe): verifies the design digest, the
  candidate generation seal + quarantine status + per-family encoder
  digests, the strong-config snapshot identity, and materializes the
  per-cell SAC genesis config for (--seed, --arm) WITHOUT constructing
  any model or env; prints the cell identity packet.
* execution mode: REFUSED unless the authorization flag names an
  existing dispatch document AND CUDA is visible; the SAC training
  loop itself is deliberately NOT implemented until that dispatch
  exists — the refusal path is the implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import (  # noqa: E402
    load_generation, sha256_file, sha256_obj)
from agent_plugins.grouped_architecture import (  # noqa: E402
    snapshot_effective_config)

DESIGN_PATH = (REPO / "docs/audits/evidence/"
               "PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json")
REGISTER_PATH = (REPO / "docs/audits/evidence/"
                 "GENERATION_QUARANTINE_REGISTER.json")


class DispatchRefused(RuntimeError):
    pass


def verify_cell(design: dict, pretrain_dir: Path, seed: int,
                arm: str) -> dict:
    if arm not in design["arms"]:
        raise DispatchRefused(f"unknown arm {arm!r}")
    if seed not in design["seeds"]:
        raise DispatchRefused(f"seed {seed} not in the design")
    _ckpt, manifest, generation = load_generation(pretrain_dir)
    seal = json.loads((pretrain_dir / "generation.json").read_text())
    if REGISTER_PATH.exists():
        register = json.loads(REGISTER_PATH.read_text())
        if (register.get("entries") or {}).get(seal["manifest_sha256"]):
            raise DispatchRefused(
                "candidate generation is QUARANTINED — dispatch "
                "refused")
    bound = design["shared_bindings"]["pretrain_generation"]
    if seal["manifest_sha256"] != bound["seal_manifest_sha256"]:
        raise DispatchRefused(
            "generation seal differs from the design binding — the "
            "design must be regenerated from the real seal")
    for family, digest in bound["per_family_encoder_digests"].items():
        actual = sha256_file(
            pretrain_dir / f"branch_{family}_encoder.pt")
        if actual != digest:
            raise DispatchRefused(
                f"encoder digest drift for {family}")
    snapshot = snapshot_effective_config(
        REPO / design["shared_bindings"]["strong_config"])
    if snapshot["materialized"]["architecture_digest"] != \
            design["shared_bindings"]["architecture_digest"]:
        raise DispatchRefused("strong-config architecture drift")
    trial = next(t for t in design["trial_ledger"]
                 if t["genesis"]["seed"] == seed
                 and t["genesis"]["arm"] == arm)
    cell = {
        "schema": "agent_multi.paired_sac_cell_genesis.v1",
        "trial_id": trial["trial_id"],
        "genesis_sha256": trial["genesis_sha256"],
        "arm": arm, "seed": seed,
        "mechanism": design["arms"][arm],
        "strong_config": design["shared_bindings"]["strong_config"],
        "architecture_digest":
            design["shared_bindings"]["architecture_digest"],
        "pretrain_generation_seal": seal["manifest_sha256"],
        "sac": design["shared_bindings"]["sac"],
        "envelope": design["shared_bindings"]["execution_envelope"],
        "data_roles": design["shared_bindings"]["data_roles"],
        "evaluation": design["shared_bindings"]["evaluation"],
        "predeclared_refusals": design["predeclared"],
        "status": "MATERIALIZED_NOT_LAUNCHED",
    }
    cell["cell_sha256"] = sha256_obj(cell)
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--gpu-authorized-by-musashi", default=None,
                        metavar="DISPATCH_DOC")
    args = parser.parse_args()
    design = json.loads(DESIGN_PATH.read_text())
    cell = verify_cell(design, Path(args.pretrain_dir), args.seed,
                       args.arm)
    print(json.dumps(cell, indent=1))
    if args.gpu_authorized_by_musashi is None:
        print("NOT_LAUNCHED: dry-run only — GPU execution requires "
              "--gpu-authorized-by-musashi <dispatch-doc>",
              file=sys.stderr)
        return 0
    dispatch_doc = Path(args.gpu_authorized_by_musashi)
    if not dispatch_doc.is_file():
        raise DispatchRefused(
            "authorization document does not exist — refused")
    import torch
    if not torch.cuda.is_available():
        raise DispatchRefused(
            "no CUDA visibility — the operator has not granted a GPU")
    raise DispatchRefused(
        "SAC training loop deliberately NOT implemented until "
        "Musashi's dispatch document is audited into this driver — "
        "NOT_LAUNCHED by construction")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DispatchRefused as exc:
        print(f"DISPATCH REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
