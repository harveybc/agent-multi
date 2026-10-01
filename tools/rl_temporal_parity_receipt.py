#!/usr/bin/env python
"""Torch-side parity receipt for a modular_encoder_export.v1 npz (torch env only).

Imports the export into a torch ModularTemporalEncoder built from the export's own config,
checks the identity block is present and the config unknown-key rules pass, runs the reference
input, and writes a receipt with fidelity and both environments' versions.
    python tools/rl_temporal_parity_receipt.py --export X.npz --out receipt.json [--scaler scaler.json]
"""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from rl_temporal.keras_import import fidelity_report, import_export_into_encoder, load_export
from rl_temporal.modular_torch import ModularTemporalEncoder, modular_config_sha256, parameter_accounting

ap = argparse.ArgumentParser(); ap.add_argument("--export", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--scaler", help="normalization artifact to bind to the export identity (sha256 recorded)")
a = ap.parse_args()
exp = load_export(a.export); meta = exp["meta"]
enc = ModularTemporalEncoder(meta["modular_config"])
rep = import_export_into_encoder(enc, exp)
fid = fidelity_report(enc, exp)
core = enc.config["core"]["params"]
doc = {"schema": "rl_temporal.parity_receipt.v1", "export_sha256": hashlib.sha256(Path(a.export).read_bytes()).hexdigest(),
       "identity_block_present": bool(meta.get("identity")), "identity": meta.get("identity"),
       "architecture": {"d_model": core["d_model"], "heads": core["heads"], "blocks": core["blocks"], "ff_dim": core["ff_dim"],
                        "features": len(enc.config["feature_names"]), "branches": len(enc.config["branches"]),
                        "window": enc.config["window"], "time_factors": core["time_factors"]},
       "imported_tensors": rep["imported_tensors"], "unmatched_keras_weights": rep["unmatched_keras_weights"],
       "fidelity": fid, "tolerance": {"fused_abs": 1e-5, "latent_abs": 1e-4},
       "passes": fid["fused_max_abs_diff"] < 1e-5 and fid["latent_max_abs_diff"] < 1e-4 and not rep["unmatched_keras_weights"],
       "modular_config_sha256_torch": modular_config_sha256(meta["modular_config"]),
       "parameters_torch": parameter_accounting(enc)["total"],
       "versions": {"export_env": meta["versions"], "import_env": {"torch": torch.__version__}},
       "scaler_binding": ({"file": a.scaler, "sha256": hashlib.sha256(Path(a.scaler).read_bytes()).hexdigest()} if a.scaler else "NOT_BOUND")}
Path(a.out).write_text(json.dumps(doc, indent=2, default=str) + "\n")
print(json.dumps({k: doc[k] for k in ("passes", "fidelity", "architecture", "imported_tensors", "identity_block_present")}))
