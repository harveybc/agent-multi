"""Keras engine -> torch extractor weight import, through a framework-neutral export.

The engine bundles and donors at pin 3ecdb256 are saved under the campaign
environment's Keras (3.13.x) and ``load_bundle`` refuses a major.minor
mismatch by design. The RL environment runs Keras 3.15 / torch. So the Keras
side happens where the bundle is loadable (``export_encoder_npz`` run with
the pinned interpreter) and produces ``modular_encoder_export.v1``: an .npz
with every weight by Keras path, the normalized config, a fixed reference
input and the reference fused/latent outputs, plus versions and shas. This
module consumes that artifact and reports per-stage fidelity. A failed
fidelity blocks R1/R2 only; R0 never needs an import.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

EXPORT_SCHEMA = "modular_encoder_export.v1"


# ----------------------------------------------------------------------------
# export side (needs the Keras engine importable; run with the pinned interpreter)
# ----------------------------------------------------------------------------

def export_encoder_npz(bundle, path, *, seed: int = 0, batch: int = 3) -> Dict[str, Any]:
    """Write the framework-neutral export of a ``ModularBundle``."""
    import keras
    import tensorflow as tf

    from .modular_torch import normalize_modular_config

    c = normalize_modular_config(bundle.config)
    arrays: Dict[str, np.ndarray] = {}
    for name, model in bundle.branch_models.items():
        for w in model.weights:
            arrays[f"branch:{name}:{w.path}"] = np.asarray(w.numpy(), dtype=np.float32)
    for w in bundle.core_model.weights:
        arrays[f"core:{w.path}"] = np.asarray(w.numpy(), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=(int(batch), c["window"], len(c["feature_names"]))).astype(np.float32)
    arrays["reference:input"] = x
    arrays["reference:fused"] = np.asarray(bundle.fusion_model(x), dtype=np.float32)
    arrays["reference:latent"] = np.asarray(bundle.encoder_model(x), dtype=np.float32)
    meta = {"schema": EXPORT_SCHEMA, "modular_config": c,
            "versions": {"keras": keras.__version__, "tensorflow": tf.__version__},
            "weights_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, __meta__=np.array(json.dumps(meta, sort_keys=True, default=str)), **arrays)
    meta["path"] = str(path)
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return meta


def load_export(path) -> Dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        if meta.get("schema") != EXPORT_SCHEMA:
            raise ValueError(f"unsupported export schema {meta.get('schema')!r}")
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
    for k, v in arrays.items():
        if hashlib.sha256(v.tobytes()).hexdigest() != meta["weights_sha256"][k]:
            raise ValueError(f"export array {k} does not match its recorded sha256")
    return {"meta": meta, "arrays": arrays}


# ----------------------------------------------------------------------------
# import side (torch only)
# ----------------------------------------------------------------------------

def _assign(param: torch.nn.Parameter, value: np.ndarray, used: List[str], key: str) -> None:
    v = torch.as_tensor(np.asarray(value, dtype=np.float32))
    if tuple(v.shape) != tuple(param.shape):
        raise ValueError(f"{key}: keras shape {tuple(v.shape)} != torch shape {tuple(param.shape)}")
    with torch.no_grad():
        param.copy_(v)
    used.append(key)


def _conv_kernel(value: np.ndarray) -> np.ndarray:
    # keras (k, in, out) -> torch (out, in, k)
    return np.transpose(value, (2, 1, 0))


def _find(arrays: Dict[str, np.ndarray], prefix: str, suffix: str) -> str:
    matches = [k for k in arrays if k.startswith(prefix) and k.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one export weight {prefix}*{suffix}; found {matches}")
    return matches[0]


def import_export_into_encoder(encoder, export: Dict[str, Any]) -> Dict[str, Any]:
    from .modular_torch import modular_config_sha256

    arrays, meta = export["arrays"], export["meta"]
    if modular_config_sha256(meta["modular_config"]) != modular_config_sha256(encoder.config):
        raise ValueError("export modular config does not match the torch encoder config")
    used: List[str] = []
    for name, branch in encoder.branches.items():
        pre = f"branch:{name}:"
        _assign(branch.conv.conv.weight, _conv_kernel(arrays[_find(arrays, pre, "/kernel")]), used, _find(arrays, pre, "/kernel"))
        _assign(branch.conv.conv.bias, arrays[_find(arrays, pre, "/bias")], used, _find(arrays, pre, "/bias"))
    core = encoder.core
    pre = "core:"
    _assign(core.model_projection.weight, arrays[_find(arrays, pre, "model_projection/kernel")].T, used,
            _find(arrays, pre, "model_projection/kernel"))
    _assign(core.model_projection.bias, arrays[_find(arrays, pre, "model_projection/bias")], used,
            _find(arrays, pre, "model_projection/bias"))
    for i, block in enumerate(core.blocks):
        att = block.attention
        for torch_param, kpath in ((att.query_kernel, f"attention_{i}/query/kernel"),
                                   (att.query_bias, f"attention_{i}/query/bias"),
                                   (att.key_kernel, f"attention_{i}/key/kernel"),
                                   (att.key_bias, f"attention_{i}/key/bias"),
                                   (att.value_kernel, f"attention_{i}/value/kernel"),
                                   (att.value_bias, f"attention_{i}/value/bias"),
                                   (att.output_kernel, f"attention_{i}/attention_output/kernel"),
                                   (att.output_bias, f"attention_{i}/attention_output/bias")):
            key = _find(arrays, pre, kpath)
            _assign(torch_param, arrays[key], used, key)
        for ln, kname in ((block.attention_norm, f"attention_norm_{i}"), (block.ffn_norm, f"ffn_norm_{i}")):
            _assign(ln.weight, arrays[_find(arrays, pre, f"{kname}/gamma")], used, _find(arrays, pre, f"{kname}/gamma"))
            _assign(ln.bias, arrays[_find(arrays, pre, f"{kname}/beta")], used, _find(arrays, pre, f"{kname}/beta"))
        for lin, kname in ((block.ffn_expand, f"ffn_expand_{i}"), (block.ffn_project, f"ffn_project_{i}")):
            _assign(lin.weight, arrays[_find(arrays, pre, f"{kname}/kernel")].T, used, _find(arrays, pre, f"{kname}/kernel"))
            _assign(lin.bias, arrays[_find(arrays, pre, f"{kname}/bias")], used, _find(arrays, pre, f"{kname}/bias"))
    for i, stage in enumerate(core.stages):
        s = f"stage_{i}_"
        for conv, kname in ((stage.skip_downsample.conv, s + "skip_downsample"),
                            (stage.block_projection.conv, s + "block_projection"),
                            (stage.temporal_conv.conv, s + "temporal_conv"),
                            (stage.temporal_projection.conv, s + "temporal_projection")):
            kk, kb = _find(arrays, pre, f"{kname}/kernel"), _find(arrays, pre, f"{kname}/bias")
            _assign(conv.weight, _conv_kernel(arrays[kk]), used, kk)
            _assign(conv.bias, arrays[kb], used, kb)
        for ln, kname in ((stage.downsample_norm, s + "downsample_norm"), (stage.temporal_norm, s + "temporal_norm")):
            _assign(ln.weight, arrays[_find(arrays, pre, f"{kname}/gamma")], used, _find(arrays, pre, f"{kname}/gamma"))
            _assign(ln.bias, arrays[_find(arrays, pre, f"{kname}/beta")], used, _find(arrays, pre, f"{kname}/beta"))
    unmatched = sorted(k for k in arrays if not k.startswith("reference:") and k not in used)
    return {"imported_tensors": len(used), "unmatched_keras_weights": unmatched, "versions_export": meta["versions"]}


def fidelity_report(encoder, export: Dict[str, Any]) -> Dict[str, float]:
    arrays = export["arrays"]
    x = torch.as_tensor(arrays["reference:input"])
    encoder.eval()
    with torch.no_grad():
        stages = encoder.stage_outputs(x)
    fused = float(np.max(np.abs(stages["fused"].numpy() - arrays["reference:fused"])))
    latent = float(np.max(np.abs(stages["latent"].numpy() - arrays["reference:latent"])))
    return {"fused_max_abs_diff": fused, "latent_max_abs_diff": latent}


def import_keras_bundle(extractor, source, *, export_path: Optional[Path] = None) -> Dict[str, Any]:
    """Import weights into ``extractor`` from a ``ModularBundle`` (exports first, in-process)
    or from an existing export .npz path. Returns the import + fidelity report."""
    encoder = getattr(extractor, "encoder", extractor)
    if isinstance(source, (str, Path)) and str(source).endswith(".npz"):
        export = load_export(source)
    else:
        import tempfile

        export_path = Path(export_path) if export_path else Path(tempfile.mkdtemp()) / "encoder_export.npz"
        export_encoder_npz(source, export_path)
        export = load_export(export_path)
    report = import_export_into_encoder(encoder, export)
    report["fidelity"] = fidelity_report(encoder, export)
    report["versions_import"] = {"torch": torch.__version__}
    return report
