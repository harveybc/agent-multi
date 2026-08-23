"""Checkpoint-coherent state bundle (findings 307/308/309).

One improvement event writes ONE coherent bundle: the selected model
artifact, the replay buffer AS OF THAT EPOCH, the monitor and inner
validation traces that scored it, exact named-tensor state hashes of
every learned component, and an immutable manifest binding them with
epoch, config/data/observation identity and RNG facts. Handoff and
EN-F continuity consume ONLY the manifest — never a mutable path, and
never state from two different epochs.

Continuity is EXACT named-state equality (309): every tensor of the
actor, critics, target critics, entropy coefficient and every
optimizer state is hashed under deterministic framing
``name|dtype|shape|bytes``; two different tensors with equal L1 norm
are different maps. Scalar-norm comparison is diagnostic only.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

BUNDLE_SCHEMA = "agent_multi.selected_checkpoint_bundle.v1"


class BundleError(ValueError):
    pass


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_digest(name: str, tensor) -> str:
    import torch

    if not torch.is_tensor(tensor):
        payload = json.dumps(tensor, sort_keys=True,
                             default=repr).encode()
        frame = f"{name}|scalar|-|".encode() + payload
        return hashlib.sha256(frame).hexdigest()
    t = tensor.detach().cpu().contiguous()
    frame = (f"{name}|{t.dtype}|{tuple(t.shape)}|".encode()
             + t.numpy().tobytes())
    return hashlib.sha256(frame).hexdigest()


def _state_dict_hashes(prefix: str, state: Dict[str, Any]
                       ) -> Dict[str, str]:
    out = {}
    for key in sorted(state):
        value = state[key]
        if isinstance(value, dict):
            out.update(_state_dict_hashes(f"{prefix}.{key}", value))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    out.update(_state_dict_hashes(
                        f"{prefix}.{key}[{i}]", item))
                else:
                    out[f"{prefix}.{key}[{i}]"] = _tensor_digest(
                        f"{prefix}.{key}[{i}]", item)
        else:
            out[f"{prefix}.{key}"] = _tensor_digest(
                f"{prefix}.{key}", value)
    return out


def named_state_hashes(model) -> Dict[str, str]:
    """Exact per-tensor hashes of every learned component (309)."""
    out: Dict[str, str] = {}
    out.update(_state_dict_hashes("policy", model.policy.state_dict()))
    for label in ("actor", "critic"):
        opt = getattr(getattr(model, label, None), "optimizer", None)
        if opt is not None:
            out.update(_state_dict_hashes(
                f"{label}.optimizer", opt.state_dict()))
    ent_opt = getattr(model, "ent_coef_optimizer", None)
    if ent_opt is not None:
        out.update(_state_dict_hashes("ent_coef.optimizer",
                                      ent_opt.state_dict()))
    log_ent = getattr(model, "log_ent_coef", None)
    if log_ent is not None:
        out["log_ent_coef"] = _tensor_digest("log_ent_coef", log_ent)
    return out


def replay_facts(model) -> Dict[str, Any]:
    buffer = getattr(model, "replay_buffer", None)
    if buffer is None:
        return {"present": False}
    return {"present": True,
            "size": int(buffer.size()),
            "pos": int(getattr(buffer, "pos", -1)),
            "full": bool(getattr(buffer, "full", False)),
            "capacity": int(getattr(buffer, "buffer_size", -1)),
            "observation_space": repr(
                getattr(buffer, "observation_space", None)),
            "action_space": repr(
                getattr(buffer, "action_space", None))}


def rng_facts() -> Dict[str, str]:
    import numpy as np
    import torch

    return {"torch_rng_sha256": hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "numpy_rng_sha256": hashlib.sha256(
                json.dumps([repr(x) for x in np.random.get_state()],
                           sort_keys=True).encode()).hexdigest()}


def write_bundle(
    bundle_dir: Path, *, epoch: int, model,
    model_artifact: Path, trace_sources: Dict[str, Path],
    config_sha256: str, data_sha256: str,
    observation_contract: Any,
) -> Path:
    """Write the coherent bundle at an improvement event.

    Everything is snapshotted NOW, from the live model that was just
    saved: replay as of this epoch (308), this epoch's scoring traces
    (307), exact named-state hashes (309)."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    replay_path = bundle_dir / "selected_replay.pkl"
    model.save_replay_buffer(str(replay_path))
    traces = {}
    for role, src in trace_sources.items():
        src = Path(src)
        if not src.is_file():
            raise BundleError(
                f"selected-checkpoint trace for role {role!r} missing "
                f"at {src}; an unbound trace cannot authorize handoff")
        dst = bundle_dir / f"selected_{role}_trace.csv"
        shutil.copyfile(src, dst)
        traces[role] = {"path": str(dst), "sha256": _sha_file(dst)}
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "epoch": int(epoch),
        "model": {"path": str(Path(model_artifact).resolve()),
                  "sha256": _sha_file(Path(model_artifact))},
        "named_state_sha256": named_state_hashes(model),
        "replay": {"path": str(replay_path),
                   "sha256": _sha_file(replay_path),
                   **replay_facts(model)},
        "traces": traces,
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "observation_contract": observation_contract,
        "rng": rng_facts(),
    }
    path = bundle_dir / "selected_checkpoint_manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    tmp.replace(path)
    return path


def load_manifest(path: Path) -> Dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != BUNDLE_SCHEMA:
        raise BundleError(f"foreign bundle schema {doc.get('schema')!r}")
    return doc


def verify_loaded_model(model, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """After warm-start load: EXACT named-state map equality (309)."""
    expected = manifest["named_state_sha256"]
    actual = named_state_hashes(model)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(k for k in set(expected) & set(actual)
                     if expected[k] != actual[k])
    if missing or extra or changed:
        raise BundleError(
            "loaded model state does not match the selected bundle: "
            f"missing={missing[:4]} extra={extra[:4]} "
            f"changed={changed[:4]} (counts {len(missing)}/"
            f"{len(extra)}/{len(changed)})")
    return {"tensors_verified": len(expected), "exact": True}
