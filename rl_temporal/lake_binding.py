"""Bind a run to a selected-feature manifest and prove the read is causal.

The manifest (``selected_feature_manifest.v1``, lane B) names the task, the
split, the resources with their sha256 and the feature variants. The binding
resolves the selected variant's feature order, verifies every resolvable
resource against its sha, and refuses a real-data pilot unless the manifest
is FROZEN. The causal probe builds the observation at a step from the
preprocessor and shows that rows at or after that step cannot move it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

MANIFEST_SCHEMA = "selected_feature_manifest.v1"
STATUSES = ("FROZEN", "DRAFT_NOT_FROZEN")


class PilotRefused(RuntimeError):
    pass


class ResourceIdentityMismatch(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SelectedFeatureBinding:
    manifest_path: str
    manifest_sha256: str
    status: str
    selected_variant: str
    feature_order: Tuple[str, ...]
    task: Dict[str, Any]
    split: Dict[str, Any]
    resources: Dict[str, Any]

    @classmethod
    def from_manifest(cls, path, *, selected_variant: Optional[str] = None,
                      expected_sha256: Optional[str] = None) -> "SelectedFeatureBinding":
        """``selected_variant`` may come from the manifest or from the task declaration
        (explicitly, never guessed); ``expected_sha256`` pins the manifest bytes."""
        p = Path(path)
        doc = json.loads(p.read_text(encoding="utf-8"))
        if expected_sha256 and _sha(p) != expected_sha256:
            raise ValueError(f"manifest sha256 {_sha(p)[:12]}... != expected {expected_sha256[:12]}...")
        if doc.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"unsupported manifest schema {doc.get('schema')!r}")
        status = doc.get("status")
        if status not in STATUSES:
            raise ValueError(f"manifest status must be one of {STATUSES}; got {status!r}")
        variant = selected_variant or doc.get("selected_variant")
        if not variant:
            raise ValueError("manifest declares no selected_variant; the RL task must name one explicitly")
        variants = doc.get("variants") or {}
        if variant not in variants:
            raise ValueError(f"selected_variant {variant!r} is not among {sorted(variants)}")
        features = variants[variant].get("features") or []
        if not features or len(set(features)) != len(features):
            raise ValueError("selected variant must list unique feature names")
        return cls(str(p), _sha(p), status, variant, tuple(str(f) for f in features),
                   dict(doc.get("task") or {}), dict(doc.get("split") or {}), dict(doc.get("resources") or {}))

    def _resolve(self, raw: str) -> Optional[Path]:
        candidates = [Path(raw), Path(self.manifest_path).parent / raw]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def verify_resources(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, res in self.resources.items():
            if not isinstance(res, dict) or "sha256" not in res:
                out[name] = {"verified": False, "reason": "NO_SHA_DECLARED"}
                continue
            raw = str(res.get("path", ""))
            path = self._resolve(raw)
            if path is None:
                out[name] = {"verified": False, "reason": "NOT_RESOLVABLE", "path": raw, "sha256": res["sha256"]}
                continue
            actual = _sha(path)
            if actual != res["sha256"]:
                raise ResourceIdentityMismatch(
                    f"resource {name} at {path} has sha256 {actual[:12]}..., manifest declares {res['sha256'][:12]}...")
            out[name] = {"verified": True, "path": str(path), "sha256": actual}
        return out

    def env_overrides(self) -> Dict[str, Any]:
        return {"feature_columns": list(self.feature_order),
                "selected_feature_manifest_sha256": self.manifest_sha256,
                "selected_feature_manifest_status": self.status,
                "selected_variant": self.selected_variant}

    def refuse_real_data_pilot(self) -> None:
        if self.status != "FROZEN":
            raise PilotRefused(f"manifest {Path(self.manifest_path).name} is {self.status}: no real-data pilot "
                               "may start from an unfrozen selection")


def causal_timestamp_probe(make_env, cfg: Dict[str, Any], *, step: int) -> Dict[str, Any]:
    """Does any row at or after ``step`` move the observation at ``step``?"""
    env = make_env(dict(cfg))
    try:
        base_df = env.dataframe
        window = int(env.config.get("window_size"))
        cols = list(env.config.get("feature_columns") or [])
        price = env.config.get("price_column", "CLOSE")
        state = {"position": 0, "equity": float(env.initial_cash), "initial_cash": float(env.initial_cash),
                 "price": float(base_df[price].iloc[step - 1]), "bar_index": step, "total_bars": len(base_df)}

        def observe(df):
            obs = env.preprocessor_plugin.make_observation(data=df, step=step, bridge_state=state, config=env.config)
            return {k: np.asarray(v, dtype=np.float64).copy() for k, v in obs.items()}

        base = observe(base_df)
        future = base_df.copy()
        future.loc[future.index[step:], cols + [price]] += 1000.0
        past = base_df.copy()
        past.loc[past.index[max(0, step - window):step], cols] += 1000.0
        same = lambda a, b: all(np.allclose(a[k], b[k]) for k in a)
        return {"step": step, "observation_rows": [step - window, step],
                "future_rows_influence": not same(base, observe(future)),
                "past_rows_influence": not same(base, observe(past)),
                "feature_order": cols}
    finally:
        env.close()
