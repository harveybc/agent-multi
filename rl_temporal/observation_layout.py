"""Where each block of the flattened gym-fx observation lives.

gym-fx emits a ``Dict`` observation; ``FlattenObservation`` concatenates its
blocks in the Dict's key order (gymnasium sorts the keys at construction).
A policy that wants the ``(window, features)`` tensor back must know exactly
which slice holds it and in which feature order; guessing from the length is
how 83 columns in the wrong order go unnoticed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

SCHEMA = "rl_temporal.observation_layout.v1"


@dataclass(frozen=True)
class ObservationLayout:
    blocks: Tuple[Tuple[str, int, int, Tuple[int, ...]], ...]  # (name, start, stop, shape)
    feature_order: Tuple[str, ...]
    window: int
    n_features: int

    # ------------------------------------------------------------------ build
    @classmethod
    def from_space(cls, space, *, feature_order: Sequence[str]) -> "ObservationLayout":
        from gymnasium import spaces

        if not isinstance(space, spaces.Dict):
            raise ValueError("ObservationLayout.from_space needs the env's Dict observation space")
        if "features" not in space.spaces:
            raise ValueError("observation space has no 'features' block; the run is not feature-aware")
        blocks = []
        cursor = 0
        for name, sub in space.spaces.items():  # Dict preserves its (sorted) key order
            size = int(np.prod(sub.shape))
            blocks.append((str(name), cursor, cursor + size, tuple(int(s) for s in sub.shape)))
            cursor += size
        fshape = space.spaces["features"].shape
        if len(fshape) != 2:
            raise ValueError(f"features block must be (window, n_features); got {fshape}")
        window, n_features = int(fshape[0]), int(fshape[1])
        feature_order = tuple(str(f) for f in feature_order)
        if len(feature_order) != n_features:
            raise ValueError(
                f"feature_order has {len(feature_order)} names but the features block has {n_features} columns")
        return cls(tuple(blocks), feature_order, window, n_features)

    @classmethod
    def synthetic(cls, *, window: int, feature_order: Sequence[str], extras: int) -> "ObservationLayout":
        """A layout with the feature window first and one extras block; for fixtures."""
        feature_order = tuple(str(f) for f in feature_order)
        f = window * len(feature_order)
        blocks = [("features", 0, f, (window, len(feature_order)))]
        if extras:
            blocks.append(("extras", f, f + int(extras), (int(extras),)))
        return cls(tuple(blocks), feature_order, int(window), len(feature_order))

    # -------------------------------------------------------------- geometry
    @property
    def total_dim(self) -> int:
        return max(stop for _, _, stop, _ in self.blocks)

    @property
    def features_slice(self) -> slice:
        for name, start, stop, _ in self.blocks:
            if name == "features":
                return slice(start, stop)
        raise ValueError("no features block")

    @property
    def extras_blocks(self) -> Tuple[Tuple[str, int, int, Tuple[int, ...]], ...]:
        return tuple(b for b in self.blocks if b[0] != "features")

    @property
    def extras_dim(self) -> int:
        return self.total_dim - self.window * self.n_features

    def feature_index(self, step: int, feature: int) -> int:
        """Flat index of feature ``feature`` at window step ``step`` (row-major (W, F))."""
        if not 0 <= step < self.window or not 0 <= feature < self.n_features:
            raise IndexError((step, feature))
        return self.features_slice.start + step * self.n_features + feature

    def split(self, flat):
        """``(B, total) -> ((B, W, F), (B, extras))`` for torch tensors or numpy arrays."""
        fs = self.features_slice
        feats = flat[:, fs].reshape(flat.shape[0], self.window, self.n_features)
        parts = [flat[:, start:stop] for _, start, stop, _ in self.extras_blocks]
        if parts:
            try:
                import torch

                if isinstance(flat, torch.Tensor):
                    extras = torch.cat(parts, dim=1)
                else:
                    extras = np.concatenate(parts, axis=1)
            except ImportError:
                extras = np.concatenate(parts, axis=1)
        else:
            extras = flat[:, 0:0]
        return feats, extras

    # -------------------------------------------------------------- identity
    def declaration(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "blocks": [list(b[:3]) + [list(b[3])] for b in self.blocks],
                "feature_order": list(self.feature_order), "window": self.window,
                "n_features": self.n_features, "total_dim": self.total_dim}

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.declaration(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
