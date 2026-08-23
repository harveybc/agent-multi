"""Strict configuration helpers for nested neural-network components."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def deep_merge_strict(
    defaults: Mapping[str, Any],
    override: Mapping[str, Any] | None,
    *,
    path: str = "config",
    allow_new: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Merge nested mappings while refusing accidental parameter names."""
    result = deepcopy(dict(defaults))
    for key, value in dict(override or {}).items():
        if key not in result and key not in allow_new:
            raise ValueError(f"unknown {path} key: {key!r}")
        current = result.get(key)
        if isinstance(current, Mapping):
            if not isinstance(value, Mapping):
                raise TypeError(f"{path}.{key} must be an object")
            result[key] = deep_merge_strict(
                current, value, path=f"{path}.{key}", allow_new=allow_new
            )
        else:
            result[key] = deepcopy(value)
    return result

