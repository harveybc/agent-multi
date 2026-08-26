"""Shared typed topology validator (DATA-SOTA-334): every branch/fusion
plugin validates its gene domain here BEFORE model construction —
invalid DOIN genes refuse, never construct."""
from __future__ import annotations

from typing import Any


class TopologyError(ValueError):
    pass


def strict_int(value: Any, label: str, minimum: int = 1) -> int:
    """DATA-SOTA-335: EXACT integral non-boolean type. bool is an int
    subclass in Python and "8"/8.0 coerce silently — none of them is a
    valid topology gene at a JSON/config boundary."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TopologyError(f"{label} must be a non-boolean integer, "
                            f"got {type(value).__name__} {value!r}")
    if value < minimum:
        raise TopologyError(f"{label} must be >= {minimum}, got {value}")
    return value


def strict_real(value: Any, label: str) -> float:
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TopologyError(f"{label} must be a non-boolean finite "
                            f"number, got {type(value).__name__} "
                            f"{value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise TopologyError(f"{label} must be finite, got {result!r}")
    return result


def require_positive_int(config: dict[str, Any], key: str,
                         minimum: int = 1) -> int:
    return strict_int(config.get(key), key, minimum)


def require_int_list(config: dict[str, Any], key: str,
                     minimum: int = 1) -> list[int]:
    value = config.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        raise TopologyError(f"{key} must be a non-empty list of "
                            f"integers, got {value!r}")
    return [strict_int(v, f"{key}[{i}]", minimum)
            for i, v in enumerate(value)]


def require_dropout(config: dict[str, Any], key: str = "dropout"
                    ) -> float:
    value = strict_real(config.get(key, 0.0), key)
    if not (0.0 <= value < 1.0):
        raise TopologyError(f"{key} must lie in [0, 1), got {value}")
    return value


def require_heads_divide(config: dict[str, Any], dim_key: str,
                         heads_key: str) -> tuple[int, int]:
    dim = require_positive_int(config, dim_key)
    heads = require_positive_int(config, heads_key)
    if dim % heads:
        raise TopologyError(f"{dim_key}={dim} must be divisible by "
                            f"{heads_key}={heads}")
    return dim, heads


def require_odd_kernel(config: dict[str, Any], key: str = "kernel"
                       ) -> int:
    kernel = require_positive_int(config, key)
    if kernel % 2 == 0:
        raise TopologyError(
            f"{key}={kernel} must be ODD: same-shape padding "
            f"(kernel//2) silently changes the output length for even "
            f"kernels")
    return kernel


def require_window(window_size: Any, minimum: int, plugin: str) -> int:
    return strict_int(window_size, f"{plugin} window_size", minimum)


def require_patch_coverage(window_size: Any, patch_len: int,
                           stride: int) -> int:
    window_size = strict_int(window_size, "window_size", 1)
    if patch_len < 1 or stride < 1 or patch_len > window_size:
        raise TopologyError("invalid patching: need 1 <= patch_len <= "
                            f"window ({patch_len} vs {window_size}) "
                            f"and stride >= 1 ({stride})")
    return 1 + (window_size - patch_len) // stride


def require_spectral_viability(window_size: int, top_k: int) -> int:
    bins = window_size // 2  # nonzero rFFT bins
    if bins < 1:
        raise TopologyError(f"window_size={window_size} has no nonzero "
                            f"spectral bin; TimesNet-style folding is "
                            f"degenerate")
    if top_k > bins:
        raise TopologyError(f"top_k={top_k} exceeds available nonzero "
                            f"spectral bins ({bins}) for window "
                            f"{window_size}")
    return top_k


def require_param_ceiling(estimate: int, config: dict[str, Any],
                          key: str = "max_parameters") -> None:
    ceiling = config.get(key)
    if ceiling is not None:
        ceiling = strict_int(ceiling, key, 1)
    if ceiling is not None and estimate > int(ceiling):
        raise TopologyError(f"estimated parameters {estimate} exceed "
                            f"declared ceiling {ceiling}")
