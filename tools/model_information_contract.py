#!/usr/bin/env python3
"""Strict measurement helpers for the model-information pilot.

The module deliberately keeps four ideas separate: model description length,
data description length, empirical memorization, and residual capacity.  A
compressed byte count is an estimator-specific upper bound, never an exact
Kolmogorov complexity or a measurement of intelligence.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import lzma
import math
import struct
import zlib
from collections.abc import Mapping
from typing import Any

import numpy as np


MODEL_DESCRIPTION_SCHEMA = "agent_multi.model_description.v1"
DATA_DESCRIPTION_SCHEMA = "agent_multi.data_description.v1"
MEASUREMENT_SCHEMA = "agent_multi.model_information_measurement.v1"


class ModelInformationError(ValueError):
    """Typed refusal for malformed or semantically invalid evidence."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ModelInformationError(f"not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_int(path: str, value: Any, *, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise ModelInformationError(f"{path}: expected integer, not bool")
    if minimum is not None and value < minimum:
        raise ModelInformationError(f"{path}: must be >= {minimum}")
    return value


def require_number(
    path: str,
    value: Any,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelInformationError(f"{path}: expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ModelInformationError(f"{path}: expected finite number")
    if minimum is not None and result < minimum:
        raise ModelInformationError(f"{path}: must be >= {minimum}")
    return result


def require_digest(path: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ModelInformationError(f"{path}: expected canonical sha256")
    if value != value.lower() or any(c not in "0123456789abcdef" for c in value):
        raise ModelInformationError(f"{path}: expected canonical sha256")
    return value


def seal_document(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in document:
        raise ModelInformationError(f"{field}: digest field already present")
    sealed = dict(document)
    sealed[field] = sha256_bytes(canonical_json_bytes(document))
    return sealed


def verify_sealed_document(
    document: Mapping[str, Any],
    *,
    field: str,
    schema: str,
    exact_keys: set[str],
) -> None:
    if set(document) != exact_keys:
        missing = sorted(exact_keys - set(document))
        extra = sorted(set(document) - exact_keys)
        raise ModelInformationError(
            f"document keys mismatch: missing={missing}, extra={extra}"
        )
    if document.get("schema") != schema:
        raise ModelInformationError("schema mismatch")
    expected = require_digest(field, document[field])
    body = {key: value for key, value in document.items() if key != field}
    actual = sha256_bytes(canonical_json_bytes(body))
    if actual != expected:
        raise ModelInformationError(f"{field}: self-digest mismatch")


def _as_finite_array(path: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_:
        raise ModelInformationError(f"{path}: boolean array is not numeric evidence")
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelInformationError(f"{path}: not numeric") from exc
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ModelInformationError(f"{path}: empty or non-finite")
    return array


def _entropy_bits(values: np.ndarray, bins: int = 256) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if np.all(flat == flat[0]):
        return 0.0
    counts, _ = np.histogram(flat, bins=bins)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def _compressed_lengths(blob: bytes) -> dict[str, int]:
    return {
        "uncompressed_bytes": len(blob),
        "zlib_bytes": len(zlib.compress(blob, level=9)),
        "bz2_bytes": len(bz2.compress(blob, compresslevel=9)),
        "lzma_bytes": len(lzma.compress(blob, preset=9)),
    }


def _array_blob(name: str, array: np.ndarray) -> bytes:
    canonical = np.ascontiguousarray(array, dtype="<f4")
    header = canonical_json_bytes(
        {"name": name, "shape": list(canonical.shape), "dtype": "float32-le"}
    )
    return struct.pack("<Q", len(header)) + header + canonical.tobytes(order="C")


def _quantized_blob(name: str, array: np.ndarray, bits: int) -> tuple[bytes, float]:
    if bits not in (4, 8, 16):
        raise ModelInformationError("quantization bits must be one of 4, 8, 16")
    source = np.asarray(array, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1
    max_abs = float(np.max(np.abs(source)))
    scale = max_abs / qmax if max_abs else 1.0
    quantized = np.rint(source / scale).clip(-qmax, qmax)
    dtype = np.int8 if bits <= 8 else np.int16
    quantized = np.ascontiguousarray(quantized, dtype=dtype)
    restored = quantized.astype(np.float64) * scale
    mse = float(np.mean((source - restored) ** 2))
    header = canonical_json_bytes(
        {
            "name": name,
            "shape": list(source.shape),
            "bits": bits,
            "scale_hex": float(scale).hex(),
            "packing": "one-signed-integer-per-value",
        }
    )
    blob = struct.pack("<Q", len(header)) + header + quantized.tobytes(order="C")
    return blob, mse


def _matrix_spectrum(name: str, array: np.ndarray) -> dict[str, Any] | None:
    if array.ndim != 2 or min(array.shape) == 0:
        return None
    singular = np.linalg.svd(np.asarray(array, dtype=np.float64), compute_uv=False)
    energy = singular**2
    total = float(np.sum(energy))
    if total == 0:
        effective_rank = 0.0
        stable_rank = 0.0
    else:
        p = energy[energy > 0] / total
        effective_rank = float(np.exp(-np.sum(p * np.log(p))))
        stable_rank = float(total / max(float(energy[0]), 1e-300))
    return {
        "name": name,
        "rows": int(array.shape[0]),
        "columns": int(array.shape[1]),
        "spectral_norm": float(singular[0]) if singular.size else 0.0,
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
    }


def describe_model_arrays(arrays: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ModelInformationError("model arrays must be a non-empty mapping")
    normalized: dict[str, np.ndarray] = {}
    for name in sorted(arrays):
        if not isinstance(name, str) or not name:
            raise ModelInformationError("model array names must be non-empty strings")
        normalized[name] = _as_finite_array(f"model.{name}", arrays[name])

    flat = np.concatenate([value.ravel() for value in normalized.values()])
    raw_blob = b"".join(_array_blob(name, normalized[name]) for name in normalized)
    quantized: dict[str, Any] = {}
    for bits in (4, 8, 16):
        pieces: list[bytes] = []
        weighted_mse = 0.0
        count = 0
        for name, array in normalized.items():
            blob, mse = _quantized_blob(name, array, bits)
            pieces.append(blob)
            weighted_mse += mse * array.size
            count += array.size
        qblob = b"".join(pieces)
        quantized[str(bits)] = {
            "serialization": _compressed_lengths(qblob),
            "parameter_mse": float(weighted_mse / count),
        }

    spectra = [
        spectrum
        for name, array in normalized.items()
        if (spectrum := _matrix_spectrum(name, array)) is not None
    ]
    body = {
        "schema": MODEL_DESCRIPTION_SCHEMA,
        "interpretation": "estimator_specific_description_length_upper_bound",
        "parameter_count": int(flat.size),
        "exact_zero_fraction": float(np.mean(flat == 0.0)),
        "near_zero_fraction_1e_6": float(np.mean(np.abs(flat) <= 1e-6)),
        "exact_repeated_value_fraction": float(1.0 - np.unique(flat).size / flat.size),
        "histogram_entropy_bits_per_parameter": _entropy_bits(flat),
        "raw_serialization": _compressed_lengths(raw_blob),
        "quantized": quantized,
        "matrix_spectra": spectra,
        "residual_capacity_inferred": False,
    }
    return seal_document(body, "description_sha256")


def describe_data_arrays(arrays: Mapping[str, Any], *, repeated_exposures: int) -> dict[str, Any]:
    require_int("repeated_exposures", repeated_exposures, minimum=1)
    if not isinstance(arrays, Mapping) or not arrays:
        raise ModelInformationError("data arrays must be a non-empty mapping")
    normalized = {
        name: _as_finite_array(f"data.{name}", arrays[name])
        for name in sorted(arrays)
    }
    blob = b"".join(_array_blob(name, normalized[name]) for name in normalized)
    values = np.concatenate([value.ravel() for value in normalized.values()])
    body = {
        "schema": DATA_DESCRIPTION_SCHEMA,
        "interpretation": "canonical_dataset_view_code_length",
        "unique_scalar_observations": int(values.size),
        "repeated_training_exposures": repeated_exposures,
        "independent_information_multiplier_from_repetition": 1,
        "histogram_entropy_bits_per_scalar": _entropy_bits(values),
        "serialization": _compressed_lengths(blob),
    }
    return seal_document(body, "description_sha256")


MODEL_DESCRIPTION_KEYS = {
    "schema",
    "interpretation",
    "parameter_count",
    "exact_zero_fraction",
    "near_zero_fraction_1e_6",
    "exact_repeated_value_fraction",
    "histogram_entropy_bits_per_parameter",
    "raw_serialization",
    "quantized",
    "matrix_spectra",
    "residual_capacity_inferred",
    "description_sha256",
}


DATA_DESCRIPTION_KEYS = {
    "schema",
    "interpretation",
    "unique_scalar_observations",
    "repeated_training_exposures",
    "independent_information_multiplier_from_repetition",
    "histogram_entropy_bits_per_scalar",
    "serialization",
    "description_sha256",
}


def verify_model_description(document: Mapping[str, Any]) -> None:
    verify_sealed_document(
        document,
        field="description_sha256",
        schema=MODEL_DESCRIPTION_SCHEMA,
        exact_keys=MODEL_DESCRIPTION_KEYS,
    )
    require_int("parameter_count", document["parameter_count"], minimum=1)
    for field in (
        "exact_zero_fraction",
        "near_zero_fraction_1e_6",
        "exact_repeated_value_fraction",
        "histogram_entropy_bits_per_parameter",
    ):
        require_number(field, document[field], minimum=0.0)
    if document["residual_capacity_inferred"] is not False:
        raise ModelInformationError(
            "description length or repeated weights cannot infer residual capacity"
        )
    if document["interpretation"] != "estimator_specific_description_length_upper_bound":
        raise ModelInformationError("model description overclaims its interpretation")


def verify_data_description(document: Mapping[str, Any]) -> None:
    verify_sealed_document(
        document,
        field="description_sha256",
        schema=DATA_DESCRIPTION_SCHEMA,
        exact_keys=DATA_DESCRIPTION_KEYS,
    )
    require_int("unique_scalar_observations", document["unique_scalar_observations"], minimum=1)
    require_int("repeated_training_exposures", document["repeated_training_exposures"], minimum=1)
    if document["independent_information_multiplier_from_repetition"] != 1:
        raise ModelInformationError("repeated epochs are not independent data information")
    if document["interpretation"] != "canonical_dataset_view_code_length":
        raise ModelInformationError("data description overclaims its interpretation")
