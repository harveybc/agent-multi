"""Validate a pinned curator receipt before gym-fx can load financial rows.

This verifies transport/content integrity, not the honesty of the curator. The
expected manifest digest must come from the admitted inventory, not this file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("CURATED_TIMESTAMP_TIMEZONE_REQUIRED")
    return result.astimezone(timezone.utc)


def validate_curated_rl_input(config):
    if not config.get("require_curated_rl_input", False):
        return None
    manifest_path = config.get("curated_input_manifest")
    expected = config.get("curated_input_manifest_sha256")
    if not manifest_path or not expected or _digest(manifest_path) != expected:
        raise ValueError("CURATED_MANIFEST_DIGEST_MISMATCH")
    manifest = json.loads(Path(manifest_path).read_text())
    if (manifest.get("schema") != "curated_rl_input.v1"
            or manifest.get("status") != "ADMITTED"
            or manifest.get("availability_verified") is not True
            or manifest.get("open_session_coverage_verified") is not True):
        raise ValueError("CURATED_INPUT_NOT_ADMITTED")
    if manifest.get("split") != config.get("curated_input_split"):
        raise ValueError("CURATED_SPLIT_MISMATCH")
    columns = config.get("feature_columns")
    if not columns or len(set(columns)) != len(columns) or columns != manifest.get("feature_columns"):
        raise ValueError("CURATED_FEATURE_ORDER_MISMATCH")
    for name in ("selection_sha256", "calendar_sha256", "inventory_sha256"):
        value = manifest.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("CURATED_LINEAGE_DIGEST_REQUIRED")
    path = config["input_data_file"]
    if _digest(path) != manifest.get("input_sha256"):
        raise ValueError("CURATED_INPUT_DIGEST_MISMATCH")
    if (manifest.get("timestamp_semantics") != "completed_bar_available_at"
            or not config.get("headers", True)
            or config.get("max_rows") is not None):
        raise ValueError("CURATED_FULL_POPULATION_REQUIRED")
    date_column = config.get("date_column", "DATE_TIME")
    start, end = _time(manifest["period_start"]), _time(manifest["period_end"])
    timestamp_hash = hashlib.sha256()
    previous = None
    count = 0
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(columns) | {date_column, "OPEN", "HIGH", "LOW", "CLOSE"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CURATED_REQUIRED_COLUMNS_MISSING")
        for row in reader:
            stamp = _time(row[date_column])
            if not start <= stamp < end or (previous is not None and stamp <= previous):
                raise ValueError("CURATED_TIMESTAMP_SUPPORT_MISMATCH")
            numbers = {key: float(row[key]) for key in required - {date_column}}
            if not all(math.isfinite(value) for value in numbers.values()):
                raise ValueError("CURATED_NONFINITE_VALUE")
            op, hi, lo, cl = (numbers[k] for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
            if not 0 < lo <= min(op, cl) <= max(op, cl) <= hi:
                raise ValueError("CURATED_INVALID_OHLC")
            timestamp_hash.update((stamp.isoformat() + "\n").encode())
            previous = stamp
            count += 1
    if (not count or count != manifest.get("row_count")
            or timestamp_hash.hexdigest() != manifest.get("expected_timestamps_sha256")):
        raise ValueError("CURATED_TIMESTAMP_SUPPORT_MISMATCH")
    return {"manifest_sha256": expected, "input_sha256": manifest["input_sha256"],
            "split": manifest["split"], "row_count": count,
            **{key: manifest[key] for key in
               ("selection_sha256", "calendar_sha256", "inventory_sha256")}}
