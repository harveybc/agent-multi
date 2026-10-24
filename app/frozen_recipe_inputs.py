"""Frozen selected market inputs for SAC; no labels, fitted statistics or state."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_KEYS = ("store", "list", "tensor", "scaler", "representation",
                 "regularization", "split", "business", "labels")


def require(condition, code):
    if not condition:
        raise ValueError("FROZEN_RECIPE_" + code)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_recipe(rec):
    require(rec.get("schema") == "expert_guided_consumer_recipe.v2", "SCHEMA")
    identity = "recipe_" + hashlib.sha256(json.dumps(
        {k: rec[k] for k in IDENTITY_KEYS}, sort_keys=True).encode()).hexdigest()[:16]
    require(rec.get("identity") == identity and tuple(rec.get("identity_covers", ())) == IDENTITY_KEYS, "IDENTITY")
    cols = rec["tensor"]["columns"]
    names = [c["store_column"] for c in cols]
    tensor = [c["tensor_name"] for c in cols]
    require(len(names) > 0 and len(set(names)) == len(names)
            and names == rec["list"]["columns"] and len(names) == rec["list"]["n"], "COLUMNS")
    for c in cols:
        require(c["consumer_ops"] in ([], ["DIFF"]), "CONSUMER_OP")
        require(c["tensor_name"] == c["store_column"] + ("@diff" if c["consumer_ops"] else ""), "TENSOR_NAME")
    sc = rec["scaler"]
    mean, std = np.asarray(sc["mean"], float), np.asarray(sc["std"], float)
    require(sc["policy"] == "fixed_train" and sc["features"] == tensor
            and mean.shape == std.shape == (len(cols),)
            and np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(), "SCALER")
    w = rec["tensor"]["window"]
    require(all(type(w[k]) is int and w[k] > 0 for k in ("observations", "stride_rows"))
            and w["includes_current_completed_bar"] is True, "WINDOW")
    g = rec["representation"]["resolved"]
    require(len(g["names"]) == len(set(g["names"])) == len(g["indices"]) == len(g["units"]) == len(g["kinds"])
            and sorted(i for group in g["indices"] for i in group) == list(range(len(cols)))
            and all(group for group in g["indices"])
            and all(type(u) is int and u > 0 for u in g["units"])
            and all(k in ("conv", "pointwise") for k in g["kinds"]), "GROUPS")
    require(rec["business"]["agent_state"].startswith("none:"), "FEEDBACK")
    end = pd.Timestamp(rec["split"]["test_closed_from"], tz="UTC")
    intervals = sorted((pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"))
                       for a, b in rec["split"]["periods"].values())
    require(all(a < b <= end for a, b in intervals)
            and all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), "SPLITS")


@dataclass(frozen=True)
class FrozenRecipeInputs:
    recipe: dict
    row_ts_ns: np.ndarray
    slot_ts_ns: np.ndarray
    windows: np.ndarray
    valid: np.ndarray
    period: np.ndarray

    @classmethod
    def load(cls, recipe, store, manifest):
        validate_recipe(recipe)
        require(file_sha(store) == recipe["store"]["sha256"], "STORE_PIN")
        require(file_sha(manifest) == recipe["store"]["manifest_sha256"], "MANIFEST_PIN")
        cols = [c["store_column"] for c in recipe["tensor"]["columns"]]
        end = pd.Timestamp(recipe["split"]["test_closed_from"], tz="UTC")
        frame = pd.read_parquet(store, columns=["timestamp"] + cols,
                                filters=[("timestamp", "<", end)])
        return cls.from_frame(recipe, frame)

    @classmethod
    def from_frame(cls, recipe, frame):
        validate_recipe(recipe)
        cols = recipe["tensor"]["columns"]
        names = [c["store_column"] for c in cols]
        ts = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).as_unit("ns")
        require(len(ts) > 0 and (np.diff(ts.asi8) > 0).all(), "TIMESTAMPS")
        require((ts < pd.Timestamp(recipe["split"]["test_closed_from"], tz="UTC")).all(), "TEST_ROWS")
        x = frame[names].to_numpy(dtype=np.float64, copy=True)
        finite = np.isfinite(x).all(axis=1)
        require(finite.any(), "NO_FINITE_ROW")
        # Match the declared consumer warm-up origin; never remove interior gaps.
        first = np.flatnonzero(finite)[0]
        x, ts = x[first:], ts[first:]
        for j, c in enumerate(cols):
            if c["consumer_ops"] == ["DIFF"]:
                x[:, j] = np.r_[np.nan, np.diff(x[:, j])]
        sc, w = recipe["scaler"], recipe["tensor"]["window"]
        z = (x - np.asarray(sc["mean"], np.float64)) / np.asarray(sc["std"], np.float64)
        slots = np.arange(len(x))[:, None] - w["stride_rows"] * np.arange(w["observations"] - 1, -1, -1)
        complete = slots[:, 0] >= 0
        safe_slots = slots.clip(0, len(x) - 1)
        windows = z[safe_slots].astype(np.float32)
        valid = complete & np.isfinite(windows).all(axis=(1, 2))
        # The store timestamps are bar starts; the feature vector is available at close.
        row_ts = ts.asi8.copy() + 3600 * 10**9
        period = np.full(len(x), "", dtype=object)
        for name, (a, b) in recipe["split"]["periods"].items():
            period[(row_ts >= pd.Timestamp(a, tz="UTC").value) & (row_ts < pd.Timestamp(b, tz="UTC").value)] = name
        slot_ts = row_ts[safe_slots]
        for array in (row_ts, slot_ts, windows, valid, period):
            array.flags.writeable = False
        return cls(copy.deepcopy(recipe), row_ts, slot_ts, windows, valid, period)

    def observation(self, row):
        require(type(row) in (int, np.int64) and 0 <= row < len(self.valid) and self.valid[row], "WINDOW_NOT_VALID")
        return {"features": self.windows[row].copy()}
