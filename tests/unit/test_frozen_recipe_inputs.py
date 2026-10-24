import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from app.frozen_recipe_inputs import FrozenRecipeInputs, validate_recipe


def recipe():
    rec = {"schema": "expert_guided_consumer_recipe.v2",
           "store": {}, "list": {"n": 2, "columns": ["x", "y"]},
           "tensor": {"columns": [{"store_column": c, "tensor_name": c, "consumer_ops": []} for c in ("x", "y")],
                      "window": {"observations": 3, "stride_rows": 2, "includes_current_completed_bar": True}},
           "scaler": {"policy": "fixed_train", "features": ["x", "y"], "mean": [2., 3.], "std": [2., 4.]},
           "representation": {"resolved": {"names": ["g"], "indices": [[0, 1]], "units": [2], "kinds": ["conv"]},
                              "model": {"units": 2, "core": None, "depth": 2, "head": 16}},
           "regularization": {"dropout": .3, "wd": .01},
           "split": {"test_closed_from": "2025-01-01", "periods": {"TRAIN": ["2020-01-01", "2024-01-01"]}},
           "business": {"agent_state": "none: no agent feedback in the observation"}, "labels": {"mode": "zigzag"}}
    return seal(rec)


def seal(rec):
    keys = ["store", "list", "tensor", "scaler", "representation", "regularization", "split", "business", "labels"]
    rec["identity_covers"] = keys
    rec["identity"] = "recipe_" + hashlib.sha256(json.dumps({k: rec[k] for k in keys}, sort_keys=True).encode()).hexdigest()[:16]
    return rec


def frame():
    ts = pd.date_range("2020-01-03", periods=12, freq="h", tz="UTC")
    ts = ts.where(np.arange(12) < 6, ts + pd.Timedelta(days=2))
    return pd.DataFrame({"timestamp": ts, "x": np.arange(12.), "y": np.arange(12.) + 10})


def test_exact_constants_current_bar_closures_no_clipping_or_feedback():
    f = frame()
    f.loc[8, "x"] = 1e5
    data = FrozenRecipeInputs.from_frame(recipe(), f)
    expected = ((f[["x", "y"]].to_numpy()[[4, 6, 8]] - [2, 3]) / [2, 4]).astype(np.float32)
    np.testing.assert_array_equal(data.observation(8)["features"], expected)
    assert set(data.observation(8)) == {"features"}
    assert expected[-1, 0] > 1e4
    assert data.slot_ts_ns[8, -1] == data.row_ts_ns[8]
    assert data.slot_ts_ns[8, -1] - data.slot_ts_ns[8, 0] > 4 * 3600 * 10**9
    with pytest.raises(ValueError, match="WINDOW_NOT_VALID"):
        data.observation(3)


def test_future_perturbation_nan_and_unknown_operations():
    f = frame()
    a = FrozenRecipeInputs.from_frame(recipe(), f)
    f.loc[9:, "x"] += 500
    b = FrozenRecipeInputs.from_frame(recipe(), f)
    np.testing.assert_array_equal(a.observation(8)["features"], b.observation(8)["features"])
    f.loc[8, "x"] = np.nan
    c = FrozenRecipeInputs.from_frame(recipe(), f)
    assert not c.valid[8]
    bad = recipe()
    bad["tensor"]["columns"][0]["consumer_ops"] = ["CLIP"]
    with pytest.raises(ValueError, match="CONSUMER_OP"):
        validate_recipe(seal(bad))


@pytest.mark.parametrize("field", ["scale", "columns", "groups", "feedback"])
def test_contract_refuses_mutated_recipe(field):
    r = recipe()
    if field == "scale":
        r["scaler"]["std"][0] = 0
    elif field == "columns":
        r["list"]["columns"].reverse()
    elif field == "groups":
        r["representation"]["resolved"]["indices"] = [[0, 0]]
    else:
        r["business"]["agent_state"] = "full"
    with pytest.raises(ValueError):
        validate_recipe(seal(r))
    r = recipe()
    r["scaler"]["mean"][0] = 55
    with pytest.raises(ValueError, match="IDENTITY"):
        validate_recipe(r)


def test_loader_pins_files_and_never_selects_test_rows(tmp_path):
    r = recipe()
    f = pd.concat([frame(), pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01", tz="UTC")], "x": [999.], "y": [999.]})])
    p = tmp_path / "store.parquet"
    f.to_parquet(p, index=False)
    m = tmp_path / "MANIFEST.json"
    m.write_text("{}")
    r["store"] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "manifest_sha256": hashlib.sha256(m.read_bytes()).hexdigest()}
    seal(r)
    out = FrozenRecipeInputs.load(r, p, m)
    assert len(out.row_ts_ns) == 12
    m.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="MANIFEST_PIN"):
        FrozenRecipeInputs.load(r, p, m)
