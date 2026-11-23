"""Non-governing TRAIN diagnostic. No execution, optimizer or predictor."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def measure(root):
    hour = 3600 * 10**9
    cutoff = np.datetime64("2023-07-01", "ns").astype("int64")
    with np.load(root / "dataset_sdelta.npz") as d, np.load(root / "stl_series.npz") as b:
        bt, close, seasonal = b["avail_ns"], b["close"].astype(float), b["seasonal"]
        assert np.all(np.diff(bt) > 0)
        for group, horizons in (("short", range(1, 7)), ("long", (24, 48))):
            for col, h in enumerate(horizons):
                mask = d["pool"] & d["ok_" + group] & (d["ts"] + h * hour < cutoff)
                assert not (mask & (d["monitor"] | d["holdout"])).any()
                ts = d["ts"][mask]
                i, j = np.searchsorted(bt, ts), np.searchsorted(bt, ts + h * hour)
                assert np.array_equal(bt[i], ts) and np.array_equal(bt[j], ts + h * hour)
                ds, dc = seasonal[j] - seasonal[i], close[j] - close[i]
                np.testing.assert_allclose(ds, d["y_" + group][mask, col], atol=1e-8)
                assert np.isfinite(ds).all() and np.isfinite(dc).all()
                nz = (ds != 0) & (dc != 0)
                yield (h, len(ts), nz.sum(), np.mean(np.sign(ds[nz]) == np.sign(dc[nz])),
                       np.mean(np.abs(dc)), np.mean(np.abs(dc - ds)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    output = csv.writer(sys.stdout, lineterminator="\n")
    output.writerow(("hours", "origins", "non_ties", "direction_agreement", "close_persistence_mae",
                     "close_plus_ideal_seasonal_delta_mae"))
    output.writerows(measure(parser.parse_args().artifact_dir))
