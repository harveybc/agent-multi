"""Reconcile terminal receipts and reconstruct archived pre-2024 input windows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runs(root):
    aggregate = read(root / "COMPARISON.json")["results"]
    records = []
    for seed in (7, 19, 43):
        for init in ("pretrained", "random"):
            for mode in ("restore", "fork"):
                arm = f"{init}-{mode}"
                where = root / f"seed-{seed}" / arm
                done = read(where / "DONE.json")
                cfg = read(where / "CONFIG.json")
                if (done != aggregate[str(seed)][arm] or
                        (cfg["seed"], cfg["init"], cfg["mode"]) != (seed, init, mode)):
                    raise ValueError(f"receipt mismatch {where}")
                years = [*cfg["train_years"], cfg["val_year"]]
                accounts = dict(zip(years, done["accounts"], strict=True))
                val = accounts[cfg["val_year"]]
                records.append(dict(seed=seed, arm=arm, champ=done["champ_epoch"],
                    epochs=done["epochs"], passes=done["passes"], stop=done["stop"],
                    promoted=done["promoted"], val_trades=val["trades"], val_wins=val["wins"],
                    val_dd=val["dd"], val_sharpe=val["sharpe"],
                    **{f"ret_pct_{year}": a["ret"] for year, a in accounts.items()},
                    checkpoint_sha256=digest(where / "champ.pt")))
    return pd.DataFrame(records)


def check_windows(recipe, archive, prefix, store):
    rec = read(recipe)
    cols = rec["tensor"]["columns"]
    names = [c["store_column"] for c in cols]
    sc = rec["scaler"]
    offsets = np.array(rec["tensor"]["window"]["slot_offsets_rows"])
    if (names != sc["features"] or any(c["consumer_ops"] for c in cols)
            or offsets.tolist() != list(range(-24, 1, 2)) or len(names) != 61):
        raise ValueError("not the archived 61-channel recipe")
    with np.load(archive, allow_pickle=False) as arc:
        ts = pd.to_datetime(arc["ts"], utc=True)
        if not ts.is_unique or not ts.is_monotonic_increasing:
            raise ValueError("decision times")
        keep = ts.year < 2024
        windows = arc["windows"][keep]
        ts = ts[keep]
    if windows.shape != (len(ts), 13, 61) or set(ts.year) != set(range(2016, 2024)):
        raise ValueError("coverage")
    frames = [pd.read_parquet(prefix, columns=names),
              pd.read_parquet(store, columns=["timestamp", *names],
                filters=[("timestamp", "<", pd.Timestamp("2024-01-01", tz="UTC"))]
                ).set_index("timestamp")]
    result = {}
    for mask, frame in zip((ts.year < 2020, ts.year >= 2020), frames):
        close = pd.DatetimeIndex(frame.index) + pd.Timedelta(hours=1)
        if not close.is_unique or not close.is_monotonic_increasing:
            raise ValueError("source ordering")
        stamps = ts[mask]
        ends = close.get_indexer(stamps)
        slots = ends[:, None] + offsets
        if slots.min() < 0 or (close[slots.ravel()].asi8.reshape(slots.shape)
                                  > stamps.asi8[:, None]).any():
            raise ValueError("unmatched or future source")
        raw = frame[names].to_numpy(dtype=np.float64)
        z = ((raw - sc["mean"]) / sc["std"]).astype(np.float32)
        expected = z[slots]
        got = windows[mask]
        for year in sorted(set(stamps.year)):
            idx = stamps.year == year
            error = np.abs(expected[idx] - got[idx])
            if not np.isfinite(error).all():
                raise ValueError("nonfinite archived input")
            result[int(year)] = dict(windows=int(idx.sum()), differing=int(np.count_nonzero(error)),
                                     max_abs=float(error.max()))
    return dict(years=result, scaler_fit=sc["fit_period"],
                learned_inputs={c["store_column"]: c["upstream"]["fit_scope"]["fit_window"]
                                for c in cols if c["upstream"].get("fit_scope")},
                sha256={k:digest(v) for k,v in dict(recipe=recipe, archive=archive,
                                                    prefix=prefix, store=store).items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("runs", "recipe", "archive", "prefix", "store", "out"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    trials = runs(a.runs)
    evidence = check_windows(a.recipe, a.archive, a.prefix, a.store)
    a.out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(a.out / "trials.csv", index=False)
    (a.out / "windows.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(evidence["years"])
    if any(v["differing"] for v in evidence["years"].values()):
        raise SystemExit("archived windows differ from independent reconstruction")
