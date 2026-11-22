"""Build indexed conditional-decoder examples; no duplicated or padded history."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(recipe_path, spec, ts, bars, profiles, dest):
    recipe = json.loads(recipe_path.read_text())
    columns = recipe["list"]["columns"]
    if columns != recipe["scaler"]["features"] or any(
            c["consumer_ops"] for c in recipe["tensor"]["columns"]):
        raise ValueError("unsupported tensor recipe")
    offsets = np.asarray(recipe["tensor"]["window"]["slot_offsets_rows"])
    if offsets[-1] != 0 or np.any(offsets > 0):
        raise ValueError("history is not causal")
    raw = pd.read_parquet(spec["producer_settings"]["store"], columns=["timestamp", *columns],
                          filters=[("timestamp", ">=", ts[0]), ("timestamp", "<=", ts[-1])])
    x = raw.set_index("timestamp").loc[ts, columns].to_numpy()
    x = (x - np.asarray(recipe["scaler"]["mean"])) / np.asarray(recipe["scaler"]["std"])
    good = np.isfinite(x).all(1)
    a, b = profiles.start.to_numpy(), profiles.end.to_numpy()
    indices = np.concatenate([np.arange(i, j) for i, j in zip(a, b)])
    segment = np.repeat(np.arange(len(profiles)), b-a)
    if np.any(indices+offsets.min()<0) or len(set(indices)) != len(indices):
        raise ValueError("overlapping samples or missing warmup")
    keep = good[indices[:, None]+offsets].all(1)
    i, s = indices[keep], segment[keep]
    o, h, l, c = bars[i+1].T
    prev = bars[i, 3]
    if not (np.all(h>=np.maximum(o,c)) and np.all(l<=np.minimum(o,c)) and np.all(l>0)):
        raise ValueError("invalid OHLC")
    y = np.stack([np.log(o/prev), np.log(c/o), np.log(h/np.maximum(o,c)),
                  np.log(np.minimum(o,c)/l)], axis=1)
    restored_o = prev*np.exp(y[:,0])
    restored_c = restored_o*np.exp(y[:,1])
    restored = np.stack([restored_o,np.maximum(restored_o,restored_c)*np.exp(y[:,2]),
                         np.minimum(restored_o,restored_c)*np.exp(-y[:,3]),restored_c], axis=1)
    np.testing.assert_allclose(restored, bars[i+1], atol=1e-14, rtol=0)
    targets = ["log_open_over_prev_close","log_close_over_open","log_upper_wick","log_lower_wick"]
    if len(columns)==71:
        for name, path in sorted(spec["external"]["paths"]["fx"].items()):
            fx = pd.read_parquet(path,columns=["timestamp","close"]).set_index("timestamp").close.reindex(ts)
            r = np.log(fx).diff().to_numpy()[i+1]
            y = np.column_stack([y,r])
            targets.append(name+"_logret")
    eligible = np.isfinite(y).all(1)
    i, s, y = i[eligible], s[eligible], y[eligible]
    if ts[i+1].max().year>=2023 or not np.all(ts[i+1]-ts[i]==pd.Timedelta(hours=1)):
        raise ValueError("split violation or closure inside pivot leg")
    keys = ["state","side","n","displacement","vol","retrace","retrace_at","curvature"]
    side=profiles.side.to_numpy()[s]
    anchor=np.where(side>0,bars[a[s],2],bars[a[s],1])
    desired=anchor*np.exp(profiles.displacement.to_numpy()[s])
    distance=np.log(desired/bars[i,3])
    left=b[s]-i
    np.savez_compressed(dest/"EXAMPLES.npz", history_bank=x.astype("float32"),
                        clock_ns=ts.as_unit("ns").asi8,offsets=offsets,origin=i,segment=s,
                        controls=profiles[keys].to_numpy(),phase=(i-a[s])/(b[s]-a[s]),
                        targets=y,remaining=distance,left=left)
    return dict(columns=columns,targets=targets,controls=keys,eligible_steps=len(i),
                eligible_profiles=int(np.unique(s).size),excluded_history=int((~keep).sum()),
                excluded_targets=int((~eligible).sum()),last_target=str(ts[i+1].max()),
                reconstruction_error=float(abs(restored-bars[indices[keep]+1]).max()),
                recipe_hash=hashlib.sha256(recipe_path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ["spec","bank","profiles","out"]:
        parser.add_argument("--"+k, type=Path, required=True)
    parser.add_argument("--recipes",nargs="+",type=Path,required=True)
    args=parser.parse_args()
    spec=json.loads(args.spec.read_text())
    profiles=pd.read_parquet(args.profiles)
    with np.load(args.bank) as f:
        grid=pd.to_datetime(f["ts"],utc=True)
        mask=grid.year<2023
        ts,bars=grid[mask],f["real"][mask]
    if not np.array_equal(ts.as_unit("ns").asi8[profiles.start],profiles.start_ns):
        raise ValueError("profile index")
    args.out.mkdir(parents=True,exist_ok=False)
    for path in args.recipes:
        width=len(json.loads(path.read_text())["list"]["columns"])
        dest=args.out/str(width)
        dest.mkdir()
        report=prepare(path,spec,ts,bars,profiles,dest)
        (dest/"REPORT.json").write_text(json.dumps(report,indent=2)+"\n")
        print(width,{k:v for k,v in report.items() if k not in ("columns","controls","targets")})


if __name__ == "__main__":
    main()
