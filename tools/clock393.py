"""Audit historical H1 timestamps; agreement is not proof of exchange opening hours."""
import argparse
import json
from pathlib import Path

import pandas as pd


def grid(path,key):
    times=pd.DatetimeIndex(pd.read_parquet(path,columns=[key])[key])
    if times.tz is None:
        raise ValueError("missing time zone")
    times=times.tz_convert("UTC")
    times=times[(times.year>=2016)&(times.year<2020)]
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("unordered or duplicate bars")
    return times


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["clock","spec","out"]:
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    ref=grid(a.clock,"Time")
    times={"eurusd":ref}
    for sym,path in json.loads(a.spec.read_text())["external"]["paths"]["fx"].items():
        times[sym]=grid(path,"timestamp")
    votes=pd.concat({s:pd.Series(True,index=i,dtype=bool) for s,i in times.items()},axis=1,sort=True)
    votes=votes.eq(True)
    diff=votes[~votes.all(axis=1)]
    a.out.mkdir(exist_ok=False,parents=True)
    diff.to_csv(a.out/"CLOCK_DISAGREEMENTS.csv",index_label="utc")
    print(json.dumps(dict(status="observed_clock_not_certified",eurusd_rows=len(ref),
                          disputed_rows=len(diff),count_by_symbol=votes.sum().to_dict())))
