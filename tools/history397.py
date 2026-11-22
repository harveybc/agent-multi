"""Build the historical extension for the 61-input arm, never inventing a bar."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.audit_g3_producers import authenticate, producers, rebuild, sha
from tools.calendar395 import LIMIT, START, schedule


def utc(s):
    return pd.Timestamp(s,tz="UTC")


def business(native):
    clock, prior = schedule()
    span = pd.date_range(LIMIT,utc("2020-01-01"),inclusive="left",freq="h")
    ny = span.tz_convert("America/New_York")
    active = (ny.weekday<4)|((ny.weekday==4)&(ny.hour<17))|((ny.weekday==6)&(ny.hour>=17))
    # Research-only: avoid uncertain hours entirely, not a claim of venue closure.
    exclusions = [("2018-12-24","2018-12-27"),("2018-12-31","2019-01-03"),
                  ("2019-12-24","2019-12-27"),("2019-12-31","2020-01-02")]
    holiday = np.zeros(len(span),bool)
    for lo,hi in exclusions:
        holiday |= (span>=utc(lo))&(span<utc(hi))
    allowed = clock.append(span[active & ~holiday])
    grid = pd.date_range(START,utc("2020-01-01"),inclusive="left",freq="h")
    closed = pd.Series(~grid.isin(allowed),index=grid)
    starts = closed.index[closed & ~closed.shift(fill_value=False)]
    ends = closed.index[~closed & closed.shift(fill_value=False)].append(pd.DatetimeIndex([utc("2020-01-01")]))
    intervals = [(a,b) for a,b in prior if b<=START] + list(zip(starts,ends,strict=True))
    intervals += [(pd.Timestamp(a),pd.Timestamp(b)) for a,b in native]
    merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]:
            merged[-1][1]=max(hi,merged[-1][1])
        else:
            merged.append([lo,hi])
    return allowed, [[a.isoformat(),b.isoformat()] for a,b in merged],exclusions


def fxcm(path):
    f=pd.read_csv(path)
    f.index=pd.to_datetime(f.DateTime,utc=True,format="%m/%d/%Y %H:%M:%S.%f")
    if not f.index.is_unique or not f.index.is_monotonic_increasing:
        raise ValueError("FXCM timestamp")
    return f[["BidOpen","BidHigh","BidLow","BidClose"]].set_axis(["OPEN","HIGH","LOW","CLOSE"],axis=1)


def verified_bar(t,minutes,hourly):
    part=minutes.loc[t:t+pd.Timedelta(minutes=59)]
    expected=pd.date_range(t,periods=60,freq="min")
    # A native H1 candle may have minutes without updates at weekly reopening.
    reopening = t.tz_convert("America/New_York")
    sparse_open = (reopening.weekday()==6 and reopening.hour==17 and len(part)>0
                   and part.index[0]==t and part.index[-1]==expected[-1]
                   and part.index.isin(expected).all())
    if not part.index.equals(expected) and not sparse_open:
        raise ValueError(f"incomplete source minutes: {t}")
    bar=np.array([part.OPEN.iloc[0],part.HIGH.max(),part.LOW.min(),part.CLOSE.iloc[-1]])
    np.testing.assert_array_equal(hourly.loc[t].to_numpy(),bar)
    return bar,len(part)


def run(a):
    read=lambda p:json.loads(p.read_text())
    cols=read(a.recipe)["list"]["columns"]
    spec=read(a.spec)
    sett=spec["producer_settings"]
    spec["pins"][str(a.factor)]=spec["pins"].pop(sett["factor_state"])
    sett["factor_state"]=str(a.factor)
    authenticate(spec)
    naive_clock,closed,policy=business(read(a.closures))
    q=pd.read_parquet(a.h1).set_index("Time").loc[:'2019']
    q=q[["Open","High","Low","Close"]].set_axis(["OPEN","HIGH","LOW","CLOSE"],axis=1)
    q.index=q.index.as_unit("ns").rename("timestamp")
    half=pd.read_parquet(a.m30,columns=["Time"]).Time
    if not half.is_unique or not q.index.is_unique:
        raise ValueError("duplicate source bars")
    counts=half.groupby(half.dt.floor("h")).size().reindex(naive_clock,fill_value=0)
    fix=counts.index[(counts!=2)|~counts.index.isin(q.index)]
    receipts=[]
    for t in fix:
        label={2016:"2016w35",2019:"2019w22"}[t.year]
        paths=[a.cache/f"fxcm397-{label}{suffix}.csv.gz" for suffix in ("","-h1")]
        bar,minute_count=verified_bar(t,*map(fxcm,paths))
        q.loc[t]=bar
        receipts.append(dict(time=t.isoformat(),minutes=minute_count,primary_halves=int(counts.loc[t]),
                             minute_sha256=sha(paths[0]),native_hour_sha256=sha(paths[1])))
    bars=q.loc[naive_clock].sort_index()
    if (not np.isfinite(bars).all().all() or not bars.index.equals(naive_clock)
        or not ((bars.LOW>0)&(bars.LOW<=bars.min(axis=1))&(bars.HIGH>=bars.max(axis=1))).all()):
        raise ValueError("bad bar geometry or coverage")
    pre=q.loc[q.index<naive_clock[0]].sort_index().tail(512)
    f=rebuild(pd.concat([pre,bars]),cols,read(Path(sett["scope"])),closed,producers(spec)).loc[bars.index]
    if len(cols)!=61 or list(f.columns)!=cols or not f.index.equals(bars.index):
        raise ValueError("producer contract")
    a.out.mkdir(exist_ok=False,parents=True)
    f.to_parquet(a.out/"FEATURES.parquet")
    bars.to_parquet(a.out/"OHLC.parquet")
    pre.to_parquet(a.out/"WARMUP_OHLC.parquet")
    (a.out/"CLOSURES.json").write_text(json.dumps(closed)+"\n")
    r=dict(arm="historical_61_conservative_holidays",training_launched=False,
           source_hashes={k:sha(getattr(a,k)) for k in ("h1","m30","recipe","spec","closures")},
           years=bars.groupby(bars.index.year).size().to_dict(),rows=len(f),repairs=receipts,
           real_only=True,all_finite=int(np.isfinite(f).all(axis=1).sum()),last=str(bars.index[-1]),
           research_blackouts=policy,quoted_excluded_by_research_blackouts=sum(int(((q.index>=utc(l))&(q.index<utc(h))).sum()) for l,h in policy),
           outputs={k:sha(a.out/k) for k in ("OHLC.parquet","WARMUP_OHLC.parquet","FEATURES.parquet","CLOSURES.json")})
    (a.out/"RESULT.json").write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("h1","m30","spec","recipe","factor","cache","closures","out"):
        p.add_argument("--"+k,type=Path,required=True)
    run(p.parse_args())
