"""Materialize the first EURUSD historical-calendar synthetic partition (not a SAC release)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tools.audit_g3_producers import authenticate, producers, rebuild, sha
from tools.probe393_diffusion import Decoder
from tools.rollout393 import sample


def run(a):
    torch.set_num_threads(1)
    spec=json.loads(a.spec.read_text())
    if sha(a.spec)!="a37df7503445d0eddff3207b08e6f9f5c424d4f64fef1b8dd309472f15d95813":
        raise ValueError("spec identity")
    sett=spec["producer_settings"]
    spec["pins"][str(a.factor)]=spec["pins"].pop(sett["factor_state"])
    sett["factor_state"]=str(a.factor)
    authenticate(spec)
    prods=producers(spec)
    state=json.loads(Path(sett["scope"]).read_text())
    recipe=json.loads(a.recipe.read_text())
    cols=recipe["list"]["columns"]
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    if len(cols)!=61 or ck["dims"][0]!=61 or ck["objective"]!="velocity" or not ck["bridge"]:
        raise ValueError("wrong arm")
    model=Decoder(*ck["dims"]); model.load_state_dict(ck["state"]); model.eval()
    script=pd.read_parquet(a.script)
    cutoff=pd.Timestamp("2016-01-01",tz="UTC")
    seed=pd.read_parquet(a.quotes,columns=["Time","Open","High","Low","Close"],
                         filters=[("Time","<",cutoff)]).set_index("Time").tail(512)
    seed.columns=["OPEN","HIGH","LOW","CLOSE"]
    seed.index=seed.index.as_unit("ns").rename("timestamp")
    clock=pd.DatetimeIndex(pd.read_parquet(a.quotes,columns=["Time"],filters=[
        ("Time",">=",cutoff),("Time","<",pd.Timestamp("2020-01-01",tz="UTC"))]).Time).as_unit("ns")
    digest=hashlib.sha256(clock.asi8.tobytes()).hexdigest()
    if (len(seed)!=512 or digest!=json.loads((a.script.parent/"REPORT.json").read_text())["calendar_sha256"]
            or seed.index[-1]>=clock[0] or not np.array_equal(clock[script.start_row],script.start)):
        raise ValueError("seed or scenario calendar")
    # Use observed weekend boundaries on this provisional seed grid. Do not
    # turn feed outages into declared sessions or change the real SAC calendar.
    ix=seed.index.append(clock[:a.hours])
    weekends=[(l+pd.Timedelta(hours=1),r) for l,r in zip(ix[:-1],ix[1:]) if
              r-l>=pd.Timedelta(hours=40) and l.weekday() in (3,4) and r.weekday() in (6,0)]
    if not weekends or not seed.index[-1]<clock[0] or (clock[1:a.hours]-clock[:a.hours-1]!=pd.Timedelta(hours=1)).any():
        raise ValueError("this partition may not cross another reopening")
    # Include a closing boundary after this bounded partition.
    fri=clock[0].tz_convert("America/New_York").normalize()+pd.offsets.Week(weekday=4)+pd.Timedelta(hours=17)
    closures=[[l.isoformat(),r.isoformat()] for l,r in weekends]+[
        [fri.tz_convert("UTC").isoformat(),(fri+pd.Timedelta(days=2)).tz_convert("UTC").isoformat()]]
    offsets=np.asarray(recipe["tensor"]["window"]["slot_offsets_rows"])
    frame=seed.copy()
    features=rebuild(frame,cols,state,closures,prods)
    rng=torch.Generator().manual_seed(41)
    k=ck["dims"][1]-10
    starts=script.start_row.to_numpy()
    # Bank of actual TRAIN reopening OPEN/previous CLOSE gaps, used only on
    # the first reopening; ordinary intrabar gaps are not weekend gaps.
    with np.load(a.realbank) as f:
        ts=pd.to_datetime(f["ts"],utc=True)
        bars=f["real"]
    gaps=(ts[1:]-ts[:-1]>=pd.Timedelta(hours=40)) & (ts[1:].year<2023)
    pool=np.flatnonzero(gaps)+1
    source=int(np.random.default_rng(41).choice(pool))
    opening_gap=np.log(bars[source,0]/bars[source-1,3])
    for i,t in enumerate(clock[:a.hours]):
        p=script.iloc[np.searchsorted(starts,i,side="right")-1]
        left=int(p.desired_bars)-(i-int(p.start_row))
        anchor=frame.iloc[len(seed)-1+int(p.start_row)]
        distance=np.log((anchor.LOW if p.side>0 else anchor.HIGH)/frame.CLOSE.iloc[-1])+p.displacement
        rate=distance/left
        control=np.r_[np.eye(k)[int(p.regime)],p.side,p.desired_bars,p.displacement,p.vol,p.retrace,
                      p.retrace_at,p.curvature,1-left/p.desired_bars,distance,rate]
        c=(torch.tensor(control,dtype=torch.float32)-ck["cm"])/ck["cs"]
        x=(features.iloc[len(frame)-1+offsets].to_numpy()-recipe["scaler"]["mean"])/recipe["scaler"]["std"]
        if not np.isfinite(x).all():
            raise ValueError("incomplete recursive context")
        y=sample(model,ck,torch.tensor(x[None],dtype=torch.float32),c[None],rng)
        o=frame.CLOSE.iloc[-1]*np.exp(opening_gap if i==0 else y[0])
        z=o*np.exp(y[1]+rate)
        frame.loc[t]=[o,max(o,z)*np.exp(y[2]),min(o,z)*np.exp(-y[3]),z]
        features=rebuild(frame,cols,state,closures,prods)
    result=features.loc[clock[:a.hours]]
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("invalid prefix")
    a.out.mkdir(exist_ok=False,parents=True)
    result.to_parquet(a.out/"FEATURES61.parquet")
    frame.loc[result.index].to_parquet(a.out/"OHLC.parquet")
    report=dict(stage="PROVISIONAL_FIRST_PARTITION",sac_release=False,rows=len(result),cols=result.shape[1],
                first=str(result.index[0]),last=str(result.index[-1]),seed_end=str(seed.index[-1]),
                first_reopen_train_donor=str(ts[source]),clock_sha256=digest,
                provisional_closures=closures,files={p.name:sha(p) for p in
                (a.checkpoint,a.recipe,a.script,a.realbank,a.out/"OHLC.parquet",a.out/"FEATURES61.parquet")})
    (a.out/"REPORT.json").write_text(json.dumps(report,indent=2)+"\n")
    print({k:report[k] for k in ("stage","rows","cols","first","last","seed_end")})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("spec","recipe","script","quotes","realbank","checkpoint","factor","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--hours",type=int,default=120)
    args=parser.parse_args()
    if not 1<=args.hours<=120:
        parser.error("first observed trading week only")
    run(args)
