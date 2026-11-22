"""Short free-running integration probe on an admitted seed grid, not the 4y prefix."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tools.audit_g3_producers import authenticate, load_external, producers, rebuild, sha
from tools.probe393_diffusion import Decoder


@torch.no_grad()
def sample(m, ck, x, c, rng):
    beta=ck["beta"]
    alpha=(1-beta).cumprod(0)
    y=torch.randn(1,len(ck["ym"]),generator=rng)
    for j in range(len(beta)-1,-1,-1):
        v=m(x,c,y,torch.tensor([j]))
        clean=alpha[j].sqrt()*y-(1-alpha[j]).sqrt()*v
        prev=alpha[j-1] if j else torch.tensor(1.)
        y=(prev.sqrt()*beta[j]*clean+(1-beta[j]).sqrt()*(1-prev)*y)/(1-alpha[j])
        if j:
            y+=(beta[j]*(1-prev)/(1-alpha[j])).sqrt()*torch.randn(y.shape,generator=rng)
    y=(ck["ym"]+ck["ys"]*y).numpy()[0].astype(float)
    y[2:4] **= 2
    return y


def run(a):
    torch.set_num_threads(1)
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    assert ck["objective"]=="velocity"
    m=Decoder(*ck["dims"])
    m.load_state_dict(ck["state"])
    m.eval()
    recipe=json.loads(a.recipe.read_text())
    cols=recipe["list"]["columns"]
    offsets=np.array(recipe["tensor"]["window"]["slot_offsets_rows"])
    spec=json.loads(a.spec.read_text())
    assert sha(a.spec)=="a37df7503445d0eddff3207b08e6f9f5c424d4f64fef1b8dd309472f15d95813"
    sett=spec["producer_settings"]
    spec["pins"][str(a.factor)]=spec["pins"].pop(sett["factor_state"])
    sett["factor_state"]=str(a.factor)
    authenticate(spec)
    loaded=producers(spec)
    state=json.loads(Path(sett["scope"]).read_text())
    closures=json.loads(Path(spec["closures"]).read_text())
    script=pd.read_parquet(a.script)
    offset=a.start_row
    if offset<0:
        change=int(np.flatnonzero(script.episode.diff().fillna(0))[0])
        offset=int(script.start_row.iloc[change-1])
    bridge=ck.get("bridge",False)
    k=ck["dims"][1]-8-2*int(bridge)
    with np.load(a.bank) as f:
        clock=pd.to_datetime(f["ts"],utc=True)
        ohlc=f["real"]
    window=a.hours
    origins=np.flatnonzero((clock[window:]-clock[:-window])==pd.Timedelta(hours=window))
    origin=int(origins[origins>=512][0])
    frame=pd.DataFrame(ohlc[:origin+1].copy(),index=clock[:origin+1],
                       columns=["OPEN","HIGH","LOW","CLOSE"])
    frame.index.name="timestamp"
    ext=load_external(spec) if len(cols)==71 else None
    if ext is not None:
        ext["fx"]={name:series.reindex(frame.index).copy() for name,series in ext["fx"].items()}
    rng=torch.Generator().manual_seed(a.seed)
    def reconstruct():
        return rebuild(frame,cols,state,closures,loaded,ext)
    feat=reconstruct()
    archived=pd.read_parquet(sett["store"],columns=["timestamp",*cols],
                             filters=[("timestamp","==",frame.index[-1])])
    np.testing.assert_allclose(archived[cols].iloc[0],feat.iloc[-1],rtol=0,atol=1e-9)
    rows=[]
    for i in range(window):
        at=offset+i
        p=script.iloc[int(np.searchsorted(script.start_row,at,side="right"))-1]
        phase=(at-p.start_row)/p.desired_bars
        c=np.r_[np.eye(k)[int(p.regime)],p.side,p.desired_bars,p.displacement,p.vol,p.retrace,
                p.retrace_at,p.curvature,phase]
        if bridge:
            ai=origin+int(p.start_row)-offset
            if ai<origin:
                raise ValueError("bridge must start at a segment boundary")
            anchor=frame.iloc[ai].LOW if p.side>0 else frame.iloc[ai].HIGH
            distance=np.log(anchor/frame.CLOSE.iloc[-1])+p.displacement
            rate=distance/(p.desired_bars-(at-p.start_row))
            c=np.r_[c,distance,rate]
        c=(torch.tensor(c,dtype=torch.float32)-ck["cm"])/ck["cs"]
        x=(feat.iloc[len(feat)-1+offsets].to_numpy()-recipe["scaler"]["mean"])/recipe["scaler"]["std"]
        if not np.isfinite(x).all():
            raise ValueError("nonfinite generated history; refusing imputation")
        y=sample(m,ck,torch.tensor(x[None],dtype=torch.float32),c[None],rng)
        if bridge:
            y[1]+=rate
        t=clock[origin+i+1]
        op=frame.CLOSE.iloc[-1]*np.exp(y[0])
        close=op*np.exp(y[1])
        frame.loc[t]=[op,max(op,close)*np.exp(y[2]),min(op,close)*np.exp(-y[3]),close]
        if ext is not None:
            for v,name in zip(y[4:],sorted(ext["fx"]),strict=True):
                fx=ext["fx"][name]
                fx.loc[t]=fx.iloc[-1]*np.exp(v)
        feat=reconstruct()
        if not np.isfinite(feat.iloc[-1]).all():
            raise ValueError("incomplete generated row")
        rows.append((t,int(p.start_row),int(p.regime),int(p.side),float(frame.CLOSE.iloc[-1])))
    a.out.mkdir(parents=True,exist_ok=False)
    frame.iloc[-window:].to_parquet(a.out/"OHLC.parquet")
    feat.iloc[-window:].to_parquet(a.out/"FEATURES.parquet")
    pd.DataFrame(rows,columns=["time","script_start","state","side","close"]).to_csv(a.out/"trace.csv",index=False)
    evidence=[]
    for p in script[(script.start_row>=offset)&(script.start_row+script.bars<=offset+window)].itertuples():
        l=origin+p.start_row-offset
        r=l+p.bars
        first,last=frame.iloc[l],frame.iloc[r]
        got=np.log(last.HIGH/first.LOW if p.side>0 else last.LOW/first.HIGH)
        evidence.append(dict(start=int(p.start_row),desired_bps=10000*p.displacement,
                             generated_bps=10000*got,
                             close_direction=int(np.sign(np.log(last.CLOSE/first.CLOSE)))))
    report=dict(stage="RECURSION_SMOKE_ONLY",width=len(cols),steps=len(rows),
                          segments=len(set(r[1] for r in rows)),regimes=len(set(r[2] for r in rows)),
                          complete_legs=evidence,
                          source="real prefix then generated FX and price; historical macro context")
    (a.out/"REPORT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["recipe","spec","bank","script","checkpoint","factor","out"]:
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--hours",type=int,default=24)
    p.add_argument("--start-row",type=int,default=0,help="negative: start one leg before first regime change")
    p.add_argument("--seed",type=int,default=41)
    run(p.parse_args())
