"""Resumable CPU materializer for the declared 61-input synthetic clock; no SAC release."""
import argparse
import fcntl
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tools.audit_g3_producers import authenticate, producers, rebuild, sha
from tools.probe393_diffusion import Decoder
from tools.rollout393 import sample


def write(path, value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    tmp.replace(path)


def run(a):
    torch.set_num_threads(1)
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/"lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binding={str(p):sha(p) for p in (Path(__file__),a.recipe,a.spec,a.factor,a.checkpoint,a.quotes,a.bank,
                a.calendar/"CLOCK.parquet",a.calendar/"SCRIPT.parquet",a.calendar/"CLOSURES.json",
                Path(sample.__code__.co_filename),Path(Decoder.forward.__code__.co_filename))}
    cfg=a.out/"BINDING.json"
    if cfg.exists() and json.loads(cfg.read_text())!=binding:
        raise ValueError("run identity changed")
    write(cfg,binding)
    spec=json.loads(a.spec.read_text())
    if sha(a.spec)!="a37df7503445d0eddff3207b08e6f9f5c424d4f64fef1b8dd309472f15d95813":
        raise ValueError("producer contract")
    sett=spec["producer_settings"]
    spec["pins"][str(a.factor)]=spec["pins"].pop(sett["factor_state"])
    sett["factor_state"]=str(a.factor)
    authenticate(spec)
    loaded=producers(spec)
    state=json.loads(Path(sett["scope"]).read_text())
    closures=json.loads((a.calendar/"CLOSURES.json").read_text())
    recipe=json.loads(a.recipe.read_text())
    cols=recipe["list"]["columns"]
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    if len(cols)!=61 or ck["dims"][0]!=61 or ck["objective"]!="velocity" or not ck["bridge"]:
        raise ValueError("wrong arm")
    model=Decoder(*ck["dims"]); model.load_state_dict(ck["state"]); model.eval()
    clock=pd.DatetimeIndex(pd.read_parquet(a.calendar/"CLOCK.parquet").timestamp).as_unit("ns")
    script=pd.read_parquet(a.calendar/"SCRIPT.parquet")
    cutoff=pd.Timestamp("2016-01-01",tz="UTC")
    seed=pd.read_parquet(a.quotes,columns=["Time","Open","High","Low","Close"],
                         filters=[("Time","<",cutoff)]).set_index("Time").tail(512)
    seed.columns=["OPEN","HIGH","LOW","CLOSE"];seed.index=seed.index.as_unit("ns").rename("timestamp")
    if not (len(seed)==512 and seed.index[-1]<cutoff<=clock[0] and clock.is_unique and
            clock.is_monotonic_increasing and script.bars.sum()==len(clock) and
            np.array_equal(clock[script.start_row],script.start)):
        raise ValueError("calendar/script/seed")
    def full(f): return rebuild(f,cols,state,closures,loaded)
    frame=seed.copy()
    hist=full(frame)
    offsets=np.asarray(recipe["tensor"]["window"]["slot_offsets_rows"])
    mu,sd=np.asarray(recipe["scaler"]["mean"]),np.asarray(recipe["scaler"]["std"])
    starts=script.start_row.to_numpy()
    with np.load(a.bank) as f:
        ts=pd.to_datetime(f["ts"],utc=True); real=f["real"]
    donors=np.flatnonzero((ts[1:]-ts[:-1]>=pd.Timedelta(hours=40)) & (ts[1:].year<2023))+1
    rng=torch.Generator().manual_seed(41)
    gap_rng=np.random.default_rng(41)
    done=0
    for part in sorted(a.out.glob("part-[0-9]*")):
        rec=json.loads((part/"STATE.json").read_text())
        prices,features=(pd.read_parquet(part/name) for name in ("OHLC.parquet","FEATURES.parquet"))
        if (rec["start"]!=done or not prices.index.equals(clock[done:rec["end"]]) or
            not prices.index.equals(features.index) or list(features.columns)!=cols or
            any(sha(part/n)!=h for n,h in rec["hashes"].items())):
            raise ValueError("corrupt resume")
        done=rec["end"]
        frame=pd.concat([frame,prices]);hist=pd.concat([hist,features])
        rng.set_state(torch.tensor(rec["torch_rng"],dtype=torch.uint8))
        gap_rng.bit_generator.state=rec["gap_rng"]
    base=done
    begin=time.monotonic()
    while done<len(clock):
        end=min(done+a.block,len(clock))
        # Local frames are discarded on failure. Commit only after an independent
        # full-prefix producer pass has checked every emitted channel in the block.
        rng_before=rng.get_state(); gap_before=gap_rng.bit_generator.state
        for canonical in (False,True):
            rng.set_state(rng_before);gap_rng.bit_generator.state=gap_before
            bars=frame.copy(); recent=hist.tail(25).copy(); emitted=[]; reopens=[]
            for i in range(done,end):
                t=clock[i]; p=script.iloc[np.searchsorted(starts,i,side="right")-1]
                s=int(p.start_row); left=int(p.desired_bars)-(i-s)
                anchor=bars.iloc[len(seed)-1+s]
                dist=np.log((anchor.LOW if p.side>0 else anchor.HIGH)/bars.CLOSE.iloc[-1])+p.displacement
                rate=dist/left
                ctl=np.r_[np.eye(ck["dims"][1]-10)[int(p.regime)],p.side,p.desired_bars,p.displacement,
                          p.vol,p.retrace,p.retrace_at,p.curvature,1-left/p.desired_bars,dist,rate]
                c=(torch.tensor(ctl,dtype=torch.float32)-ck["cm"])/ck["cs"]
                x=(recent.iloc[len(recent)-1+offsets].to_numpy()-mu)/sd
                if not np.isfinite(x).all(): raise ValueError(f"missing history {i}")
                y=sample(model,ck,torch.tensor(x[None],dtype=torch.float32),c[None],rng)
                if t-bars.index[-1]>pd.Timedelta(hours=1):
                    j=int(gap_rng.choice(donors)); y[0]=np.log(real[j,0]/real[j-1,3])
                    reopens.append([i,str(ts[j])])
                o=bars.CLOSE.iloc[-1]*np.exp(y[0]); z=o*np.exp(y[1]+rate)
                candle=[o,max(o,z)*np.exp(y[2]),min(o,z)*np.exp(-y[3]),z]
                if not np.isfinite(candle).all() or min(candle)<=0: raise ValueError(f"invalid bar {i}")
                bars.loc[t]=candle
                last=full(bars if canonical else bars.tail(512)).tail(1)
                emitted.append(last);recent=pd.concat([recent.tail(24),last])
            candidate=pd.concat(emitted)
            reference=full(bars).iloc[-len(emitted):]
            err=np.abs(reference.to_numpy()-candidate.to_numpy()).max()
            if np.isfinite(err) and err<=1e-9: break
        else: raise ValueError("full-prefix producer parity")
        part=Path(tempfile.mkdtemp(prefix=".pending-",dir=a.out))
        bars.iloc[-len(candidate):].to_parquet(part/"OHLC.parquet")
        candidate.to_parquet(part/"FEATURES.parquet")
        write(part/"STATE.json",dict(start=done,end=end,torch_rng=rng.get_state().tolist(),
              gap_rng=gap_rng.bit_generator.state,reopens=reopens,max_delta=float(err),canonical=canonical,
              hashes={n:sha(part/n) for n in ("OHLC.parquet","FEATURES.parquet")}))
        part.rename(a.out/f"part-{done:06d}")
        frame=bars; hist=pd.concat([hist,candidate]); done=end
        progress=dict(rows=done,scheduled_rows=len(clock),last=str(clock[done-1]),sac_release=False,
                      last_block_delta=float(err),seconds_per_row=(time.monotonic()-begin)/(done-base))
        write(a.out/"PROGRESS.json",progress)
        print(json.dumps(progress),flush=True)
        if a.stop_after and done-base>=a.stop_after: break
    if done==len(clock): write(a.out/"CALENDAR_LIMIT.json",dict(last=str(clock[-1]),rows=done,sac_release=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("spec","factor","recipe","checkpoint","bank","calendar","quotes","out"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--block",type=int,default=120)
    p.add_argument("--stop-after",type=int,default=0)
    a=p.parse_args()
    if a.block<1 or a.stop_after<0: p.error("invalid limits")
    run(a)
