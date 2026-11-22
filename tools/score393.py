"""Re-extract retrospective pivots independently of the generator and scripted labels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pandas as pd
from app.pivot_segment_generator import chunks, match
from app.zigzag_opportunities import engine


def run(a):
    engine(a.engine)
    src=subprocess.check_output(["git","-C",str(a.source),"cat-file","blob",
                                 "1aaea5cb2262130cf4ce8ff7c87a745ccbe82290"])
    if hashlib.sha256(src).hexdigest()!="c06592f6478f3887213e985a80de5823dd953d3887ce25bec0272197860affda":
        raise ValueError("unexpected ZigZag")
    z=types.ModuleType("app.short_horizon.zz393")
    z.__package__="app.short_horizon"
    sys.modules[z.__name__]=z
    exec(compile(src,"frozen_zigzag","exec"),z.__dict__)
    with np.load(a.bank) as f:
        ts=pd.to_datetime(f["ts"],utc=True)
        real=pd.DataFrame(f["real"],index=ts,columns=["OPEN","HIGH","LOW","CLOSE"])
    script=pd.read_parquet(a.root/"macro/SCRIPT.parquet").set_index("start_row")
    out=[]
    for path in sorted(a.root.glob("*-*/path-*/REPORT.json")):
        rep=json.loads(path.read_text())
        bars=pd.read_parquet(path.parent/"OHLC.parquet")
        trace=pd.read_csv(path.parent/"trace.csv")
        offset=int(trace.script_start.iloc[0])
        frame=pd.concat([real[real.index<bars.index[0]],bars])
        origin=len(frame)-len(bars)-1
        lo,hi=list(chunks(frame.index))[-1]
        p=z.hindsight_pivots(frame.HIGH.values[lo:hi],frame.LOW.values[lo:hi],
                             z.Params(depth=4,backstep=2,deviation_pips=5))
        got=[(lo+i-origin,k) for i,k,_ in p if 0<lo+i-origin<len(bars)]
        want=[(r["start"]-offset+int(script.loc[r["start"],"bars"]),int(script.loc[r["start"],"side"]))
              for r in rep["complete_legs"] if r["start"]-offset+int(script.loc[r["start"],"bars"])<len(bars)]
        errs=[abs(r["desired_bps"]-r["generated_bps"]) for r in rep["complete_legs"]]
        out.append(dict(arm=path.parent.parent.name,path=path.parent.name,wanted=len(want),found=len(got),
                        exact=match([want],[got],0),near=match([want],[got],1),sum_abs_bps=sum(errs),
                        legs=len(errs)))
    if len(out)!=16:
        raise ValueError("incomplete paired panel")
    df=pd.DataFrame(out)
    df.to_csv(a.root/"PIVOTS.csv",index=False)
    panel=df.groupby("arm").sum(numeric_only=True)
    panel["mae_bps"]=panel.sum_abs_bps/panel.legs
    panel.to_csv(a.root/"SUMMARY.csv")
    print(panel.to_string())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["root","engine","source","bank"]:
        p.add_argument("--"+key,type=Path,required=True)
    run(p.parse_args())
