"""Construct a TRAIN-only eight-year market, then replay the actual frozen actor."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from app.frozen_recipe_inputs import FrozenRecipeInputs, file_sha
from app.selected_input_market import market_for
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment, evaluate
from app.frozen_signal_masks import prepared_signals
from tools.account395 import stitch_market, take
from tools.prepare_g3_joint import synthetic_market as market_from_frames

read=lambda p:json.loads(p.read_text())


def assemble(parent, bundle, data):
    rec=read(bundle/"RECIPE.json");report=read(data/"RESULT.json");spec=read(parent/"SPEC.json")
    for name,h in report["outputs"].items():
        if file_sha(data/name)!=h: raise ValueError(name)
    if report["source_hashes"]["recipe"]!=file_sha(bundle/"RECIPE.json") or not report["real_only"]:
        raise ValueError("source identity")
    real=market_for(parent,spec,rec)
    from app.short_horizon.expert_guided.data import calendar_from_closures
    cfg=rec["business"]["calendar"]
    cal=calendar_from_closures(read(data/"CLOSURES.json"),cfg["entry_delay_after_open_hours"],
        cfg["force_flat_before_close_hours"],cfg["entry_cutoff_before_flat_hours"],report["arm"])
    bars=pd.read_parquet(data/"OHLC.parquet"); features=pd.read_parquet(data/"FEATURES.parquet")
    if not bars.index.equals(features.index) or not (bars.index.year<2020).all(): raise ValueError("prefix")
    left=market_from_frames(real,bars.rename_axis("timestamp"),features.rename_axis("timestamp"))
    from app.short_horizon.zigzag import causal_sigma, causal_range_sigma
    warm=pd.read_parquet(data/"WARMUP_OHLC.parquet")
    if len(warm)!=512 or warm.index[-1]>=bars.index[0]: raise ValueError("risk warmup")
    history=pd.concat([warm,bars])
    left.sigma=causal_sigma(history)[len(warm):]
    left.sigma_rule=causal_range_sigma(history)[len(warm):]
    if not all(np.isfinite(v).all() and (v>0).all() for v in (left.sigma,left.sigma_rule)):
        raise ValueError("risk scale")
    x=left.inputs
    period=np.full(len(x.valid),"TRAIN",dtype=object);period.flags.writeable=False
    left.inputs=FrozenRecipeInputs(x.recipe,x.row_ts_ns,x.slot_ts_ns,x.windows,x.valid,period)
    right=copy.copy(real)
    hi=pd.Timestamp("2024-01-01",tz="UTC")
    mask=real.bars.index+pd.Timedelta(hours=1)<hi
    right.bars=real.bars[mask];right.sigma=real.sigma[mask];right.sigma_rule=real.sigma_rule[mask]
    right.inputs=take(real.inputs,real.inputs.row_ts_ns<hi.value)
    joined=stitch_market(left,right,cal)
    if not (np.all(joined.inputs.period=="TRAIN") and joined.inputs.row_ts_ns.max()<hi.value
            and np.array_equal(right.inputs.valid,joined.inputs.valid[-len(right.inputs.valid):])):
        raise ValueError("split")
    return joined,real


def run(a):
    import torch
    torch.set_num_threads(1)
    model,_=load_bundle(a.bundle,"cpu")
    joined,real=assemble(a.parent,a.bundle,a.data)
    risk=read(a.parent/"RISK.json")
    ready=read(a.bundle.parent/"READY.json")
    if ready["start"]!=file_sha(a.bundle/"model.pt"):raise ValueError("model identity")
    logits,_=emit(model,joined.inputs);pre=prepared_signals(logits)
    report={}
    for key,interval in (("control",["2020-01-01","2024-01-01"]),("eight",["2016-01-01","2024-01-01"])):
        env=environment(joined,risk,6,interval,model.effective["gamma"])
        metrics=evaluate(env,logits,prepared=pre)
        report[key]={k:metrics[k] for k in ("return_pct","trades")}
        if key=="control":
            ref=ready["initial"]["TRAIN"]
            if metrics["trades"]!=ref["trades"] or abs(metrics["return_pct"]-ref["return_pct"])>1e-5:
                raise ValueError(("sealed control not reproduced",metrics["trades"],metrics["return_pct"],ref))
        else:
            equity=np.asarray(env.equity_trace)
            cutoff=pd.Timestamp("2020-01-01",tz="UTC").value
            ix=np.searchsorted(equity[:,0],cutoff)
            if not (0<ix<len(equity) and env.finished and np.all(np.diff(equity[:,0])>0)):
                raise ValueError("incomplete chronological trace")
            report["seam_equity"]=equity[ix-1:ix+1,1].tolist()
            report["equity_points"]=len(equity)
    report.update(trained=False,prefix_origin="historical",bars=len(joined.bars),
                  prefix_valid_windows=int(joined.inputs.valid[joined.inputs.row_ts_ns<pd.Timestamp("2020",tz="UTC").value].sum()),
                  validation_included=False,model_sha256=ready["start"])
    a.out.mkdir(exist_ok=False,parents=True)
    (a.out/"RESULT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("parent","bundle","data","out"):
        p.add_argument("--"+k,type=Path,required=True)
    run(p.parse_args())
