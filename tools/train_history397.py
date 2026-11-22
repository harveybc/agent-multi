"""Single-account historical counterpart to the four-epoch 391 pilot."""
import argparse
import fcntl
from pathlib import Path

import numpy as np
import torch

from tools import pilot391 as p
from tools.account397 import assemble
from app.three_signal_learning import evaluate
from app.frozen_signal_masks import prepared_signals


def run(a):
    bundle=a.pilot/"bundle"
    receipt, proof, spec=(p.read(a.pilot/"READY.json"),p.read(a.proof/"RESULT.json"),p.read(a.parent/"SPEC.json"))
    if (receipt["start"]!=proof["model_sha256"] or receipt["start"]!=p.file_sha(bundle/"model.pt")
        or receipt["risk"]!=p.file_sha(a.parent/"RISK.json") or receipt["spec"]!=p.file_sha(a.parent/"SPEC.json")
        or proof["trained"] or proof["validation_included"]):
        raise ValueError("custody")
    p.verify_files(a.parent,{k:v for k,v in spec["files"].items() if k.startswith(("data/","engine/")) or k=="SUPPORT_RECIPE.json"})
    p.configure(7)
    device="cuda" if a.execute else "cpu"
    if a.execute and not torch.cuda.is_available():raise ValueError("GPU required")
    torch.backends.cudnn.allow_tf32=False
    model,_=p.load_bundle(bundle,device)
    train,val=assemble(a.parent,bundle,a.data)
    design=dict(spec["design"],max_epochs=4,matched_budget=True,initialize_epoch_zero=True,
                allow_ineligible_epoch_zero=True,head_only=False)
    risk=p.read(a.parent/"RISK.json")
    train_period=["2016-01-01","2024-01-01"];val_period=model.recipe["split"]["periods"]["VALIDATION"]
    if val_period!=["2024-01-01","2025-01-01"]:raise ValueError("validation years")
    def env(m,interval):
        return p.environment(m,risk,design["horizon"],interval,model.effective["gamma"])
    def assess(candidate):
        result={}
        for key,market,interval in (("TRAIN",train,train_period),("VALIDATION",val,val_period)):
            logits,_=p.emit(candidate,market.inputs)
            result[key]=evaluate(env(market,interval),logits,prepared=prepared_signals(logits))
        return result
    zero=assess(model)
    for key,ref in (("TRAIN",proof["eight"]),("VALIDATION",receipt["initial"]["VALIDATION"])):
        if zero[key]["trades"]!=ref["trades"] or abs(zero[key]["return_pct"]-ref["return_pct"])>1e-5:
            raise ValueError(("epoch zero",key))
    binding=dict(model=receipt["start"],proof=p.file_sha(a.proof/"RESULT.json"),data=p.file_sha(a.data/"RESULT.json"),
                 risk=receipt["risk"],spec=receipt["spec"],train=train_period,validation=val_period,
                 epochs=4,seed=7,accounts_per_epoch=1,replay_quota=None,design=design,
                 code={k:p.file_sha(k) for k in (__file__,assemble.__code__.co_filename,p.fit_epochs.__code__.co_filename)})
    a.out.mkdir(exist_ok=True,parents=True)
    with (a.out/"lock").open("a") as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key=p.identity(binding)
        dest=a.out/"CONTRACT.json"
        if dest.exists() and p.read(dest)!=binding:raise ValueError("contract changed")
        p.write(dest,binding)
        if not a.execute:
            p.write(a.out/"PREFLIGHT.json",dict(identity=key,metrics=zero,training=False))
            return
        if (a.out/"RESULT.json").exists():return
        ix=np.flatnonzero(train.inputs.valid)[::761]
        state=p.fit_epochs(model,lambda:env(train,train_period),assess,design,7,a.out,key,train.inputs.windows[ix].copy())
        p.write(a.out/"RESULT.json",dict(identity=key,epochs=state["epoch"],updates=state["updates"],promoted=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for k in ("pilot","parent","proof","data","out"):
        parser.add_argument("--"+k,type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    run(parser.parse_args())
