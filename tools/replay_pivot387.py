"""Native account smoke on each TRAIN scenario, not an augmentation-training result."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment,emit,evaluate
from tools.prepare_g3_joint import synthetic_market
from tools.audit_g3_producers import write


def run(a):
    def read(p):return json.loads(p.read_text())
    manifest=read(a.data/'COHORTS.json')
    if not manifest['release_for_training'] or file_sha(a.bundle/'model.pt')!=manifest['source']:
        raise ValueError('scenario/source identity')
    model,_=load_bundle(a.bundle,'cpu')
    original=market_for(a.parent,read(a.parent/'SPEC.json'),model.recipe)
    risk=read(a.parent/'RISK.json')
    interval=model.recipe['split']['periods']['TRAIN']
    def replay(m):
        idx=np.flatnonzero(m.inputs.valid & (m.inputs.period=='TRAIN'))
        logits,_=emit(model,m.inputs,idx)
        env=environment(m,risk,6,interval,model.effective['gamma'])
        metrics=evaluate(env,logits,prepared=prepared_signals(logits))
        keys={(int(env.engine.v[t['entry_bar_abs']]),int(t['direction'])) for t in env.engine.trades}
        return metrics,keys
    real,rkeys=replay(original)
    if real['trades']!=745 or abs(real['return_pct']-16.05822355)>1e-6:raise ValueError('incumbent did not reproduce')
    rows={'real':real}
    for seed in manifest['seeds']['16']:
        path=a.data/str(seed)
        if file_sha(path/'RESULT.json')!=manifest['receipts'][str(seed)]:raise ValueError('path receipt')
        verify_files(path,read(path/'RESULT.json')['sha256'])
        m=synthetic_market(original,pd.read_parquet(path/'OHLC.parquet'),pd.read_parquet(path/'FEATURES71.parquet'))
        metrics,keys=replay(m)
        rows[str(seed)]=dict(metrics,new_entry_time_direction=len(keys-rkeys))
    write(a.data/'NATIVE_TRAIN_REPLAY.json',rows)
    print([(k,v['trades'],v['return_pct']) for k,v in rows.items()],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('parent','bundle','data'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
