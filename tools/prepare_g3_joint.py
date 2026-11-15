"""Materialize and qualify one conditional joint synthetic TRAIN path."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from app.g3_joint_sampler import ConditionalJointSampler,distribution_gate,innovations
from app.g3_producer_audit import identity,parity,support_report
from tools.audit_g3_producers import authenticate,load_external,producers,rebuild,write
from tools.recover_stl_price_warmup import sha


def condition_frame(frame,external):
    cross = external['cross']; params = external['cfg']['params']
    parts = []
    for name,series in external['daily'].items():
        features = cross.daily_features(series,name,external['cfg']['daily'][name]['kind'])
        names = {'us10y':['dly.us10y.chg_1d','dly.us10y.chg_5d'],
                 'spx':['dly.spx.ret_1d'],'dax':['dly.dax.ret_5d']}[name]
        parts.append(cross.asof_join(frame.index,features[names],cross.daily_available(features.index,
            params['daily_lag_days'],params['release_offset_hours'][name]),
            pd.Timedelta(days=params['daily_max_stale_days'])))
    f = cross.cot_features(external['cot'])[['cot.eur.net_long','cot.eur.net_share','cot.eur.net_z156']]
    parts.append(cross.asof_join(frame.index,f,cross.cot_available(f.index,params['release_offset_hours']['cot']),
                                pd.Timedelta(days=params['cot_max_stale_days'])))
    return pd.concat(parts,axis=1)


def synthetic_market(original,bars,features):
    from app.frozen_recipe_inputs import FrozenRecipeInputs
    from app.short_horizon.zigzag import causal_sigma,causal_range_sigma
    market = copy.copy(original)
    market.bars = bars.copy()
    market.inputs = FrozenRecipeInputs.from_frame(original.recipe,features.reset_index())
    market.sigma = causal_sigma(bars)
    market.sigma_rule = causal_range_sigma(bars)
    return market


def run(spec_path,parent_root,bundle,output):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return locked_run(spec_path,parent_root,bundle,output)


def locked_run(spec_path,parent_root,bundle,output):
    started = time.monotonic()
    spec = json.loads(spec_path.read_text()); authenticate(spec)
    source = Path(__file__).resolve().parents[1]
    cfg = dict(schema='g3_conditional_joint.v1',audit_spec_sha256=sha(spec_path),
        parent_spec_sha256=sha(parent_root/'SPEC.json'),recipe_sha256=sha(bundle/'RECIPE.json'),
        seed=19,neighbors=10,block_rows=4,warmup_rows=256,
        code={p:sha(source/p) for p in ('app/g3_joint_sampler.py','tools/prepare_g3_joint.py',
            'tools/audit_g3_producers.py','app/g3_producer_audit.py')})
    cfg['identity'] = identity(cfg)
    config_path = output/'CONFIG.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != cfg:
            raise ValueError('G3_GENERATION_RESUME_IDENTITY')
        if (output/'DONE.json').exists():
            done = json.loads((output/'DONE.json').read_text())
            if any(sha(output/k)!=v for k,v in done['files'].items()):
                raise ValueError('G3_GENERATION_TERMINAL_DIGEST')
            return json.loads((output/'RESULT.json').read_text())
    else:
        if any(p.name!='worker.lock' for p in output.iterdir()):
            raise ValueError('G3_NEW_GENERATION_ROOT')
        write(config_path,cfg)
    def emit(stage,**kw):
        row=dict(stage=stage,elapsed_seconds=time.monotonic()-started,**kw)
        write(output/'PROGRESS.json',row);print(json.dumps(row),flush=True)
    recipe=json.loads((bundle/'RECIPE.json').read_text()); cols=recipe['list']['columns']
    if sha(bundle/'RECIPE.json') != sha(spec['recipe']):
        raise ValueError('G3_GENERATION_RECIPE')
    end=pd.Timestamp(spec['train_end'])
    frame=pd.read_parquet(spec['producer_settings']['bars'],filters=[('timestamp','<',end)]).set_index('timestamp')
    frame=frame[['BidOpen','BidHigh','BidLow','BidClose']].rename(columns={
        'BidOpen':'OPEN','BidHigh':'HIGH','BidLow':'LOW','BidClose':'CLOSE'})
    external=load_external(spec);loaded=producers(spec)
    state=json.loads(Path(spec['producer_settings']['scope']).read_text())
    closures=json.loads(Path(spec['closures']).read_text())
    original=pd.read_parquet(spec['producer_settings']['store'],columns=['timestamp']+cols,
                             filters=[('timestamp','<',end)]).set_index('timestamp')
    emit('ORIGINAL71_REPLAY')
    baseline=rebuild(frame,cols,state,closures,loaded,external)
    check=parity(baseline,original)
    if not all(r['passed'] for r in check.values()):raise ValueError('G3_BASELINE_CHANGED')
    crosses=pd.DataFrame({k:v.reindex(frame.index) for k,v in external['fx'].items()},index=frame.index)
    macro=condition_frame(frame,external)
    sampler=ConditionalJointSampler(cfg['neighbors'],cfg['block_rows'],cfg['warmup_rows'])
    sampler.fit(frame,crosses,macro,end)
    emit('GENERATING_JOINT_TRAIN',donors=int(sampler.valid.sum()))
    bars,fx,donor=sampler.generate(frame,crosses,macro,cfg['seed'])
    joblib.dump(sampler,output/'SAMPLER.joblib')
    restored=joblib.load(output/'SAMPLER.joblib')
    prefix,pfx,pdonor=restored.generate(frame.iloc[:1024],crosses.iloc[:1024],macro.iloc[:1024],cfg['seed'])
    if not (np.array_equal(prefix,bars.iloc[:1024]) and np.array_equal(pfx,fx.iloc[:1024],equal_nan=True)
            and np.array_equal(pdonor,donor[:1024])):raise ValueError('G3_GENERATOR_COLD_PREFIX')
    real=innovations(frame,crosses)[:,[0,4,5,6,7,8,9]]
    syn=innovations(bars,fx)[:,[0,4,5,6,7,8,9]]
    distribution=distribution_gate(real,syn,frame.index)
    emit('RECOMPUTING_SYNTHETIC71',distribution_passed=distribution['passed'])
    generated=dict(external,fx={k:fx[k] for k in fx})
    features=rebuild(bars,cols,state,closures,loaded,generated)
    features.index.name=bars.index.name='timestamp'
    prefix_features=rebuild(bars.iloc[:1024].copy(),cols,state,closures,loaded,generated)
    temporal=parity(prefix_features,features.iloc[:1024])
    if not all(r['exact'] for r in temporal.values()):raise ValueError('G3_SYNTHETIC_FEATURE_PREFIX')
    bars.reset_index().to_parquet(output/'SYNTHETIC_BARS.parquet',index=False)
    features.reset_index().to_parquet(output/'SYNTHETIC_FEATURES.parquet',index=False)
    np.savez_compressed(output/'DONORS.npz',donor=donor,timestamp=bars.index.as_unit('ns').asi8)
    write(output/'DISTRIBUTION.json',distribution)
    # Native control accounts validate reward/accounting, not learned utility.
    from app.selected_input_market import market_for
    from app.three_signal_learning import environment,evaluate
    from app.frozen_signal_masks import prepared_signals
    original_market=market_for(parent_root,json.loads((parent_root/'SPEC.json').read_text()),recipe)
    market=synthetic_market(original_market,bars,features)
    risk=json.loads((parent_root/'RISK.json').read_text())
    native={}
    for name in ('NOP','BUY','SELL','ALTERNATE'):
        logits=np.zeros((len(market.inputs.valid),3),np.float32)
        if name=='ALTERNATE':
            actions=1+(np.arange(len(logits))//24)%2
        else:actions=np.full(len(logits),('NOP','BUY','SELL').index(name))
        logits[np.arange(len(logits)),actions]=1
        env=environment(market,risk,6,recipe['split']['periods']['TRAIN'],.99)
        native[name]=evaluate(env,logits,prepared=prepared_signals(logits))
    if native['NOP']['trades']!=0 or any(v['cash_error']>1e-6 for v in native.values()):
        raise ValueError('G3_NATIVE_REWARD_ACCOUNTING')
    generated_rows=int((donor>=0).sum())
    exact_original=int(np.sum((donor>=0)&(donor==np.arange(len(donor)))))
    report=dict(identity=cfg['identity'],status='GENERATOR_QUALIFIED' if distribution['passed'] else 'DISTRIBUTION_REJECTED',
        elapsed_seconds=time.monotonic()-started,selected_features=len(cols),generated_rows=generated_rows,
        same_index_donors=exact_original,original71_passed=True,cold_prefix_exact=True,
        synthetic71_prefix_exact=True,distribution_passed=distribution['passed'],
        synthetic_support=support_report(features,cols,recipe['tensor']['window']['slot_offsets_rows']),
        native_controls=native,training_started=False,model_promoted=False,stl_resumed=False,
        scope='Conditional FX augmentation given original TRAIN macro/calendar path; no new observed market information')
    authenticate(spec)
    write(output/'RESULT.json',report)
    write(output/'DONE.json',dict(identity=cfg['identity'],files={p.name:sha(p) for p in output.iterdir()
        if p.name in ('CONFIG.json','RESULT.json','DISTRIBUTION.json','SYNTHETIC_BARS.parquet',
                      'SYNTHETIC_FEATURES.parquet','SAMPLER.joblib','DONORS.npz')}))
    emit(report['status'],generated_rows=generated_rows)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','parent-root','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=run(a.spec,a.parent_root,a.bundle,a.output)
    raise SystemExit(0 if result['status']=='GENERATOR_QUALIFIED' else 2)
