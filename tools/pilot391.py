"""Four-epoch paired hierarchical augmentation pilot. Never promotes a policy."""
import argparse
import copy
import fcntl
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from app.frozen_recipe_inputs import file_sha
from app.g3_producer_audit import identity
from app.rl_subset import subset_recipe
from app.selected_input_market import market_for
from app.selected_warm_start import target_surfaces
from app.structural_refit import build_model, export_bundle, load_bundle, verify_files
from app.three_signal_learning import environment, emit
from tools.augmentation389 import PairedReplay, read, write
from tools.prepare_g3_joint import synthetic_market
from tools.run_discrete_curve_probe import configure
from tools.run_structural_sac_refit import assess_full, fit_epochs

COHORT = 'c3fb74587de1ba319e1caa6cb011f3471a64e1eb9a05b3be5401291bba45cc5b'
SOURCE = '69b5254489d224d4b20ac358956e4dd3b9a67a2f214a9aff98045c5def25a536'


def trainable(name):
    return name.startswith(('actor_network_params.','qvalue_network_params.')) and any(
        s in name for s in ('.head.','.encoder.core.','.encoder.post_fusion.'))


def project(source, cols, probe):
    rec = subset_recipe(source.recipe,cols)
    eff = copy.deepcopy(source.effective)
    eff['architecture']['fixed_fusion_offset'] = True
    dest = build_model(rec,eff)
    orig = source.recipe['representation']['resolved']
    names = rec['representation']['resolved']['names']
    keep = [orig['names'].index(n) for n in names]
    if any(any(source.recipe['list']['columns'][j] not in cols for j in orig['indices'][i]) for i in keep):
        raise ValueError('391_EXPECT_WHOLE_GROUPS')
    width = eff['architecture']['branch_width']
    ix = [i*width+j for i in keep for j in range(width)]
    src = source.loss.state_dict()
    with torch.no_grad():
        for key,v in dest.loss.state_dict().items():
            if re.search(r'\.encoder\.indices_\d+$',key) or key.endswith('fixed_fusion_offset'): continue
            origin = re.sub(r'(\.encoder\.branches\.)(\d+)\.',lambda m:m[1]+str(keep[int(m[2])])+'.',key)
            w = src[origin]
            if key.endswith('.encoder.fusion.conv.weight'):
                w = w.index_select(-2,torch.tensor(ix))
            if w.shape!=v.shape:raise ValueError(key)
            v.copy_(w)
        for family,count in [('actor_network_params',1),('qvalue_network_params',2),('target_qvalue_network_params',2)]:
            actor=count==1
            for i in range(count):
                par=getattr(source.loss,family); par=par if actor else par[i]
                with par.to_module(source.actor if actor else source.loss.qvalue_network):
                    enc=(source.actor_net if actor else source.q_net).encoder
                    z=[b(torch.zeros(1,len(indices),enc.shape[0])) for b,indices in zip(enc.branches,orig['indices'])]
                    for j in keep:z[j].zero_()
                    v=torch.nn.functional.conv1d(torch.cat(z,1),enc.fusion.conv.weight)[0]
                key=next(k for k in dest.loss.state_dict() if k.startswith(family+'.') and k.endswith('fixed_fusion_offset'))
                buf=dest.loss.state_dict()[key]
                (buf if actor else buf[i]).copy_(v)
        old=dict(source.loss.named_parameters())
        for n,p in dest.loss.named_parameters():
            p.requires_grad_(trainable(n))
            if trainable(n) and old[n] in source.optimizer.state:
                dest.optimizer.state[p]=copy.deepcopy(source.optimizer.state[old[n]])
        for old_group,new_group in zip(source.optimizer.param_groups,dest.optimizer.param_groups,strict=True):
            new_group.update({k:v for k,v in old_group.items() if k!='params'})
        sel=[source.recipe['list']['columns'].index(c) for c in cols]
        px=torch.tensor(probe); full=torch.zeros(len(px),px.shape[1],len(source.recipe['list']['columns']))
        full[:,:,sel]=px
        err={}
        for family,func in [('actor',lambda m,x:m.surfaces(x)),('twins',lambda m,x:m.q_surfaces(x)),
                            ('target_twins',target_surfaces)]:
            one,two=func(source,full),func(dest,px)
            torch.testing.assert_close(one,two,rtol=1e-5,atol=1e-5)
            if not torch.equal(one.argmax(-1),two.argmax(-1)):raise ValueError(family+'_decision')
            err[family]=float((one-two).abs().max())
    return dest,err


def prepare(a):
    banks=a.data/'banks';manifest=read(banks/'COHORTS.json')
    if file_sha(banks/'COHORTS.json')!=COHORT or file_sha(a.source/'model.pt')!=SOURCE:raise ValueError('pins')
    mode='price' if a.width==61 else 'context'
    paths=[banks/(mode+str(s)) for s in manifest['seeds']['8']]
    for path in paths:
        row=manifest['paths'][path.name]
        if row['physics_failed'] or row['n_columns']!=a.width:raise ValueError(path)
        verify_files(path,row['sha256'])
    spec=read(a.parent/'SPEC.json');risk=read(a.parent/'RISK.json')
    source,_=load_bundle(a.source,'cpu')
    cols=list(pd.read_parquet(paths[0]/f'FEATURES{a.width}.parquet').columns)
    recipe=subset_recipe(source.recipe,cols) if a.width==61 else source.recipe
    market=market_for(a.parent,spec,recipe)
    take=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))[::157]
    probe=market.inputs.windows[take].copy()
    dest=a.root/str(a.width);dest.mkdir(parents=True,exist_ok=True)
    if a.width==61:
        model,parity=project(source,cols,probe)
    else:model,parity=source,{}
    for n,p in model.loss.named_parameters():p.requires_grad_(trainable(n))
    path=dest/'bundle'
    if not path.exists():export_bundle(path,model,probe[:32],dict(kind='391_paired_start',source=SOURCE))
    model,_=load_bundle(path,'cpu')
    initial=assess_full(model,market,risk,spec['design'])
    if a.width==71 and (abs(initial['VALIDATION']['return_pct']-18.36994575)>1e-5 or initial['VALIDATION']['trades']!=207):
        raise ValueError('incumbent mismatch')
    ids=[]
    for path in paths:
        out=dest/path.name;out.mkdir(exist_ok=True)
        m=synthetic_market(market,pd.read_parquet(path/'OHLC.parquet'),pd.read_parquet(path/f'FEATURES{a.width}.parquet'))
        logits,_=emit(model,m.inputs)
        env=environment(m,risk,6,recipe['split']['periods']['TRAIN'],model.effective['gamma'])
        obs=env.reset(); rng=np.random.default_rng(391);xs=[];meta=[]
        while not env.finished:
            v=logits[env.rows[env.bar]].astype(float);p=np.exp(v-v.max());action=int(rng.choice(3,p=p/p.sum()))
            xs.append(obs['features']);obs,reward,done,steps=env.step(action)
            meta.append((action,reward,steps,done))
            if done:xs.append(obs['features'])
        if len(xs)!=len(meta)+1 or sum(row[-1] for row in meta)!=1 or not meta[-1][-1]:raise ValueError('terminal')
        np.save(out/'X.npy',np.stack(xs));np.save(out/'META.npy',np.asarray(meta));del xs,meta
        ids.append(dict(name=path.name,files={n:file_sha(out/n) for n in ('X.npy','META.npy')}))
    write(dest/'READY.json',dict(width=a.width,banks=ids,source=SOURCE,cohort=COHORT,parity=parity,
        start=file_sha(dest/'bundle/model.pt'),initial=initial,risk=file_sha(a.parent/'RISK.json'),
        spec=file_sha(a.parent/'SPEC.json'),support=identity(market.inputs.row_ts_ns[market.inputs.valid].tolist())))


def train(a):
    if not torch.cuda.is_available():raise ValueError('GPU required')
    configure(7);torch.backends.cudnn.allow_tf32=False
    folder=a.root/str(a.width);rec=read(folder/'READY.json')
    if (rec['source']!=SOURCE or rec['cohort']!=COHORT or rec['start']!=file_sha(folder/'bundle/model.pt')
        or rec['spec']!=file_sha(a.parent/'SPEC.json') or rec['risk']!=file_sha(a.parent/'RISK.json')):raise ValueError('391_binding')
    model,_=load_bundle(folder/'bundle','cuda');spec=read(a.parent/'SPEC.json')
    verify_files(a.parent,{k:v for k,v in spec['files'].items() if k.startswith(('data/','engine/')) or k in ('RISK.json','SUPPORT_RECIPE.json')})
    market=market_for(a.parent,spec,model.recipe);risk=read(a.parent/'RISK.json')
    if rec['support']!=identity(market.inputs.row_ts_ns[market.inputs.valid].tolist()):raise ValueError('support')
    data=[]
    if a.dose:
        for d in rec['banks']:
            path=folder/d['name'];verify_files(path,d['files'])
            data.append(tuple(np.load(path/n,mmap_mode='r') for n in ('X.npy','META.npy')))
    design=dict(spec['design'],max_epochs=4,matched_budget=True,initialize_epoch_zero=True,
                allow_ineligible_epoch_zero=True,head_only=False)
    key=identity(dict(ready=file_sha(folder/'READY.json'),code=file_sha(__file__),dose=a.dose,design=design))
    out=folder/('mix' if a.dose else 'real');out.mkdir(exist_ok=True)
    with (out/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(out/'CONTRACT.json',dict(start=rec['start'],support=rec['support'],seed=7,epochs=4,dose=a.dose,key=key))
        if (out/'RESULT.json').exists():return
        def assess(m):
            metrics=assess_full(m,market,risk,design)
            if not (out/'EPOCH_ZERO.json').exists():
                for split,old in rec['initial'].items():
                    if (metrics[split]['trades']!=old['trades'] or abs(metrics[split]['return_pct']-old['return_pct'])>1e-5):
                        raise ValueError(('391_epoch_zero',split,old['return_pct'],metrics[split]['return_pct']))
            return metrics
        ix=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))[::761]
        state=fit_epochs(model,lambda:environment(market,risk,6,model.recipe['split']['periods']['TRAIN'],model.effective['gamma']),
            assess,design,7,out,key,market.inputs.windows[ix].copy(),replay_factory=lambda real:PairedReplay(real,data,207,design['batch_size']))
        write(out/'RESULT.json',dict(epochs=state['epoch'],updates=state['updates'],best=state['best_bundle'],promoted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','train'])
    parser.add_argument('--width',type=int,choices=[61,71],required=True)
    parser.add_argument('--dose',type=int,choices=[0,10],default=0)
    for k in ('root','parent','source','data'):
        parser.add_argument('--'+k,type=Path,required=k in ('root','parent'))
    a=parser.parse_args()
    (train if a.mode=='train' else prepare)(a)
