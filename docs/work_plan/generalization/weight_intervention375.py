"""Offline historical diagnostic; invoke with the repo on PYTHONPATH, CUDA hidden."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.sac_training_reliability import module_key
from app.structural_refit import build_model
from app.three_signal_learning import emit, evaluate, environment
from tools.run_nested_history_training import market_for,input_parity


def read(p):
    return json.loads(p.read_text())


def actor_candidate(root,c,months,early,mode):
    from tools.run_structural_sac_refit import assess_full
    r=root/f'seed_{c["seed"]}'/f'months_{months}_seed_{c["seed"]}'
    spec=read(r/'SPEC.json');history={x['epoch']:x for x in read(r/'HISTORY.json')}
    market=market_for(c,months);input_parity(c,months,market)
    model=build_model(market.recipe,spec['effective'],c['seed'],'cpu')
    epochs=[40,45,50,55,60] if mode=='average' else [early,60];states=[]
    for e in epochs:
        snap=history[e]['weight_snapshot'];path=r/snap['path']
        assert file_sha(path)==snap['sha256']
        states.append(torch.load(path,map_location='cpu',weights_only=True))
    state=dict(states[-1])
    for k,p in model.loss.named_parameters():
        if k.startswith('actor_network_params.') and p.requires_grad:
            if mode=='average':
                state[k]=torch.stack([s[k].double() for s in states]).mean(0).to(state[k])
            elif module_key(k) in ('actor.attention','actor.core'):
                state[k]=states[0][k]
    model.loss.load_state_dict(state)
    metrics=assess_full(model,market,read(Path(c['parent'])/'RISK.json'),spec['design'])
    (root/f'{mode}_{months}.json').write_text(json.dumps(dict(epochs=epochs,metrics=metrics),indent=2))
    print(months,mode,[(v['return_pct'],v['trades']) for v in metrics.values()],flush=True)


def run(root,c,months,early,late):
    r=root/f'seed_{c["seed"]}'/f'months_{months}_seed_{c["seed"]}'
    spec=read(r/'SPEC.json'); history={x['epoch']:x for x in read(r/'HISTORY.json')}
    assert spec['recipe']==file_sha(Path(c['stores'])/f'months_{months}/RECIPE.json')
    market=market_for(c,months); support=input_parity(c,months,market)
    assert support['support_sha256']==read(r/'QUALIFICATION.json')['support_sha256']
    model=build_model(market.recipe,spec['effective'],c['seed'],'cpu')
    params=dict(model.loss.named_parameters())
    groups={}
    for k in params:
        if k.startswith('actor_network_params.'):
            g=module_key(k).split('.')[1];groups.setdefault(g,[]).append(k)
    assert set(groups)=={'branch','attention','fusion','head','core'},set(groups)
    def snapshot(e):
        s=history[e]['weight_snapshot'];p=r/s['path']
        assert file_sha(p)==s['sha256']
        return torch.load(p,map_location='cpu',weights_only=True)
    E,L=snapshot(early),snapshot(late)
    risk=read(Path(c['parent'])/'RISK.json');results={};ref={}
    for base_epoch,base,donor in [(early,E,L),(late,L,E)]:
        for group in ['intact',*sorted(groups)]:
            state=dict(base); state.update({k:donor[k] for k in groups.get(group,[])})
            model.loss.load_state_dict(state,strict=True)
            logits,_=emit(model,market.inputs);prepared=prepared_signals(logits)
            if group=='intact':ref[base_epoch]=prepared[0].copy()
            metrics={}
            for period,interval in market.recipe['split']['periods'].items():
                m=evaluate(environment(market,risk,spec['design']['horizon'],interval,spec['effective']['gamma']),logits,prepared=prepared)
                if group=='intact':
                    orig=history[base_epoch]['metrics'][period]
                    for k in ['return_pct','trades','buy','sell','close_reasons','costs']:
                        assert m[k]==orig[k],(base_epoch,period,k,m[k],orig[k])
                    print('intact signal count delta CPU minus historical GPU',months,base_epoch,period,
                          [m['signals'].get(str(i),0)-orig['signals'].get(str(i),0) for i in range(3)],flush=True)
                ix=market.inputs.valid&(market.inputs.period==period)
                metrics[period]={k:m[k] for k in ('return_pct','net_profit_usd','sharpe','trades','orders_per_week','costs','close_reasons','buy','sell')}
                metrics[period]['decision_flip_fraction']=float(np.mean(prepared[0][ix]!=ref[base_epoch][ix]))
            results[f'{base_epoch}:{group}']=metrics
            print(months,base_epoch,group,*[(round(v['return_pct'],2),v['trades'],round(v['decision_flip_fraction'],3)) for v in metrics.values()],flush=True)
    return dict(epochs=[early,late],seed=c['seed'],months=months,hashes=[history[e]['weight_snapshot']['sha256'] for e in [early,late]],results=results)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('config',type=Path);ap.add_argument('months',type=int);ap.add_argument('early',type=int);ap.add_argument('late',type=int)
    ap.add_argument('--candidate',choices=['average','paired_rollback'])
    a=ap.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)
    if a.candidate:
        actor_candidate(a.root,read(a.config),a.months,a.early,a.candidate)
        raise SystemExit(0)
    result=run(a.root,read(a.config),a.months,a.early,a.late)
    (a.root/f'results_{a.months}.json').write_text(json.dumps(result,indent=2)+'\n')
