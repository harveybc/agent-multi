"""One-action counterfactuals on the frozen 369 TRAIN panel, not new policy training."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tools.run_frozen_critic_repair import (cached, configure, digest, file_sha, load_bundle,
    market_for, read, support_hash, verify_files)


def grad(prob, utility):
    return prob*(utility-(prob*utility).sum(axis=-1,keepdims=True))


def run(a):
    configure(369)
    cfg=read(a.panel/'CONFIG.json'); identity=digest(cfg)
    assert cfg['source']==file_sha(a.bundle/'model.pt') and cfg['parent']==file_sha(a.parent/'SPEC.json')
    spec=read(a.parent/'SPEC.json')
    verify_files(a.parent,{k:v for k,v in spec['files'].items() if k.startswith(('data/','engine/')) or k in ('SUPPORT_RECIPE.json','RISK.json')})
    model,_=load_bundle(a.bundle,'cpu')
    market=market_for(a.parent,spec,model.recipe)
    assert support_hash(market.inputs)==cfg['support']
    panel=cached(a.panel/'PANEL.json',identity)['anchors']
    records=[cached(a.panel/'rows'/f"{s['row']}.json",identity) for s in panel]
    assert [r['anchor'] for r in records]==panel
    assert all(s['year']<=2023 for s in panel)
    x=torch.as_tensor(market.inputs.windows[[s['row'] for s in panel]].copy())
    with torch.no_grad():
        z=model.surfaces(x).double()
        logp=z.log_softmax(-1).numpy(); p=np.exp(logp)
        q=model.q_surfaces(x).min(0).values.double().numpy()
        alpha=float(model.loss.log_alpha.exp())
    def target(name):
        return np.array([[o[name] for o in r['greedy']] for r in records])
    def mc(name):
        return np.array([[[o[name] for o in r['stochastic'][str(s)]] for s in cfg['tapes']] for r in records])
    cash=target('net_usd'); reward=target('reward_return')
    tapes=mc('soft_return'); soft=tapes.mean(1); sampled=mc('reward_return').mean(1)
    assert np.isfinite([cash,reward,soft,sampled,q,p]).all()
    sac_grad=grad(p,q-alpha*logp)
    raw=z.clone().requires_grad_(True)
    autograd=torch.autograd.grad((raw.softmax(-1)*(torch.tensor(q)-alpha*raw.log_softmax(-1))).sum(),raw)[0]
    np.testing.assert_allclose(sac_grad,autograd.numpy(),atol=1e-10)
    ref=cached(a.panel/'BASELINE.json',identity)
    results=[]; geometry=[]; sensitivity=[]
    for split in ('fit','monitor'):
        mask=np.array([s['role']==split for s in panel])
        for i, tape in enumerate(cfg['tapes']):
            act=np.delete(tapes,i,axis=1).mean(1).argmax(1)
            delta=cash.max(1)-cash[np.arange(len(act)),act]
            sensitivity.append(dict(split=split,omitted=tape,n=int(mask.sum()),
                mean_cash_regret_usd=float(delta[mask].mean()),suboptimal_cash=int((delta[mask]>1e-8).sum())))
        for policy,score in [('actor',p),('minQ',q),('MC_soft',soft),('MC_sampled_reward',sampled),
                             ('greedy_reward',reward),('greedy_cash',cash)]:
            act=score.argmax(1)
            regrets={}
            for name,truth in [('soft',soft),('greedy_reward',reward),('cash_usd',cash)]:
                delta=truth.max(1)-truth[np.arange(len(act)),act]
                regrets['mean_regret_'+name]=float(delta[mask].mean())
                regrets['suboptimal_'+name]=int((delta[mask]>1e-8).sum())
            if policy=='minQ':
                np.testing.assert_allclose(regrets['mean_regret_soft'],ref['metrics']['soft'][split]['all']['mean_regret'],atol=1e-6)
            if split=='monitor' and policy=='actor':
                np.testing.assert_allclose(regrets['mean_regret_greedy_reward'],ref['controls']['greedy']['actor']['mean_regret'],atol=1e-6)
            results.append(dict(split=split,policy=policy,n=int(mask.sum()),**regrets))
        for name,u in [('MC_soft_including_first_entropy',soft-alpha*logp),
                       ('greedy_continuation_cash',cash)]:
            g=grad(p,u); dot=(g*sac_grad).sum(1)
            norm=np.linalg.norm(g,axis=1)*np.linalg.norm(sac_grad,axis=1)
            valid=mask & (norm>1e-10)
            cos=dot[valid]/norm[valid]
            geometry.append(dict(split=split,benchmark=name,informative=len(cos),
                                 negative=int((cos<0).sum()),median_cos=float(np.median(cos))))
    a.out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('choices',results),('gradients',geometry),('leave_one_tape_out',sensitivity)]:
        with (a.out/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps([r for r in results+geometry if r['split']=='monitor'],indent=1))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('parent','bundle','panel','out'):
        parser.add_argument(key,type=Path)
    run(parser.parse_args())
