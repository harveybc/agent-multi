"""CPU-only counterfactuals on immutable 399 checkpoint snapshots; no model selection."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from tools import pilot391 as p
from tools.account397 import assemble
from app.frozen_signal_masks import prepared_signals
from app.three_signal_learning import evaluate
from app.selected_warm_start import target_surfaces


def main(root,work,arm,actor_device):
    p.configure(7)
    torch.backends.cudnn.allow_tf32=False
    base=root/'model/61/bundle'
    parent=Path('/var/tmp/sac70-warm-v3-20261108')
    m,_=p.load_bundle(base,actor_device)
    train,market=assemble(parent,base,root/'history397-61-r2')
    risk=p.read(parent/'RISK.json')
    cfg=m.recipe['split']['periods']['VALIDATION']
    folder=work/arm
    early=next((folder/'bundles').iterdir()); late=next((folder/'recovery').iterdir())
    p.verify_files(late,p.read(late/'MANIFEST.json')['files'])
    p.verify_files(early,p.read(early/'MANIFEST.json')['files'])
    early_s,late_s=[torch.load(path/'model.pt',map_location='cpu',weights_only=False)['loss'] for path in (early,late)]
    ve=p.environment(market,risk,6,cfg,m.effective['gamma'])
    ve.reset()
    ix=ve.rows[ve.bar:ve.engine.last][ve.valid[ve.bar:ve.engine.last]]
    report=dict(arm=arm,early=early.name,late=late.name,rows=len(ix),actor_device=actor_device,
                hashes={k:p.file_sha(v/'model.pt') for k,v in [('early',early),('late',late)]})
    groups=('head','core','post_fusion')
    def group(k): return next((g for g in groups if '.'+g+'.' in k),None)
    diff=[k for k in late_s if k.startswith('actor_network_params.') and not torch.equal(early_s[k],late_s[k])]
    if not all(group(k) for k in diff): raise ValueError('unaccounted actor differences')
    def summary(metrics):
        return {k:metrics[k] for k in ['return_pct','trades','signals','mean_signal_entropy','buy','sell','costs','close_reasons']}
    combo={}
    for bits in itertools.product([0,1],repeat=3):
        use={g for g,b in zip(groups,bits) if b}
        marker=''.join(map(str,bits))
        m.loss.load_state_dict({k:early_s[k] if k in diff and group(k) not in use else v for k,v in late_s.items()})
        logits,_=p.emit(m,market.inputs,indices=ix if actor_device=='cpu' else None)
        env=p.environment(market,risk,6,cfg,m.effective['gamma'])
        r=evaluate(env,logits,prepared=prepared_signals(logits))
        z=logits[ix]; prob=torch.from_numpy(z).softmax(-1).numpy()
        combo[marker]=dict(**summary(r),mean_probs=prob.mean(axis=0,dtype=np.float64).tolist(),
            median_nop_margin=float(np.median(z[:,0]-z[:,1:].max(axis=1))),
            gross=sum(t['gross_usd'] for t in env.engine.trades))
        if marker in ('000','111'):
            ep=int((early if marker=='000' else late).name.split('_')[-1])
            ref=p.read(folder/'HISTORY.json')[ep-1]['metrics']['VALIDATION']
            if r['signals']!=ref['signals'] or abs(r['return_pct']-ref['return_pct'])>1e-7:
                raise ValueError(('endpoint replay mismatch',marker,r['signals'],ref['signals']))
    report['late_modules_head_core_attention']=combo
    if actor_device!='cpu': m,_=p.load_bundle(base,'cpu')
    m.load(late/'model.pt')
    learning=torch.load(late/'learning.pt',map_location='cpu',weights_only=False)
    storage=learning['replay']['_storage']
    d=storage['_storage']; n=storage['_len']
    ids=np.flatnonzero(train.inputs.valid) if arm.startswith('8') else np.flatnonzero(
        market.inputs.valid & (market.inputs.period=='TRAIN'))
    windows=(train if arm.startswith('8') else market).inputs.windows
    actions=d['action'][:n].reshape(-1);rewards=d['next.reward'][:n].reshape(-1)
    report['replay']=dict(n=n,actions=actions.bincount(minlength=3).tolist(),
        nonzero=int(rewards.count_nonzero()),max_abs_reward=float(rewards.abs().max()),
        alpha=float(m.loss._alpha),log_alpha_trainable=m.loss.log_alpha.requires_grad)
    def batch(seed):
        idx=np.random.default_rng(seed).choice(n,128,replace=False)
        def obs(k):
            return torch.tensor(windows[ids[d[k][idx].reshape(-1).numpy()]])
        td=TensorDict(dict(features=obs('features'),mask=d['mask'][idx],action=d['action'][idx],
                steps_to_next_obs=d['steps_to_next_obs'][idx]),[len(idx)])
        td['next']=TensorDict(dict(features=obs('next.features'),mask=d['next.mask'][idx],
            reward=d['next.reward'][idx],done=d['next.done'][idx],terminated=d['next.terminated'][idx]),[len(idx)])
        return td
    fixed=batch(57)
    def qprobe():
        with torch.no_grad():
            q=m.q_surfaces(fixed['features']); target=m.loss._compute_target(fixed)
            pr=m.surfaces(fixed['features']).softmax(-1)
            return dict(q=q.min(0).values.mean(0).tolist(),td_target=float(target.mean()),
                p_nop=float(pr[:,0].mean()),q_argmax=q.min(0).values.argmax(-1).bincount(minlength=3).tolist(),
                target_twins=target_surfaces(m,fixed['features']).mean(dim=1).tolist())
    report['early_late_q']={}
    for path in (early,late):
        m.load(path/'model.pt')
        report['early_late_q'][path.name]=qprobe()
    if arm=='8y-s19':
        trials={}
        for condition in ['original','no_critic_dropout','no_bootstrap']:
            m.load(late/'model.pt')
            dp=[mod for mod in m.q_net.encoder.core.modules() if isinstance(mod,torch.nn.Dropout)]
            assert dp and all(mod.p==.1 for mod in dp)
            if condition=='no_critic_dropout':
                for mod in dp: mod.p=0.
            before=qprobe()
            losses=[]
            for i in range(64):
                td=batch(411+i)
                if condition=='no_bootstrap':
                    td['next','done'].fill_(True);td['next','terminated'].fill_(True)
                torch.manual_seed(731+i)
                losses.append(m.update(td))
            trials[condition]=dict(before=before,after=qprobe(),losses={k:np.mean([r[k] for r in losses]) for k in losses[0]})
            for mod in dp: mod.p=.1
        report['offline_paired_64_updates']=trials
    dest=folder/'DIAG.json'
    dest.write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('arm')
    ap.add_argument('--actor-device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--root',type=Path,default=Path('/var/tmp/sac399-series'))
    ap.add_argument('--work',type=Path,default=Path('/var/tmp/diag400'))
    args=ap.parse_args()
    main(args.root,args.work,args.arm,args.actor_device)
