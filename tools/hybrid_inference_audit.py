"""Finite paired inference-only audit of categorical policies; no model fitting."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from app.frozen_recipe_inputs import file_sha, require
from tools.recipe_hybrid_queue import Market, authenticated, verify, configure
from tools.recipe_sac_queue import read, write, now, evaluate, clean


def timestamp_uniform(seed,bar_timestamp_ns):
    digest=hashlib.sha256(f'{seed}:{bar_timestamp_ns}'.encode('ascii')).digest()
    return (int.from_bytes(digest[:8],'big')>>11)/2**53


def frozen_probabilities(probabilities,mask):
    p=np.asarray(probabilities,dtype=np.float64)
    m=np.asarray(mask,dtype=bool)
    require(p.shape==m.shape==(5,) and np.isfinite(p).all() and (p>=0).all(),'PROBABILITY_VECTOR')
    p=np.where(m,p,0.)
    require(p.sum()>0,'EMPTY_MASKED_PROBABILITY')
    return p/p.sum()


def categorical_command(p,u):
    p=frozen_probabilities(p,np.ones(5,bool))
    require(0<=u<1,'UNIFORM_RANGE')
    cdf=np.cumsum(p);cdf[-1]=1.
    return int(np.searchsorted(cdf,u,side='right'))


class ObservedMarket:
    def __init__(self,market):
        self.market=market;self.recipe=market.recipe;self.active=None

    def env(self,period):
        self.active=self.market.env(period)
        return self.active


class InferencePolicy:
    def __init__(self,model,market,mode,seed,constants):
        self.model,self.market,self.mode,self.seed,self.constants=model,market,mode,seed,constants
        self.policy=model.policy
        self.policy.set_training_mode(False)

    def predict(self,obs,deterministic=True):
        if self.mode=='ARGMAX':
            return self.model.predict(obs,deterministic=True)
        mask=obs['action_mask']
        if self.mode=='BLIND_TRAIN_MARGINAL':
            route='long' if mask[3] else 'short' if mask[4] else 'flat'
            p=frozen_probabilities(self.constants[route],mask)
        else:
            import torch
            tensor,_=self.policy.obs_to_tensor(obs)
            with torch.no_grad():
                p=self.policy.get_distribution(tensor).cat.probs[0].cpu().numpy()
            p=frozen_probabilities(p,mask)
        e=self.market.active
        u=timestamp_uniform(self.seed,int(e.engine.v[e.bar]))
        command=categorical_command(p,u)
        return np.array([command,.5,.5,.5],np.float32),None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve();audit=a.audit.resolve()
    require(not out.is_relative_to(root) and not out.is_relative_to(audit),'OUTPUT_ISOLATION')
    spec=verify(root);sha=file_sha(root/'SPEC.private.json')
    require(read(root/'CLOSE.json')['state']=='COMPLETE','PARENT_CLOSED')
    seal=read(audit/'CLOSE.json')
    require(seal['state']=='COMPLETE' and seal['spec_sha256']==sha,'AUDIT_PARENT')
    for name,digest in seal['artifacts'].items():
        path=(audit/name).resolve()
        require(path.is_relative_to(audit) and file_sha(path)==digest,'AUDIT_ARTIFACT')
    design=dict(training_seeds=[7,19,43],evaluation_seeds=[101,202,303],
                modes=['ARGMAX','SAMPLE','BLIND_TRAIN_MARGINAL'],periods=['TRAIN','INNER_A'],
                arm='CAT_FIXED',checkpoint=102400,fits=0,expert=False,test=False,
                randomization='sha256(eval_seed:bar_start_ns), common across policies',
                constants='mean probabilities on frozen TRAIN common support by route',
                decision='diagnostic only; no checkpoint, seed or strategy selection')
    identity=dict(design=design,parent_spec_sha256=sha,audit_seal_sha256=file_sha(audit/'CLOSE.json'),
                  tool_sha256=file_sha(Path(__file__)))
    if (out/'IDENTITY.json').exists():require(read(out/'IDENTITY.json')==identity,'AUDIT_IDENTITY')
    else:write(out/'IDENTITY.json',identity)
    identity_sha=file_sha(out/'IDENTITY.json')
    configure()
    from stable_baselines3 import PPO
    market=ObservedMarket(Market(spec,'CAT_FIXED'))
    status=read(root/'STATUS.json');completed=[];started=time.monotonic()
    for seed in design['training_seeds']:
        unit=f'CAT_FIXED_s{seed}';path=root/status['completed'][unit]
        parent=authenticated(path,sha)
        audit_unit=read(audit/unit/'AUDIT.json')
        require(audit_unit['result_sha256']==file_sha(path),'AUDIT_UNIT_PIN')
        checkpoint=audit_unit['probes'][-1]
        archive=path.parent/'model-102400.zip'
        require(checkpoint['step']==102400 and checkpoint['archive_sha256']==file_sha(archive),'AUDIT_CHECKPOINT')
        constants={r['route']:r['probability_mean'] for r in checkpoint['periods']['TRAIN']}
        model=PPO.load(archive,device='cuda')
        for mode in design['modes']:
            for eval_seed in ([None] if mode=='ARGMAX' else design['evaluation_seeds']):
                key=f'{unit}-{mode}-{eval_seed}'
                folder=out/key;result=folder/'RESULT.json'
                if result.exists():
                    r=read(result);require(r['identity_sha256']==identity_sha,'RESULT_IDENTITY')
                    for n,d in r['artifacts'].items():
                        require(Path(n).name==n and file_sha(folder/n)==d,'RESULT_ARTIFACT')
                    completed.append(str(result.relative_to(out)));continue
                reports=[]
                wrapper=InferencePolicy(model,market,mode,eval_seed,constants)
                for period in design['periods']:
                    write(out/'STATUS.json',dict(state='EVALUATING',unit=key,period=period,
                        complete=len(completed),total=21,heartbeat_at=now(),seconds=time.monotonic()-started))
                    report=evaluate(wrapper,market,period,folder)
                    if mode=='ARGMAX':
                        require(file_sha(folder/(period+'.trades.json'))==file_sha(path.parent/(period+'.trades.json')),
                                'DETERMINISTIC_TRADES_REPLAY')
                        source=next(r for r in parent['reports'] if r['period']==period)
                        for field in ('net_profit_usd','sharpe_daily_sqrt365','max_drawdown_pct','trades','win_rate'):
                            require(report[field]==source[field],'DETERMINISTIC_METRIC_REPLAY')
                    report.update(training_seed=seed,evaluation_seed=eval_seed,inference_mode=mode,
                                  unit=key,scope='INFERENCE_DIAGNOSTIC_NOT_SELECTED_STRATEGY')
                    write(folder/(period+'.json'),report);reports.append(report)
                write(result,dict(state='COMPLETE',identity_sha256=identity_sha,source_result_sha256=file_sha(path),
                    checkpoint_sha256=file_sha(archive),reports=reports,
                    artifacts={p.name:file_sha(p) for p in folder.iterdir() if p.is_file()},finished_at=now()))
                completed.append(str(result.relative_to(out)))
                print(key,'complete',round(time.monotonic()-started,1),flush=True)
    write(out/'SUMMARY.json',clean([read(out/name) for name in completed]))
    write(out/'STATUS.json',dict(state='COMPLETE',units=len(completed),seconds=time.monotonic()-started,finished_at=now()))
    write(out/'CLOSE.json',dict(state='COMPLETE',identity_sha256=identity_sha,
        artifacts={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*.json') if p.name!='CLOSE.json'},
        finished_at=now()))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
