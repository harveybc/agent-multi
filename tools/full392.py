"""Train on a full real four-year account AND one full synthetic four-year account."""
import argparse
import fcntl
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from app.g3_replay import restore_buffer
from tools import pilot391 as prior
from tools.prepare_g3_joint import synthetic_market


class TwoAccounts:
    def __init__(self, real, synthetic):
        self.accounts=(real,synthetic)
        self.engine=SimpleNamespace(last=sum(e.engine.last for e in self.accounts))

    def reset(self):
        self.current,self.last_source=0,None
        self.n=[0,0]
        self.obs=[env.reset() for env in self.accounts]
        self.expected=[int(env.valid[env.bar:env.engine.last].sum()) for env in self.accounts]
        return self.obs[0]

    @property
    def bar(self):
        return sum(e.bar for e in self.accounts)

    @property
    def finished(self):
        return all(e.finished for e in self.accounts)

    def step(self, act):
        i=self.current
        assert not self.accounts[i].finished
        nxt,reward,done,steps=self.accounts[i].step(act)
        self.n[i]+=1
        self.obs[i]=nxt
        self.last_source=i
        if not self.accounts[1-i].finished:self.current=1-i
        if self.finished and self.n!=self.expected:raise ValueError('incomplete traversal')
        return nxt,reward,done,steps

    def coverage(self):
        return dict(real=self.n[0],synthetic=self.n[1],expected_real=self.expected[0],expected_synthetic=self.expected[1])

    def collection_observation(self):
        return self.obs[self.current]


class Balanced:
    _storage=None
    def __init__(self,buffer,size,batch,active):
        if batch%2 or batch<2:raise ValueError('even batch required')
        self.buffer,self.batch,self.active=buffer,batch,active
        self.tags=np.full(size,-1,np.int8)
        self.rng=np.random.default_rng(207)
        self.samples=[0,0];self.collected=[0,0]

    def __len__(self):return len(self.buffer)

    def extend(self,item):
        ix=self.buffer.extend(item)
        i=self.active().last_source
        self.tags[np.asarray(ix)]=i
        self.collected[i]+=1

    def sample(self):
        picks=[]
        for i in range(2):
            remaining=self.batch//2
            if not self.collected[i]:raise ValueError('empty stratum')
            while remaining:
                candidates=self.rng.integers(len(self.buffer),size=remaining*3+8)
                chosen=candidates[self.tags[candidates]==i][:remaining]
                picks.extend(chosen)
                remaining-=len(chosen)
            self.samples[i]+=self.batch//2
        return self.buffer._storage.get(torch.as_tensor(self.rng.permutation(picks))).clone()

    def receipt(self):
        return dict(real_samples=self.samples[0],synthetic_samples=self.samples[1],
                    real_transitions=self.collected[0],synthetic_transitions=self.collected[1])

    def state_dict(self):
        return dict(real=self.buffer.state_dict(),tags=self.tags,samples=self.samples,
                    collected=self.collected,rng=self.rng.bit_generator.state)

    def load_state_dict(self,state):
        restore_buffer(self.buffer,state['real']);self.tags=state['tags']
        self.samples,self.collected=state['samples'],state['collected']
        self.rng.bit_generator.state=state['rng']


def run(a):
    read,sha=prior.read,prior.file_sha
    if not torch.cuda.is_available():raise ValueError('GPU required')
    prior.configure(7);torch.backends.cudnn.allow_tf32=False
    root=a.root/str(a.width);source=a.pilot/str(a.width);bundle=source/'bundle'
    mode='price7' if a.width==61 else 'context7'
    if sha(a.data/'COHORTS.json')!=prior.COHORT:raise ValueError('cohort')
    ready=read(source/'READY.json')
    if (ready['source']!=prior.SOURCE or ready['start']!=sha(bundle/'model.pt') or
        ready['risk']!=sha(a.parent/'RISK.json') or ready['spec']!=sha(a.parent/'SPEC.json')):raise ValueError('model or real data')
    spec=read(a.parent/'SPEC.json')
    prior.verify_files(a.parent,{k:v for k,v in spec['files'].items() if k.startswith(('data/','engine/')) or k in ('RISK.json','SUPPORT_RECIPE.json')})
    report=read(a.data/'COHORTS.json')['paths'][mode]
    if report['physics_failed'] or report['n_columns']!=a.width:raise ValueError('synthetic gate')
    prior.verify_files(a.data/mode,report['sha256'])
    model,_=prior.load_bundle(bundle,'cuda')
    real=prior.market_for(a.parent,spec,model.recipe);risk=read(a.parent/'RISK.json')
    if ready['support']!=prior.identity(real.inputs.row_ts_ns[real.inputs.valid].tolist()):raise ValueError('real support')
    periods=model.recipe['split']['periods']
    if periods!={'TRAIN':['2020-01-01','2024-01-01'],'VALIDATION':['2024-01-01','2025-01-01']}:raise ValueError(periods)
    path=a.data/mode;bars=pd.read_parquet(path/'OHLC.parquet')
    if not bars.index.equals(real.bars[(real.bars.index.year>=2020)&(real.bars.index.year<2024)].index):raise ValueError('four-year clock')
    synth=synthetic_market(real,bars,pd.read_parquet(path/f'FEATURES{a.width}.parquet'))
    if np.any(synth.inputs.period=='VALIDATION'):raise ValueError('synthetic validation')
    design=dict(spec['design'],max_epochs=4,matched_budget=True,initialize_epoch_zero=True,
                allow_ineligible_epoch_zero=True,head_only=False)
    binding=dict(start=ready['start'],support=ready['support'],source_file=sha(__file__),runner=sha(prior.fit_epochs.__code__.co_filename),
        seed=7,epochs=4,real_train=periods['TRAIN'],synthetic_train=periods['TRAIN'],validation_real=periods['VALIDATION'],
        synthetic_path=mode,synthetic_files=report['sha256'],bars_per_calendar=len(bars),
        batch_real=design['batch_size']//2,batch_synthetic=design['batch_size']//2)
    key=prior.identity(binding)
    root.mkdir(exist_ok=True,parents=True)
    with (root/'lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'CONTRACT.json').exists() and read(root/'CONTRACT.json')!=binding:raise ValueError('run collision')
        prior.write(root/'CONTRACT.json',binding)
        if (root/'RESULT.json').exists():return
        active=None
        def make():
            nonlocal active
            active=TwoAccounts(*(prior.environment(m,risk,6,periods['TRAIN'],model.effective['gamma']) for m in (real,synth)))
            return active
        def assess(m):
            metrics=prior.assess_full(m,real,risk,design)
            if not (root/'EPOCH_ZERO.json').exists():
                for k,ref in ready['initial'].items():
                    if metrics[k]['trades']!=ref['trades'] or abs(metrics[k]['return_pct']-ref['return_pct'])>1e-5:
                        raise ValueError('real account parity:'+k)
            return metrics
        def hook(m,row):
            if not active.finished or active.n!=active.expected:raise ValueError('coverage')
            row['collection']=active.coverage()
        ix=np.flatnonzero(real.inputs.valid & (real.inputs.period=='TRAIN'))[::761]
        st=prior.fit_epochs(model,make,assess,design,7,root,key,real.inputs.windows[ix].copy(),
            replay_factory=lambda buf:Balanced(buf,design['replay_size'],design['batch_size'],lambda:active),epoch_hook=hook)
        prior.write(root/'RESULT.json',dict(key=key,epochs=st['epoch'],updates=st['updates'],promoted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ('pilot','root','data','parent'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--width',choices=[61,71],type=int,required=True)
    run(parser.parse_args())
