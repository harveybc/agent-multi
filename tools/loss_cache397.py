"""Opt-in, non-governing speed experiment for the fixed history397 bundle."""
import argparse
import hashlib
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torchrl.objectives.utils import _vmap_func

from app.frozen_signal_masks import prepared_signals
from app.three_signal_learning import evaluate
from agent_plugins.sac_critic_regularization import module_mode
from tools import pilot391 as p
from tools.account397 import assemble
from tools.run_discrete_curve_probe import transition


def prefix_keys(name):
    return any(s in name for s in ('.encoder.indices_', '.encoder.branches.',
               '.encoder.fusion.', '.encoder.fixed_fusion_offset'))


def fingerprint(model):
    h = hashlib.sha256()
    for k,v in model.loss.state_dict().items():
        if prefix_keys(k):
            h.update(k.encode())
            h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tail(enc,x,norm):
    x=enc.post_fusion(x)
    if norm is None: return enc.core(x)
    for i,block in enumerate(enc.core): x=block(x,first_norm=norm if i==0 else None)
    return x


class Bank:
    def __init__(self,m,inputs):
        self.before=fingerprint(m)
        self.ids=np.flatnonzero(inputs.valid)
        self.mapping=np.full(len(inputs.valid),-1,np.int64)
        self.mapping[self.ids]=np.arange(len(self.ids))
        self.banks=[]
        loss=m.loss
        frozen=[v for k,v in loss.named_parameters() if prefix_keys(k)]
        if not frozen or any(v.requires_grad for v in frozen):
            raise ValueError('cache requires frozen functional parameters')
        if any(v.requires_grad for group in m.optimizer.param_groups for v in group['params']
               if any(v is f for f in frozen)):
            raise ValueError('optimizer has a trainable prefix')
        pairs=[(m.actor_net.encoder,m.actor,loss.actor_network_params)]
        for params in (loss.qvalue_network_params,loss.target_qvalue_network_params):
            pairs.extend((m.q_net.encoder,loss.qvalue_network,params[i]) for i in range(2))
        with torch.no_grad():
            for enc,module,params in pairs:
                with params.to_module(module):
                    if enc.context is not None: raise ValueError('unsupported context')
                    if any(isinstance(b,torch.nn.Dropout) and b.p for b in enc.branches.modules()):
                        raise ValueError('stochastic prefix')
                    chunks=[]
                    for i in range(0,len(self.ids),128):
                        x=torch.as_tensor(inputs.windows[self.ids[i:i+128]],device=m.device).transpose(1,2)
                        grouped=[branch(x.index_select(1,getattr(enc,f'indices_{j}')))
                                 for j,branch in enumerate(enc.branches)]
                        chunks.append(enc.fusion_activation(enc.fusion(torch.cat(grouped,1))+enc.fixed_fusion_offset))
                    self.banks.append(torch.cat(chunks))
        if not all(torch.isfinite(b).all() for b in self.banks): raise ValueError('nonfinite cache')

    def row_env(self,env):
        def observe():
            ix=self.mapping[env.rows[env.bar]]
            if ix<0: raise ValueError('invalid row')
            env.last_observation=dict(features=np.array([ix],np.int64),mask=np.ones(3,bool))
            return env.last_observation
        env.observation=observe
        return env

    @contextmanager
    def use(self,m):
        a,q=m.actor_net.encoder,m.q_net.encoder
        af,qf=a.sequence,q.sequence
        loss=m.loss
        native=loss._vmap_qnetworkN0
        twins=_vmap_func(loss.qvalue_network,(0,0),randomness=loss.vmap_randomness,
                         pseudo_vmap=loss.deactivate_vmap)
        def act(obs,core_first_norm=None):
            x=obs['features']
            if x.dtype!=torch.long: return af(obs,core_first_norm)
            return tail(a,self.banks[0].index_select(0,x[:,0]),core_first_norm)
        def crit(obs,core_first_norm=None):
            x=obs['features']
            if tuple(x.shape[1:])==q.transport_shape: return qf(obs,core_first_norm)
            if x.shape[1:]!=self.banks[1].shape[1:]: raise ValueError('unknown critic input')
            return tail(q,x,core_first_norm)
        def paired(td,params):
            x=td['features']
            if x.dtype!=torch.long: return native(td,params)
            offset=3 if params is loss.target_qvalue_network_params else 1
            if offset==1 and params is not loss.qvalue_network_params and params is not loss._cached_detached_qvalue_params:
                raise ValueError('unknown critic family')
            fused=torch.stack([b.index_select(0,x[:,0]) for b in self.banks[offset:offset+2]])
            view=td.expand(2,*td.batch_size).clone(False)
            view.set('features',fused)
            return twins(view,params)
        with ExitStack() as stack:
            for obj,key,value in ((a,'sequence',act),(q,'sequence',crit),(loss,'_vmap_qnetworkN0',paired)):
                stack.enter_context(patch.object(obj,key,value))
            yield


def gate(bundle,train,bank,make_env):
    ref,_=p.load_bundle(bundle,'cuda')
    cand,_=p.load_bundle(bundle,'cuda')
    env=make_env()
    obs=env.reset()
    rows=[]; records=[]
    for i in range(600):
        ix=bank.mapping[env.rows[env.bar]]
        nxt,r,done,dt=env.step(i%3)
        records.append(transition(obs,i%3,nxt,r,done,dt))
        rows.append([ix,bank.mapping[env.rows[env.bar]]])
        obs=nxt
    source=torch.cat(records).to('cuda')
    ids=torch.as_tensor(rows,device='cuda')
    diffs={k:0. for k in ('actor','critic','alpha','loss_alpha','grad','state','logits','twin_q','target_q')}
    def check(k,u,v,atol=2e-6):
        torch.testing.assert_close(u,v,atol=atol,rtol=1e-5,msg=lambda s:k+': '+s)
        diffs[k]=max(diffs[k],float((u-v).abs().max()))
    with bank.use(cand):
        for step in range(16):
            idx=np.random.default_rng(step+53).integers(0,len(rows),128)
            b=source[idx].clone()
            c=b.clone()
            c['features']=ids[idx,:1]
            c['next','features']=ids[idx,1:]
            with torch.no_grad():
                check('logits',ref.surfaces(b['features']),cand.surfaces(c['features']))
                for family,k in [('qvalue_network_params','twin_q'),('target_qvalue_network_params','target_q')]:
                    def qs(m,td):
                        torch.manual_seed(3000+step)
                        with module_mode(m.loss.qvalue_network,k=='twin_q'):
                            return m.loss._vmap_qnetworkN0(td.select('features'),getattr(m.loss,family))['action_value']
                    check(k,qs(ref,b),qs(cand,c))
            vals=[]
            for m,td in [(ref,b),(cand,c)]:
                torch.manual_seed(1000+step)
                cap=[]
                def store(_,args,result):
                    cap.append(result['loss_alpha'].detach().clone())
                hook=m.loss.register_forward_hook(store)
                try: scalars=m.update(td)
                finally: hook.remove()
                vals.append((scalars,cap[0]))
            for k in ('actor','critic','alpha'):
                check(k,*[torch.tensor(v[0][k]) for v in vals])
            check('loss_alpha',vals[0][1],vals[1][1])
            for (name,u),(other,v) in zip(ref.loss.named_parameters(),cand.loss.named_parameters(),strict=True):
                if other!=name or (u.grad is None)!=(v.grad is None): raise ValueError(name)
                if u.grad is not None: check('grad',u.grad,v.grad)
            refstate=ref.loss.state_dict()
            for k,v in cand.loss.state_dict().items():
                if v.is_floating_point(): check('state',refstate[k],v)
                else: assert torch.equal(refstate[k],v)
        if bank.before!=fingerprint(cand): raise ValueError('prefix changed')
    return diffs


def main(out,epoch):
    root=Path('/var/tmp/pilot397')
    parent=Path('/var/tmp/sac70-warm-v3-20261108')
    bundle=root/'model/61/bundle'
    p.configure(7)
    torch.backends.cudnn.allow_tf32=False
    m,_=p.load_bundle(bundle,'cuda')
    train,val=assemble(parent,bundle,root/'history397-61-r2')
    d=dict(p.read(parent/'SPEC.json')['design'],max_epochs=1,matched_budget=True,initialize_epoch_zero=True,
           allow_ineligible_epoch_zero=True,head_only=False)
    risk=p.read(parent/'RISK.json')
    def env(market,interval):
        return p.environment(market,risk,d['horizon'],interval,m.effective['gamma'])
    tr=['2016-01-01','2024-01-01']
    def make(): return env(train,tr)
    def assess(model):
        result={}
        for key,market,interval in [('TRAIN',train,tr),('VALIDATION',val,m.recipe['split']['periods']['VALIDATION'])]:
            logits,_=p.emit(model,market.inputs)
            result[key]=evaluate(env(market,interval),logits,prepared=prepared_signals(logits))
        return result
    out.mkdir(exist_ok=False,parents=True)
    start=time.monotonic()
    bank=Bank(m,train.inputs)
    setup=time.monotonic()-start
    diffs=gate(bundle,train,bank,make)
    rec=dict(non_governing=True,model_sha256=p.file_sha(bundle/'model.pt'),rows=len(bank.ids),setup_seconds=setup,
             bytes=sum(x.nbytes for x in bank.banks),paired_updates=16,max_deltas=diffs,scope='TRAIN only')
    p.write(out/'GATE.json',rec)
    if not epoch: return
    np.random.seed(7)
    torch.manual_seed(7)
    m,_=p.load_bundle(bundle,'cuda')
    assess(m)  # Same preflight as train_history397.
    with bank.use(m):
        ix=np.flatnonzero(train.inputs.valid)[::761]
        state=p.fit_epochs(m,lambda:bank.row_env(make()),assess,d,7,out,p.identity(rec),train.inputs.windows[ix].copy())
    if bank.before!=fingerprint(m): raise ValueError('post epoch prefix changed')
    row=state['history'][-1]
    ref=p.read(root/'out/HISTORY.json')[0]
    if any(row[k]!=ref[k] for k in ['epoch_updates','decisions','steps']): raise ValueError('work budget differs')
    p.write(out/'COMPARISON.json',dict(reference_seconds=ref['epoch_seconds'],cached_seconds=row['epoch_seconds'],
            speedup=ref['epoch_seconds']/row['epoch_seconds'],setup_seconds=setup,updates=row['epoch_updates'],
            metrics={s:{k:[ref['metrics'][s][k],row['metrics'][s][k]] for k in ('trades','return_pct','sharpe')}
            for s in ('TRAIN','VALIDATION')}))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('out',type=Path)
    cli.add_argument('--epoch',action='store_true')
    a=cli.parse_args()
    main(a.out,a.epoch)
