"""Finite shared/independent encoder contrast; unchanged categorical business contract."""
import argparse
import copy
import fcntl
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agent_plugins.hybrid_orders import ElapsedRolloutBuffer
from agent_plugins.qualified_orders import QualifiedOrderPolicy
from agent_plugins.independent_order_encoders import IndependentOrderPolicy
from app.order_qualification import OrderSignal, conditional_probe, passes
from app.frozen_recipe_inputs import file_sha, require
from tools.recipe_hybrid_queue import DESIGN, QUALIFICATION, Market, verify as verify_parent, business_behavior
from tools.recipe_hybrid_queue import configure
from tools.recipe_sac_queue import ROOT, read,write,now,source_pins,software,tensor_hash,preserve_rng,evaluate,clean
from tools.hybrid_gradient_audit import inspect_rollout


def study_design():
    return dict(training=copy.deepcopy(DESIGN),seeds=[7,19,43],arms=['SHARED','INDEPENDENT'],
        units=[f'{a}_s{s}' for s in (7,19,43) for a in ('SHARED','INDEPENDENT')],
        periods=['TRAIN','INNER_A'],checkpoints=[25600,51200,76800,102400],
        qualification=copy.deepcopy(QUALIFICATION),reward='equity',learned_risk=False,
        evaluation='fixed final checkpoint, deterministic; no 2024 selection or TEST',
        contrast='encoder sharing only; independent critic starts as exact copy, extra parameters reported')


class InstrumentedPPO(PPO):
    def train(self):
        folder=getattr(self,'diagnostic_output',None)
        before={n:p.detach().clone() for n,p in self.policy.features_extractor.named_parameters()} if folder else None
        probes=None
        if folder and self._n_updates%50==0:
            with preserve_rng():probes=inspect_rollout(self)
        super().train()
        if folder:
            delta=sum(float((p.detach()-before[n]).double().square().sum())
                      for n,p in self.policy.features_extractor.named_parameters())**.5
            write(Path(folder)/f'UPDATE_{self._n_updates:05d}.json',clean(dict(
                updates=self._n_updates,steps=self.num_timesteps,metrics=dict(self.logger.name_to_value),
                actor_encoder_update_norm=delta,pre_update_gradient_probe=probes,
                probe_scope='shared actor features; independent critic has zero derivative here by construction')))


def build_model(env,recipe,seed,device,arm,n_steps=512):
    require(arm in ('SHARED','INDEPENDENT'),'ARM')
    cls=QualifiedOrderPolicy if arm=='SHARED' else IndependentOrderPolicy
    d=DESIGN
    return InstrumentedPPO(cls,env,policy_kwargs=dict(recipe=recipe,learned_risk=False),
        rollout_buffer_class=ElapsedRolloutBuffer,device=device,seed=seed,n_steps=n_steps,
        n_epochs=d['n_epochs'],batch_size=min(d['batch_size'],n_steps),learning_rate=d['learning_rate'],
        gamma=d['gamma'],gae_lambda=d['gae_lambda'],ent_coef=d['ent_coef'],clip_range=d['clip_range'],
        vf_coef=d['vf_coef'],max_grad_norm=d['max_grad_norm'],verbose=0)


def authenticate(path,sha,unit):
    r=read(path)
    require(r['state']=='COMPLETE' and r['spec_sha256']==sha and r['unit']==unit,'RESULT_IDENTITY')
    for name,digest in r['artifacts'].items():
        p=(path.parent/name).resolve()
        require(p.is_relative_to(path.parent.resolve()) and file_sha(p)==digest,'RESULT_ARTIFACT')
    if unit.startswith('QUAL_'):
        require('QUALIFICATION.json' in r['artifacts'] and read(path.parent/'QUALIFICATION.json')['passed'],
                'QUALIFICATION_RECEIPT')
    else:
        require('reports' in r,'FINANCIAL_REPORTS_REQUIRED')
        require(len(r['reports'])==2 and {v['period'] for v in r['reports']}=={'TRAIN','INNER_A'},'REPORT_PERIODS')
        arm,seed=unit.rsplit('_s',1)
        for report in r['reports']:
            require(report['arm']==arm and report['seed']==int(seed),'REPORT_IDENTITY')
            require(report==read(path.parent/(report['period']+'.json')),'NATIVE_REPORT')
        require('model-102400.zip' in r['artifacts'],'FINAL_CHECKPOINT')
    return r


def verify(root):
    s=read(root/'SPEC.private.json')
    require(file_sha(root/'SPEC.private.json')==read(root/'IDENTITY.json')['spec_sha256'],'SPEC_PIN')
    require(s['schema']=='hybrid_encoder_contrast.v1' and s['study']==study_design(),'DESIGN_PIN')
    require(s['software']==software(),'SOFTWARE_PIN')
    for role,base in [('agent',Path(s['agent_root'])),('engine',Path(s['engine_root']))]:
        for name,digest in s['code'][role].items():require(file_sha(base/name)==digest,'CODE_PIN')
    require(Path(s['agent_root']).resolve()==ROOT.resolve(),'RUNTIME_ROOT')
    parent=verify_parent(Path(s['parent_root']))
    require(file_sha(Path(s['parent_root'])/'SPEC.private.json')==s['parent_spec_sha256'],'PARENT_PIN')
    require(s['inputs']==parent['inputs'] and s['recipe_sha256']==parent['recipe_sha256'],'INPUTS_CHANGED')
    return s


def prepare(root,parent,audit):
    require(not root.exists(),'NEW_CAMPAIGN_REQUIRED')
    p=verify_parent(parent)
    require(read(parent/'CLOSE.json')['state']=='COMPLETE','PARENT_CLOSED')
    seal=read(audit/'CLOSE.json');ident=read(audit/'IDENTITY.json')
    require(seal['state']=='COMPLETE' and seal['identity_sha256']==file_sha(audit/'IDENTITY.json')
            and ident['parent_spec_sha256']==file_sha(parent/'SPEC.private.json'),'GRADIENT_PARENT')
    for n,h in seal['artifacts'].items():
        path=(audit/n).resolve();require(path.is_relative_to(audit.resolve()) and file_sha(path)==h,'GRADIENT_ARTIFACT')
    results=read(audit/'SUMMARY.json')
    require(len(results)==9 and all(r['optimizer_steps']==0 for r in results),'GRADIENT_COMPLETENESS')
    spec=dict(schema='hybrid_encoder_contrast.v1',study=study_design(),inputs=p['inputs'],
        recipe_sha256=p['recipe_sha256'],engine_root=p['engine_root'],agent_root=str(ROOT),
        parent_root=str(parent),parent_spec_sha256=file_sha(parent/'SPEC.private.json'),
        gradient_seal_sha256=file_sha(audit/'CLOSE.json'),software=software(),
        code={'agent':source_pins(ROOT,['app','env_plugins','agent_plugins','tools']),'engine':p['code']['engine']},
        max_attempts=2,task_timeout_seconds=7200,device='cuda')
    write(root/'SPEC.private.json',spec)
    write(root/'IDENTITY.json',dict(spec_sha256=file_sha(root/'SPEC.private.json'),created_at=now()))


def heartbeat(root,unit,start,budget):
    class Progress(BaseCallback):
        def _on_step(self):
            self.model.rollout_buffer.pending_duration=np.asarray([v['elapsed_bars'] for v in self.locals['infos']],np.int32)
            if self.num_timesteps%512==0:
                write(root/'PROGRESS.json',dict(state='TRAINING',unit=unit,steps=self.num_timesteps,budget=budget,
                    heartbeat_at=now(),seconds=time.monotonic()-start,gpu_peak_bytes=torch.cuda.max_memory_allocated()))
            return True
    return Progress()


def worker(root,unit,attempt):
    spec=verify(root);sha=file_sha(root/'SPEC.private.json');configure()
    require(torch.cuda.is_available(),'CUDA_REQUIRED')
    folder=root/unit/f'attempt-{attempt}';folder.mkdir(parents=True,exist_ok=False)
    rec=read(spec['inputs']['recipe']);started=time.monotonic()
    if unit.startswith('QUAL_'):
        seed,lag=map(int,unit.removeprefix('QUAL_').split('_'))
        require(dict(seed=seed,lag=lag) in QUALIFICATION['cases'],'QUALIFICATION_UNIT')
        env=OrderSignal((rec['tensor']['window']['observations'],rec['list']['n']),lag=lag)
        model=build_model(env,rec,seed,'cuda','INDEPENDENT',n_steps=64)
        model.learn(QUALIFICATION['steps'],callback=heartbeat(root,unit,started,QUALIFICATION['steps']))
        rows=conditional_probe(model,env);model.save(folder/'model.zip')
        loaded=PPO.load(folder/'model.zip',device='cuda');replay=conditional_probe(loaded,env)
        write(folder/'QUALIFICATION.json',dict(rows=rows,reload_exact=rows==replay,passed=passes(rows) and rows==replay,
            seed=seed,lag=lag,steps=QUALIFICATION['steps'],seconds=time.monotonic()-started))
        require(passes(rows) and rows==replay,'SYNTHETIC_LEARNING_FAILED')
        reports=None
    else:
        require(unit in spec['study']['units'],'FINANCIAL_UNIT')
        status=read(root/'STATUS.json')
        for q in QUALIFICATION['cases']:
            key=f'QUAL_{q["seed"]}_{q["lag"]}'
            require(key in status['completed'],'QUALIFICATION_REQUIRED')
            path=root/status['completed'][key];authenticate(path,sha,key)
            require(read(path.parent/'QUALIFICATION.json')['passed'],'QUALIFICATION_REQUIRED')
        arm,seed=unit.rsplit('_s',1);seed=int(seed)
        market=Market(spec,'CAT_FIXED');env=market.env('TRAIN')
        model=build_model(env,rec,seed,'cuda',arm)
        with preserve_rng():
            control=build_model(market.env('TRAIN'),rec,seed,'cuda','SHARED')
            for name,value in control.policy.state_dict().items():
                require(torch.equal(value,model.policy.state_dict()[name]),'INITIAL_COMMON_TENSOR_PARITY')
            obs,_=market.env('TRAIN').reset();tensor,_=model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                d,v=model.policy._parts(tensor);cd,cv=control.policy._parts(tensor)
                require(torch.equal(d.cat.probs,cd.cat.probs) and torch.equal(v,cv),'INITIAL_FUNCTION_PARITY')
            reference_hash=tensor_hash(control);del control
        write(folder/'EFFECTIVE_MODEL.json',dict(arm=arm,seed=seed,design=spec['study'],
            policy_class=type(model.policy).__name__,initial_common_policy_sha256=reference_hash,
            initial_policy_sha256=tensor_hash(model),parameters=sum(p.numel() for p in model.policy.parameters()),
            actor_encoder_parameters=sum(p.numel() for p in model.policy.features_extractor.parameters()),
            source_parent_spec=spec['parent_spec_sha256']))
        model.diagnostic_output=str(folder/'learning')
        for step in spec['study']['checkpoints']:
            model.learn(25600,reset_num_timesteps=(step==25600),callback=heartbeat(root,unit,started,102400))
            require(model.num_timesteps==step,'STEP_COUNT')
            model.save(folder/f'model-{step}.zip')
            write(folder/f'TRAINING_{step}.json',clean(dict(step=step,metrics=dict(model.logger.name_to_value),
                elapsed_seconds=time.monotonic()-started,policy_tensor_sha256=tensor_hash(model))))
        loaded=PPO.load(folder/'model-102400.zip',device='cuda')
        require(tensor_hash(model)==tensor_hash(loaded),'RELOAD_TENSOR_PARITY');del loaded
        reports=[]
        for period in spec['study']['periods']:
            write(root/'PROGRESS.json',dict(state='EVALUATING',unit=unit,period=period,heartbeat_at=now()))
            report=evaluate(model,market,period,folder)
            counts=read(folder/(period+'.orders.json'))['counts']
            report.update(business_behavior(report,counts))
            report.update(arm=arm,seed=seed,step=102400,scope='PAIRED_ENCODER_CONTRAST')
            write(folder/(period+'.json'),report);reports.append(report)
    result=dict(state='COMPLETE',spec_sha256=sha,unit=unit,seconds=time.monotonic()-started,
        artifacts={str(p.relative_to(folder)):file_sha(p) for p in folder.rglob('*') if p.is_file()},finished_at=now())
    if reports is not None:result['reports']=reports
    write(folder/'RESULT.json',result)


def tick(root):
    spec=verify(root);sha=file_sha(root/'SPEC.private.json')
    with (root/'QUEUE.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        if (root/'CLOSE.json').exists():return
        path=root/'STATUS.json'
        status=read(path) if path.exists() else dict(state='RUNNING',completed={},attempts={},failures={},started_at=now())
        order=[f'QUAL_{q["seed"]}_{q["lag"]}' for q in QUALIFICATION['cases']]+spec['study']['units']
        for unit in order:
            if unit in status['completed']:
                authenticate(root/status['completed'][unit],sha,unit);continue
            # Adopt a sealed success if a parent died after child completion.
            for result in sorted((root/unit).glob('attempt-*/RESULT.json')):
                authenticate(result,sha,unit);status['completed'][unit]=str(result.relative_to(root))
            if unit in status['completed']:write(path,status);continue
            while status['attempts'].get(unit,0)<spec['max_attempts']:
                attempt=status['attempts'].get(unit,0)+1;status['attempts'][unit]=attempt
                status.update(state='RUNNING',active_unit=unit,heartbeat_at=now());write(path,status)
                with (root/f'{unit}-attempt-{attempt}.log').open('x') as log:
                    try:
                        p=subprocess.run([sys.executable,'-m','tools.hybrid_encoder_queue','worker','--root',str(root),
                            '--unit',unit,'--attempt',str(attempt)],stdout=log,stderr=subprocess.STDOUT,
                            timeout=spec['task_timeout_seconds'])
                        rc=p.returncode
                    except subprocess.TimeoutExpired:rc=124
                result=root/unit/f'attempt-{attempt}'/'RESULT.json'
                if rc==0 and result.exists():
                    authenticate(result,sha,unit);status['completed'][unit]=str(result.relative_to(root));write(path,status);break
                status['failures'][f'{unit}/{attempt}']=dict(rc=rc,finished_at=now());write(path,status)
            if unit not in status['completed'] and unit.startswith('QUAL_'):
                status.update(state='FAILED_QUALIFICATION',finished_at=now());write(path,status)
                write(root/'CLOSE.json',dict(state=status['state'],spec_sha256=sha,completed=status['completed'],finished_at=now()))
                return
        complete=all(u in status['completed'] for u in order)
        status.update(state='COMPLETE' if complete else 'COMPLETE_WITH_FAILURES',active_unit=None,finished_at=now())
        write(path,status)
        write(root/'SUMMARY.json',[read(root/status['completed'][u]) for u in spec['study']['units'] if u in status['completed']])
        write(root/'CLOSE.json',dict(state=status['state'],spec_sha256=sha,completed=status['completed'],finished_at=now()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','tick','worker'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--audit',type=Path)
    p.add_argument('--unit');p.add_argument('--attempt',type=int)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.root.resolve(),a.parent.resolve(),a.audit.resolve())
    elif a.command=='tick':tick(a.root.resolve())
    else:worker(a.root.resolve(),a.unit,a.attempt)


if __name__=='__main__':main()
