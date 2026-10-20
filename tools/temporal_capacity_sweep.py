"""Finite adaptive Conv1D capacity ladder over the native selected SAC queue."""
import argparse
import base64
import copy
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

from tools import selected_rl_comparison as selected
from tools import selected_rl_regularization_sequence as sequence
from tools.curated_rl_campaign import digest
from tools.paired_checkpoint_diagnostics import read,write
from tools.verify_rl_predecessor import predecessor_ready,resources_ready

SCHEMA='temporal_capacity_sweep.v1'
BASE_ARCH=dict(net_arch=[8],branch_filters=2,core_filters=4,kernel_size=3,core_layers=1)
RULE=dict(min_daily_gain=1e-8,numeric_tolerance=1e-8,max_fresh_fit_attempts=2)
METADATA=('SPEC.json','SWEEP_STATE.json','WORK_STATUS.json','PUBLICATION_STATUS.json',
          'SUMMARY.json','RESULTS.csv','RESULTS.md','SWEEP_COMPLETE.json')


def stage_architectures(stage,incumbent):
    field,values=[('core_filters',(8,16)),('net_arch',([16],[32])),
                  ('branch_filters',(4,8)),('core_layers',(2,))][stage]
    return [dict(copy.deepcopy(incumbent),**{field:value}) for value in values]


def replications(arch):
    return [] if arch==BASE_ARCH else [dict(role=role,seed=seed,
        architecture=copy.deepcopy(BASE_ARCH if role=='baseline' else arch))
        for seed in (17,29) for role in ('baseline','finalist')]


def improves(candidate,incumbent):
    keys=('train_growth','validation_growth','max_drawdown_pct','trades_closed')
    if not all(type(b.get(k)) in (int,float) and math.isfinite(b[k]) for b in (candidate,incumbent) for k in keys):
        raise ValueError('SWEEP_FINITE_OUTCOME_REQUIRED')
    gap=lambda b:b['train_growth']-b['validation_growth']
    return (candidate['validation_growth']>incumbent['validation_growth']+RULE['min_daily_gain']
        and gap(candidate)<=max(gap(incumbent),0)+RULE['numeric_tolerance']
        and candidate['max_drawdown_pct']<=incumbent['max_drawdown_pct']+RULE['numeric_tolerance']
        and candidate['trades_closed']>0)


def _commit():
    return subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()


def _node(root,arch,seed):
    return dict(root=str(Path(root).resolve()),contract_sha256=digest(Path(root)/'CONTRACT.json'),
                architecture=copy.deepcopy(arch),seed=seed)


def outcome(node):
    root=Path(node['root'])
    if predecessor_ready(root,node['contract_sha256'])!='READY':raise ValueError('SWEEP_OUTCOME_NOT_CLOSED')
    body=read(root/'CONTRACT.json');arch=node['architecture'];cfg=body['common_config']
    if (body['seeds']!=[node['seed']] or body['arms']!=['SAC_PER_FEATURE_CONV1D']
            or cfg['net_arch']!=arch['net_arch']
            or any(cfg.get('temporal_'+k,1)!=arch[k] for k in ('branch_filters','core_filters','kernel_size','core_layers'))):
        raise ValueError('SWEEP_OUTCOME_ARCH_SEED_MISMATCH')
    train=next(t for t in read(root/'STATUS.json')['tasks'] if t['split']=='train')
    monitor=Path(train['config']).parent/'paired-monitor'
    final=read(monitor/'FINAL.json')
    curves=read(monitor/'diagnostics/CURVES.json')
    if (final.get('selected_step')!=100000 or final.get('evaluations')!=4
            or final.get('stop_reason')!='FIXED_BUDGET' or len(curves['rows'])!=4
            or curves['rows'][-1]['step']!=100000):raise ValueError('SWEEP_FIXED_FINAL_REQUIRED')
    metrics={t['split']:read(t['summary']) for t in read(root/'STATUS.json')['tasks']}
    row=curves['rows'][-1]
    result=dict(train_growth=row['train_growth'],validation_growth=row['validation_growth'],
        max_drawdown_pct=metrics['validation']['max_drawdown_pct'],trades_closed=metrics['validation']['trades_closed'],
        metrics={role:{k:values.get(k) for k in ('net_profit','total_return','max_drawdown_pct',
            'sharpe_ratio','trades_closed','win_rate_pct','profit_factor')} for role,values in metrics.items()},
        close_sha256=digest(root/'CLOSE.json'),wall_seconds=sum(t['wall_seconds'] for t in read(root/'STATUS.json')['tasks']))
    if not all(math.isfinite(result[k]) for k in ('train_growth','validation_growth','max_drawdown_pct')):
        raise ValueError('SWEEP_NONFINITE_OUTCOME')
    return result


def prepare(base_request,root,baseline,baseline_pin):
    root=selected._output_root(root);baseline=Path(baseline).resolve();base_request=Path(base_request).resolve()
    if baseline.is_relative_to(root) or root.is_relative_to(baseline):raise ValueError('SWEEP_SEPARATE_ROOT_REQUIRED')
    base=read(base_request)
    if base['schema']!='selected_rl_comparison_request.v4':raise ValueError('SWEEP_BASE_V4_REQUIRED')
    old=_node(baseline,BASE_ARCH,7)
    if old['contract_sha256']!=baseline_pin:raise ValueError('SWEEP_BASELINE_PIN_MISMATCH')
    result=outcome(old)
    spec=dict(schema=SCHEMA,runtime=_commit(),base_request=str(base_request),base_request_sha256=digest(base_request),
              baseline=old,baseline_outcome=result,rule=RULE,stages=4,maximum_new_fits=11)
    sequence.immutable(root/'SPEC.json',spec)
    if not (root/'SWEEP_STATE.json').exists():
        write(root/'SWEEP_STATE.json',dict(schema=SCHEMA,spec_sha256=digest(root/'SPEC.json'),
            state='READY',stages=[],trials=[],incumbent=dict(name='baseline',**old),replications_declared=False))
    return spec


def plan(root):
    root=Path(root).resolve();spec=read(root/'SPEC.json')
    if (spec['schema']!=SCHEMA or spec['runtime']!=_commit() or spec['rule']!=RULE
            or digest(spec['base_request'])!=spec['base_request_sha256']
            or digest(Path(spec['baseline']['root'])/'CONTRACT.json')!=spec['baseline']['contract_sha256']):
        raise ValueError('SWEEP_SPEC_OR_RUNTIME_DRIFT')
    state=read(root/'SWEEP_STATE.json')
    if state['spec_sha256']!=digest(root/'SPEC.json'):raise ValueError('SWEEP_STATE_IDENTITY_DRIFT')
    incumbent=dict(name='baseline',**spec['baseline'])
    for index,decision in enumerate(state['stages']):
        if read(root/'decisions'/f'stage-{index+1}.json')!=decision or decision['previous']!=incumbent['name']:
            raise ValueError('SWEEP_DECISION_DRIFT')
        group=[t for t in state['trials'] if t['phase']==index]
        if [t['architecture'] for t in group]!=stage_architectures(index,incumbent['architecture']):
            raise ValueError('SWEEP_STAGE_ARCHITECTURE_DRIFT')
        if decision['winner']!=incumbent['name']:
            chosen=next(t for t in group if t['name']==decision['winner'])
            incumbent=dict(name=chosen['name'],**chosen['attempts'][-1])
    if incumbent!=state['incumbent']:raise ValueError('SWEEP_INCUMBENT_DRIFT')
    return spec,state


def _save(root,state):write(Path(root)/'SWEEP_STATE.json',state)


def _candidate(root,spec,trial):
    attempt=len(trial['attempts'])+1
    name=trial['name']+f'-a{attempt}'
    request=copy.deepcopy(read(spec['base_request']))
    from agent_plugins.fixed_budget_monitor import SETTINGS
    request.update(schema='selected_rl_comparison_request.v9',experiment_seed=trial['seed'],
        temporal_encoder=trial['architecture'],early_stopping=SETTINGS)
    request['publication']['code_commit']=spec['runtime']
    request['publication']['experiment_set_key']+='-temporal-ladder-'+name
    request_file=Path(root)/'requests'/(name+'.json');sequence.immutable(request_file,request)
    output=Path(root)/'runs'/name
    selected.materialize(request_file,output)
    # Match the complete admitted baseline, not merely a hand-picked subset of settings.
    baseline=read(Path(spec['baseline']['root'])/'CONTRACT.json');body=read(output/'CONTRACT.json')
    a,b=copy.deepcopy(baseline['common_config']),copy.deepcopy(body['common_config'])
    for values in (a,b):
        for key in ('net_arch','temporal_branch_filters','temporal_core_filters','temporal_kernel_size','temporal_core_layers'):
            values.pop(key,None)
    if a!=b or body['selected_input_binding']!=baseline['selected_input_binding']:
        raise ValueError('SWEEP_NONARCHITECTURAL_CHANGE')
    node=_node(output,trial['architecture'],trial['seed']);node.update(attempt=attempt,
        request_file=str(request_file),request_sha256=digest(request_file))
    trial['attempts'].append(node)
    return node


def _failed(root,trial,node,cause):
    target=Path(node['root'])/'SWEEP_FAILURE.json'
    sequence.immutable(target,dict(state='PARTIAL_FAILED_NOT_A_FINAL_RESULT',cause=cause,
        contract_sha256=node['contract_sha256'],attempt=node['attempt']))
    if len(trial['attempts'])>=RULE['max_fresh_fit_attempts']:
        trial.update(state='FAILED_EXHAUSTED',cause=cause)


def _advance(root,spec,state):
    while True:
        active=[t for t in state['trials'] if t['state'] not in ('CLOSED','FAILED_EXHAUSTED')]
        if active:
            trial=active[0]
            if not trial['attempts']:
                node=_candidate(root,spec,trial);_save(root,state);return node
            node=trial['attempts'][-1];r=Path(node['root'])
            if (r/'CLOSE.json').exists():
                trial.update(state='CLOSED',outcome=outcome(node));_save(root,state);summarize(root,spec,state);continue
            journal=read(r/'STATUS.json') if (r/'STATUS.json').exists() else {}
            current=next((t for t in journal.get('tasks',[]) if t['state']!='ENQUEUED'),None)
            bad=(r/'SWEEP_FAILURE.json').exists()
            if current and (current['state']=='RUNNING' or current['state'].startswith('FAILED') or current['state']=='ATTEMPTS_EXHAUSTED'):
                if current['split']=='train':bad=True
                elif current['state']=='ATTEMPTS_EXHAUSTED' or current['attempt']>=2:
                    _failed(root,trial,node,'TERMINAL_VALIDATION_EXHAUSTED');trial.update(state='FAILED_EXHAUSTED');_save(root,state);continue
            if bad:
                _failed(root,trial,node,'TRAIN_INTERRUPTED_OR_FAILED')
                if trial['state']=='FAILED_EXHAUSTED':_save(root,state);continue
                node=_candidate(root,spec,trial);_save(root,state);return node
            if journal and all(t['state']=='ENQUEUED' for t in journal.get('tasks',[])):
                return None
            return node
        stage=len(state['stages'])
        if stage<4:
            group=[t for t in state['trials'] if t['phase']==stage]
            if not group:
                for index,arch in enumerate(stage_architectures(stage,state['incumbent']['architecture'])):
                    state['trials'].append(dict(name=f's{stage+1}-v{index+1}',phase=stage,seed=7,
                        architecture=arch,state='PENDING',attempts=[]))
                _save(root,state);continue
            incumbent=state['incumbent'];old=outcome(incumbent)
            eligible=[t for t in group if t['state']=='CLOSED' and improves(t['outcome'],old)]
            if eligible:
                winner=max(eligible,key=lambda t:(t['outcome']['validation_growth'],-t['outcome']['max_drawdown_pct']))
                incumbent=dict(name=winner['name'],**winner['attempts'][-1])
            decision=dict(stage=stage+1,previous=state['incumbent']['name'],winner=incumbent['name'],
                candidate_outcomes=[dict(name=t['name'],state=t['state'],outcome=t.get('outcome')) for t in group],rule=RULE)
            sequence.immutable(Path(root)/'decisions'/f'stage-{stage+1}.json',decision)
            state['stages'].append(decision);state['incumbent']=incumbent;_save(root,state);continue
        if not state['replications_declared']:
            for item in replications(state['incumbent']['architecture']):
                state['trials'].append(dict(name=f"confirm-{item['role']}-s{item['seed']}",phase=4,
                    seed=item['seed'],role=item['role'],architecture=item['architecture'],state='PENDING',attempts=[]))
            state['replications_declared']=True;_save(root,state);continue
        state['state']='COMPLETED_WITH_FAILED_CANDIDATES' if any(t['state']=='FAILED_EXHAUSTED' for t in state['trials']) else 'ALL_TRIALS_CLOSED'
        _save(root,state);summarize(root,spec,state)
        sequence.immutable(Path(root)/'SWEEP_COMPLETE.json',dict(schema=SCHEMA,state=state['state'],
            spec_sha256=digest(Path(root)/'SPEC.json'),state_sha256=digest(Path(root)/'SWEEP_STATE.json'),
            summary_sha256=digest(Path(root)/'SUMMARY.json'),trials=len(state['trials']),
            successful=sum(t['state']=='CLOSED' for t in state['trials'])))
        return None


def smoke_model(arch,device):
    import tempfile
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from agent_plugins.sac_agent import Plugin,_policy_tensor_hash
    class Synthetic(gym.Env):
        observation_space=gym.spaces.Dict(dict(features=gym.spaces.Box(-10,10,(13,15),dtype=np.float32),
            state=gym.spaces.Box(-10,10,(17,),dtype=np.float32)))
        action_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed);return {k:np.zeros(s.shape,dtype=np.float32) for k,s in self.observation_space.spaces.items()},{}
        def step(self,action):return self.reset()[0],-float(np.square(action).sum()),False,False,{}
    cfg=dict(observation_encoder='per_feature_conv1d',net_arch=arch['net_arch'],
        temporal_branch_filters=arch['branch_filters'],temporal_core_filters=arch['core_filters'],
        temporal_kernel_size=3,temporal_core_layers=arch['core_layers'],train_seed=7,device=device,
        buffer_size=64,batch_size=8,learning_starts=0)
    env=Synthetic();model=Plugin(cfg).build(env,cfg);before=_policy_tensor_hash(model.policy);model.learn(16)
    if before==_policy_tensor_hash(model.policy) or not all(torch.isfinite(p).all().item() for p in model.policy.parameters()):
        raise ValueError('SWEEP_SMOKE_GRADIENT_FAILED')
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'policy.zip';model.save(path);loaded=SAC.load(path,env=env,device=device);obs=env.reset()[0]
        if (_policy_tensor_hash(model.policy)!=_policy_tensor_hash(loaded.policy)
                or not np.array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])):
            raise ValueError('SWEEP_SMOKE_RELOAD_FAILED')
    env.close();return dict(gradient_updates=model._n_updates,policy_reload_equal=True,
        actor_parameters=sum(p.numel() for p in model.actor.parameters()),critic_parameters=sum(p.numel() for p in model.critic.parameters()))


def smoke(node):
    from tools.curated_rl_campaign import validate_gpu_evidence
    import torch
    root=Path(node['root']);body=selected.validate(root);validate_gpu_evidence(body['execution'])
    uuid=body['execution']['gpu_evidence']['assigned_gpu_uuid']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('SWEEP_ASSIGNED_GPU_REQUIRED')
    path=root/'CAPACITY_GPU_SMOKE.json'
    identity=dict(contract_sha256=node['contract_sha256'],architecture=node['architecture'],gpu_uuid=uuid)
    if path.exists():
        old=read(path)
        if old['identity']!=identity:raise ValueError('SWEEP_SMOKE_IDENTITY_CHANGED')
        if old['state']=='VERIFIED':return
    else:old={}
    attempt=old.get('attempt',0)+1
    if attempt>2:raise ValueError('SWEEP_GPU_ADMISSION_EXHAUSTED')
    write(path,dict(state='STARTED',attempt=attempt,identity=identity))
    try:result=smoke_model(node['architecture'],'cuda')
    except RuntimeError:
        write(path,dict(state='FAILED',attempt=attempt,identity=identity,error_type='RuntimeError'))
        if attempt>=2:raise ValueError('SWEEP_GPU_ADMISSION_EXHAUSTED')
        return False
    write(path,dict(state='VERIFIED',attempt=attempt,identity=identity,classification='SYNTHETIC_ENGINEERING_ONLY',**result))
    return True


def work(root):
    root=Path(root);spec,state=plan(root)
    if (root/'SWEEP_COMPLETE.json').exists():return 0
    with (root/'.worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return 0
        uuid=read(spec['base_request'])['execution']['gpu_evidence']['assigned_gpu_uuid']
        ready=resources_ready(root,uuid)
        if ready!='READY':write(root/'WORK_STATUS.json',dict(state=ready));return 0
        node=_advance(root,spec,state)
        if node is None:
            write(root/'WORK_STATUS.json',dict(state=state['state'] if (root/'SWEEP_COMPLETE.json').exists() else 'WAIT_PUBLICATION'));return 0
        write(root/'WORK_STATUS.json',dict(state='GPU_ADMISSION',current_root=node['root']))
        if smoke(node) is False:return 2
        write(root/'WORK_STATUS.json',dict(state='RUNNING',current_root=node['root']))
        code=selected.run(Path(node['root']))
        write(root/'WORK_STATUS.json',dict(state='WAIT_PUBLICATION_OR_RETRY',current_root=node['root'],returncode=code))
        return code


def publish(root):
    from tools.fixed_budget_diagnostics import refresh
    root=Path(root);_,state=plan(root);pending=0
    for trial in state['trials']:
        for node in trial['attempts']:
            r=Path(node['root'])
            if not (r/'CLOSE.json').exists():refresh(r)
            pending+=selected.publish_and_close(r)!=0
    write(root/'PUBLICATION_STATUS.json',dict(state='PENDING' if pending else 'CURRENT_REPORTS_VERIFIED',pending_attempts=pending))
    return 2 if pending else 0


def summarize(root,spec,state):
    items=[dict(name='baseline',seed=7,architecture=BASE_ARCH,state='CLOSED',outcome=spec['baseline_outcome'])]
    items.extend({k:t.get(k) for k in ('name','seed','architecture','state','outcome')} for t in state['trials'])
    changed=bool(replications(state['incumbent']['architecture']))
    replication=[dict(seed=7,validation_daily_growth_difference=outcome(state['incumbent'])['validation_growth']-
        spec['baseline_outcome']['validation_growth'],nonworse_risk_and_gap=improves(outcome(state['incumbent']),spec['baseline_outcome']))] if changed else []
    for seed in (17,29):
        pair={t.get('role'):t for t in state['trials'] if t['phase']==4 and t['seed']==seed}
        if len(pair)==2 and all(t['state']=='CLOSED' for t in pair.values()):
            replication.append(dict(seed=seed,validation_daily_growth_difference=
                pair['finalist']['outcome']['validation_growth']-pair['baseline']['outcome']['validation_growth'],
                nonworse_risk_and_gap=improves(pair['finalist']['outcome'],pair['baseline']['outcome'])))
    verdict='SEARCH_IN_PROGRESS' if len(state['stages'])<4 else 'NOT_NEEDED_NO_PROMOTION' if not changed else 'PROMISING_SHARED_SEEDS_DEVELOPMENT_ONLY' if (
        len(replication)==3 and all(r['nonworse_risk_and_gap'] for r in replication)) else 'INCONCLUSIVE'
    payload=dict(schema=SCHEMA,state=state['state'],scope='DEVELOPMENT_NO_TEST',incumbent=state['incumbent'],trials=items,
        replication=replication,confirmation=verdict,
        mean_paired_daily_growth_difference=statistics.mean(r['validation_daily_growth_difference'] for r in replication) if len(replication)==3 else None)
    write(Path(root)/'SUMMARY.json',payload)
    stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['trial','seed','split','state','architecture','profit_usd','return_fraction','max_dd_pct','sharpe_daily_nonannualized','trades','win_pct','profit_factor'])
    lines=['# Temporal Capacity Results','','Development only;all final100000;TEST not read.','',
        '| Trial | Seed | TRAIN return | VALIDATION return | V maxDD | State |','| --- | --- | --- | --- | --- | --- |']
    for item in items:
        metrics=(item.get('outcome') or {}).get('metrics',{})
        for role in ('train','validation'):
            m=metrics.get(role,{})
            writer.writerow([item['name'],item['seed'],role,item['state'],json.dumps(item['architecture'],sort_keys=True),
                *[m.get(k) for k in ('net_profit','total_return','max_drawdown_pct','sharpe_ratio','trades_closed','win_rate_pct','profit_factor')]])
        def fmt(role,key,scale=1):
            value=metrics.get(role,{}).get(key);return 'unavailable' if value is None else f'{value*scale:.4f}'
        lines.append(f"| {item['name']} | {item['seed']} | {fmt('train','total_return',100)}% | {fmt('validation','total_return',100)}% | {fmt('validation','max_drawdown_pct')}% | {item['state']} |")
    _text(Path(root)/'RESULTS.csv',stream.getvalue());_text(Path(root)/'RESULTS.md','\n'.join(lines)+'\n')


def _text(path,value):
    partial=path.with_suffix(path.suffix+'.partial')
    with partial.open('w') as f:f.write(value);f.flush();os.fsync(f.fileno())
    os.replace(partial,path)


def status(root):
    spec,state=plan(root);current=read(Path(root)/'WORK_STATUS.json') if (Path(root)/'WORK_STATUS.json').exists() else dict(state='READY')
    progress=None
    if current.get('current_root') and not (Path(current['current_root'])/'CLOSE.json').exists():progress=selected.read_status(Path(current['current_root']))
    finished=sum(t['state'] in ('CLOSED','FAILED_EXHAUSTED') for t in state['trials'])
    durations=[t['outcome']['wall_seconds'] for t in state['trials'] if t['state']=='CLOSED'] or [spec['baseline_outcome']['wall_seconds']]
    remaining=max(0,7-sum(t['phase']<4 and t['state'] in ('CLOSED','FAILED_EXHAUSTED') for t in state['trials']))
    confirm_remaining=sum(t['phase']==4 and t['state'] not in ('CLOSED','FAILED_EXHAUSTED') for t in state['trials'])
    upper=remaining+(confirm_remaining if state['replications_declared'] else 4)
    blocked=current['state'].startswith(('WAIT_RAM','WAIT_DISK','WAIT_GPU','BLOCKED'))
    complete=(Path(root)/'SWEEP_COMPLETE.json').exists()
    return dict(schema=SCHEMA,state=state['state'],operation=current,completed_trials=finished,declared_trials=len(state['trials']),
        maximum_new_fits=11,stages_completed=len(state['stages']),incumbent=state['incumbent'],current=progress,
        estimated_remaining_seconds=None if blocked else [0,0] if complete else [remaining*statistics.mean(durations),upper*statistics.mean(durations)],
        eta_scope='PROVISIONAL_BASED_ON_COMPLETED_FULL_TRIALS_INCLUDES_CURRENT_TRIAL',
        trials=[dict(name=t['name'],state=t['state'],architecture=t['architecture'],seed=t['seed'],outcome=t.get('outcome')) for t in state['trials']])


def render(root,directory,environment_file):
    spec,_=plan(root);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    runtime=Path(__file__).resolve().parents[1];python=sys.executable
    uuid=read(spec['base_request'])['execution']['gpu_evidence']['assigned_gpu_uuid']
    paths=[]
    for role in ('worker','publication'):
        action='work' if role=='worker' else 'publish'
        command=['/usr/bin/env','CUDA_VISIBLE_DEVICES='+uuid if role=='worker' else 'CUDA_VISIBLE_DEVICES=',
            'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',python,'-m','tools.temporal_capacity_sweep',action,'--root',root]
        service='\n'.join(['[Unit]','Description=Finite adaptive temporal SAC '+role,
            'After=network-online.target','ConditionPathExists=!'+str(Path(root)/'SWEEP_COMPLETE.json'),
            'ConditionPathExists=!'+str(Path(root)/'BLOCKED.json'),'StartLimitIntervalSec=0','','[Service]',
            'Type=exec' if role=='worker' else 'Type=oneshot','WorkingDirectory='+sequence._setting_path(runtime),
            'EnvironmentFile='+sequence._setting_path(environment_file),'ExecStart='+' '.join(sequence._q(x) for x in command),
            'SuccessExitStatus=2','MemoryHigh='+('5G' if role=='worker' else '768M'),
            'MemoryMax='+('6G' if role=='worker' else '1500M'),'MemorySwapMax=0',
            'CPUQuota='+('200%' if role=='worker' else '100%'),'TasksMax=256','KillMode=control-group',
            'RuntimeMaxSec=21600' if role=='worker' else 'TimeoutStartSec=300','UMask=0077',''])
        paths.extend(sequence._unit(directory,'rl-temporal-capacity-'+role,service))
    return paths


def export(root):
    root=Path(root)
    return {name:dict(sha256=digest(root/name),data=base64.b64encode((root/name).read_bytes()).decode()) for name in METADATA if (root/name).exists()}


def collect(host,source_root,runtime,archive,state_root,remote_python='python3'):
    from tools.collect_closed_raw_rl import collect as closed_collect
    import hashlib
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',host):raise ValueError('SWEEP_SAFE_SSH_ALIAS_REQUIRED')
    archive=selected._output_root(archive);state_root=selected._output_root(state_root)
    if (archive.is_relative_to(state_root) or state_root.is_relative_to(archive)
            or not Path(source_root).is_absolute() or str(source_root)=='/' or not Path(runtime).is_absolute()):
        raise ValueError('SWEEP_SEPARATE_PRIVATE_COLLECTION_ROOTS_REQUIRED')
    command='cd '+shlex.quote(str(runtime))+' && '+shlex.quote(remote_python)+' -m tools.temporal_capacity_sweep export --root '+shlex.quote(str(source_root))
    packet=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,command],timeout=30))
    raw={name:base64.b64decode(item['data'],validate=True) for name,item in packet.items()}
    for name,data in raw.items():
        if name not in METADATA or hashlib.sha256(data).hexdigest()!=packet[name]['sha256']:raise ValueError('SWEEP_EXPORT_CHANGED')
    state=json.loads(raw['SWEEP_STATE.json']);spec=json.loads(raw['SPEC.json'])
    if state['spec_sha256']!=packet['SPEC.json']['sha256']:raise ValueError('SWEEP_COLLECT_IDENTITY')
    archive=Path(archive);state_root=Path(state_root);state_root.mkdir(parents=True,exist_ok=True)
    pending=0;count=0
    nodes=[dict(name='baseline',**spec['baseline'])]
    nodes.extend(dict(name=t['name'],**t['attempts'][-1]) for t in state['trials'] if t['state']=='CLOSED')
    for node in nodes:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',node['name']):raise ValueError('SWEEP_COLLECTION_PATH_ESCAPE')
        dest=archive/'runs'/node['name'];proof=state_root/node['name']
        if closed_collect(host,node['root'],dest,proof)!=0:pending+=1;continue
        if read(dest/'CLOSE.json')['contract_sha256']!=node['contract_sha256']:raise ValueError('SWEEP_COLLECT_CONTRACT_CHANGED')
        count+=1
    for name,data in raw.items():
        archive.mkdir(parents=True,exist_ok=True)
        _text(archive/name,data.decode())
    complete='SWEEP_COMPLETE.json' in raw
    if complete:
        end=json.loads(raw['SWEEP_COMPLETE.json'])
        if end['state_sha256']!=packet['SWEEP_STATE.json']['sha256'] or end['summary_sha256']!=packet['SUMMARY.json']['sha256']:
            raise ValueError('SWEEP_CLOSE_EXPORT_CHANGED')
    result=dict(state='COLLECTION_COMPLETE' if complete and not pending else 'WAIT_SOURCE_COMPLETION_OR_COPY',
        copied_closed_runs=count,pending=pending,scope='CLOSED_RUNS_AND_TRIAL_LEDGER_PARTIAL_FAILED_ROOTS_RETAINED_AT_SOURCE')
    write(state_root/'COLLECTION_STATUS.json',result)
    if result['state']=='COLLECTION_COMPLETE':write(state_root/'COLLECTION_COMPLETE.json',result)
    return 2 if pending else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','render','work','publish','status','export','collect'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--base-request',type=Path)
    p.add_argument('--baseline',type=Path);p.add_argument('--baseline-sha256')
    p.add_argument('--units-dir',type=Path);p.add_argument('--environment-file',type=Path)
    p.add_argument('--host');p.add_argument('--runtime',type=Path);p.add_argument('--archive',type=Path);p.add_argument('--state-root',type=Path)
    p.add_argument('--remote-python',default='python3')
    a=p.parse_args()
    try:
        if a.action=='prepare':result=prepare(a.base_request,a.root,a.baseline,a.baseline_sha256)
        elif a.action=='render':result=[str(v) for v in render(a.root,a.units_dir,a.environment_file)]
        elif a.action=='work':return work(a.root)
        elif a.action=='publish':return publish(a.root)
        elif a.action=='status':result=status(a.root)
        elif a.action=='export':result=export(a.root)
        else:return collect(a.host,a.root,a.runtime,a.archive,a.state_root,a.remote_python)
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        if a.action=='work':write(a.root/'BLOCKED.json',dict(state='BLOCKED_INTEGRITY_OR_RUNTIME',error_type=type(exc).__name__))
        if a.action=='collect':write(a.state_root/'COLLECTION_STATUS.json',dict(state='COLLECTION_REFUSED',error_type=type(exc).__name__))
        print(json.dumps(dict(state='SWEEP_REFUSED',error_type=type(exc).__name__)));return 78


if __name__=='__main__':raise SystemExit(main())
