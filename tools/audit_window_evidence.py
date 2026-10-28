"""Read-only audit of the archived fine-window study; no fitting or dispatch."""
import argparse
import copy
import csv
import hashlib
import zipfile
from pathlib import Path,PurePosixPath

from tools.collect_closed_raw_rl import verify
from tools.paired_checkpoint_diagnostics import read,write,digest,trace_metrics
from tools.window_common_support import trace_score,require_equal_support

OPERATIONAL_KEYS=('config_manifest_file','curated_input_manifest','input_data_file',
    'paired_campaign_root','paired_campaign_sha256','paired_source_unit','progress_file',
    'resolved_config_file','results_file','return_trace_file','save_config','save_model',
    'session_flatten_custody_root','training_progress_file')


def scientific_config(config):
    result=copy.deepcopy(config)
    for key in OPERATIONAL_KEYS:result.pop(key,None)
    result.pop('window_size',None)
    if result.get('mode')=='inference':result.pop('load_model',None)
    result.setdefault('temporal_core_layers',1)
    if 'data' in result:result['data'].pop('input_data_file',None)
    evidence=result.get('queue_execution',{}).get('gpu_evidence',{})
    for key in ('path','sha256'):evidence.pop(key,None)
    return result


def differences(a,b):
    return sorted(k for k in a.keys()|b.keys() if (k not in a or k not in b or a[k]!=b[k]))


def policy_identity(task,summary):
    if task['split']=='train':pin=summary['checkpoint_policy_tensor_sha256']
    else:
        static=summary['static_policy']
        if static['before']!=static['after']:raise ValueError('AUDIT_POLICY_CHANGED_IN_EVALUATION')
        pin=static['before']
    if task.get('policy_tensor_sha256',pin)!=pin:raise ValueError('AUDIT_POLICY_IDENTITY_MISMATCH')
    return pin


def relocated(root,source,path):
    path=PurePosixPath(path)
    if not path.is_absolute() or '..' in path.parts:raise ValueError('AUDIT_PATH_ESCAPE')
    result=Path(root).joinpath(*path.relative_to(source).parts)
    if not result.resolve().is_relative_to(Path(root).resolve()):raise ValueError('AUDIT_PATH_ESCAPE')
    return result


def audit(archive):
    import math
    from stable_baselines3 import SAC
    from agent_plugins.sac_agent import _policy_tensor_hash
    archive=Path(archive).resolve();spec=read(archive/'SPEC.json');state=read(archive/'SWEEP_STATE.json')
    if state['spec_sha256']!=digest(archive/'SPEC.json'):raise ValueError('AUDIT_SPEC_CHANGED')
    nodes=[dict(name='baseline',**spec['baseline'],expected_close=spec['baseline_outcome']['close_sha256'])]
    nodes.extend(dict(name=t['name'],**t['attempts'][-1],expected_close=t['outcome']['close_sha256'])
        for t in state['trials'] if t['state']=='CLOSED')
    if len(nodes)!=5:raise ValueError('AUDIT_COMPLETE_FIVE_RUNS_REQUIRED')
    rows=[];reference=None
    for node in nodes:
        root=archive/'runs'/node['name'];source=PurePosixPath(node['root'])
        proofs=verify(root,source,node['expected_close'])
        body=read(root/'CONTRACT.json');journal=read(root/'STATUS.json');roles={};effective={};common={}
        if digest(root/'CONTRACT.json')!=node['contract_sha256']:raise ValueError('AUDIT_CONTRACT_CHANGED')
        for task in journal['tasks']:
            role=task['split'];cfg=read(relocated(root,source,task['effective_config']))
            summary=read(relocated(root,source,task['summary']))
            data=relocated(root,source,cfg['input_data_file'])
            if digest(data)!=summary['curated_input_identity']['input_sha256']:raise ValueError('AUDIT_INPUT_CHANGED')
            trace=relocated(root,source,task['return_trace_file'])
            measured=trace_metrics(trace,cfg['initial_cash'],cfg['entry_day_timezone'])
            warmup=1+(cfg['window_size']-1)*cfg['observation_stride_bars']
            if (measured['rows']!=summary['curated_input_identity']['row_count']-warmup
                    or summary['evaluation_truncated_by_cap'] is not False
                    or not math.isclose(measured['return_pct']/100,summary['total_return'],abs_tol=1e-12)):
                raise ValueError('AUDIT_METRIC_OR_ROWS')
            common[role]=trace_score(trace,cfg['initial_cash'],cfg['entry_day_timezone'],35-warmup)
            effective[role]=scientific_config(cfg)
            if cfg['train_seed']!=7 or cfg['eval_seed']!=7:raise ValueError('AUDIT_SEED_MISMATCH')
            roles[role]=dict(input=summary['curated_input_identity'],native_rows=measured['rows'],
                native_return_pct=measured['return_pct'],policy_tensor_sha256=policy_identity(task,summary))
            if role=='train':
                compute=summary['compute_contract'];stop=summary['paired_early_stopping']
                if (compute['training_transitions_observed']!=100000 or compute['gradient_updates']!=99000
                        or stop['selected_step']!=100000 or stop['evaluations']!=4
                        or stop['stop_reason']!='FIXED_BUDGET'):raise ValueError('AUDIT_TRAINING_BUDGET')
                checkpoint=relocated(root,source,task['checkpoint'])
                with zipfile.ZipFile(checkpoint) as zipped:
                    runtime_info_sha256=hashlib.sha256(zipped.read('system_info.txt')).hexdigest()
                policy=SAC.load(checkpoint,device='cpu')
                if _policy_tensor_hash(policy.policy)!=policy_identity(task,summary):raise ValueError('AUDIT_WEIGHTS')
                architecture=dict(actor_parameters=sum(p.numel() for p in policy.actor.parameters()),
                    critic_parameters=sum(p.numel() for p in policy.critic.parameters()),
                    feature_shape=list(policy.observation_space['features'].shape))
                del policy
        if roles['train']['policy_tensor_sha256']!=roles['validation']['policy_tensor_sha256']:
            raise ValueError('AUDIT_EVALUATED_ANOTHER_POLICY')
        comparable=dict(common=scientific_config(body['common_config']),effective=effective,
            selection=body['selected_input_binding'],inputs={r:v['input'] for r,v in roles.items()},
            gym=body['execution']['gym_code_commit'],gpu=body['execution']['gpu_evidence']['assigned_gpu_uuid'],
            saved_runtime_info_sha256=runtime_info_sha256)
        if reference is None:reference=copy.deepcopy(comparable);ref_support=common
        unexpected=differences(reference,comparable)
        if unexpected:raise ValueError('AUDIT_UNEXPECTED_SCIENTIFIC_DIFFERENCE:'+','.join(unexpected))
        for role in common:require_equal_support(common[role],ref_support[role])
        rows.append(dict(name=node['name'],window=body['common_config']['window_size'],
            contract_sha256=node['contract_sha256'],close_sha256=node['expected_close'],
            artifacts_verified=len(proofs),source_code_commit=body['publication']['code_commit'],
            actual_training_transitions=100000,actual_gradient_updates=99000,
            saved_runtime_info_sha256=runtime_info_sha256,
            common_support=common,roles=roles,architecture=architecture,
            unexpected_scientific_differences=unexpected))
    return dict(schema='window_evidence_audit.v1',state='ARCHIVED_CONTRACTS_METRICS_WEIGHTS_VERIFIED',
        runs=rows,auditor_sha256=digest(__file__),operational_keys_normalized=list(OPERATIONAL_KEYS),
        control_reused_not_repeated=True,seeds=[7],optimal_window_established=False,
        differences_requiring_interpretation=['head_parameter_count','warmup_and_training_episode_start',
            'shape_dependent_initialization','baseline_older_code_with_default_depth1'],
        limits=['No fresh control fit in this audit','No multi-seed superiority established'],TEST_read=False)


def initialization_probe(windows):
    import inspect
    import gymnasium as gym
    import numpy as np
    import torch
    from agent_plugins.sac_agent import Plugin,_policy_tensor_hash
    from agent_plugins.temporal_features import PerFeatureTemporalExtractor
    if not windows or any(type(n) is not int or n not in (9,11,13,15,17) for n in windows):
        raise ValueError('AUDIT_DECLARED_INITIALIZATION_WINDOWS')
    def tensor_hash(module):
        h=hashlib.sha256()
        for name,value in module.state_dict().items():
            h.update(name.encode());h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    torch.set_num_threads(1);rows=[]
    for n in windows:
        class Synthetic(gym.Env):
            observation_space=gym.spaces.Dict(dict(features=gym.spaces.Box(-10,10,(n,15),dtype=np.float32),
                state=gym.spaces.Box(-10,10,(17,),dtype=np.float32)))
            action_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)
            def reset(self,*,seed=None,options=None):
                super().reset(seed=seed)
                return {k:np.zeros(v.shape,dtype=np.float32) for k,v in self.observation_space.spaces.items()},{}
        cfg=dict(observation_encoder='per_feature_conv1d',net_arch=[8],temporal_branch_filters=2,
            temporal_core_filters=4,temporal_core_layers=1,temporal_kernel_size=3,
            train_seed=7,device='cpu',buffer_size=64,batch_size=8,learning_starts=0)
        model=Plugin(cfg).build(Synthetic(),cfg)
        rows.append(dict(window=n,policy_tensor_sha256=_policy_tensor_hash(model.policy),
            actor_extractor_sha256=tensor_hash(model.actor.features_extractor),
            critic_extractor_sha256=tensor_hash(model.critic.features_extractor),
            actor_parameters=sum(v.numel() for v in model.actor.parameters()),
            critic_parameters=sum(v.numel() for v in model.critic.parameters()),gradient_updates=model._n_updates))
    return dict(classification='SYNTHETIC_INITIALIZATION_ONLY',runs=rows,
        source_sha256={cls.__name__:digest(inspect.getsourcefile(cls)) for cls in (Plugin,PerFeatureTemporalExtractor)})


def business_trace_hash(path):
    h=hashlib.sha256()
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream)
        fields=[k for k in reader.fieldnames if k not in ('run_id','episode_id')]
        for row in reader:
            h.update(repr(tuple((k,row[k]) for k in fields)).encode())
    return h.hexdigest()


def compare_control(archive,repeat,source,contract_pin):
    archive=Path(archive);repeat=Path(repeat)
    if not (repeat/'CLOSE.json').exists():return dict(state='WAIT_REPEAT_CLOSURE',TEST_read=False)
    spec=read(archive/'SPEC.json');ref=archive/'runs/baseline'
    verify(ref,PurePosixPath(spec['baseline']['root']),spec['baseline_outcome']['close_sha256'])
    verify(repeat,PurePosixPath(source),digest(repeat/'CLOSE.json'))
    if digest(repeat/'CONTRACT.json')!=contract_pin:raise ValueError('AUDIT_REPEAT_CONTRACT')
    pairs={};differences_found=[]
    for label,root,origin in [('reference',ref,spec['baseline']['root']),('repeat',repeat,source)]:
        body=read(root/'CONTRACT.json')
        if body['common_config']['window_size']!=13 or body['seeds']!=[7]:raise ValueError('AUDIT_NOT_CONTROL13')
        roles={}
        for task in read(root/'STATUS.json')['tasks']:
            role=task['split'];summary=read(relocated(root,origin,task['summary']))
            cfg=read(relocated(root,origin,task['effective_config']))
            roles[role]=dict(scientific_config=scientific_config(cfg),
                input_identity=summary['curated_input_identity'],policy_tensor_sha256=policy_identity(task,summary),
                business_trace_sha256=business_trace_hash(relocated(root,origin,task['return_trace_file'])),
                metrics={k:summary[k] for k in ('net_profit','total_return','max_drawdown_pct',
                    'trades_closed','win_rate_pct','profit_factor','sharpe_ratio')})
        pairs[label]=roles
    for role in ('train','validation'):
        a,b=pairs['reference'][role],pairs['repeat'][role]
        differences_found.extend(role+'.'+k for k in differences(a,b))
    return dict(schema='window_control_repeat.v1',
        state='REPRODUCED_BITWISE_POLICY_AND_BUSINESS_TRACES' if not differences_found else 'REPRODUCTION_MISMATCH',
        differences=differences_found,pairs=pairs,comparison='Exact policy/business traces/metrics; identifiers excluded from trace hash only',
        single_seed=True,optimal_window_established=False,TEST_read=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeat-archive',type=Path);p.add_argument('--repeat-source');p.add_argument('--repeat-contract')
    p.add_argument('--initialization-windows',type=int,nargs='+')
    a=p.parse_args()
    if a.initialization_windows:
        write(a.output,initialization_probe(a.initialization_windows));print('SYNTHETIC_INITIALIZATION_ONLY');return
    if a.archive is None:p.error('--archive required for retained evidence audit')
    if a.output.resolve().is_relative_to(a.archive.resolve()):raise ValueError('AUDIT_SEPARATE_OUTPUT')
    if a.repeat_archive:
        if not a.repeat_source or not a.repeat_contract:p.error('repeat requires source and contract pin')
        if a.output.resolve().is_relative_to(a.repeat_archive.resolve()):raise ValueError('AUDIT_SEPARATE_OUTPUT')
        result=compare_control(a.archive,a.repeat_archive,a.repeat_source,a.repeat_contract)
    else:result=audit(a.archive)
    write(a.output,result);print(result['state'])


if __name__=='__main__':main()
