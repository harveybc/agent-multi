"""Prepare pinned sequential ablations; native queue/timers own all learning."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import sys

from agent_plugins.sac_regularization import PROFILES
from tools import selected_rl_comparison as selected
from tools.curated_rl_campaign import digest
from tools.verify_rl_predecessor import predecessor_ready
from tools.verify_rl_static_engineering import save_status


SCHEMA='selected_rl_regularization_sequence.v1'
PREFIX='rl-selected-reg'


def immutable(path,body):
    data=json.dumps(body,sort_keys=True,indent=2)+'\n'
    if path.exists():
        if path.read_text()!=data:
            raise ValueError('SEQUENCE_EXISTING_IDENTITY_DIFFERS')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:
            stream.write(data)


def prepare(base_request,root,previous,previous_pin):
    root=selected._output_root(root)
    previous=Path(previous).resolve()
    if (not re.fullmatch('[0-9a-f]{64}',previous_pin)
            or previous.is_relative_to(root) or root.is_relative_to(previous)):
        raise ValueError('SEQUENCE_SEPARATE_PINNED_PREDECESSOR_REQUIRED')
    base_request=Path(base_request).resolve()
    base=json.loads(base_request.read_text())
    if base.get('schema')!='selected_rl_comparison_request.v4':
        raise ValueError('SEQUENCE_PAIRED_BASE_REQUIRED')
    code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),
                                         'rev-parse','HEAD'],text=True).strip()
    nodes=[]
    for name,profile in PROFILES.items():
        request=copy.deepcopy(base)
        request.update(schema='selected_rl_comparison_request.v5',regularization=profile)
        request['publication']['code_commit']=code_commit
        request['publication']['experiment_set_key']=base['publication']['experiment_set_key']+'-'+name
        path=root/'requests'/f'{name}.json'
        immutable(path,request)
        output=root/name
        contract=selected.materialize(path,output)
        pin=digest(contract)
        nodes.append(dict(profile=name,root=str(output),contract_sha256=pin,
            request_file=str(path),request_sha256=digest(path),
            predecessor_root=str(previous),predecessor_sha256=previous_pin))
        previous,previous_pin=output,pin
    spec=dict(schema=SCHEMA,base_request_file=str(base_request),
              base_request_sha256=digest(base_request),nodes=nodes)
    immutable(root/'SEQUENCE.json',spec)
    return spec


def checked_nodes(spec):
    if spec.get('schema')!=SCHEMA:
        raise ValueError('SEQUENCE_SCHEMA_REQUIRED')
    for node in spec['nodes']:
        root=Path(node['root'])
        if digest(root/'CONTRACT.json')!=node['contract_sha256']:
            raise ValueError('SEQUENCE_CONTRACT_DRIFT')
        if digest(node['request_file'])!=node['request_sha256']:
            raise ValueError('SEQUENCE_REQUEST_DRIFT')
        yield node,root


def status(spec):
    nodes=[]
    now=datetime.now(timezone.utc)
    for node,root in checked_nodes(spec):
        result=selected.read_status(root)
        closed=(root/'CLOSE.json').exists()
        result['authenticated_closed']=closed and predecessor_ready(root,node['contract_sha256'])=='READY'
        result['eta_seconds_to_cap_estimate']=None
        state=json.loads((root/'STATUS.json').read_text()) if (root/'STATUS.json').exists() else {}
        current=next((t for t in state.get('tasks',[]) if t.get('state')=='RUNNING'),None)
        pairs=[json.loads(p.read_text()) for p in sorted(root.glob('*/*/paired-monitor/step-*/PAIR.json'))]
        if current and pairs:
            step=pairs[-1]['step']
            elapsed=(now-datetime.fromisoformat(current['started_at'].replace('Z','+00:00'))).total_seconds()
            if step>0 and elapsed>0:
                result['eta_seconds_to_cap_estimate']=max(0,(100000-step)*elapsed/step)+pairs[-1]['wall_seconds']
        nodes.append(dict(profile=node['profile'],**result))
    return dict(schema=SCHEMA,completed=sum(n['authenticated_closed'] for n in nodes),
                declared=len(nodes),experiments=nodes,eta_kind='PROVISIONAL_CURRENT_ARM_TO_CAP_ONLY')


def publish(spec,spec_path):
    complete=True
    for node,root in checked_nodes(spec):
        if selected.publish_and_close(root)!=0:
            complete=False
        elif predecessor_ready(root,node['contract_sha256'])!='READY':
            complete=False
    if complete:
        save_status(Path(spec_path).parent/'SEQUENCE_COMPLETE.json',dict(
            schema=SCHEMA,state='ALL_FOUR_CLOSED_AND_WAREHOUSE_READBACK_VERIFIED',
            spec_sha256=digest(spec_path),profiles=list(PROFILES)))
    return 0 if complete else 2


def _q(value):
    value=str(value)
    if any(c in value for c in '\n\r\0'):
        raise ValueError('SEQUENCE_UNIT_ARGUMENT_INVALID')
    return json.dumps(value.replace('%','%%'))


def _setting_path(value):
    path=str(Path(value).resolve())
    _q(path)
    return path.replace('%','%%')


def render(spec,directory,environment_file,spec_path=None,runtime_root=None,*,
           prefix=PREFIX,sequence_module='tools.selected_rl_regularization_sequence',
           smoke_module='agent_plugins.sac_regularization'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    runtime=Path(runtime_root).resolve() if runtime_root else Path(__file__).resolve().parents[1]
    runtime_pin=subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip()
    paths=[]
    for node,root in checked_nodes(spec):
        contract=json.loads((root/'CONTRACT.json').read_text())
        if contract['publication']['code_commit']!=runtime_pin:
            raise ValueError('SEQUENCE_RENDER_RUNTIME_PIN_MISMATCH')
        uuid=contract['execution']['gpu_evidence']['assigned_gpu_uuid']
        if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',uuid):
            raise ValueError('SEQUENCE_GPU_UUID_REQUIRED')
        name=prefix+'-'+node['profile']
        gate=[sys.executable,'-m','tools.verify_rl_predecessor',
              '--predecessor-root',node['predecessor_root'],'--predecessor-sha256',node['predecessor_sha256'],
              '--root',root,'--gpu-uuid',uuid]
        start=['/usr/bin/env','CUDA_VISIBLE_DEVICES='+uuid,'OMP_NUM_THREADS=1',
               'OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',sys.executable,
               '-m','tools.selected_rl_comparison','run','--root',root]
        smoke=['/usr/bin/env','CUDA_VISIBLE_DEVICES='+uuid,'OMP_NUM_THREADS=1',
               'OPENBLAS_NUM_THREADS=1',sys.executable,'-m',
               smoke_module,'--smoke-root',root]
        service='\n'.join(['[Unit]','Description=Finite SAC regularization '+node['profile'],
            'After=network-online.target','ConditionPathExists=!'+str(root/'CLOSE.json'),
            'StartLimitIntervalSec=0','','[Service]','Type=exec',
            'WorkingDirectory='+_setting_path(runtime),'EnvironmentFile='+_setting_path(environment_file),
            'ExecCondition='+' '.join(_q(x) for x in gate),
            'ExecCondition='+' '.join(_q(x) for x in smoke),
            'ExecStart='+' '.join(_q(x) for x in start),
            'Restart=on-failure','RestartSec=60','RestartPreventExitStatus=78',
            'MemoryHigh=5G','MemoryMax=6G','MemorySwapMax=0','CPUQuota=200%',
            'TasksMax=256','RuntimeMaxSec=43200','KillMode=control-group','UMask=0077',''])
        paths.extend(_unit(directory,name,service))
    path=Path(spec_path) if spec_path else directory.parent/'SEQUENCE.json'
    command=['/usr/bin/env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1',
             'OPENBLAS_NUM_THREADS=1',sys.executable,'-m',
             sequence_module,'publish','--spec',path]
    publisher='\n'.join(['[Unit]','Description=Sequential SAC checkpoint publication and readback',
        'ConditionPathExists=!'+str(path.parent/'SEQUENCE_COMPLETE.json'),'',
        '[Service]','Type=oneshot','WorkingDirectory='+_setting_path(runtime),
        'EnvironmentFile='+_setting_path(environment_file),'ExecStart='+' '.join(_q(x) for x in command),
        'SuccessExitStatus=2','MemoryHigh=768M','MemoryMax=1500M','MemorySwapMax=0',
        'CPUQuota=100%','TasksMax=64','TimeoutStartSec=300','UMask=0077',''])
    paths.extend(_unit(directory,prefix+'-publication',publisher))
    return paths


def _unit(directory,name,service):
    timer='\n'.join(['[Unit]','Description=Finite resource-aware SAC timer',
        '','[Timer]','OnActiveSec=20','OnUnitInactiveSec=60','AccuracySec=5',
        'Unit='+name+'.service','','[Install]','WantedBy=timers.target',''])
    paths=[]
    for suffix,data in (('.service',service),('.timer',timer)):
        path=directory/(name+suffix)
        if path.exists() and path.read_text()!=data:
            raise ValueError('SEQUENCE_UNIT_DRIFT')
        path.write_text(data);paths.append(path)
    return paths


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','render','status','publish'))
    p.add_argument('--spec',type=Path)
    p.add_argument('--base-request',type=Path)
    p.add_argument('--root',type=Path)
    p.add_argument('--predecessor-root',type=Path)
    p.add_argument('--predecessor-sha256')
    p.add_argument('--environment-file',type=Path)
    p.add_argument('--runtime-root',type=Path)
    p.add_argument('--units-dir',type=Path)
    a=p.parse_args(argv)
    if a.action=='prepare':
        if not all((a.base_request,a.root,a.predecessor_root,a.predecessor_sha256)):
            p.error('prepare requires base request, root and pinned predecessor')
        result=prepare(a.base_request,a.root,a.predecessor_root,a.predecessor_sha256)
    else:
        if not a.spec:p.error('--spec required')
        spec=json.loads(a.spec.read_text())
        if a.action=='render':
            if not a.environment_file:p.error('--environment-file required')
            result=[str(x) for x in render(spec,a.units_dir or a.spec.parent/'units',
                                           a.environment_file,a.spec,a.runtime_root)]
        elif a.action=='status':result=status(spec)
        else:return publish(spec,a.spec)
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
