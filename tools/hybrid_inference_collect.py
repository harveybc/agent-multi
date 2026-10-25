"""Mirror sealed inference diagnostics and publish through the existing outbox."""
import argparse
import fcntl
from pathlib import Path
import subprocess

from app.frozen_recipe_inputs import file_sha,require
from tools.recipe_sac_queue import read,write,now
from tools.recipe_hybrid_queue import authenticated
from tools import rl_warehouse


def verify_seal(root):
    seal=read(root/'CLOSE.json')
    require(seal['state']=='COMPLETE' and seal['identity_sha256']==file_sha(root/'IDENTITY.json'),'DIAGNOSTIC_SEAL')
    for name,digest in seal['artifacts'].items():
        path=(root/name).resolve()
        require(path.is_relative_to(root.resolve()) and file_sha(path)==digest,'DIAGNOSTIC_ARTIFACT')
    return seal


def collect(spec):
    root=Path(spec['mirror']);state=Path(spec['state_root']);parent=Path(spec['parent_mirror'])
    root.mkdir(parents=True,exist_ok=True);state.mkdir(parents=True,exist_ok=True)
    with (state/'COLLECT.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        subprocess.run(['rsync','-a','--checksum','--timeout=60','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',
            f'{spec["host"]}:{spec["remote_root"]}/',str(root)+'/'],check=True,timeout=300)
        if not (root/'CLOSE.json').exists():
            write(state/'STATUS.json',dict(state='MIRRORED_WAITING_FOR_CLOSE',updated_at=now()));return
        seal=verify_seal(root);identity=read(root/'IDENTITY.json');sha=seal['identity_sha256']
        require(identity['tool_sha256']==spec['tool_sha256'] and
                identity['parent_spec_sha256']==file_sha(parent/'SPEC.private.json'),'DIAGNOSTIC_PARENT')
        recipe=Path(spec['local_recipe']);parent_spec=read(parent/'SPEC.private.json')
        require(file_sha(recipe)==parent_spec['recipe_sha256'],'RECIPE_PIN')
        dataset_sha=read(recipe)['store']['sha256']
        pub=spec['publication'];rl_warehouse.prepare(pub,state)
        done=read(state/'ENQUEUED.json') if (state/'ENQUEUED.json').exists() else {}
        results=read(root/'SUMMARY.json')
        require(len(results)==21,'DIAGNOSTIC_COMPLETENESS')
        for result in results:
            require(result['state']=='COMPLETE' and result['identity_sha256']==sha,'RESULT_IDENTITY')
            for report in result['reports']:
                seed=report['training_seed'];period=report['period']
                require(seed in (7,19,43) and period in ('TRAIN','INNER_A'),'REPORT_SCOPE')
                source=parent/f'CAT_FIXED_s{seed}'/'attempt-1'/'RESULT.json'
                authenticated(source,identity['parent_spec_sha256'])
                require(file_sha(source)==result['source_result_sha256'],'SOURCE_RESULT')
                require(file_sha(source.parent/'model-102400.zip')==result['checkpoint_sha256'],'SOURCE_CHECKPOINT')
                key=f'{sha[:16]}-{report["unit"]}-{period}'
                if key in done:continue
                metrics=[dict(metric=k,value=v,split=period.lower(),horizon=None,
                    unit='USD' if k.endswith('_usd') or k=='initial_capital' else 'percent' if k.endswith('_pct')
                    else 'fraction' if k=='win_rate' else 'ratio') for k,v in report.items() if type(v) in (int,float) or v is None]
                request=dict(source_unit=key,report=dict(experiment_key=key,experiment_set_key='hybrid-v2-inference-diagnostic',
                    actor='musashi',lake='financial-data',lineage='UNVERIFIED',project='agent-multi',phase='DEVELOPMENT',
                    config_sha256=sha,tags=dict(scope='INFERENCE_DIAGNOSTIC_NOT_SELECTED_STRATEGY',fits='0',
                        mode=report['inference_mode'],training_seed=str(seed),evaluation_seed=str(report['evaluation_seed']),
                        parent_spec_sha256=identity['parent_spec_sha256'],checkpoint_archive_sha256=result['checkpoint_sha256'],
                        test='CLOSED',costs='ASSUMED_NOT_BROKER_VERIFIED'),
                    datasets=[dict(lake='financial-data',resource='shared-feature-store-r3',sha256=dataset_sha,
                                   role=period.lower(),lineage='UNVERIFIED')],metrics=metrics))
                target=state/'reports'/(key+'.json');write(target,request)
                rl_warehouse.invoke(pub,state,'enqueue',target);done[key]=file_sha(target);write(state/'ENQUEUED.json',done)
        pump=state/'PUMP.private.json';write(pump,{k:pub[k] for k in ('warehouse_url','token_env')})
        result=rl_warehouse.invoke(pub,state,'pump',pump)
        write(state/'STATUS.json',dict(state='COPIED_AND_PUMPED',reports=len(done),warehouse=result,updated_at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    collect(read(p.parse_args().spec))
