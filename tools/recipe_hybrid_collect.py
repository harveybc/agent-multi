"""Independent checked mirror and existing warehouse publication for hybrid queue."""
import argparse
import fcntl
from pathlib import Path
import subprocess

from app.frozen_recipe_inputs import file_sha,require
from tools.recipe_sac_queue import read,write,now
from tools.recipe_hybrid_queue import authenticated,ARMS,SEEDS
from tools import rl_warehouse


def request_for(path,spec,sha,report):
    result=authenticated(path,sha)
    arm,seed=result['arm'],result['seed']
    require(arm in ARMS and seed in SEEDS and report in result['reports'],'REPORT_BINDING')
    key=f'{sha[:16]}-{result["unit"]}-{path.parent.name}-{report["period"]}'
    metrics=[]
    for name,value in report.items():
        if type(value) in (int,float) or value is None:
            unit=('USD' if name.endswith('_usd') or name=='initial_capital' else 'percent' if name.endswith('_pct')
                  else 'fraction' if name=='win_rate' else 'seconds' if name=='seconds' else 'ratio')
            metrics.append(dict(metric=name,value=value,split=report['period'].lower(),horizon=None,unit=unit))
    return dict(source_unit=key,report=dict(experiment_key=key,experiment_set_key='recipe-r3-hybrid-objectives',
        actor='musashi',lake='financial-data',lineage='UNVERIFIED',project='agent-multi',phase='DEVELOPMENT',
        config_sha256=sha,tags=dict(arm=arm,seed=str(seed),algorithm='PPO_CONDITIONAL_BETA',
            checkpoint='102400',checkpoint_archive_sha256=file_sha(path.parent/'model-102400.zip'),
            policy_tensor_sha256=report['policy_tensor_sha256'],recipe_sha256=spec['recipe_sha256'],
            expert='NONE',feedback='ROUTING_MASK_ONLY',scaling='FROZEN_TRAIN',test='CLOSED',
            policy_family=spec.get('design',{}).get('policy_family','legacy'),
            behavior_eligible=str(report.get('business_behavior_eligible','not_measured')),
            costs='ASSUMED_NOT_BROKER_VERIFIED',objective=ARMS[arm]['reward_mode']),
        datasets=[dict(lake='financial-data',resource='shared-feature-store-r3',
            sha256=read(spec['inputs']['recipe'])['store']['sha256'],role=report['period'].lower(),lineage='UNVERIFIED')],
        metrics=metrics))


def collect(spec):
    state,mirror=Path(spec['state_root']),Path(spec['mirror'])
    state.mkdir(parents=True,exist_ok=True);mirror.mkdir(parents=True,exist_ok=True)
    with (state/'COLLECT.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return
        subprocess.run(['rsync','-a','--checksum','--timeout=60','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',
            f'{spec["host"]}:{spec["remote_root"]}/',str(mirror)+'/'],check=True,timeout=300)
        sha=file_sha(mirror/'SPEC.private.json')
        require(sha==spec['campaign_sha256']==read(mirror/'IDENTITY.json')['spec_sha256'],'MIRROR_IDENTITY')
        campaign=read(mirror/'SPEC.private.json')
        # Recipe is already an independently pinned local file, not an untrusted
        # remote absolute path read during publication.
        recipe=Path(spec['local_recipe'])
        require(file_sha(recipe)==campaign['recipe_sha256'],'LOCAL_RECIPE_PIN')
        campaign['inputs']={**campaign['inputs'],'recipe':str(recipe)}
        pub=spec['publication'];rl_warehouse.prepare(pub,state)
        enqueued=read(state/'ENQUEUED.json') if (state/'ENQUEUED.json').exists() else {}
        for path in sorted(mirror.glob('*/attempt-*/RESULT.json')):
            result=authenticated(path,sha)
            for report in result['reports']:
                request=request_for(path,campaign,sha,report)
                key=request['source_unit']
                if key in enqueued: continue
                target=state/'reports'/(key+'.json');write(target,request)
                rl_warehouse.invoke(pub,state,'enqueue',target)
                enqueued[key]=file_sha(target);write(state/'ENQUEUED.json',enqueued)
        pump=state/'PUMP.private.json';write(pump,{k:pub[k] for k in ('warehouse_url','token_env')})
        result=rl_warehouse.invoke(pub,state,'pump',pump)
        write(state/'STATUS.json',dict(state='COPIED_AND_PUMPED',updated_at=now(),reports_enqueued=len(enqueued),
            warehouse=result,campaign_closed=(mirror/'CLOSE.json').exists()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    collect(read(p.parse_args().spec))


if __name__=='__main__': main()
