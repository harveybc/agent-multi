"""Independent copy and governed publication of paired encoder contrast results."""
import argparse
import fcntl
from pathlib import Path
import subprocess

from app.frozen_recipe_inputs import file_sha,require
from tools.recipe_sac_queue import read,write,now
from tools.hybrid_encoder_queue import authenticate,study_design
from tools import rl_warehouse


def collect(spec):
    root=Path(spec['mirror']);state=Path(spec['state_root'])
    root.mkdir(parents=True,exist_ok=True);state.mkdir(parents=True,exist_ok=True)
    with (state/'COLLECT.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        subprocess.run(['rsync','-a','--checksum','--timeout=60','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',
            f'{spec["host"]}:{spec["remote_root"]}/',str(root)+'/'],check=True,timeout=300)
        sha=file_sha(root/'SPEC.private.json');campaign=read(root/'SPEC.private.json')
        require(sha==spec['campaign_sha256'] and sha==read(root/'IDENTITY.json')['spec_sha256'],'CAMPAIGN_PIN')
        require(campaign['study']==study_design(),'DESIGN_PIN')
        status=read(root/'STATUS.json') if (root/'STATUS.json').exists() else {'completed':{}}
        recipe=Path(spec['local_recipe']);require(file_sha(recipe)==campaign['recipe_sha256'],'RECIPE_PIN')
        dataset_sha=read(recipe)['store']['sha256']
        pub=spec['publication'];rl_warehouse.prepare(pub,state)
        done=read(state/'ENQUEUED.json') if (state/'ENQUEUED.json').exists() else {}
        complete=0
        for unit in campaign['study']['units']:
            if unit not in status['completed']:continue
            path=(root/status['completed'][unit]).resolve()
            require(path.is_relative_to(root.resolve()),'RESULT_PATH')
            result=authenticate(path,sha,unit);complete+=1
            arm,seed=unit.rsplit('_s',1)
            for report in result['reports']:
                period=report['period']
                require(report['arm']==arm and report['seed']==int(seed),'REPORT_IDENTITY')
                key=f'{sha[:16]}-{unit}-{period}'
                if key in done:continue
                metrics=[dict(metric=k,value=v,split=period.lower(),horizon=None,
                    unit='USD' if k.endswith('_usd') or k=='initial_capital' else 'percent' if k.endswith('_pct')
                    else 'fraction' if k=='win_rate' else 'ratio') for k,v in report.items() if type(v) in (int,float) or v is None]
                request=dict(source_unit=key,report=dict(experiment_key=key,experiment_set_key='hybrid-encoder-contrast',
                    actor='musashi',lake='financial-data',lineage='UNVERIFIED',project='agent-multi',phase='DEVELOPMENT',
                    config_sha256=sha,tags=dict(scope='PAIRED_ENCODER_CONTRAST',arm=arm,seed=seed,
                        parent_spec_sha256=campaign['parent_spec_sha256'],result_sha256=file_sha(path),
                        test='CLOSED',costs='ASSUMED_NOT_BROKER_VERIFIED'),
                    datasets=[dict(lake='financial-data',resource='shared-feature-store-r3',sha256=dataset_sha,
                                   role=period.lower(),lineage='UNVERIFIED')],metrics=metrics))
                target=state/'reports'/(key+'.json');write(target,request)
                rl_warehouse.invoke(pub,state,'enqueue',target);done[key]=file_sha(target);write(state/'ENQUEUED.json',done)
        pump=state/'PUMP.private.json';write(pump,{k:pub[k] for k in ('warehouse_url','token_env')})
        result=rl_warehouse.invoke(pub,state,'pump',pump)
        write(state/'STATUS.json',dict(state='COPIED_AND_PUMPED',completed_fits=complete,expected_fits=6,
            reports=len(done),warehouse=result,campaign_state=status.get('state','PREPARED'),updated_at=now()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--inherit',type=Path)
    p.add_argument('--campaign',type=Path)
    p.add_argument('--remote-root');p.add_argument('--mirror');p.add_argument('--state-root')
    a=p.parse_args()
    if a.inherit:
        base=read(a.inherit)
        require(not a.spec.exists(),'NEW_COLLECTOR_CONFIG')
        cfg={k:base[k] for k in ('host','local_recipe','publication')}
        cfg.update(remote_root=a.remote_root,mirror=a.mirror,state_root=a.state_root,campaign_sha256=file_sha(a.campaign))
        write(a.spec,cfg)
    else:collect(read(a.spec))


if __name__=='__main__':main()
