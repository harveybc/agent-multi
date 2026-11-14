"""Automatically wait for a sealed replication, then independently score all endpoints."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.robust_adamw_direction import numpy_heads
from app.training_forensics import tree_digest
from tools.audit_fixed_actor_critics import covering_schedule
from tools.run_bayesian_head_contrast import verify_done,seal
from tools.run_discrete_curve_probe import read,write
from tools.verify_twin_complete_run import quality


def wait_complete(root,seconds):
    if not np.isfinite(seconds) or seconds<0:raise ValueError('VERIFY_WAIT_BUDGET')
    deadline=time.monotonic()+seconds
    while not (root/'DONE.json').exists():
        if (root/'PROGRESS.json').exists() and read(root/'PROGRESS.json').get('stage')=='FAILED':
            raise ValueError('VERIFY_PRODUCER_FAILED')
        remaining=deadline-time.monotonic()
        if remaining<=0:raise ValueError('VERIFY_RUN_NOT_COMPLETE')
        time.sleep(min(10.,remaining))


def run(a):
    wait_complete(a.run,a.wait_seconds);cfg=read(a.run/'CONFIG.json');verify_done(a.run,cfg['identity'])
    if cfg['schema']!='twin_protection_replication.v2' or cfg['transaction_guard']!='original_batched_plus_cpu_gpu_single':
        raise ValueError('VERIFY_TRANSACTION_VERSION')
    for name in ('source','probe','coverage'):
        parent=read(getattr(a,name)/'CONFIG.json')
        if parent['identity']!=cfg['parents'][name]:raise ValueError('VERIFY_PARENT_IDENTITY')
        verify_done(getattr(a,name),parent['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('VERIFY_SOURCE_ACTOR')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    result=read(a.run/'RESULT.json');units=result['units']
    keys={(u['fold'],u['seed'],u['method']) for u in units}
    expected={(f,s,m) for f in ('A','B') for s in (19,43) for m in ('active','complete')}
    if keys!=expected or len(units)!=8:raise ValueError('VERIFY_UNITS')
    support=read(a.coverage/'prepared/SUPPORT.json')
    z=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')['z'].double().numpy()
    y=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']])
    checked=[];table=[];transfer=[]
    for fold in ('A','B'):
        for seed in (19,43):
            source_path=a.source/'units'/f'{fold}_{seed}_cost'/'LATEST.pt'
            source=torch.load(source_path,weights_only=False,map_location='cpu')
            snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
            info=support['folds'][fold];fit=np.array(info['expanded']);old=np.array(info['masks']['fit'])
            scopes=dict(fit=fit,stop=np.array(info['masks']['stop']),audit=np.array(info['masks']['audit']))
            if set(fit)&(set(scopes['stop'])|set(scopes['audit'])):raise ValueError('VERIFY_LABEL_OVERLAP')
            pair=[next(u for u in units if (u['fold'],u['seed'],u['method'])==(fold,seed,m)) for m in ('active','complete')]
            for key in ('source_sha256','schedule_sha256','stable_added_indices'):
                if pair[0][key]!=pair[1][key]:raise ValueError('VERIFY_UNPAIRED_SUPPORT')
            stable=np.array(pair[0]['stable_added_indices'])
            if len(set(stable))!=len(stable) or not set(stable)<=set(fit)-set(old):raise ValueError('VERIFY_STABLE_SUPPORT')
            schedule=fit[covering_schedule(len(fit),32768,cfg['batch'],seed)]
            if pair[0]['schedule_sha256']!=tree_digest(schedule):raise ValueError('VERIFY_SCHEDULE')
            sourceq=numpy_heads(source['state'],z)
            transfer.append(dict(fold=fold,seed=seed,method='source',offset=0,
                scopes={k:quality(sourceq[:,ix],y[ix]) for k,ix in scopes.items()}))
            row=dict(fold=fold,seed=seed,source_regret=quality(sourceq[:,stable],y[stable])['regret'])
            for u in pair:
                method=u['method'];out=a.run/'units'/f'{fold}_{seed}_{method}';verify_done(out,cfg['identity'])
                if u['source_sha256']!=file_sha(source_path):raise ValueError('VERIFY_SOURCE_CHECKPOINT')
                if [p['offset'] for p in u['curve']]!=list(range(0,257,32)):raise ValueError('VERIFY_CURVE')
                for offset in (64,256):
                    cp=torch.load(out/f'ENDPOINT_{offset}.pt',weights_only=False,map_location='cpu')
                    want=next(p for p in u['curve'] if p['offset']==offset)
                    if cp['curve'][-1]!=want or cp['identity']!=cfg['identity'] or cp['source_sha256']!=u['source_sha256']:
                        raise ValueError('VERIFY_CHECKPOINT_RECEIPT')
                    if any(int(v['step'])!=source['accepted']+cp['accepted'] for v in cp['optimizer']['state'].values()):
                        raise ValueError('VERIFY_OPTIMIZER_STEPS')
                    q=numpy_heads(cp['state'],z);qo=q.min(0)[old];actions=snap['actions'];rr=np.arange(len(old))
                    gaps=qo[rr,actions,None]-qo;gaps[rr,actions]=np.inf
                    surplus=float(np.min(gaps-snap['margin'][:,None]))
                    if not np.array_equal(qo.argmax(1),actions) or surplus<0:raise ValueError('VERIFY_PROTECTED_DECISIONS')
                    assessment=want['assessment']
                    for ix,score in ((fit,assessment['observation']['fit']['numpy']),(stable,assessment['stable_added']['numpy'])):
                        actual=quality(q[:,ix],y[ix])
                        if any(abs(actual[k]-score[k])>1e-12 for k in actual):raise ValueError('VERIFY_METRICS')
                    checked.append(dict(fold=fold,seed=seed,method=method,offset=offset,min_margin_surplus=surplus))
                    transfer.append(dict(fold=fold,seed=seed,method=method,offset=offset,
                        scopes={k:quality(q[:,ix],y[ix]) for k,ix in scopes.items()}))
                    if offset==256:row[method]=dict(accepted=cp['accepted'],stable_added=quality(q[:,stable],y[stable]),
                        geometry_agrees=assessment['geometry_agrees'],scopes=transfer[-1]['scopes'])
            table.append(row)
    summary=dict(identity=cfg['identity'],seconds=result['seconds'],states_verified=len(checked),
        verification_code_sha256=file_sha(Path(__file__)),
        paired_cases=4,cold_replays=sum(u['cold_replays'] for u in units),
        resumed_transitions=sum(u['resumed_transitions'] for u in units),table=table,checks=checked,transfer=transfer,
        actor_unchanged=True,model_promoted=False,
        transfer_scope='Internal TRAIN; encoder pretrained on full TRAIN. No held-label fitting or checkpoint selection.')
    write(a.output/'RESULT.json',summary)
    lines=['# Twin Protection Paired Replication','','Fixed256 native head updates; not trading returns.','',
        '| Case | Active accepted | Complete accepted | Source regret | Active regret | Complete regret |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in table:
        lines.append(f"| {r['fold']}{r['seed']} | {r['active']['accepted']} | {r['complete']['accepted']} | {r['source_regret']:.9f} | {r['active']['stable_added']['regret']:.9f} | {r['complete']['stable_added']['regret']:.9f} |")
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');seal(a.output,cfg['identity']);print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','source','probe','coverage','bundle','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--wait-seconds',type=float,default=0);run(p.parse_args())
