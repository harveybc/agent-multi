"""Two CPU label workers followed by four resumable native-head GPU fits."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from app.action_ranking_contrast import ranking_metrics
from app.critic_blocked_stopping import bounded_snapshot
from app.critic_fit_probe import fit_metrics
from app.critic_target_contract import contract_loss
from app.decision_attribution import continuation,state_view
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.native_fit_coverage import compact_support,validate_record
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest,support_hash
from app.role_routed_critic import RoleHeads
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment
from app.training_forensics import tree_digest,rng_state,restore_rng
from env_plugins.discrete_curve_native import HOUR
from tools.audit_fixed_actor_critics import covering_schedule
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_teacher_coverage import metrics
from tools.run_training_forensics import template,checkpoint

ROOT=Path(__file__).resolve().parents[1]


def prepare(a,cfg):
    out=a.output/'prepared'
    if (out/'DONE.json').exists():verify_done(out,cfg['identity']);return
    source=torch.load(a.coverage/'cache/DATA.pt',weights_only=True)
    original=torch.load(a.previous/'cache/DATA.pt',weights_only=True)
    support=read(a.coverage/'cache/SUPPORT.json');ix,folds=compact_support(support)
    records=read(a.labels/'panel/RESULT.json')['records']
    rows=np.array(support['rows'])[ix];n=len(records)
    if n!=239 or rows[:n].tolist()!=[r['row'] for r in records]:raise ValueError('NATIVE_ORIGINAL_ROWS')
    z=source['z'][:,ix]
    if not torch.equal(z[:,:n],original['z']):raise ValueError('NATIVE_ORIGINAL_CACHE')
    if tree_digest(source['initial'])!=tree_digest(original['initial']):raise ValueError('NATIVE_INITIAL')
    real=np.stack([validate_record(r) for r in records])
    if not np.array_equal(real,original['targets'].numpy()):raise ValueError('NATIVE_ORIGINAL_TARGET')
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    model,_=load_bundle(a.bundle,'cuda');market=market_for(a.parent,spec,model.recipe)
    if support_hash(market.inputs)!=cfg['support'] or model.recipe['list']['n']!=71:raise ValueError('NATIVE_SOURCE')
    frozen=tree_digest(model.loss.state_dict());flags={k:v.requires_grad for k,v in model.loss.named_parameters()}
    train=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    logits=emit_actor_reference(model,market.inputs,train,flags)[0];signals=prepared_signals(logits)[0]
    entries=[]
    for row in rows[n:]:
        ts=int(market.inputs.row_ts_ns[row]);year=pd.Timestamp(ts,tz='UTC').year
        if not market.inputs.valid[row] or market.inputs.period[row]!='TRAIN' or year not in (2020,2021):
            raise ValueError('NATIVE_NEW_ROW')
        entries.append(dict(row=int(row),timestamp=ts,year=year,window_start=int(market.inputs.slot_ts_ns[row,0])))
    replays=[];zg=original['z'].to('cuda')
    for fold,info in folds.items():
        old_masks=read(a.previous/f'cache/FOLD_{fold}.json')
        if any(info['masks'][k]!=old_masks[k] for k in ('fit','stop','audit')):raise ValueError('NATIVE_MASKS')
        for seed in cfg['seeds']:
            d=a.previous/f'units/{fold}_{seed}_base';r=read(d/'RESULT.json');verify_done(d,r['identity'])
            h=RoleHeads(template('cuda'),1,seed)
            for endpoint in ('initial','final'):
                state=original['initial'][seed] if endpoint=='initial' else torch.load(d/'LATEST.pt',weights_only=False)['model']
                h.load_state_dict(state)
                with torch.no_grad():q=h(zg).mean(2).cpu().numpy()
                if {k:fit_metrics(q[:,v],real[v])['minimum'] for k,v in info['masks'].items()}!=r[endpoint]:
                    raise ValueError('NATIVE_CONTROL_REPLAY')
                replays.append(dict(fold=fold,seed=seed,endpoint=endpoint,exact=True))
    if frozen!=tree_digest(model.loss.state_dict()):raise ValueError('NATIVE_SOURCE_WEIGHTS')
    if not np.array_equal(logits[train],emit_actor_reference(model,market.inputs,train,flags)[0][train]):
        raise ValueError('NATIVE_SOURCE_SIGNALS')
    checkpoint(out/'DATA.pt',dict(z=z,initial=original['initial'],signals=torch.from_numpy(signals)))
    write(out/'SUPPORT.json',dict(rows=rows.tolist(),old_count=n,folds=folds,new=entries,original=records))
    write(out/'VERIFY.json',dict(control_replays=replays,source_exact=True,actor_exact=True,
        original_embeddings_exact=True,original_targets_exact=True,new_rows=len(entries)))
    seal(out,cfg['identity'])


def label_worker(a):
    configure(318);cfg=read(a.output/'CONFIG.json');root=a.output/'labels'/str(a.shard)
    if (root/'DONE.json').exists():verify_done(root,cfg['identity']);return
    verify_done(a.output/'prepared',cfg['identity']);support=read(a.output/'prepared/SUPPORT.json')
    signals=torch.load(a.output/'prepared/DATA.pt',weights_only=True)['signals'].numpy()
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('NATIVE_WORKER_SOURCE')
    model,_=load_bundle(a.bundle,'cpu');market=market_for(a.parent,spec,model.recipe)
    if support_hash(market.inputs)!=cfg['support']:raise ValueError('NATIVE_WORKER_SUPPORT')
    wanted={r['row']:r for i,r in enumerate(support['new']) if i%cfg['workers']==a.shard}
    witnesses={}
    for year in (2020,2021):
        r=next(r for r in support['original'] if r['year']==year);witnesses[r['row']]=r
    env=environment(market,read(a.parent/'RISK.json'),spec['design']['horizon'],
                    ['2020-01-01','2023-01-01'],cfg['gamma']);env.reset()
    done={};verified=[];started=time.monotonic()
    while not env.finished:
        row=int(env.rows[env.bar])
        if row in wanted or row in witnesses:
            original=witnesses.get(row);meta=wanted.get(row,original);year=meta['year']
            state=state_view(env);bounded=bounded_snapshot(env,pd.Timestamp(f'{year+1}-01-01',tz='UTC').value)
            terminal=int(bounded.engine.v[bounded.engine.last])+HOUR
            if original is not None:
                if digest(state)!=original['state_sha256'] or terminal!=original['terminal']:
                    raise ValueError('NATIVE_WITNESS_STATE')
                if [continuation(bounded,signals,j) for j in range(3)]!=original['outcomes']:
                    raise ValueError('NATIVE_WITNESS_OUTCOMES')
                verified.append(row)
            if row in wanted:
                path=root/'rows'/f'{row}.json';position=0 if env.engine.pos is None else env.engine.pos.direction
                metadata=dict(**wanted[row],identity=cfg['identity'],terminal=terminal,
                              state_sha256=digest(state),position=position)
                if path.exists():
                    rec=read(path)
                    if any(rec.get(k)!=v for k,v in metadata.items()):raise ValueError('NATIVE_LABEL_RESUME')
                else:
                    outcomes=[continuation(bounded,signals,j) for j in range(3)]
                    allowed=all(o['first_effect']['entries']>0 for o in outcomes[1:])
                    kind=('open_long' if position==1 else 'open_short') if position else (
                        'flat_both_allowed' if allowed else 'flat_restricted')
                    rec=dict(**metadata,kind=kind,outcomes=outcomes,outcomes_sha256=digest(outcomes))
                    validate_record(rec);write(path,rec)
                validate_record(rec);done[row]=rec
                write(root/'PROGRESS.json',dict(completed=len(done),total=len(wanted),at=now(),seconds=time.monotonic()-started))
            if state_view(env)!=state:raise ValueError('NATIVE_WORKER_PARENT_CHANGED')
        if len(done)==len(wanted) and len(verified)==len(witnesses):break
        env.step(int(signals[row]))
    if set(done)!=set(wanted) or set(verified)!=set(witnesses):raise ValueError('NATIVE_LABEL_MISSING')
    write(root/'RESULT.json',dict(identity=cfg['identity'],records=[done[r] for r in sorted(done)],
                                 witnesses=verified,seconds=time.monotonic()-started))
    seal(root,cfg['identity'])


def collect(a,cfg,progress):
    jobs=[]
    try:
        for shard in range(cfg['workers']):
            root=a.output/'labels'/str(shard)
            if (root/'DONE.json').exists():verify_done(root,cfg['identity']);continue
            root.mkdir(parents=True,exist_ok=True)
            log=(a.output/f'label_worker_{shard}.log').open('a')
            command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--stage','labels','--shard',str(shard)]
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            jobs.append((p,log))
        while any(p.poll() is None for p,_ in jobs):
            if any(p.poll() not in (None,0) for p,_ in jobs):raise RuntimeError('NATIVE_LABEL_WORKER_FAILED')
            states=[read(p) for p in (a.output/'labels').glob('*/PROGRESS.json')]
            progress('LABELS',completed=sum(r['completed'] for r in states),total=294,workers=cfg['workers'])
            time.sleep(2)
        if any(p.returncode!=0 for p,_ in jobs):raise RuntimeError('NATIVE_LABEL_WORKER_FAILED')
    finally:
        for p,log in jobs:
            if p.poll() is None:p.terminate()
            p.wait();log.close()
    support=read(a.output/'prepared/SUPPORT.json');byrow={}
    for shard in range(cfg['workers']):
        root=a.output/'labels'/str(shard);verify_done(root,cfg['identity'])
        for r in read(root/'RESULT.json')['records']:
            if r['row'] in byrow:raise ValueError('NATIVE_LABEL_DUPLICATE')
            byrow[r['row']]=r
    records=support['original']+[byrow[row] for row in support['rows'][support['old_count']:]]
    if set(byrow)!={r['row'] for r in support['new']}:raise ValueError('NATIVE_LABEL_COVERAGE')
    write(a.output/'PANEL.json',dict(records=records,kind_counts=dict(Counter(r['kind'] for r in records))))
    return records,support


def fit(a,cfg,records,support,progress):
    data=torch.load(a.output/'prepared/DATA.pt',weights_only=True);z=data['z'].to('cuda')
    targets=np.stack([validate_record(r) for r in records]);y=torch.tensor(targets,dtype=torch.float32,device='cuda')
    nold=support['old_count'];results=[]
    for fold,info in support['folds'].items():
        masks={k:np.array(v) for k,v in info['masks'].items()}
        for kind in ('flat_both_allowed','flat_restricted','open_long','open_short'):
            ix=np.array([i for i in masks['audit'] if records[i]['kind']==kind],dtype=int)
            if len(ix):masks['audit_'+kind]=ix
        fitrows=np.array(info['expanded'])
        for seed in cfg['seeds']:
            out=a.output/'units'/f'{fold}_{seed}';out.mkdir(parents=True,exist_ok=True)
            if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
            head=RoleHeads(template('cuda'),1,seed);head.load_state_dict(data['initial'][seed])
            opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            schedule=fitrows[covering_schedule(len(fitrows),32768,128,seed)]
            latest=out/'LATEST.pt';curve=[];start=0;clipped=0
            if latest.exists():
                s=torch.load(latest,weights_only=False)
                if s['identity']!=cfg['identity']:raise ValueError('NATIVE_FIT_RESUME')
                head.load_state_dict(s['state']);opt.load_state_dict(s['optimizer']);restore_rng(s['rng'])
                curve=s['curve'];start=s['step'];clipped=s['clipped']
            for step in range(start,32769):
                if step==start and latest.exists():continue
                if step:
                    ix=schedule[step-1];opt.zero_grad(set_to_none=True)
                    loss=contract_loss(head(z[:,ix]).mean(2),y[ix],'mse');loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(head.parameters(),10.,error_if_nonfinite=True)
                    clipped+=int(float(norm)>10);opt.step()
                if step not in (0,64,256,1024) and step%2048:continue
                curve.append(dict(step=step,metrics=metrics(head,z,targets,fitrows,masks,nold)))
                snap=dict(identity=cfg['identity'],state=head.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
                          step=step,curve=curve,clipped=clipped)
                checkpoint(latest,snap);write(out/'CURVE.json',curve)
                if step in (8192,32768):checkpoint(out/f'ENDPOINT_{step}.pt',snap)
                progress('FIT',fold=fold,seed=seed,step=step,total=32768,completed=len(results),arms=4)
            endpoints={}
            for step,label in ((8192,'equal_updates'),(32768,'equal_presentations')):
                saved=torch.load(out/f'ENDPOINT_{step}.pt',weights_only=False)
                if any(g['lr']!=cfg['lr'] or g['weight_decay']!=cfg['weight_decay'] for g in saved['optimizer']['param_groups']):
                    raise ValueError('NATIVE_OPTIMIZER')
                cold=RoleHeads(template('cuda'),1,seed);cold.load_state_dict(saved['state'])
                actual=metrics(cold,z,targets,fitrows,masks,nold)
                if actual!=next(r['metrics'] for r in curve if r['step']==step):raise ValueError('NATIVE_COLD_RELOAD')
                endpoints[label]=actual
            cpath=a.previous/f'units/{fold}_{seed}_base'
            control=RoleHeads(template('cuda'),1,seed)
            control.load_state_dict(torch.load(cpath/'LATEST.pt',weights_only=False)['model'])
            with torch.no_grad():q=control(z[:,:nold]).mean(2).cpu().numpy()
            baseline={k:fit_metrics(q[:,ix],targets[ix])['minimum'] for k,ix in masks.items()}
            actor={k:ranking_metrics(np.eye(3)[data['signals'].numpy()[np.array(support['rows'])[ix]]],targets[ix])
                   for k,ix in masks.items() if k.startswith('audit')}
            r=dict(identity=cfg['identity'],fold=fold,seed=seed,fit_rows=len(fitrows),cold_exact=True,
                   initial=curve[0]['metrics'],initialization_sha256=tree_digest(data['initial'][seed]),
                   schedule_sha256=tree_digest(schedule),clipped_updates=clipped,control=baseline,actor=actor,
                   fit_kind_counts=dict(Counter(records[i]['kind'] for i in fitrows)),
                   constant={k:fit_metrics(np.tile(targets[fitrows].mean(0),(2,len(ix),1)),targets[ix])['minimum']
                             for k,ix in masks.items()},**endpoints)
            write(out/'RESULT.json',r);seal(out,cfg['identity']);results.append(r)
    return results


def run(a):
    started=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('NATIVE_DEVICE')
    pc=read(a.previous/'CONFIG.json');cc=read(a.coverage/'CONFIG.json');lc=read(a.labels/'CONFIG.json')
    for path,c in ((a.previous,pc),(a.coverage,cc),(a.labels,lc)):verify_done(path,c['identity'])
    if cc['forensics']!=pc['identity'] or pc['labels']!=lc['identity']:raise ValueError('NATIVE_LINEAGE')
    cfg=dict(schema='native_fit_coverage.v1',previous=pc['identity'],coverage=cc['identity'],labels=lc['identity'],
             source_sha256=cc['source_sha256'],support=cc['support'],seeds=[19,43],factor=4,workers=2,
             updates=[8192,32768],batch=128,lr=lc['lr'],weight_decay=.01,gamma=lc['gamma'],
             code={k:file_sha(ROOT/k) for k in ('tools/run_native_fit_coverage.py','app/native_fit_coverage.py',
                 'app/decision_attribution.py','app/critic_blocked_stopping.py','tools/run_teacher_coverage.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('NATIVE_CHECKPOINT')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('NATIVE_RUN_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**kw):write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    progress('PREPARE');prepare(a,cfg);torch.cuda.empty_cache()
    records,support=collect(a,cfg,progress);results=fit(a,cfg,records,support,progress)
    comparisons=[]
    for r in results:
        for endpoint in ('equal_updates','equal_presentations'):
            new,old=r[endpoint],r['control'];flat='audit_flat_both_allowed'
            passed=(new['audit']['mean_regret']<old['audit']['mean_regret'] and
                    new['audit']['rmse']<=old['audit']['rmse'] and new[flat]['mean_regret']<=old[flat]['mean_regret'])
            comparisons.append(dict(fold=r['fold'],seed=r['seed'],endpoint=endpoint,passes=passed,
                regret=new['audit']['mean_regret'],control_regret=old['audit']['mean_regret'],
                rmse=new['audit']['rmse'],control_rmse=old['audit']['rmse'],
                flat_regret=new[flat]['mean_regret'],control_flat_regret=old[flat]['mean_regret']))
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('NATIVE_SOURCE_CHANGED')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],arms=results,comparisons=comparisons,
        source_preserved=True,actor_updates=0,encoder_updates=0,model_promoted=False,validation_read=False,test_read=False))
    report(a.output,results,comparisons)
    seal(a.output,cfg['identity']);progress('COMPLETE',arms=4,completed=4)


def report(root,results,comparisons):
    lines=['# Native FIT Coverage Results','','Internal native-label critic calibration; not trading returns.','',
           '| Fold | Seed | Endpoint | Control regret | New regret | Control RMSE | New RMSE | Pass |',
           '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for c in comparisons:
        lines.append(f"| {c['fold']} | {c['seed']} | {c['endpoint']} | {c['control_regret']:.6f} | {c['regret']:.6f} | {c['control_rmse']:.6f} | {c['rmse']:.6f} | {c['passes']} |")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,r in zip(axes.flat,results):
        curve=read(root/f"units/{r['fold']}_{r['seed']}/CURVE.json")
        for key in ('expanded_fit','stop','audit'):
            ax.plot([v['step'] for v in curve],[v['metrics'][key]['mean_regret'] for v in curve],label=key)
        ax.set(title=f"{r['fold']}/{r['seed']}",xlabel='Updates',ylabel='Action regret');ax.legend()
    fig.tight_layout();fig.savefig(root/'NATIVE_COVERAGE_CURVES.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','previous','coverage','labels','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);p.add_argument('--stage',choices=('run','labels'),default='run')
    p.add_argument('--shard',type=int,choices=(0,1),default=0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=='labels':label_worker(a)
    else:
        with (a.output/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:run(a)
            except Exception as exc:write(a.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise
