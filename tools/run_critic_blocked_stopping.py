"""Sequential bounded-label, FIT-only critic stopping contrast; actor immutable."""
import argparse
from copy import deepcopy
import csv
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from app.action_ranking_contrast import ranking_metrics
from app.critic_blocked_stopping import RegretStopper,bounded_snapshot,fold_indices
from app.critic_fit_probe import fit_metrics
from app.critic_target_contract import contract_loss
from app.decision_attribution import continuation,state_view
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest,support_hash
from app.role_routed_critic import RoleHeads
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment
from env_plugins.discrete_curve_native import HOUR
from tools.audit_fixed_actor_critics import covering_schedule
from tools.diagnose_critic_coverage_drift import INTERVALS
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_role_routed_critic import cache_embeddings

ROOT=Path(__file__).resolve().parents[1]
FOLDS=(('A',[2020],2021,2022),('B',[2020,2021],2022,2023))


def panel(a,cfg,cc,dc,spec,market,signals,progress):
    root=a.output/'panel'
    if (root/'DONE.json').exists():
        verify_done(root,cfg['identity']);return read(root/'RESULT.json')['records']
    old=read(a.coverage/'PARTITION.json')['records']
    if digest(old)!=cc['records_sha256']:raise ValueError('STOPPING_RECORD_IDENTITY')
    wanted={r['timestamp']:(i,r) for i,r in enumerate(old)}
    if len(wanted)!=239:raise ValueError('STOPPING_ANCHOR_COUNT')
    references=read(a.diagnostic/'design/PANEL.json')['reference_cash']
    risk=read(a.parent/'RISK.json');records=[];excluded=[];checked=0;witnesses={}
    for segment,interval in INTERVALS:
        env=environment(market,risk,spec['design']['horizon'],interval,cfg['gamma']);env.reset()
        while not env.finished:
            row=int(env.rows[env.bar]);ts=int(market.inputs.row_ts_ns[row])
            if ts in wanted:
                index,original=wanted[ts];checked+=1;state=state_view(env)
                expected=original.get('state_sha256',digest(original.get('state')))
                if original['row']!=row or digest(state)!=expected:
                    raise ValueError('STOPPING_NATIVE_STATE')
                year=original['year'];start=pd.Timestamp(f'{year}-01-01',tz='UTC').value
                end=pd.Timestamp(f'{year+1}-01-01',tz='UTC').value
                bounded=bounded_snapshot(env,end);terminal=int(bounded.engine.v[bounded.engine.last])+HOUR
                if original['window_start']<start:
                    excluded.append(dict(index=index,reason='feature_window_crosses_year'))
                else:
                    path=root/'rows'/f'{index:04d}.json'
                    metadata={k:original[k] for k in ('row','timestamp','window_start','year')}
                    position=0 if env.engine.pos is None else env.engine.pos.direction
                    metadata.update(identity=cfg['identity'],source_index=index,original_role=original['role'],
                        terminal=terminal,state_sha256=expected,position=position)
                    if path.exists():
                        rec=read(path)
                        if any(rec.get(k)!=v for k,v in metadata.items()) or rec['outcomes_sha256']!=digest(rec['outcomes']):
                            raise ValueError('STOPPING_CACHED_LABEL')
                    else:
                        reused=bounded.engine.last==env.engine.last
                        outcomes=original['outcomes'] if reused else [continuation(bounded,signals,j) for j in range(3)]
                        if not all(o['parent_unchanged'] and np.isfinite(o['reward_return']) for o in outcomes):
                            raise ValueError('STOPPING_LABEL_FINITE')
                        allowed=all(o['first_effect']['entries']>0 for o in outcomes[1:])
                        kind=('open_long' if position==1 else 'open_short') if position else (
                            'flat_both_allowed' if allowed else 'flat_restricted')
                        rec=dict(**metadata,kind=kind,outcomes=outcomes,outcomes_sha256=digest(outcomes),
                            reused_identical_native_terminal=reused)
                        write(path,rec)
                    if year not in witnesses:
                        repeated=[continuation(bounded,signals,j) for j in range(3)]
                        if repeated!=rec['outcomes']:raise ValueError('STOPPING_LABEL_WITNESS')
                        witnesses[year]=dict(source_index=index,exact=True)
                    records.append(rec)
                    progress('LABELS',anchors=len(records),total=239,year=year)
                if state_view(env)!=state:raise ValueError('STOPPING_PARENT_MUTATED')
            env.step(int(signals[row]))
        if abs(env.engine.balance-env.engine.cfg.capital-references[segment])>1e-6:
            raise ValueError('STOPPING_SOURCE_ACCOUNT')
    if checked!=239:raise ValueError('STOPPING_MISSING_ANCHOR')
    records.sort(key=lambda r:r['source_index'])
    for _,years,stop,audit in FOLDS:fold_indices(records,years,stop,audit)
    write(root/'RESULT.json',dict(identity=cfg['identity'],records=records,excluded=excluded,
        source_states_verified=checked,witnesses=witnesses))
    seal(root,cfg['identity']);return records


def report(root,results,cfg):
    rows=[];lines=['# Critic blocked stopping results','',
        'Internal diagnostic on frozen pretrained encoders; annual native target, not portfolio returns.','',
        '| Fold | Seed | WD | Selected update | Stop triggered | Stopped audit regret | Fixed audit regret | Actor regret |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        s=r['stopped']['audit']['minimum']['mean_regret'];f=r['fixed']['audit']['minimum']['mean_regret']
        lines.append(f"| {r['fold']} | {r['seed']} | {r['weight_decay']} | {r['selected_update']} | {r['stopped_at']} | {s:.6f} | {f:.6f} | {r['baselines']['actor']['mean_regret']:.6f} |")
        for p in r['curve']:
            rows.append(dict(fold=r['fold'],seed=r['seed'],weight_decay=r['weight_decay'],**p))
    with (root/'CURVES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(fold,wd) in zip(axes.flat,[(f,w) for f in ('A','B') for w in cfg['weight_decays']]):
        for r in results:
            if (r['fold'],r['weight_decay'])!=(fold,wd):continue
            x=[p['updates'] for p in r['curve']]
            ax.plot(x,[p['fit_regret'] for p in r['curve']],label=f"{r['seed']} fit")
            ax.plot(x,[p['stop_regret'] for p in r['curve']],label=f"{r['seed']} stop",linestyle='--')
        ax.set(title=f'{fold} WD={wd}',xlabel='Head updates',ylabel='Native action regret');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'STOPPING_CURVES.png',dpi=140);plt.close(fig)


def run(a):
    begun=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name:
        raise ValueError('STOPPING_DEVICE')
    cc=read(a.coverage/'CONFIG.json');verify_done(a.coverage,cc['identity'])
    dc=read(a.diagnostic/'CONFIG.json');verify_done(a.diagnostic,dc['identity'])
    previous=read(a.previous/'CONFIG.json');verify_done(a.previous,previous['identity'])
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    source_sha=file_sha(a.bundle/'model.pt')
    if (source_sha!=cc['source_sha256'] or source_sha!=previous['source_sha256']
            or dc['identity']!=cc['diagnostic'] or cc['identity']!=previous['coverage']):
        raise ValueError('STOPPING_LINEAGE')
    model,_=load_bundle(a.bundle,'cuda');market=market_for(a.parent,spec,model.recipe)
    if support_hash(market.inputs)!=cc['support'] or model.recipe['list']['n']!=71:
        raise ValueError('STOPPING_FEATURES')
    cfg=dict(schema='critic_blocked_stopping.v1',source_sha256=source_sha,coverage=cc['identity'],
        diagnostic=dc['identity'],previous=previous['identity'],folds=FOLDS,seeds=[19,43],
        weight_decays=[.01,.1],lr=model.effective['learning_rate'],gamma=model.effective['gamma'],
        updates=4096,batch=128,check_every=64,patience=20,min_delta=1e-8,loss='mse',mode='shared_mean3',
        code={name:file_sha(ROOT/name) for name in ('tools/run_critic_blocked_stopping.py',
            'app/critic_blocked_stopping.py','app/decision_attribution.py','app/role_routed_critic.py',
            'tools/run_role_routed_critic.py','app/critic_target_contract.py')})
    # JSON-normalize tuples so cold resumes compare the same representation.
    import json
    cfg=json.loads(json.dumps(cfg));cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('STOPPING_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**kw):write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-begun,**kw))
    progress('PREFLIGHT')
    flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    frozen={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    train=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    logits=emit_actor_reference(model,market.inputs,train,flags)[0]
    signals=prepared_signals(logits)[0]
    verify_done(a.diagnostic/'cache',dc['identity'])
    with np.load(a.diagnostic/'cache/SIGNALS.npz',allow_pickle=False) as d:
        if not np.array_equal(signals[train],d['source'][train]):raise ValueError('STOPPING_SOURCE_SIGNALS')
    records=panel(a,cfg,cc,dc,spec,market,signals,progress)
    rows=np.array([r['row'] for r in records])
    if any(not market.inputs.valid[i] or market.inputs.period[i]!='TRAIN' for i in rows):
        raise ValueError('STOPPING_NONTRAIN')
    z,template,errors=cache_embeddings(model,torch.tensor(market.inputs.windows[rows],device='cuda'))
    if template._forward_pre_hooks:raise ValueError('STOPPING_CACHE_HOOK')
    write(a.output/'CACHE.json',dict(errors=errors,dimensions=list(z.shape)))
    targets=np.array([[o['reward_return'] for o in r['outcomes']] for r in records])
    y=torch.tensor(targets,dtype=torch.float32,device='cuda');results=[]
    held={151+i for i in cc['held']}
    for fold,years,stop_year,audit_year in FOLDS:
        masks=fold_indices(records,years,stop_year,audit_year)
        if min(len(v) for v in masks.values())<12:raise ValueError('STOPPING_INSUFFICIENT_BLOCK')
        masks['audit_opening']=np.array([i for i in masks['audit'] if records[i]['kind']=='flat_both_allowed'])
        masks['old_held_descriptive']=np.array([i for i,r in enumerate(records) if r['source_index'] in held])
        if not len(masks['audit_opening']):raise ValueError('STOPPING_NO_OPENING_AUDIT')
        write(a.output/f'FOLD_{fold}.json',{k:v.tolist() for k,v in masks.items()})
        for seed in cfg['seeds']:
            schedule=masks['fit'][covering_schedule(len(masks['fit']),4096,128,seed)]
            for wd in cfg['weight_decays']:
                out=a.output/f'{fold}_{seed}_{wd}';out.mkdir(parents=True,exist_ok=True)
                if (out/'DONE.json').exists():
                    verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
                progress('FIT',fold=fold,seed=seed,weight_decay=wd,arms_complete=len(results),arms_total=8)
                heads=RoleHeads(template,3,seed).to('cuda');initial=deepcopy(heads.state_dict())
                initial_hash=digest({k:v.cpu().tolist() for k,v in initial.items()})
                opt=torch.optim.AdamW(heads.parameters(),lr=cfg['lr'],weight_decay=wd)
                stopper=RegretStopper(cfg['patience'],cfg['min_delta']);curve=[];best=None
                for step in range(4097):
                    if step:
                        ix=schedule[step-1];opt.zero_grad(set_to_none=True)
                        loss=contract_loss(heads(z[:,ix]).mean(2),y[ix],cfg['loss']);loss.backward()
                        torch.nn.utils.clip_grad_norm_(heads.parameters(),10.,error_if_nonfinite=True);opt.step()
                    if step%64==0:
                        with torch.no_grad():
                            fitq=heads(z[:,masks['fit']]).mean(2).cpu().numpy()
                            stopq=heads(z[:,masks['stop']]).mean(2).cpu().numpy()
                        fm=fit_metrics(fitq,targets[masks['fit']])['minimum']
                        sm=fit_metrics(stopq,targets[masks['stop']])['minimum']
                        if step and stopper.observe(step,sm['mean_regret']):best=deepcopy(heads.state_dict())
                        curve.append(dict(updates=step,fit_regret=fm['mean_regret'],stop_regret=sm['mean_regret'],
                            fit_rmse=fm['rmse'],stop_rmse=sm['rmse'],patience_used=stopper.bad))
                        write(out/'CURVE.json',curve)
                        progress('FIT',fold=fold,seed=seed,weight_decay=wd,updates=step,
                            selected_update=stopper.best_step,stopped_at=stopper.stopped_at,
                            arms_complete=len(results),arms_total=8)
                if best is None:raise ValueError('STOPPING_NO_SELECTED_CHECKPOINT')
                if all(torch.equal(v,initial[k]) for k,v in heads.state_dict().items()):raise ValueError('STOPPING_NO_UPDATES')
                torch.save(dict(state=best,updates=stopper.best_step),out/'SELECTED.pt')
                torch.save(dict(state=heads.state_dict(),updates=4096),out/'FIXED.pt')
                # Audit metrics never participate in the stopping decision.
                endpoints={}
                for label,path in (('stopped','SELECTED.pt'),('fixed','FIXED.pt')):
                    cold=RoleHeads(template,3,seed).to('cuda');cold.load_state_dict(torch.load(out/path,weights_only=True)['state'])
                    expected=RoleHeads(template,3,seed).to('cuda')
                    expected.load_state_dict(best if label=='stopped' else heads.state_dict())
                    with torch.no_grad():
                        q=cold(z).mean(2)
                        if not torch.equal(q,expected(z).mean(2)):raise ValueError('STOPPING_COLD_RELOAD')
                        q=q.cpu().numpy()
                    endpoints[label]={k:fit_metrics(q[:,ix],targets[ix]) for k,ix in masks.items()}
                audit=masks['audit'];truth=targets[audit]
                baselines=dict(actor=ranking_metrics(logits[rows[audit]],truth),
                    nop=ranking_metrics(np.tile([1.,0.,0.],(len(audit),1)),truth),
                    fit_mean=ranking_metrics(np.tile(targets[masks['fit']].mean(0),(len(audit),1)),truth))
                result=dict(identity=cfg['identity'],fold=fold,seed=seed,weight_decay=wd,
                    initialization_sha256=initial_hash,schedule_sha256=digest(schedule.tolist()),
                    selected_update=stopper.best_step,stopped_at=stopper.stopped_at,selection_best_regret=stopper.best,
                    curve=curve,baselines=baselines,cold_exact=True,**endpoints)
                write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
                del heads,cold,expected,opt
    for fold,_,_,_ in FOLDS:
        for seed in cfg['seeds']:
            pair=[r for r in results if (r['fold'],r['seed'])==(fold,seed)]
            if len(pair)!=2 or any(len({r[k] for r in pair})!=1 for k in ('initialization_sha256','schedule_sha256')):
                raise ValueError('STOPPING_UNPAIRED')
    if (any(not torch.equal(v,frozen[k]) for k,v in model.loss.state_dict().items())
            or not np.array_equal(emit_actor_reference(model,market.inputs,train,flags)[0][train],logits[train])
            or file_sha(a.bundle/'model.pt')!=source_sha):raise ValueError('STOPPING_SOURCE_MUTATED')
    comparisons=[]
    for wd in cfg['weight_decays']:
        group=[r for r in results if r['weight_decay']==wd]
        passed=all(r['stopped']['audit']['minimum']['mean_regret']<r['fixed']['audit']['minimum']['mean_regret']
            and r['stopped']['audit_opening']['minimum']['mean_regret']<=r['fixed']['audit_opening']['minimum']['mean_regret']
            for r in group)
        comparisons.append(dict(weight_decay=wd,improves_all_folds_seeds_with_opening_guard=passed))
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],arms=results,comparisons=comparisons,
        source_exact=True,actor_exact=True,model_promoted=False,validation_read=False,test_read=False,
        stl_included=False,historical_mixed_cot=True))
    report(a.output,results,cfg);seal(a.output,cfg['identity']);progress('COMPLETE',arms_complete=8,arms_total=8)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','coverage','diagnostic','previous','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise
