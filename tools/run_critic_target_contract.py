"""Paired marginal/routed critic targets with native TRAIN visitation weights."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from app.critic_fit_probe import fit_metrics
from app.critic_representation_control import state_kind as record_kind
from app.critic_target_contract import visitation_probabilities,contract_loss,stratum
from app.decision_attribution import state_view
from app.frozen_recipe_inputs import file_sha
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest,support_hash
from app.role_routed_critic import RoleHeads
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment
from tools.audit_fixed_actor_critics import covering_schedule
from tools.diagnose_critic_coverage_drift import INTERVALS
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_role_routed_critic import cache_embeddings,predicted
from tools.run_stratified_critic import state_kind

POINTS=(0,64,256,1024,2048,4096)


def role_id(kind):
    return 1 if kind=='open_long' else 2 if kind=='open_short' else 0


def export(root,results,identity):
    comparisons=[]
    for sampling in ('uniform','visitation'):
        for loss in ('smooth_l1','mse'):
            pairs=[]
            for seed in (19,43):
                a,b=[next(r for r in results if (r['seed'],r['sampling'],r['loss'],r['mode'])==
                    (seed,sampling,loss,mode)) for mode in ('shared_mean3','routed')]
                ma,mb=a['curve'][-1]['metrics'],b['curve'][-1]['metrics']
                h0,h1=ma['held']['minimum'],mb['held']['minimum']
                f0,f1=ma['held_flat_both_allowed']['minimum'],mb['held_flat_both_allowed']['minimum']
                pairs.append(dict(seed=seed,held_regret_delta=h1['mean_regret']-h0['mean_regret'],
                    held_accuracy_delta=h1['pairwise_accuracy']-h0['pairwise_accuracy'],
                    opening_regret_delta=f1['mean_regret']-f0['mean_regret'],
                    promising=h1['mean_regret']<h0['mean_regret'] and f1['mean_regret']<=f0['mean_regret']))
            comparisons.append(dict(sampling=sampling,loss=loss,seeds=pairs,
                promising_for_design=all(p['promising'] for p in pairs)))
    write(root/'RESULT.json',dict(identity=identity,arms=results,comparisons=comparisons,
        actor_exact=True,source_exact=True,model_promoted=False,stl_included=False,
        validation_read=False,test_read=False,historical_mixed_cot=True))
    lines=['# Critic target contract results','',
        'All16 fixed endpoints; same55 previously inspected TRAIN anchors. No actor promotion.','',
        '| Seed | Sampling | Loss | Output | Held accuracy | Held regret | Opening regret | Q RMSE |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    flat=[]
    for r in results:
        m=r['curve'][-1]['metrics'];h=m['held']['minimum'];o=m['held_flat_both_allowed']['minimum']
        lines.append(f"| {r['seed']} | {r['sampling']} | {r['loss']} | {r['mode']} | {h['pairwise_accuracy']:.6f} | {h['mean_regret']:.6f} | {o['mean_regret']:.6f} | {h['rmse']:.6f} |")
        for p in r['curve']:
            for population,m in p['metrics'].items():
                v=m['minimum']
                flat.append(dict(seed=r['seed'],sampling=r['sampling'],loss=r['loss'],mode=r['mode'],
                    updates=p['updates'],population=population,accuracy=v['pairwise_accuracy'],
                    regret=v['mean_regret'],rmse=v['rmse']))
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    with (root/'METRICS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(sampling,loss) in zip(axes.flat,[(s,l) for s in ('uniform','visitation') for l in ('smooth_l1','mse')]):
        for r in results:
            if (r['sampling'],r['loss'])!=(sampling,loss):continue
            ax.plot([p['updates'] for p in r['curve']],
                [p['metrics']['held']['minimum']['mean_regret'] for p in r['curve']],
                label=f"{r['seed']} {r['mode']}")
        ax.set(title=sampling+' '+loss,xlabel='Critic head updates',ylabel='Held native regret')
        ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'TARGET_CURVES.png',dpi=140);plt.close(fig)


def run(a):
    started=time.monotonic();configure(317);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name:
        raise ValueError('CONTRACT_GPU')
    cc=read(a.coverage/'CONFIG.json');verify_done(a.coverage,cc['identity'])
    ic=read(a.identifiability/'CONFIG.json');verify_done(a.identifiability,ic['identity'])
    dc=read(a.diagnostic/'CONFIG.json');verify_done(a.diagnostic,dc['identity'])
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    sha=file_sha(a.bundle/'model.pt')
    if (sha!=cc['source_sha256'] or sha!=ic['source_sha256'] or ic['coverage']!=cc['identity']
            or dc['identity']!=cc['diagnostic']):raise ValueError('CONTRACT_LINEAGE')
    model,_=load_bundle(a.bundle,'cuda');market=market_for(a.parent,spec,model.recipe)
    if model.recipe['list']['n']!=71 or support_hash(market.inputs)!=cc['support']:
        raise ValueError('CONTRACT_SUPPORT')
    records=read(a.coverage/'PARTITION.json')['records']
    if digest(records)!=cc['records_sha256']:raise ValueError('CONTRACT_RECORDS')
    fit=np.array([i for i,r in enumerate(records) if r['role']=='fit'])
    held=151+np.array(cc['held']);old_monitor=np.array([i for i,r in enumerate(records[:151]) if r['role']=='monitor'])
    if (len(records),len(fit),len(held),len(old_monitor))!=(239,146,55,38) or set(fit)&set(held):
        raise ValueError('CONTRACT_PARTITION')
    for r in records:
        if (not market.inputs.valid[r['row']] or market.inputs.period[r['row']]!='TRAIN'
                or int(market.inputs.row_ts_ns[r['row']])!=r['timestamp']):raise ValueError('CONTRACT_ROWS')
    cfg=dict(schema='critic_target_contract.v1',source_sha256=sha,coverage=cc['identity'],
        identifiability=ic['identity'],diagnostic=dc['identity'],seeds=[19,43],updates=4096,batch=128,
        modes=['shared_mean3','routed'],sampling=['uniform','visitation'],losses=['smooth_l1','mse'],
        learning_rate=model.effective['learning_rate'],weight_decay=model.effective['weight_decay'],
        actor_updates=0,encoder_updates=0,code={n:file_sha(Path(__file__).resolve().parents[1]/n) for n in (
            'tools/run_critic_target_contract.py','app/critic_target_contract.py','app/role_routed_critic.py',
            'tools/run_role_routed_critic.py','app/action_ranking_contrast.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('CONTRACT_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**kw):
        write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    verify_done(a.diagnostic/'cache',dc['identity'])
    with np.load(a.diagnostic/'cache/SIGNALS.npz',allow_pickle=False) as d:signals=d['source'].copy()
    popdir=a.output/'population'
    if (popdir/'DONE.json').exists():verify_done(popdir,cfg['identity']);pop=read(popdir/'RESULT.json')
    else:
        progress('NATIVE_VISITATION');counts=Counter();later=Counter();snapshots={r['timestamp']:r for r in records}
        reference=read(a.diagnostic/'design/PANEL.json')['reference_cash'];risk=read(a.parent/'RISK.json');checked=0
        for segment,interval in INTERVALS:
            env=environment(market,risk,spec['design']['horizon'],interval,model.effective['gamma']);env.reset()
            while not env.finished:
                row=int(env.rows[env.bar]);ts=int(market.inputs.row_ts_ns[row]);year=pd.Timestamp(ts,tz='UTC').year
                kind=state_kind(env);key=f'{year}:{kind}'
                (counts if year<2023 else later)[key]+=1
                if ts in snapshots:
                    r=snapshots[ts];state=state_view(env)
                    if (stratum(r)!=key or digest(state)!=r.get('state_sha256',digest(r.get('state')))):
                        raise ValueError('CONTRACT_STATE_REPLAY')
                    checked+=1
                env.step(int(signals[row]))
            if abs(env.engine.balance-env.engine.cfg.capital-reference[segment])>1e-6:
                raise ValueError('CONTRACT_REFERENCE_ACCOUNT')
        if checked!=len(records):raise ValueError('CONTRACT_MISSING_SNAPSHOTS')
        pop=dict(fit_counts=dict(counts),monitor_counts_not_used_in_fit=dict(later),snapshots_exact=checked)
        write(popdir/'RESULT.json',pop);seal(popdir,cfg['identity'])
    weights=visitation_probabilities(records,fit,pop['fit_counts'])
    write(a.output/'FIT_WEIGHTS.json',dict(indices=fit.tolist(),probabilities=weights.tolist(),
        fit_effective_sample_size=float(1/(weights@weights)),strata=pop['fit_counts']))
    flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    frozen={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    trainrows=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    baseline=emit_actor_reference(model,market.inputs,trainrows,flags)[0][trainrows]
    cf=[]
    for bank in read(a.identifiability/'RESULT.json')['origins']:
        if bank['status']!='COMPLETE':continue
        for first,outcomes in zip(bank['first'],bank['outcomes']):
            p=first['state']['position'];direction=0 if p is None else p['direction']
            cf.append(dict(row=bank['next_row'],kind='open_long' if direction==1 else 'open_short' if direction==-1 else 'flat',
                outcomes=outcomes,year=bank['anchor']['year'],bank=bank['anchor']['row']))
    if len(cf)!=48:raise ValueError('CONTRACT_COUNTERFACTUAL_COUNT')
    allrecords=records+cf;rows=np.array([r['row'] for r in allrecords])
    if any(not market.inputs.valid[i] or market.inputs.period[i]!='TRAIN' for i in rows):raise ValueError('CONTRACT_CACHE_ROWS')
    unique=sorted(set(rows.tolist()));lookup={r:i for i,r in enumerate(unique)}
    progress('CACHE_EMBEDDINGS')
    cached,template,errors=cache_embeddings(model,torch.tensor(market.inputs.windows[unique],device='cuda'))
    z=cached[:,[lookup[r] for r in rows]]
    roles=torch.tensor([role_id(record_kind(r)) for r in allrecords],device='cuda')
    targets=np.array([[o['reward_return'] for o in r['outcomes']] for r in allrecords])
    y=torch.tensor(targets,device='cuda',dtype=torch.float32)
    masks=dict(fit=fit,held=held,old_monitor=old_monitor,counterfactual=np.arange(len(records),len(allrecords)))
    for kind in sorted({record_kind(records[i]) for i in held}):masks['held_'+kind]=np.array([i for i in held if record_kind(records[i])==kind])
    for year in range(2020,2024):masks[f'held_{year}']=np.array([i for i in held if records[i]['year']==year])
    write(a.output/'CACHE.json',dict(rows=unique,embedding_errors=errors,dimensions=list(z.shape)))
    results=[]
    for seed in cfg['seeds']:
        for sampling in cfg['sampling']:
            schedule=(fit[covering_schedule(len(fit),4096,128,seed)] if sampling=='uniform' else
                np.random.default_rng(seed).choice(fit,size=(4096,128),p=weights))
            if set(schedule.ravel())!=set(fit):raise ValueError('CONTRACT_SCHEDULE_COVERAGE')
            np.save(a.output/f'SCHEDULE_{seed}_{sampling}.npy',schedule)
            schedule_hash=digest(schedule.tolist())
            for loss in cfg['losses']:
                for mode in cfg['modes']:
                    out=a.output/f'{seed}_{sampling}_{loss}_{mode}';out.mkdir(parents=True,exist_ok=True)
                    if (out/'DONE.json').exists():
                        verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
                    heads=RoleHeads(template,3,seed).to('cuda');initial=deepcopy(heads.state_dict())
                    init_hash=digest({k:v.detach().cpu().tolist() for k,v in initial.items()})
                    effective=dict(seed=seed,sampling=sampling,loss=loss,mode=mode,
                        initialization_sha256=init_hash,schedule_sha256=schedule_hash,
                        parameters=sum(p.numel() for p in heads.parameters()),
                        learning_rate=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
                    write(out/'EFFECTIVE.json',effective);torch.save(initial,out/'INITIAL.pt')
                    opt=torch.optim.AdamW(heads.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);curve=[]
                    for step in range(4097):
                        if step:
                            ix=schedule[step-1];opt.zero_grad(set_to_none=True)
                            objective=contract_loss(predicted(heads,z[:,ix],roles[ix],mode),y[ix],loss)
                            objective.backward();torch.nn.utils.clip_grad_norm_(heads.parameters(),10.,error_if_nonfinite=True);opt.step()
                        if step in POINTS:
                            with torch.no_grad():q=predicted(heads,z,roles,mode).cpu().numpy()
                            curve.append(dict(updates=step,metrics={k:fit_metrics(q[:,ix],targets[ix]) for k,ix in masks.items()}))
                            write(out/'CURVE.json',curve);progress('FIT',**effective,updates=step,arms_complete=len(results),arms_total=16)
                    torch.save(dict(state=heads.state_dict(),effective=effective),out/'HEADS.pt')
                    cold=RoleHeads(template,3,seed).to('cuda');cold.load_state_dict(torch.load(out/'HEADS.pt',weights_only=True)['state'])
                    with torch.no_grad():
                        if not torch.equal(heads(z),cold(z)):raise ValueError('CONTRACT_COLD_RELOAD')
                    if not any(not torch.equal(v,initial[k]) for k,v in heads.state_dict().items()):raise ValueError('CONTRACT_NO_UPDATES')
                    result=dict(identity=cfg['identity'],**effective,curve=curve,cold_exact=True)
                    write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
                    del heads,cold,opt
    for seed in cfg['seeds']:
        group=[r for r in results if r['seed']==seed]
        if len({r['initialization_sha256'] for r in group})!=1 or len({r['parameters'] for r in group})!=1:
            raise ValueError('CONTRACT_UNPAIRED_INITIALIZATION')
        for sampling in cfg['sampling']:
            if len({r['schedule_sha256'] for r in group if r['sampling']==sampling})!=1:
                raise ValueError('CONTRACT_UNPAIRED_SCHEDULE')
    if (any(not torch.equal(v,frozen[k]) for k,v in model.loss.state_dict().items())
            or not np.array_equal(emit_actor_reference(model,market.inputs,trainrows,flags)[0][trainrows],baseline)
            or file_sha(a.bundle/'model.pt')!=sha):raise ValueError('CONTRACT_SOURCE_MUTATION')
    export(a.output,results,cfg['identity']);seal(a.output,cfg['identity']);progress('COMPLETE',arms_complete=16,arms_total=16)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parent','bundle','coverage','identifiability','diagnostic','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
