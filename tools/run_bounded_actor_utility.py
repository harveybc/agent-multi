"""Single350 native-critic to actor trial, sequential seeds, finite terminal gate."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time
import numpy as np
import torch
from app.bounded_actor_utility import install_native_heads,choose_fold,qualify,winner
from app.critic_fit_probe import probe_q
from app.bounded_functional_actor import BUDGET,budget_metrics,admissible,backtrack
from app.actor_learning_diagnostics import clear_optimizer_memory
from app.operational_actor_transfer import actor_loss
from app.operational_opening_critic import emit_actor_reference
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import digest,support_hash
from app.selected_input_market import market_for
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle,verify_files,export_bundle
from app.three_signal_learning import environment,evaluate
from app.training_forensics import tree_digest,rng_state,restore_rng
from app.weight_decision_forensics import native_forward
from tools.audit_fixed_actor_critics import covering_schedule
from tools.replay_first_harmful_crossings import cpu_state,require_equal
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write,configure,now
from tools.run_refit_mask_pass import account_equal
from tools.run_role_routed_critic import cache_embeddings
from tools.run_training_forensics import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def actor_parameters(model):
    named=[(n,p) for n,p in model.loss.named_parameters() if n.startswith('actor_') and '.head.' in n]
    if len(named)!=4 or not all(p.requires_grad for _,p in named):raise ValueError('UTILITY_ACTOR_LAYOUT')
    return named


def flat(params):return torch.cat([p.detach().flatten() for p in params])


def snapshot(model):
    return dict(loss=cpu_state(model.loss.state_dict()),optimizer=deepcopy(model.optimizer.state_dict()),rng=rng_state())


def restore(model,state):
    model.loss.load_state_dict(state['loss']);model.optimizer.load_state_dict(deepcopy(state['optimizer']));restore_rng(state['rng'])


def actor_step(model,x,q,ix,source_logits,source_theta):
    params=[p for _,p in actor_parameters(model)];before=[p.detach().clone() for p in params]
    oldopt=deepcopy(model.optimizer.state_dict())
    try:return _actor_step(model,x,q,ix,source_logits,source_theta)
    except BaseException:
        with torch.no_grad():
            for p,b in zip(params,before):p.copy_(b)
        model.optimizer.load_state_dict(oldopt);raise


def _actor_step(model,x,q,ix,source_logits,source_theta):
    params=[p for _,p in actor_parameters(model)];before=[p.detach().clone() for p in params]
    theta=flat(params);oldopt=deepcopy(model.optimizer.state_dict());alpha=model.loss.log_alpha.detach().exp()
    before_z=model.surfaces(x).detach()
    loss0=float(actor_loss(before_z[ix],q[ix],alpha))
    model.optimizer.zero_grad(set_to_none=True)
    loss=actor_loss(model.surfaces(x[ix]),q[ix],alpha)
    grads=torch.autograd.grad(loss,params)
    for p,g in zip(params,grads):p.grad=g
    norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
    model.optimizer.step();direction=flat(params)-theta
    with torch.no_grad():
        for p,b in zip(params,before):p.copy_(b)
    def check():
        after=model.surfaces(x).detach();value=float(actor_loss(after[ix],q[ix],alpha))
        metrics=budget_metrics(source_logits,before_z,after,source_theta,flat(params))
        moved=any(not torch.equal(p,b) for p,b in zip(params,before))
        return dict(accepted=bool(moved and metrics['step_margin_rms']>1e-12 and admissible(metrics,loss0,value)),
            metrics=metrics,loss_before=loss0,loss_after=value)
    try:result=backtrack(params,direction,check,BUDGET['backtracks'])
    except BaseException:
        with torch.no_grad():
            for p,b in zip(params,before):p.copy_(b)
        model.optimizer.load_state_dict(oldopt);raise
    if not result['accepted']:model.optimizer.load_state_dict(oldopt)
    return dict(result,gradient_norm=float(norm))


def verify_frozen(model,fixed):
    current=model.loss.state_dict()
    for k,v in fixed.items():
        if not torch.equal(v,current[k].detach().cpu()):raise ValueError('UTILITY_FROZEN:'+k)


def run(a):
    start=time.monotonic();configure(350);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('UTILITY_DEVICE')
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('critics','decision','verified','coverage')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if not parents['critics']==parents['decision']==parents['verified']:raise ValueError('UTILITY_CRITIC_VERIFICATION')
    decision=read(a.decision/'DECISION.json')
    if decision['selected']!='hybrid349' or not decision['critic_diagnostics_closed']:raise ValueError('UTILITY_DECISION')
    if parents['critics']['parents']['coverage']!=parents['coverage']['identity']:raise ValueError('UTILITY_COVERAGE_LINEAGE')
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    source=read(a.bundle.parent/'RESULT.json');source_config=read(a.bundle.parent/'CONFIG.json')
    verify_done(a.bundle.parent,source['identity'])
    if digest(source_config)!=source['identity'] or not source['qualified']:raise ValueError('UTILITY_SOURCE_QUALIFIED')
    model,_=load_bundle(a.bundle,'cuda');flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    actor_sha=file_sha(a.bundle/'model.pt')
    if actor_sha!=parents['critics']['source_actor_sha256'] or actor_sha!=parents['coverage']['source_sha256']:raise ValueError('UTILITY_SOURCE_SHA')
    market=market_for(a.parent,spec,model.recipe)
    support=read(a.coverage/'prepared/SUPPORT.json');data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')
    records=read(a.coverage/'PANEL.json')['records'];rows=np.array(support['rows']);fold=choose_fold(support)
    fit=np.array(support['folds'][fold]['expanded']);masks=support['folds'][fold]['masks']
    if (set(fit)&set(masks['stop']) or set(fit)&set(masks['audit']) or
            len(rows)!=len(records) or not market.inputs.valid[rows].all() or not (market.inputs.period[rows]=='TRAIN').all()
            or not np.array_equal(market.inputs.row_ts_ns[rows],[r['timestamp'] for r in records])):raise ValueError('UTILITY_FIT_ALIGNMENT')
    if support_hash(market.inputs)!=parents['coverage']['support'] or model.recipe['list']['n']!=71:raise ValueError('UTILITY_SUPPORT')
    if model.recipe['split']['periods']!={'TRAIN':['2020-01-01','2024-01-01'],'VALIDATION':['2024-01-01','2025-01-01']}:raise ValueError('UTILITY_SPLITS')
    cfg=dict(schema='bounded_actor_utility.v1',parents={k:c['identity'] for k,c in parents.items()},
        parent_identity=spec['identity'],source_actor_sha256=actor_sha,source_receipt=file_sha(a.bundle.parent/'DONE.json'),
        source_metrics_sha256=file_sha(a.bundle.parent/'RESULT.json'),support_sha256=support_hash(market.inputs),recipe=model.recipe['identity'],
        fold=fold,fold_rule='largest_expanded_FIT_then_lexical',fit_rows=len(fit),seeds=[19,43],
        max_attempts=64,accepted_endpoint=8,points=[0,1,8],batch=128,budget=BUDGET,
        effective=model.effective,alpha=float(model.loss.log_alpha.detach().exp()),objective='categorical_SAC_native_alpha_frozen_min_Q',
        actor_only=True,critic_updates=0,features=71,full_periods=model.recipe['split']['periods'],
        code={k:file_sha(ROOT/k) for k in ('tools/run_bounded_actor_utility.py','app/bounded_actor_utility.py',
            'app/bounded_functional_actor.py','app/operational_actor_transfer.py','app/three_signal_learning.py','app/selected_input_market.py',
            'app/critic_fit_probe.py','tools/run_role_routed_critic.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('UTILITY_CONFIG')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**kw):write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-start,**kw))
    def logits(m):return emit_actor_reference(m,market.inputs,None,flags)[0]
    risk=read(a.parent/'RISK.json')
    def measure(z,out):
        prepared=prepared_signals(z);result={}
        for period,interval in model.recipe['split']['periods'].items():
            progress('ACCOUNT',account=period,path=str(out.relative_to(a.output)))
            result[period]=evaluate(environment(market,risk,spec['design']['horizon'],interval,model.effective['gamma']),z,
                prepared=prepared,save=out/period,diagnose=True)
        return result
    progress('REFERENCE');baseline=logits(model);reference=measure(baseline,a.output/'reference')
    for p in reference:account_equal(reference[p],source['metrics'][p])
    write(a.output/'REFERENCE.json',reference)
    features=torch.tensor(market.inputs.windows[rows],device='cuda')
    embeddings,_,errors=cache_embeddings(model,features)
    delta=float((embeddings.cpu()-data['z']).abs().max())
    if delta>1e-6:raise ValueError('UTILITY_EMBEDDING_PARITY:'+str(delta))
    write(a.output/'EMBEDDINGS.json',dict(shape=list(embeddings.shape),max_error=delta,native_errors=errors))
    del model;results=[]
    for seed in cfg['seeds']:
        out=a.output/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
        progress('INTEGRATION',seed=seed)
        model,_=load_bundle(a.bundle,'cuda');old=cpu_state(model.loss.state_dict())
        donor_path=a.critics/'units'/f'{fold}_{seed}'/'ENDPOINT_16.pt'
        donor=torch.load(donor_path,weights_only=False,map_location='cpu')
        if donor['identity']!=parents['critics']['identity']:raise ValueError('UTILITY_DONOR_IDENTITY')
        install_native_heads(model,donor['state']);changed=[k for k,v in model.loss.state_dict().items() if not torch.equal(v.cpu(),old[k])]
        if not changed or any(not k.startswith('qvalue_network_params.') or '.head.' not in k for k in changed):raise ValueError('UTILITY_IMPORT_SCOPE')
        if not np.array_equal(logits(model)[market.inputs.valid],baseline[market.inputs.valid]):raise ValueError('UTILITY_IMPORT_ACTOR')
        with torch.no_grad():qfull=torch.cat([probe_q(model,features[i:i+128]).cpu() for i in range(0,len(features),128)],dim=1).numpy()
        qref=native_forward(donor['state'],data['z'].double().numpy())[0]
        qerror=float(np.max(abs(qfull-qref)))
        if qerror>1e-5 or not np.array_equal(qfull[:,fit].min(0).argmax(1),qref[:,fit].min(0).argmax(1)):raise ValueError('UTILITY_Q_PARITY:'+str(qerror))
        write(out/'INTEGRATION.json',dict(donor_sha256=file_sha(donor_path),changed=changed,native_q_max_error=qerror,
            fit_actions_exact=True,actor_exact=True,source_alpha=cfg['alpha'],fold=fold))
        clear_optimizer_memory(model.optimizer);reset_learning_rng(seed)
        named=actor_parameters(model);names={n for n,_ in named};params=[p for _,p in named]
        fixed={n:v for n,v in cpu_state(model.loss.state_dict()).items() if n not in names}
        x=features[fit];q=torch.tensor(qfull[:,fit].min(0),device='cuda');source_z=model.surfaces(x).detach();theta=flat(params)
        schedule=covering_schedule(len(fit),64,128,seed)
        trace=[];accepted=0;attempt=0
        if (out/'RECOVERY.pt').exists():
            saved=torch.load(out/'RECOVERY.pt',weights_only=False,map_location='cpu')
            if saved['identity']!=cfg['identity'] or saved['schedule']!=tree_digest(schedule):raise ValueError('UTILITY_RESUME_IDENTITY')
            restore(model,saved['model']);trace=saved['trace'];accepted=saved['accepted'];attempt=saved['attempt']
            verify_frozen(model,fixed)
        while accepted<8 and attempt<64:
            before=snapshot(model);ix=torch.tensor(schedule[attempt],device='cuda');attempt+=1
            rec=actor_step(model,x,q,ix,source_z,theta);accepted+=int(rec['accepted'])
            verify_frozen(model,fixed)
            after=snapshot(model)
            if attempt==1:
                checkpoint(out/'FIRST_BEFORE.pt',before);checkpoint(out/'FIRST_AFTER.pt',after)
                restore(model,torch.load(out/'FIRST_BEFORE.pt',weights_only=False,map_location='cpu'))
                repeated=actor_step(model,x,q,ix,source_z,theta)
                require_equal(rec,repeated,'UTILITY_COLD_TRANSACTION');require_equal(after,snapshot(model),'UTILITY_COLD_STATE')
                restore(model,after)
            trace.append(dict(attempt=attempt,accepted_total=accepted,**rec))
            saved=dict(identity=cfg['identity'],schedule=tree_digest(schedule),attempt=attempt,accepted=accepted,trace=trace,model=after)
            checkpoint(out/'RECOVERY.pt',saved)
            if rec['accepted'] and accepted in (1,8):checkpoint(out/f'ACCEPTED_{accepted}.pt',saved)
            progress('ACTOR_UPDATES',seed=seed,attempt=attempt,accepted=accepted)
        points=[];final_metrics=reference;final_path=None
        for endpoint in (1,8):
            path=out/f'ACCEPTED_{endpoint}.pt'
            if not path.exists():continue
            saved=torch.load(path,weights_only=False,map_location='cpu');restore(model,saved['model']);verify_frozen(model,fixed)
            z=logits(model);model.save(out/f'MODEL_{endpoint}.pt')
            cold,_=load_bundle(a.bundle,'cuda');cold.load(out/f'MODEL_{endpoint}.pt');cz=logits(cold)
            if not np.array_equal(z[market.inputs.valid],cz[market.inputs.valid]):raise ValueError('UTILITY_COLD_LOGITS')
            metrics=measure(z,out/f'point_{endpoint}');replayed=measure(cz,out/f'cold_{endpoint}')
            for p in metrics:account_equal(metrics[p],replayed[p])
            points.append(dict(accepted=endpoint,attempt=saved['attempt'],metrics=metrics,cold_logits_exact=True,cold_accounts_exact=True))
            if endpoint==8:final_metrics=metrics;final_path=out/f'MODEL_{endpoint}.pt'
            del cold
        qualification=qualify(final_metrics,reference,accepted)
        result=dict(seed=seed,accepted=accepted,attempts=attempt,points=points,metrics=final_metrics,primary_endpoint_available=accepted==8,
            qualification=qualification,checkpoint=final_path.name if final_path else None,frozen_exact=True,
            first_transaction_cold_exact=True,donor_sha256=file_sha(donor_path),trace=trace)
        write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result);del model
    selected=winner(results);export=None
    if selected is not None:
        model,_=load_bundle(a.bundle,'cuda');model.load(a.output/f'seed_{selected}/MODEL_8.pt')
        destination=a.output/'qualified_bundle'
        if destination.exists():
            cold,_=load_bundle(destination,'cuda')
            if not np.array_equal(logits(model)[market.inputs.valid],logits(cold)[market.inputs.valid]):raise ValueError('UTILITY_EXPORT_RESUME')
            export=dict(model_sha256=file_sha(destination/'model.pt'),reload_exact=True)
        else:export=export_bundle(destination,model,market.inputs.windows[rows[:32]],dict(identity=cfg['identity'],seed=selected,accepted=8))
        cold,_=load_bundle(destination,'cuda');check=measure(logits(cold),a.output/'export_replay')
        for p in reference:account_equal(check[p],next(r for r in results if r['seed']==selected)['metrics'][p])
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if file_sha(a.bundle/'model.pt')!=actor_sha:raise ValueError('UTILITY_SOURCE_CHANGED')
    report=dict(identity=cfg['identity'],seconds=time.monotonic()-start,reference=reference,units=results,
        selected_seed=selected,export=export,terminal='QUALIFIED_EXPORT' if selected is not None else 'KEEP299_CLOSE_REFINEMENT',
        live_promoted=False,critic_diagnostics_closed=True,automatic_extension=False)
    write(a.output/'RESULT.json',report);plot(a.output,report);seal(a.output,cfg['identity']);progress('COMPLETE',selected_seed=selected,terminal=report['terminal'])


def plot(out,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,u in zip(axes,result['units']):
        points=[dict(accepted=0,metrics=result['reference'])]+u['points']
        for p in ('TRAIN','VALIDATION'):ax.plot([v['accepted'] for v in points],[v['metrics'][p]['return_pct'] for v in points],marker='o',label=p)
        ax.set(title=f"Seed {u['seed']}",xlabel='Accepted actor-head updates',ylabel='Full-period return (%)');ax.legend()
    fig.tight_layout();fig.savefig(out/'TRAINING_VALIDATION.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('parent','bundle','critics','decision','verified','coverage','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise
