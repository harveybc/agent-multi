"""Exact342/346 replay and fixed-step weight/AdamW factorial; no promotion."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time
import numpy as np
import torch
from app.crossing_step_forensics import adam_parts,flat
from app.decision_cost_objective import cost_objective
from app.frozen_recipe_inputs import file_sha
from app.interior_adamw_repair import transactional_step
from app.matched_divergence import locate_events,factorial_effects
from app.native_fit_coverage import validate_record
from app.native_head_warm_start import configure_scope
from app.rl_subset import digest
from app.training_forensics import rng_state,restore_rng,tree_digest
from app.weight_decision_forensics import native_forward,weighted_min_gradient
from tools.audit_fixed_actor_critics import covering_schedule
from tools.probe_optimal_fit_protection import support_and_repair
from tools.replay_first_harmful_crossings import cpu_state,require_equal
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_targeted_rescue_continuation import rescue
from tools.run_training_forensics import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def measure(w,z,row,preferred,other):
    qs=[native_forward(w[k],z)[0] for k in ('before','proposed','after')]
    margins=[float(q.min(0)[row,preferred]-q.min(0)[row,other]) for q in qs]
    coeff=np.zeros(qs[0].shape[1:]);coeff[row,preferred]=1;coeff[row,other]-=1
    jac=weighted_min_gradient(w['before'],z,coeff)[row]
    parts=adam_parts(w['before'],w['proposed'],w['optimizer_before'],w['optimizer_proposed'],w['gradient'])
    contributions={k:float(jac@parts[k]) for k in ('history','current','decay','rounding')}
    raw=flat(w['proposed'])-flat(w['before']);repair=flat(w['after'])-flat(w['proposed'])
    if not np.isclose(sum(contributions.values()),jac@raw,atol=1e-10,rtol=1e-10):raise ValueError('DIVERGENCE_ADAM_SUM')
    return dict(actions=[int(q.min(0).argmax(1)[row]) for q in qs],margins=margins,
        raw_finite=margins[1]-margins[0],protection_finite=margins[2]-margins[1],
        raw_linear=float(jac@raw),protection_linear=float(jac@repair),adam_contributions=contributions,
        adam_reconstruction_error=parts['reconstruction_max_error'],accepted=bool(w['receipt']['accepted']))


def restore(ctx,cfg,state,opt,rng):
    ctx.head.load_state_dict(state);optimizer,_=configure_scope(ctx.head,'full_head',cfg['lr'],cfg['weight_decay'])
    optimizer.load_state_dict(deepcopy(opt));restore_rng(rng)
    return optimizer


def execute(ctx,opt,ix,arm,offset,budget,historical):
    w=dict(before=cpu_state(ctx.head.state_dict()),optimizer_before=deepcopy(opt.state_dict()),rng_before=rng_state(),batch=ix)
    def repair(before,proposed):
        w['proposed']=cpu_state(proposed);w['optimizer_proposed']=deepcopy(opt.state_dict())
        w['gradient']=torch.cat([p.grad.detach().reshape(-1).cpu() for p in ctx.head.parameters()]).double().numpy()
        if arm=='T' and offset<=16 and historical:return rescue(ctx,cpu_state(before),cpu_state(proposed),budget)
        state,rec,details,_=support_and_repair(ctx,cpu_state(before),cpu_state(proposed))
        rec=dict(rec,details=details)
        if arm=='T':rec.update(arm='ordinary',searches_used=4)
        return state,rec
    rec=transactional_step(ctx.head,opt,ctx.z[:,ix],ctx.y[ix],repair,
        loss_fn=lambda q,t:cost_objective(q,t,'cost',ctx.temperature))
    w.update(after=cpu_state(ctx.head.state_dict()),optimizer_after=deepcopy(opt.state_dict()),rng_after=rng_state(),receipt=dict(offset=offset,**rec))
    return w


def verify(a):
    cfg=read(a.output/'CONFIG.json');verify_done(a.output,cfg['identity'])
    for k,identity in cfg['parents'].items():
        parent=read(getattr(a,k)/'CONFIG.json');verify_done(getattr(a,k),identity)
        if parent['identity']!=identity:raise ValueError('DIVERGENCE_VERIFY_PARENT')
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('DIVERGENCE_VERIFY_ACTOR')
    result=read(a.output/'RESULT.json');data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')
    support=read(a.coverage/'prepared/SUPPORT.json')
    z=data['z'].double().numpy();y=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']])
    checked=0;max_jac=0.;max_forward=0.
    for case in result['cases']:
        out=a.output/'cases'/case['case'];verify_done(out,cfg['identity'])
        traj=read(out/'TRAJECTORY.json');c=np.array(traj['C']);t=np.array(traj['T'])
        if c.shape!=(65,len(y)) or t.shape!=c.shape:raise ValueError('DIVERGENCE_VERIFY_LENGTH')
        fold=case['case'].split('_')[0];partition=case['partition'];s=support['folds'][fold]
        require_equal(case['partition_rows'],s['expanded'] if partition=='fit' else s['masks'][partition],'DIVERGENCE_PARTITION')
        source=torch.load(a.source/'units'/f"{case['case']}_cost"/'LATEST.pt',weights_only=False,map_location='cpu')
        initial=native_forward(source['state'],z)[0].min(0).argmax(1)
        for arm,root,actions in [('C',a.control,c),('T',a.treatment,t)]:
            if not np.array_equal(actions[0],initial):raise ValueError('DIVERGENCE_VERIFY_INITIAL')
            endpoint=torch.load(root/'units'/case['case']/'ENDPOINT_64.pt',weights_only=False,map_location='cpu')
            if not np.array_equal(actions[-1],native_forward(endpoint['state'],z)[0].min(0).argmax(1)):raise ValueError('DIVERGENCE_VERIFY_FINAL')
        require_equal(locate_events(c,t,y,case['partition_rows']),case['events'],'DIVERGENCE_EVENTS')
        for entry in case['factorials']:
            path=out/entry['file'];require_equal(file_sha(path),entry['sha256'],'DIVERGENCE_WITNESS_HASH')
            witness=torch.load(path,weights_only=False,map_location='cpu');row=entry['row'];p=entry['control_action'];other=entry['treatment_action']
            for key,w in witness['cells'].items():
                require_equal(w['before'],witness['origins'][key[0]]['before'],'DIVERGENCE_WEIGHT_INTERVENTION')
                require_equal(w['optimizer_before'],witness['origins'][key[1]]['optimizer_before'],'DIVERGENCE_OPT_INTERVENTION')
                require_equal(measure(w,z,row,p,other),entry['cells'][key],'DIVERGENCE_MEASURE')
                state={k:v.double().requires_grad_() for k,v in w['before'].items()};q=[]
                for twin in range(2):
                    prefix=f'twins.{twin}.0.'
                    h=torch.nn.functional.silu(torch.nn.functional.linear(torch.from_numpy(z[twin,row:row+1]),state[prefix+'0.weight'],state[prefix+'0.bias']))
                    q.append(torch.nn.functional.linear(h,state[prefix+'2.weight'],state[prefix+'2.bias']))
                qt=torch.stack(q);m=qt.min(0).values[0,p]-qt.min(0).values[0,other]
                g=torch.autograd.grad(m,tuple(state.values()));g=np.concatenate([v.detach().numpy().ravel() for v in g])
                coeff=np.zeros((len(y),3));coeff[row,p]=1;coeff[row,other]-=1
                jac=weighted_min_gradient(w['before'],z,coeff)[row]
                max_jac=max(max_jac,float(np.max(abs(g-jac))))
                qn=native_forward(w['before'],z[:,row:row+1])[0];max_forward=max(max_forward,float(np.max(abs(qt.detach().numpy()-qn))))
                if max_jac>1e-10 or max_forward>1e-11:raise ValueError('DIVERGENCE_INDEPENDENT_DERIVATIVE')
                if key in ('CC','TT'):
                    for field in ('before','after','optimizer_after','rng_after','receipt'):require_equal(w[field],witness['origins'][key[0]][field],'DIVERGENCE_DIAGONAL')
                checked+=1
            for kind,index in [('raw',1),('protected',2)]:
                effects=factorial_effects({k:v['margins'][index] for k,v in entry['cells'].items()})
                require_equal(effects,entry['effects'][kind],'DIVERGENCE_EFFECTS')
    report=dict(identity=cfg['identity'],transactions_exact=result['transactions_exact'],factorial_cells=checked,
        repeated_exact=result['repeated_exact'],max_jacobian_error=max_jac,max_forward_error=max_forward,actor_updates=0,model_promoted=False)
    write(a.verification_output/'CONFIG.json',cfg);write(a.verification_output/'VERIFICATION.json',report);seal(a.verification_output,cfg['identity'])


def run(a):
    from tools.run_decision_cost_contrast import CostContext
    start=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('DIVERGENCE_DEVICE')
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('source','probe','coverage','control','treatment')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    for arm in ('control','treatment'):
        for k in ('source','probe','coverage'):
            if parents[arm]['parents'][k]!=parents[k]['identity']:raise ValueError('DIVERGENCE_LINEAGE')
    if parents['treatment']['parents']['control']!=parents['control']['identity']:raise ValueError('DIVERGENCE_CONTROL')
    cfg=dict(schema='matched_divergence_replay.v1',parents={k:c['identity'] for k,c in parents.items()},cases=[['B',19,'fit'],['A',43,'stop']],
        code={k:file_sha(ROOT/k) for k in ('tools/replay_matched_divergence.py','app/matched_divergence.py','tools/run_targeted_rescue_continuation.py','tools/probe_optimal_fit_protection.py')},
        lr=parents['source']['lr'],weight_decay=parents['source']['weight_decay'],batch=parents['source']['batch'],source_actor_sha256=parents['source']['source_sha256'])
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('DIVERGENCE_ACTOR')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('DIVERGENCE_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu');support=read(a.coverage/'prepared/SUPPORT.json')
    records=read(a.coverage/'PANEL.json')['records'];y=np.stack([validate_record(r) for r in records]);results=[]
    for fold,seed,partition in cfg['cases']:
        name=f'{fold}_{seed}';out=a.output/'cases'/name
        if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
        src=a.source/'units'/f'{name}_cost'/'LATEST.pt';source=torch.load(src,weights_only=False,map_location='cpu')
        snap=torch.load(a.probe/'units'/f'{name}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        old=np.array(support['folds'][fold]['masks']['fit']);limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,y,support,snap,fold,seed,limit);require_equal(ctx.support_receipt,source['support'],'DIVERGENCE_SUPPORT')
        schedule=ctx.fit[covering_schedule(len(ctx.fit),32768,cfg['batch'],seed)];histories={};trajectories={};endpoint_checks=0
        for arm,root in [('C',a.control),('T',a.treatment)]:
            prior=root/'units'/name;verify_done(prior,parents['control' if arm=='C' else 'treatment']['identity'])
            final=torch.load(prior/'ENDPOINT_64.pt',weights_only=False,map_location='cpu')
            if final['source_sha256']!=file_sha(src):raise ValueError('DIVERGENCE_SOURCE')
            opt=restore(ctx,cfg,source['state'],source['optimizer'],source['rng']);budget=[0];history=[]
            actions=[native_forward(source['state'],ctx.zn)[0].min(0).argmax(1).tolist()]
            for offset in range(1,65):
                w=execute(ctx,opt,schedule[1024+8192+offset-1],arm,offset,budget,True)
                require_equal(w['receipt'],final['trace'][offset-1],'DIVERGENCE_TRANSACTION')
                history.append(w);actions.append(native_forward(w['after'],ctx.zn)[0].min(0).argmax(1).tolist())
                if (prior/f'ENDPOINT_{offset}.pt').exists():
                    cp=torch.load(prior/f'ENDPOINT_{offset}.pt',weights_only=False,map_location='cpu')
                    for field,k in [('state','after'),('optimizer','optimizer_after'),('rng','rng_after')]:require_equal(w[k],cp[field],'DIVERGENCE_ENDPOINT')
                    endpoint_checks+=1
                if offset%16==0:write(a.output/'PROGRESS.json',dict(stage='REPLAYING',case=name,arm=arm,offset=offset,seconds=time.monotonic()-start,at=now()))
            histories[arm]=history;trajectories[arm]=actions
        rows=ctx.fit.tolist() if partition=='fit' else support['folds'][fold]['masks'][partition]
        events=locate_events(np.array(trajectories['C']),np.array(trajectories['T']),y,rows);factorials=[]
        for event in events:
            offset=event['control_acquisition'];row=event['row']
            if offset is None or offset<=16:raise ValueError('DIVERGENCE_NOT_POST_SEARCH')
            origins={k:histories[k][offset-1] for k in ('C','T')};cells={};summaries={}
            for key in ('CC','CT','TC','TT'):
                weights,optimizer=origins[key[0]],origins[key[1]]
                opt=restore(ctx,cfg,weights['before'],optimizer['optimizer_before'],weights['rng_before'])
                w=execute(ctx,opt,weights['batch'],key[0],offset,[4],False)
                if key in ('CC','TT'):
                    for field in ('before','after','optimizer_after','rng_after','receipt'):require_equal(w[field],origins[key[0]][field],'DIVERGENCE_DIAGONAL')
                opt=restore(ctx,cfg,weights['before'],optimizer['optimizer_before'],weights['rng_before'])
                repeat=execute(ctx,opt,weights['batch'],key[0],offset,[4],False);require_equal(repeat,w,'DIVERGENCE_REPEAT')
                cells[key]=w;summaries[key]=measure(w,ctx.zn,row,event['control_action'],event['treatment_action'])
            path=out/f'FACTORIAL_{row}_{offset}.pt';checkpoint(path,dict(identity=cfg['identity'],origins=origins,cells=cells))
            effects={kind:factorial_effects({k:v['margins'][index] for k,v in summaries.items()}) for kind,index in [('raw',1),('protected',2)]}
            factorials.append(dict(event,kind=records[row]['kind'],file=path.name,sha256=file_sha(path),cells=summaries,effects=effects))
        result=dict(case=name,partition=partition,partition_rows=rows,events=events,factorials=factorials,
            transactions_exact=128,endpoints_exact=endpoint_checks,repeated_exact=4*len(factorials))
        write(out/'TRAJECTORY.json',trajectories);write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('DIVERGENCE_ACTOR_CHANGED')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],seconds=time.monotonic()-start,cases=results,
        transactions_exact=sum(r['transactions_exact'] for r in results),repeated_exact=sum(r['repeated_exact'] for r in results),actor_updates=0,model_promoted=False))
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','probe','coverage','control','treatment','bundle','output','verification-output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not a.verify_only:run(a)
            if (a.verification_output/'DONE.json').exists():verify_done(a.verification_output,read(a.output/'CONFIG.json')['identity'])
            else:verify(a)
        except Exception as exc:
            if not a.verify_only:write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()))
            raise
