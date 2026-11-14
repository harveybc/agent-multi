"""Automatic paired MSE/cost/hybrid refinement with geometry-aware scoring."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.decision_cost_objective import fit_temperature,cost_objective,stable_added_mask,qualifies_extension
from app.frozen_recipe_inputs import file_sha
from app.interior_adamw_repair import transactional_step
from app.native_fit_coverage import validate_record
from app.native_head_warm_start import configure_scope
from app.rl_subset import digest
from app.robust_adamw_direction import decision_guard,numpy_heads
from app.training_forensics import tree_digest,rng_state,restore_rng
from tools.audit_fixed_actor_critics import covering_schedule
from tools.probe_robust_adamw_direction import scores,fit_quality
from tools.run_automated_robust_refinement import Context
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def initialize_optimizer(head,snapshot,lr,weight_decay,mode):
    if mode not in ('inherited','fresh'):raise ValueError('COST_OPT_MODE')
    before=tree_digest(head.state_dict())
    opt,_=configure_scope(head,'full_head',lr,weight_decay)
    if mode=='inherited':opt.load_state_dict(deepcopy(snapshot))
    steps={int(v['step']) for v in opt.state.values()}
    if len(steps)>1 or (mode=='inherited' and not steps):raise ValueError('COST_OPT_START')
    if tree_digest(head.state_dict())!=before:raise ValueError('COST_OPT_CHANGED_WEIGHTS')
    return opt,next(iter(steps)) if steps else 0


class CostContext(Context):
    def __init__(self,data,targets,support,proposal,fold,seed,limit):
        fit=list(support['folds'][fold]['expanded']);old=support['folds'][fold]['masks']['fit']
        self.old_positions=np.array([fit.index(i) for i in old]);self.stable=None
        self.temperature=fit_temperature(targets[fit])
        super().__init__(data,targets,support,proposal,fold,seed,limit)
        self.head.load_state_dict(self.initial);self.cpu.load_state_dict(self.initial)
        with torch.no_grad():
            gpu=self.head(self.z[:,self.fit]).mean(2).min(0).values.cpu().numpy()
            cpu=self.cpu(self.zcpu[:,self.fit]).mean(2).min(0).values.numpy()
            single=np.concatenate([self.head(self.z[:,i:i+1]).mean(2).min(0).values.cpu().numpy() for i in self.fit])
        independent=numpy_heads(self.initial,self.zn[:,self.fit]).min(0)
        self.stable=stable_added_mask(self.old_positions,[gpu,cpu,independent,single])
        if not self.stable.any():raise ValueError('COST_EMPTY_STABLE_ADDED')
        self.baseline=self.observe(self.initial,single=True)
        self.support_receipt=dict(temperature=self.temperature,expanded_rows=len(self.fit),
            stable_added_rows=int(self.stable.sum()),stable_added_indices=self.fit[self.stable].tolist(),
            expanded_sha256=tree_digest(self.fit),original_sha256=tree_digest(self.old),
            unstable_added_rows=int(len(self.fit)-len(self.old)-self.stable.sum()))

    def quality(self,q):
        target=self.targets[self.fit];result=fit_quality(q,target)
        logits=q.min(0)/self.temperature;logits-=logits.max(1,keepdims=True)
        p=np.exp(logits);p/=p.sum(1,keepdims=True);costs=target.max(1,keepdims=True)-target
        result['cost']=float(self.temperature*np.mean(np.sum(p*costs,axis=1)))
        result['hybrid']=result['mse']+result['cost']
        if self.stable is not None:
            result['stable_added_regret']=fit_quality(q[:,self.stable],target[self.stable])['regret']
        return result

    def observe(self,state,single=False):
        self.head.load_state_dict(state);self.cpu.load_state_dict(state)
        go,gf=scores(self.head,self.z,self.fit);co,cf=scores(self.cpu,self.zcpu,self.fit)
        no=numpy_heads(state,self.zn[:,:239]);nf=numpy_heads(state,self.zn[:,self.fit])
        guards={}
        for name,old,fit in (('gpu32',go,gf),('cpu32',co,cf),('numpy64',no,nf)):
            for geom,q in (('guard',old.min(0)[self.old]),('expanded',fit.min(0)[self.old_positions])):
                guards[name+'_'+geom]=decision_guard(q,self.actions,self.margin,self.targets[self.old],self.limit)
        if single:
            with torch.no_grad():
                for name,head,z in (('gpu32',self.head,self.z),('cpu32',self.cpu,self.zcpu)):
                    q=np.concatenate([head(z[:,i:i+1]).mean(2).min(0).values.cpu().numpy() for i in self.old])
                    guards[name+'_single']=decision_guard(q,self.actions,self.margin,self.targets[self.old],self.limit)
        for g in guards.values():g['actions_sha256']=digest(g.pop('actions'))
        return dict(passed=bool(all(g['passed'] for g in guards.values())),guards=guards,
            fit=dict(gpu=self.quality(gf),cpu=self.quality(cf),numpy=self.quality(nf)))

    def assess(self,state,accepted):
        obs=self.observe(state,single=True)
        improved=all(self.baseline['fit'][k]['stable_added_regret']-obs['fit'][k]['stable_added_regret']>1e-8
                     for k in ('gpu','cpu','numpy'))
        return dict(robust=obs['passed'],accepted=accepted,stable_added_improved=bool(improved),observation=obs)


def run(a):
    started=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('COST_DEVICE')
    pc,cc,rc=[read(p/'CONFIG.json') for p in (a.probe,a.coverage,a.control)]
    for path,c in ((a.probe,pc),(a.coverage,cc),(a.control,rc)):verify_done(path,c['identity'])
    if pc['parents'][1]!=cc['identity'] or rc['probe']!=pc['identity']:raise ValueError('COST_LINEAGE')
    loss_control=None
    if a.optimizer_state=='fresh':
        if a.loss_control is None:raise ValueError('COST_RESET_NEEDS_INHERITED_CONTROL')
        lc=read(a.loss_control/'CONFIG.json');verify_done(a.loss_control,lc['identity'])
        for key,expected in (('probe',pc['identity']),('coverage',cc['identity']),('control',rc['identity']),
                ('source_sha256',pc['source_sha256']),('short_budget',256),('objectives',['mse','cost','hybrid']),
                ('temperature','median_positive_expanded_FIT_cost'),('lr',pc['lr']),
                ('weight_decay',pc['weight_decay']),('batch',pc['batch'])):
            if lc.get(key)!=expected:raise ValueError('COST_RESET_CONTROL_CONTRACT:'+key)
        if lc.get('optimizer_state','inherited')!='inherited':raise ValueError('COST_RESET_CONTROL_MODE')
        loss_control=read(a.loss_control/'RESULT.json')
        if loss_control['identity']!=lc['identity']:raise ValueError('COST_RESET_CONTROL_IDENTITY')
    elif a.loss_control is not None:raise ValueError('COST_UNEXPECTED_LOSS_CONTROL')
    cfg=dict(schema='decision_cost_contrast.v1',probe=pc['identity'],coverage=cc['identity'],control=rc['identity'],
        source_sha256=pc['source_sha256'],objectives=['mse','cost','hybrid'],short_budget=256,extended_budget=1024,
        lr=pc['lr'],weight_decay=pc['weight_decay'],batch=pc['batch'],temperature='median_positive_expanded_FIT_cost',
        code={k:file_sha(ROOT/k) for k in ('tools/run_decision_cost_contrast.py','app/decision_cost_objective.py',
            'app/native_head_warm_start.py','app/interior_adamw_repair.py','app/robust_adamw_direction.py',
            'tools/run_automated_robust_refinement.py','tools/probe_robust_adamw_direction.py')})
    cfg['optimizer_state']=a.optimizer_state
    cfg['inherited_loss_control']=loss_control['identity'] if loss_control is not None else None
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('COST_SOURCE')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('COST_RESUME_IDENTITY')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**kw):write(a.output/'PROGRESS.json',dict(stage=stage,seconds=time.monotonic()-started,at=now(),**kw))
    support=read(a.coverage/'prepared/SUPPORT.json');records=read(a.coverage/'PANEL.json')['records']
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')
    if support['old_count']!=239 or support['rows']!=[r['row'] for r in records]:raise ValueError('COST_SUPPORT')
    targets=np.stack([validate_record(r) for r in records]);cells=[(f,s) for f in ('A','B') for s in (19,43)]
    results=[];controls=read(a.control/'RESULT.json')['sequential'];baseline={};receipts={}
    def context(fold,seed):
        snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        old=np.array(support['folds'][fold]['masks']['fit']);acts=snap['actions']
        limit=float(np.mean(targets[old].max(1)-targets[old][np.arange(len(old)),acts]))
        ctx=CostContext(data,targets,support,snap,fold,seed,limit)
        return ctx,snap
    def unit(fold,seed,objective,budget):
        out=a.output/'units'/f'{fold}_{seed}_{objective}';ctx,snap=context(fold,seed)
        key=f'{fold}_{seed}';baseline[key]=ctx.baseline;receipts[key]=ctx.support_receipt
        if loss_control is not None:
            prior=torch.load(a.loss_control/'units'/out.name/'LATEST.pt',weights_only=False,map_location='cpu')
            if (prior['initial_sha256']!=tree_digest(snap['initial']) or prior['support']!=ctx.support_receipt
                    or loss_control['baselines'][key]!=ctx.baseline):raise ValueError('COST_RESET_PAIRING')
        opt,original_step=initialize_optimizer(ctx.head,snap['optimizer_before'],cfg['lr'],cfg['weight_decay'],a.optimizer_state)
        schedule=ctx.fit[covering_schedule(len(ctx.fit),32768,cfg['batch'],seed)]
        loss=lambda q,y:cost_objective(q,y,objective,ctx.temperature)
        curve=[];trace=[];start=0;accepted=0;latest=out/'LATEST.pt'
        if latest.exists():
            saved=torch.load(latest,weights_only=False,map_location='cpu')
            if (saved['identity']!=cfg['identity'] or saved['initial_sha256']!=tree_digest(snap['initial'])
                    or saved['support']!=ctx.support_receipt):raise ValueError('COST_CHECKPOINT_IDENTITY')
            ctx.head.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
            start=saved['step'];accepted=saved['accepted'];curve=saved['curve'];trace=saved['trace']
            if ctx.assess(saved['state'],accepted)!=curve[-1]['assessment']:raise ValueError('COST_RESUME_METRICS')
            if start in (64,256,1024) and not (out/f'ENDPOINT_{start}.pt').exists():checkpoint(out/f'ENDPOINT_{start}.pt',saved)
        if start>budget:return next(p for p in curve if p['step']==budget)
        for step in range(start,budget+1):
            if step==start and latest.exists():continue
            if step:
                ix=schedule[8192+step-1]
                rec=transactional_step(ctx.head,opt,ctx.z[:,ix],ctx.y[ix],lambda b,p:ctx.repair(b,p,'interior'),loss_fn=loss)
                accepted+=int(rec['accepted']);trace.append(dict(step=step,**rec))
            if step not in (0,1,8) and step%32:continue
            state={k:v.detach().cpu().clone() for k,v in ctx.head.state_dict().items()}
            point=dict(step=step,assessment=ctx.assess(state,accepted));curve.append(point)
            if any(int(v['step'])!=original_step+accepted for v in opt.state.values()):raise ValueError('COST_OPT_COUNT')
            saved=dict(identity=cfg['identity'],initial_sha256=tree_digest(snap['initial']),state=state,
                optimizer=deepcopy(opt.state_dict()),rng=rng_state(),step=step,accepted=accepted,
                trace=trace,curve=curve,support=ctx.support_receipt)
            checkpoint(latest,saved);write(out/'CURVE.json',curve)
            if step in (64,256,1024):checkpoint(out/f'ENDPOINT_{step}.pt',saved)
            progress('FIT',unit=out.name,step=step,budget=budget,accepted=accepted,finished_endpoints=len(results))
        return curve[-1]
    for fold,seed in cells:
        for objective in cfg['objectives']:
            result=unit(fold,seed,objective,256);results.append(dict(fold=fold,seed=seed,objective=objective,**result))
    gates={};qualified=[]
    for objective in (cfg['objectives'] if loss_control is not None else ('cost','hybrid')):
        backend_gates={}
        for backend in ('gpu','cpu','numpy'):
            rows=[]
            for fold,seed in cells:
                candidate=next(r for r in results if (r['fold'],r['seed'],r['objective'],r['step'])==(fold,seed,objective,256))
                control_rows=loss_control['results'] if loss_control is not None else results
                control_objective=objective if loss_control is not None else 'mse'
                control=next(r for r in control_rows if (r['fold'],r['seed'],r['objective'],r['step'])==(fold,seed,control_objective,256))
                x=candidate['assessment'];b=baseline[f'{fold}_{seed}']['fit'][backend]
                rows.append(dict(fold=fold,seed=seed,robust=x['robust'],accepted=x['accepted'],
                    before=b['stable_added_regret'],after=x['observation']['fit'][backend]['stable_added_regret'],
                    control=control['assessment']['observation']['fit'][backend]['stable_added_regret']))
            backend_gates[backend]=dict(passed=qualifies_extension(rows),rows=rows)
        passed=all(g['passed'] for g in backend_gates.values());gates[objective]=dict(passed=passed,backends=backend_gates)
        if passed:qualified.append(objective)
    write(a.output/'EXTENSION_GATE.json',gates)
    if qualified:
        for fold,seed in cells:
            for objective in (qualified if loss_control is not None else ['mse',*qualified]):
                result=unit(fold,seed,objective,1024);results.append(dict(fold=fold,seed=seed,objective=objective,**result))
    replayed=resumed=0;comparisons=[]
    for fold,seed in cells:
        for objective in cfg['objectives']:
            out=a.output/'units'/f'{fold}_{seed}_{objective}';ctx,snap=context(fold,seed)
            saved=torch.load(out/'LATEST.pt',weights_only=False,map_location='cpu')
            for path in sorted(out.glob('ENDPOINT_*.pt')):
                s=torch.load(path,weights_only=False,map_location='cpu')
                actual=ctx.assess(s['state'],s['accepted']);expected=next(p['assessment'] for p in saved['curve'] if p['step']==s['step'])
                if actual!=expected:raise ValueError('COST_COLD_REPLAY')
                replayed+=1
            s=torch.load(out/'ENDPOINT_64.pt',weights_only=False,map_location='cpu');ctx.head.load_state_dict(s['state'])
            opt,_=configure_scope(ctx.head,'full_head',cfg['lr'],cfg['weight_decay']);opt.load_state_dict(s['optimizer']);restore_rng(s['rng'])
            schedule=ctx.fit[covering_schedule(len(ctx.fit),32768,cfg['batch'],seed)]
            loss=lambda q,y:cost_objective(q,y,objective,ctx.temperature)
            for step in (65,66,67):
                ix=schedule[8192+step-1]
                actual=transactional_step(ctx.head,opt,ctx.z[:,ix],ctx.y[ix],lambda b,p:ctx.repair(b,p,'interior'),loss_fn=loss)
                if dict(step=step,**actual)!=saved['trace'][step-1]:raise ValueError('COST_RESUME_TRANSITION')
                resumed+=1
            if objective=='mse':
                actual=next(p for p in saved['curve'] if p['step']==256)['assessment']['observation']['fit']['gpu']
                prior=next(p for p in controls if (p['fold'],p['seed'],p['kind'],p['step'])==(fold,seed,'interior',256))['assessment']['observation']['fit']['gpu']
                comparisons.append(dict(fold=fold,seed=seed,prior331=prior,current={k:actual[k] for k in ('mse','regret')},
                    same_metrics=all(actual[k]==prior[k] for k in ('mse','regret'))))
            write(out/'RESULT.json',dict(fold=fold,seed=seed,objective=objective,curve=saved['curve'],support=saved['support']))
            seal(out,cfg['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('COST_SOURCE_CHANGED')
    for p,c in ((a.probe,pc),(a.coverage,cc),(a.control,rc)):verify_done(p,c['identity'])
    if loss_control is not None:verify_done(a.loss_control,loss_control['identity'])
    write(a.output/'REPLAY.json',dict(cold_endpoints_exact=replayed,resumed_transitions_exact=resumed,mse_control_comparisons=comparisons))
    result=dict(identity=cfg['identity'],seconds=time.monotonic()-started,results=results,baselines=baseline,
        supports=receipts,gates=gates,extended=qualified,source_preserved=True,model_promoted=False,
        actor_updates=0,encoder_updates=0,validation_read=False,test_read=False)
    write(a.output/'RESULT.json',result)
    lines=['# Decision Cost Contrast','','Native critic FIT metrics; not trading returns.','',
        '| Fold | Seed | Loss | Attempts | Accepted | Robust | MSE | Full regret | Stable-added regret |',
        '| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for r in results:
        x=r['assessment'];m=x['observation']['fit']['gpu']
        lines.append(f"| {r['fold']} | {r['seed']} | {r['objective']} | {r['step']} | {x['accepted']} | {x['robust']} | {m['mse']:.6f} | {m['regret']:.6f} | {m['stable_added_regret']:.6f} |")
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    plot_curves(a.output)
    seal(a.output,cfg['identity']);progress('COMPLETE',extended=qualified,endpoints=len(results))


def plot_curves(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex='col')
    for col,(fold,seed) in enumerate([(f,s) for f in ('A','B') for s in (19,43)]):
        for objective,color in (('mse','#bb6633'),('cost','#327bb5'),('hybrid','#16856a')):
            curve=read(output/'units'/f'{fold}_{seed}_{objective}'/'CURVE.json');xs=[p['step'] for p in curve]
            for row,key in enumerate(('mse','stable_added_regret')):
                axes[row,col].plot(xs,[p['assessment']['observation']['fit']['gpu'][key] for p in curve],label=objective,color=color)
                axes[row,col].grid(alpha=.2);axes[row,col].set_xlim(left=0)
        axes[0,col].set_title(f'{fold} / {seed}');axes[1,col].set_xlabel('Attempted updates')
    axes[0,0].set_ylabel('Full FIT MSE');axes[1,0].set_ylabel('Stable-added FIT regret')
    mode=read(output/'CONFIG.json').get('optimizer_state','inherited')
    axes[0,0].legend();fig.suptitle(f'Decision loss contrast ({mode} AdamW); source actor unchanged')
    fig.tight_layout();fig.savefig(output/'CURVES.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','coverage','control','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--optimizer-state',choices=('inherited','fresh'),default='inherited')
    p.add_argument('--loss-control',type=Path)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:
            write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise
