"""Cold CPU verification and matched-budget reporting for342; no fitting."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.training_forensics import tree_digest
from app.weight_decision_forensics import native_forward
from tools.replay_first_harmful_crossings import require_equal
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write


def score(q,y,temperature):
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float)
    logits=q.min(0)/temperature;logits-=logits.max(1,keepdims=True)
    probability=np.exp(logits);probability/=probability.sum(1,keepdims=True)
    costs=y.max(1,keepdims=True)-y;action=q.min(0).argmax(1)
    return dict(cost=float(temperature*(probability*costs).sum(1).mean()),
        regret=float(costs[np.arange(len(y)),action].mean()),actions_sha256=tree_digest(action))


def verify_trace(trace,baseline):
    previous={k:dict(v) for k,v in baseline.items()};accepted=0
    for offset,r in enumerate(trace,1):
        if r['offset']!=offset:raise ValueError('CONT_VERIFY_OFFSET')
        for k,v in previous.items():
            for metric in ('cost','regret'):
                if r['details']['baseline'][k][metric]!=v[metric]:raise ValueError('CONT_VERIFY_CHAIN')
        if not r['accepted']:continue
        a=r['details']['assessment']
        if not a['passed'] or not a['original_passed'] or not a['motion']>0:raise ValueError('CONT_VERIFY_FINITE')
        if set(a['metrics'])!=set(previous):raise ValueError('CONT_VERIFY_VIEWS')
        for k,v in a['metrics'].items():
            if not v['passed'] or v['lost_optimal'] or not v['cost']<previous[k]['cost'] or v['regret']>previous[k]['regret']:
                raise ValueError('CONT_VERIFY_QUALITY')
        previous=a['metrics'];accepted+=1
    return accepted,previous


def run(a):
    torch.set_num_threads(1)
    cfg=read(a.run/'CONFIG.json');verify_done(a.run,cfg['identity']);result=read(a.run/'RESULT.json')
    for k in ('source','control','coverage'):
        c=read(getattr(a,k)/'CONFIG.json');verify_done(getattr(a,k),c['identity'])
        if c['identity']!=cfg['parents'][k]:raise ValueError('CONT_VERIFY_PARENT')
    support=read(a.coverage/'prepared/SUPPORT.json');z=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True)['z'].double().numpy()
    y=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']]);rows=[];traces=0;endpoints=0;max_error=0.
    for u in result['units']:
        fold,seed=u['fold'],u['seed'];name=f'{fold}_{seed}';out=a.run/'units'/name;verify_done(out,cfg['identity'])
        src=a.source/'units'/f'{name}_cost'/'LATEST.pt';source=torch.load(src,weights_only=False,map_location='cpu')
        control_path=a.control/'units'/f'{name}_complete'/'ENDPOINT_64.pt'
        if file_sha(control_path)!=u['control_sha256']:raise ValueError('CONT_VERIFY_CONTROL')
        control=torch.load(control_path,weights_only=False,map_location='cpu')
        if control['source_sha256']!=file_sha(src):raise ValueError('CONT_VERIFY_SOURCE')
        masks={k:np.array(v) for k,v in support['folds'][fold]['masks'].items()}
        fit=np.array(support['folds'][fold]['expanded'])
        if tree_digest(fit)!=source['support']['expanded_sha256']:raise ValueError('CONT_VERIFY_SUPPORT')
        temperature=source['support']['temperature']
        def check(state,metrics):
            nonlocal max_error
            q=native_forward(state,z)[0]
            qt=[]
            for twin in range(2):
                prefix=f'twins.{twin}.0.'
                hidden=torch.nn.functional.silu(torch.nn.functional.linear(torch.from_numpy(z[twin]),state[prefix+'0.weight'].double(),state[prefix+'0.bias'].double()))
                qt.append(torch.nn.functional.linear(hidden,state[prefix+'2.weight'].double(),state[prefix+'2.bias'].double()).numpy())
            error=float(np.max(abs(q-np.stack(qt))));max_error=max(max_error,error)
            if error>1e-11:raise ValueError('CONT_VERIFY_FORWARD')
            for split,ix in [('fit',fit),('stop',masks['stop']),('audit',masks['audit'])]:
                expected=metrics['fit']['numpy'] if split=='fit' else metrics['held_internal_train'][split]
                actual=score(q[:,ix],y[ix],temperature)
                if actual['actions_sha256']!=expected['actions_sha256']:raise ValueError('CONT_VERIFY_ACTIONS')
                for metric in ('cost','regret'):
                    if abs(actual[metric]-expected[metric])>1e-12:raise ValueError('CONT_VERIFY_METRICS')
            return q
        q0=check(source['state'],u['curve'][0]['metrics']);qc=check(control['state'],u['control'])
        for offset in (16,32,48,64):
            cp=torch.load(out/f'ENDPOINT_{offset}.pt',weights_only=False,map_location='cpu')
            if cp['source_sha256']!=file_sha(src) or cp['identity']!=cfg['identity'] or cp['offset']!=offset:raise ValueError('CONT_VERIFY_CHECKPOINT')
            require_equal(cp['curve'],u['curve'][:offset//16+1],'CONT_VERIFY_CURVE')
            accepted,last=verify_trace(cp['trace'],u['curve'][0]['metrics']['fit'])
            if accepted!=cp['accepted']:raise ValueError('CONT_VERIFY_ACCEPTED')
            for v in cp['optimizer']['state'].values():
                if int(v['step'])!=source['accepted']+accepted:raise ValueError('CONT_VERIFY_OPTIMIZER')
            metrics=cp['curve'][-1]['metrics'];q=check(cp['state'],metrics);endpoints+=1
            for k,v in last.items():
                for m in ('cost','regret'):
                    if v[m]!=metrics['fit'][k][m]:raise ValueError('CONT_VERIFY_TRACE_ENDPOINT')
        traces+=len(cp['trace'])
        for label,qq,metrics in [('source',q0,u['curve'][0]['metrics']),('control64',qc,u['control']),('protected64',q,metrics)]:
            actions=qq[:,fit].min(0).argmax(1);initial=q0[:,fit].min(0).argmax(1);target=y[fit]
            delta=target[np.arange(len(fit)),actions]-target[np.arange(len(fit)),initial]
            rows.append(dict(case=name,arm=label,accepted=u['accepted'] if label=='protected64' else None,
                fit_cost=metrics['fit']['numpy']['cost'],fit_regret=metrics['fit']['numpy']['regret'],
                stop_regret=metrics['held_internal_train']['stop']['regret'],audit_regret=metrics['held_internal_train']['audit']['regret'],
                changed=int(np.sum(actions!=initial)),improved=int(np.sum(delta>0)),worsened=int(np.sum(delta<0))))
    a.output.mkdir(parents=True,exist_ok=True)
    report=dict(identity=cfg['identity'],checkpoints=endpoints,trace_steps=traces,forward_max_error=max_error,
        cold_exact_reported=sum(u['cold_exact'] for u in result['units']),resumed_exact_reported=sum(u['resumed_exact'] for u in result['units']),
        rows=rows,actor_updates=0,model_promoted=False)
    write(a.output/'VERIFICATION.json',report);write(a.output/'CONFIG.json',cfg)
    with (a.output/'SUMMARY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,u in zip(axes.flat,result['units']):
        ax.plot([c['offset'] for c in u['curve']],[c['metrics']['fit']['numpy']['regret'] for c in u['curve']],marker='o',label='Protected FIT')
        ax.axhline(u['control']['fit']['numpy']['regret'],color='red',linestyle='--',label='Saved control at64')
        ax.set(title=f"{u['fold']} / seed {u['seed']}",xlabel='Attempt',ylabel='FIT decision regret');ax.legend()
    fig.tight_layout();fig.savefig(a.output/'FIT_REGRET.png',dpi=140);plt.close(fig)
    verify_done(a.run,cfg['identity']);seal(a.output,cfg['identity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','source','control','coverage','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
