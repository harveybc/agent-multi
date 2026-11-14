"""Frozen native-head hybrids with per-tensor provenance and FIT-only gates."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from app.critic_fit_probe import fit_metrics
from app.frozen_recipe_inputs import file_sha
from app.head_component_recombination import combine_heads,fit_preservation_gate
from app.native_decision_error import decision_ledger,summarize
from app.native_fit_coverage import validate_record
from app.rl_subset import digest
from app.training_forensics import tree_digest
from tools.analyze_native_decision_error import surfaces
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def measured(s,y,expanded,masks):
    values={k:fit_metrics(s['old'][:,ix],y[ix])['minimum'] for k,ix in masks.items()}
    values['expanded_fit']=fit_metrics(s['fit'],y[expanded])['minimum']
    return values


def run(a):
    started=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('RECOMBINATION_DEVICE')
    cc,pc,dc=[read(p/'CONFIG.json') for p in (a.coverage,a.previous,a.diagnosis)]
    for p,c in ((a.coverage,cc),(a.previous,pc),(a.diagnosis,dc)):verify_done(p,c['identity'])
    if cc['previous']!=pc['identity'] or dc['coverage']!=cc['identity'] or dc['previous']!=pc['identity']:
        raise ValueError('RECOMBINATION_LINEAGE')
    if file_sha(a.bundle/'model.pt')!=cc['source_sha256']:raise ValueError('RECOMBINATION_SOURCE')
    cfg=dict(schema='native_head_recombination.v1',coverage=cc['identity'],previous=pc['identity'],
             diagnosis=dc['identity'],source_sha256=cc['source_sha256'],optimizer_steps=0,tolerance=1e-8,
             variants=['control_hidden_expanded_output','expanded_hidden_control_output'],
             code={k:file_sha(ROOT/k) for k in ('app/head_component_recombination.py',
                 'tools/run_head_component_recombination.py','tools/analyze_native_decision_error.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('RECOMBINATION_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    support=read(a.coverage/'prepared/SUPPORT.json');records=read(a.coverage/'PANEL.json')['records']
    if support['old_count']!=239 or support['rows']!=[r['row'] for r in records]:raise ValueError('RECOMBINATION_ROWS')
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True);z=data['z'].to('cuda')
    y=np.stack([validate_record(r) for r in records]);original=read(a.coverage/'RESULT.json')
    replays=[];reconstructions=[];results=[];outputs={};max_residual=0.;donors=[]
    with np.load(a.diagnosis/'OUTPUTS.npz',allow_pickle=False) as saved_outputs:
        for arm in original['arms']:
            fold,seed=arm['fold'],arm['seed'];info=support['folds'][fold]
            expanded=np.array(info['expanded']);masks={k:np.array(v) for k,v in info['masks'].items()}
            for kind in ('flat_both_allowed','flat_restricted','open_long','open_short'):
                ix=np.array([i for i in masks['audit'] if records[i]['kind']==kind],dtype=int)
                if len(ix):masks['audit_'+kind]=ix
            control=torch.load(a.previous/f'units/{fold}_{seed}_base/LATEST.pt',weights_only=False)['model']
            ch=tree_digest(control);ccstate=combine_heads(control,control)
            if tree_digest(ccstate)!=ch:raise ValueError('RECOMBINATION_SELF_STATE')
            cs=surfaces(ccstate,z,expanded);cm=measured(cs,y,expanded,masks)
            if {k:v for k,v in cm.items() if k!='expanded_fit'}!=arm['control']:
                raise ValueError('RECOMBINATION_CONTROL_REPLAY')
            if not np.array_equal(cs['old'],saved_outputs[f'{fold}_{seed}_control']):
                raise ValueError('RECOMBINATION_SELF_OUTPUT')
            replays.append(dict(fold=fold,seed=seed,endpoint='control',exact=True))
            reconstructions.append(dict(fold=fold,seed=seed,endpoint='control',state_exact=True,outputs_exact=True))
            for step,endpoint in ((8192,'equal_updates'),(32768,'equal_presentations')):
                donor=torch.load(a.coverage/f'units/{fold}_{seed}/ENDPOINT_{step}.pt',weights_only=False)['state']
                dh=tree_digest(donor);nnstate=combine_heads(donor,donor)
                if tree_digest(nnstate)!=dh:raise ValueError('RECOMBINATION_SELF_STATE')
                ns=surfaces(nnstate,z,expanded);nm=measured(ns,y,expanded,masks)
                if nm!=arm[endpoint] or not np.array_equal(ns['old'],saved_outputs[f'{fold}_{seed}_{endpoint}']):
                    raise ValueError('RECOMBINATION_ENDPOINT_REPLAY')
                replays.append(dict(fold=fold,seed=seed,endpoint=endpoint,exact=True))
                reconstructions.append(dict(fold=fold,seed=seed,endpoint=endpoint,state_exact=True,outputs_exact=True))
                combos={cfg['variants'][0]:combine_heads(control,donor),cfg['variants'][1]:combine_heads(donor,control)}
                hybrid={}
                for variant,state in combos.items():
                    name=f'{fold}_{seed}_{endpoint}_{variant}';out=a.output/'hybrids'/name
                    provenance={}
                    for key,value in state.items():
                        hidden=key.split('.')[3]=='0'
                        use_control=hidden if variant==cfg['variants'][0] else not hidden
                        expected=(control if use_control else donor)[key]
                        if not torch.equal(value,expected) or value.data_ptr()==expected.data_ptr():
                            raise ValueError('RECOMBINATION_PROVENANCE')
                        provenance[key]=dict(donor='control' if use_control else 'expanded',sha256=tree_digest(value))
                    surface=surfaces(state,z,expanded);hybrid[variant]=surface
                    metrics=measured(surface,y,expanded,masks);ledger={}
                    for scope,ix in dict(fit=masks['fit'],expanded_fit=expanded,stop=masks['stop'],audit=masks['audit']).items():
                        key='fit' if scope=='expanded_fit' else 'old'
                        q0,q1=cs[key],surface[key]
                        if key=='old':q0,q1=q0[:,ix],q1[:,ix]
                        ledger[scope]=summarize(decision_ledger(q0.min(0),q1.min(0),y[ix]))
                    fit_values=lambda m:{k:m[k]['mean_regret'] for k in ('fit','expanded_fit')}
                    passed=fit_preservation_gate(fit_values(cm),fit_values(metrics),cfg['tolerance'])
                    checkpoint(out/'HEAD.pt',dict(identity=cfg['identity'],state=state,control_sha256=ch,expanded_sha256=dh))
                    r=dict(identity=cfg['identity'],fold=fold,seed=seed,endpoint=endpoint,variant=variant,head_sha256=tree_digest(state),
                           control_sha256=ch,expanded_sha256=dh,provenance=provenance,metrics=metrics,
                           control=cm,expanded=nm,switches=ledger,fit_gate=passed)
                    write(out/'RESULT.json',r);seal(out,cfg['identity']);results.append(r)
                    outputs[name]=surface['old'];outputs[name+'_expanded_fit']=surface['fit']
                interaction={}
                for key,hkey in (('old','hidden'),('fit','fit_hidden')):
                    # This identity is per twin, before minimum aggregation.
                    ccq,nnq=cs[key].astype(float),ns[key].astype(float)
                    cn=hybrid[cfg['variants'][0]][key].astype(float)
                    nc=hybrid[cfg['variants'][1]][key].astype(float)
                    actual=nnq-nc-cn+ccq
                    expected=np.einsum('tni,tai->tna',ns[hkey].astype(float)-cs[hkey].astype(float),
                                       ns['weights'].astype(float)-cs['weights'].astype(float))
                    residual=float(np.max(np.abs(actual-expected)));max_residual=max(max_residual,residual)
                    if not np.allclose(actual,expected,atol=2e-6,rtol=1e-5):raise ValueError('RECOMBINATION_INTERACTION')
                    interaction[key]=dict(residual=residual,rms=float(np.sqrt(np.mean(actual**2))))
                if ch!=tree_digest(control) or dh!=tree_digest(donor):raise ValueError('RECOMBINATION_DONOR_MUTATED')
                donors.append(dict(fold=fold,seed=seed,endpoint=endpoint,donors_unchanged=True,interaction=interaction))
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('RECOMBINATION_SOURCE_CHANGED')
    np.savez_compressed(a.output/'OUTPUTS.npz',**outputs)
    gates={f'{endpoint}:{variant}':dict(passes=sum(r['fit_gate'] for r in results if r['endpoint']==endpoint and r['variant']==variant),total=4)
           for endpoint in ('equal_updates','equal_presentations') for variant in cfg['variants']}
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],hybrids=results,replays=replays,
        self_reconstructions=reconstructions,donors=donors,fit_gates=gates,interaction_max_residual=max_residual,
        optimizer_steps=0,source_preserved=True,model_promoted=False,validation_read=False,test_read=False))
    report(a.output,results)
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),seconds=time.monotonic()-started))


def report(root,results):
    lines=['# Native Head Component Recombination','','FIT-only diagnostic gate; stop/audit are descriptive. No model promotion.','',
           '| Fold | Seed | Endpoint | Hidden/output | Original FIT regret | Expanded FIT regret | Audit regret | FIT pass |',
           '| --- | ---: | --- | --- | ---: | ---: | ---: | --- |']
    for r in results:
        m=r['metrics'];variant='control/new' if r['variant'].startswith('control') else 'new/control'
        lines.append(f"| {r['fold']} | {r['seed']} | {r['endpoint']} | {variant} | {m['fit']['mean_regret']:.6f} | {m['expanded_fit']['mean_regret']:.6f} | {m['audit']['mean_regret']:.6f} | {r['fit_gate']} |")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','coverage','diagnosis','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    run(a)
