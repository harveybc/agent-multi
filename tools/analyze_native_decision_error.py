"""Replay sealed critic heads and attribute exact native decision switches."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from app.critic_fit_probe import fit_metrics
from app.frozen_recipe_inputs import file_sha
from app.native_decision_error import decision_ledger,summarize,fit_gap_bins,gap_labels,layer_delta
from app.native_fit_coverage import validate_record
from app.rl_subset import digest
from app.role_routed_critic import RoleHeads
from app.training_forensics import tree_digest
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import template

ROOT=Path(__file__).resolve().parents[1]


def surfaces(state,z,expanded):
    model=RoleHeads(template('cuda'),1,19);model.load_state_dict(state);model.eval()
    before=tree_digest(model.state_dict())
    with torch.no_grad():
        old=model(z[:,:239]).mean(2).cpu().numpy()
        fit=model(z[:,expanded]).mean(2).cpu().numpy()
        hidden=torch.stack([m[0][1](m[0][0](z[i,:239])) for i,m in enumerate(model.twins)]).cpu().numpy()
        fit_hidden=torch.stack([m[0][1](m[0][0](z[i,expanded])) for i,m in enumerate(model.twins)]).cpu().numpy()
        weights=torch.stack([m[0][2].weight for m in model.twins]).cpu().numpy()
        bias=torch.stack([m[0][2].bias for m in model.twins]).cpu().numpy()
    if before!=tree_digest(model.state_dict()):raise ValueError('ORDERING_STATE_MUTATED')
    return dict(old=old,fit=fit,hidden=hidden,fit_hidden=fit_hidden,weights=weights,bias=bias)


def modes(twins):
    return dict(minimum=twins.min(0),mean=twins.mean(0),twin0=twins[0],twin1=twins[1])


def analyze_scope(old,new,target,records,indices,bins,hidden,output,actor):
    # Float64 aggregation matches the existing calibration metric helper.
    old,new=np.asarray(old,dtype=float),np.asarray(new,dtype=float)
    om,nm=modes(old),modes(new);labels=gap_labels(target,bins)
    ledger=decision_ledger(om['minimum'],nm['minimum'],target)
    a0,a1=old.argmax(-1),new.argmax(-1)
    for i,r in enumerate(ledger):
        rec=records[indices[i]];oa,na=r['old_action'],r['new_action']
        r.update(row=rec['row'],timestamp=rec['timestamp'],kind=rec['kind'],position=rec['position'],
                 gap_bin=labels[i],old_twins=a0[:,i].tolist(),new_twins=a1[:,i].tolist(),
                 new_minimum_winner_neither_twin=bool(na not in a1[:,i]),
                 old_minimum_winner_neither_twin=bool(oa not in a0[:,i]),
                 new_twins_disagree=bool(a1[0,i]!=a1[1,i]),
                 old_twins_disagree=bool(a0[0,i]!=a0[1,i]),
                 hidden_layer_selected_gap_delta=(hidden[:,i,na]-hidden[:,i,oa]).tolist(),
                 output_layer_selected_gap_delta=(output[:,i,na]-output[:,i,oa]).tolist(),
                 old_min_owner=old[:,i].argmin(0).tolist(),new_min_owner=new[:,i].argmin(0).tolist(),
                 actor_action=int(actor[indices[i]]),
                 actor_regret=float(target[i].max()-target[i,actor[indices[i]]]))
    summary=summarize(ledger)
    summary['actor_regret']=float(np.mean([r['actor_regret'] for r in ledger]))
    summary['harmful_below_1e6_predicted_margin']=sum(r['harmful'] and r['new_margin']<=1e-6 for r in ledger)
    summary['changed_to_nop']=sum(r['switched'] and r['new_action']==0 for r in ledger)
    summary['changed_from_nop']=sum(r['switched'] and r['old_action']==0 for r in ledger)
    summary['new_minimum_winner_neither_twin']=sum(r['new_minimum_winner_neither_twin'] for r in ledger)
    return dict(summary=summary,
        by_kind={k:summarize([r for r in ledger if r['kind']==k]) for k in sorted({r['kind'] for r in ledger})},
        by_gap={k:summarize([r for r in ledger if r['gap_bin']==k]) for k in ('all_tied','low','mid','high','unbinned')},
        modes={k:summarize(decision_ledger(om[k],nm[k],target)) for k in om},
        new_mean_to_minimum=summarize(decision_ledger(nm['mean'],nm['minimum'],target)),
        old_mean_to_minimum=summarize(decision_ledger(om['mean'],om['minimum'],target)),
        old_action_counts=np.bincount(om['minimum'].argmax(1),minlength=3).tolist(),
        new_action_counts=np.bincount(nm['minimum'].argmax(1),minlength=3).tolist()),ledger


def run(a):
    started=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('ORDERING_DEVICE')
    cc=read(a.coverage/'CONFIG.json');pc=read(a.previous/'CONFIG.json')
    verify_done(a.coverage,cc['identity']);verify_done(a.previous,pc['identity'])
    if cc['previous']!=pc['identity'] or file_sha(a.bundle/'model.pt')!=cc['source_sha256']:
        raise ValueError('ORDERING_LINEAGE')
    cfg=dict(schema='native_decision_error.v1',coverage=cc['identity'],previous=pc['identity'],
             source_sha256=cc['source_sha256'],threshold_source='original_FIT_only',optimizer_steps=0,
             code={k:file_sha(ROOT/k) for k in ('tools/analyze_native_decision_error.py','app/native_decision_error.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('ORDERING_IDENTITY')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    support=read(a.coverage/'prepared/SUPPORT.json');records=read(a.coverage/'PANEL.json')['records']
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True)
    if support['old_count']!=239 or [r['row'] for r in records]!=support['rows']:raise ValueError('ORDERING_SUPPORT')
    y=np.stack([validate_record(r) for r in records]);z=data['z'].to('cuda')
    actor=data['signals'].numpy()[np.array(support['rows'])]
    original=read(a.coverage/'RESULT.json');results=[];replays=[];all_ledgers=[]
    max_layer_error=0.;outputs={}
    for arm in original['arms']:
        fold,seed=arm['fold'],arm['seed'];info=support['folds'][fold]
        expanded=np.array(info['expanded']);masks={k:np.array(v) for k,v in info['masks'].items()}
        for kind in ('flat_both_allowed','flat_restricted','open_long','open_short'):
            ix=np.array([i for i in masks['audit'] if records[i]['kind']==kind],dtype=int)
            if len(ix):masks['audit_'+kind]=ix
        baseline=torch.load(a.previous/f'units/{fold}_{seed}_base/LATEST.pt',weights_only=False)['model']
        old=surfaces(baseline,z,expanded);bins=fit_gap_bins(y[masks['fit']])
        if {k:fit_metrics(old['old'][:,ix],y[ix])['minimum'] for k,ix in masks.items()}!=arm['control']:
            raise ValueError('ORDERING_CONTROL_REPLAY')
        replays.append(dict(fold=fold,seed=seed,endpoint='control',exact=True))
        outputs[f'{fold}_{seed}_control']=old['old']
        for step,endpoint in ((8192,'equal_updates'),(32768,'equal_presentations')):
            state=torch.load(a.coverage/f'units/{fold}_{seed}/ENDPOINT_{step}.pt',weights_only=False)['state']
            new=surfaces(state,z,expanded)
            measured={k:fit_metrics(new['old'][:,ix],y[ix])['minimum'] for k,ix in masks.items()}
            measured['expanded_fit']=fit_metrics(new['fit'],y[expanded])['minimum']
            if measured!=arm[endpoint]:raise ValueError('ORDERING_ENDPOINT_REPLAY')
            replays.append(dict(fold=fold,seed=seed,endpoint=endpoint,exact=True))
            outputs[f'{fold}_{seed}_{endpoint}']=new['old'];scopes={}
            for scope,ix in dict(fit=masks['fit'],expanded_fit=expanded,stop=masks['stop'],audit=masks['audit']).items():
                key='fit' if scope=='expanded_fit' else 'old'
                hkey='fit_hidden' if scope=='expanded_fit' else 'hidden'
                q0,q1=old[key],new[key];h0,h1=old[hkey],new[hkey]
                hidden,output=layer_delta(h0,h1,old['weights'],new['weights'],old['bias'],new['bias'])
                residual=float(np.max(np.abs((q1.astype(float)-q0.astype(float))-(hidden+output))))
                max_layer_error=max(max_layer_error,residual)
                if not np.allclose(q1.astype(float)-q0.astype(float),hidden+output,atol=2e-6,rtol=1e-5):
                    raise ValueError('ORDERING_LAYER_IDENTITY')
                if scope!='expanded_fit':q0,q1,hidden,output=q0[:,ix],q1[:,ix],hidden[:,ix],output[:,ix]
                scope_report,ledger=analyze_scope(q0,q1,y[ix],records,ix,bins,hidden,output,actor)
                scope_report['layer_delta_rms']=dict(hidden=float(np.sqrt(np.mean(hidden**2))),output=float(np.sqrt(np.mean(output**2))))
                scopes[scope]=scope_report
                all_ledgers.extend(dict(fold=fold,seed=seed,endpoint=endpoint,scope=scope,**r) for r in ledger)
            drift={k:float((state[k]-baseline[k]).norm()/(baseline[k].norm()+1e-30)) for k in state}
            results.append(dict(fold=fold,seed=seed,endpoint=endpoint,fit_gap_bins=bins,
                                parameter_relative_drift=drift,scopes=scopes))
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('ORDERING_SOURCE_CHANGED')
    write(a.output/'LEDGER.json',all_ledgers);np.savez_compressed(a.output/'OUTPUTS.npz',**outputs)
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],comparisons=results,replays=replays,
        layer_identity_max_error=max_layer_error,ledger_rows=len(all_ledgers),optimizer_steps=0,
        source_preserved=True,model_promoted=False,validation_read=False,test_read=False))
    report(a.output,results)
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),seconds=time.monotonic()-started))


def report(root,results):
    lines=['# Native Decision Error Attribution','','Positive delta is worse. Counts are paired row evaluations, not independent trades.','',
           '| Fold | Seed | Endpoint | Switches | Helpful | Harmful | Neutral | Regret delta | Harmful despite lower MSE |',
           '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        s=r['scopes']['audit']['summary']
        lines.append(f"| {r['fold']} | {r['seed']} | {r['endpoint']} | {s['switches']} | {s['helpful']} | {s['harmful']} | {s['neutral_switches']} | {s['regret_delta']:.6f} | {s['lower_mse_but_harmful']} |")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','coverage','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    run(a)
