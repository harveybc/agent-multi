"""Sequential representation/negative controls after authenticated314; actor frozen."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.critic_fit_probe import fit_metrics, probe_q
from app.critic_representation_control import shuffled_fit_records, recommendation
from app.frozen_recipe_inputs import file_sha
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest, support_hash
from app.selected_critic_calibration import calibrate
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now


def report(root, rows, identity):
    decision=recommendation(rows)
    write(root/'RESULT.json',dict(identity=identity,arms=rows,recommendation=decision,
        model_promoted=False,actor_changed=False,validation_read=False,test_read=False,
        stl_included=False,historical_mixed_cot=True))
    lines=['# Critic representation diagnostic','',
        'Fixed endpoint4096. Same55 held-out TRAIN anchors; already inspected panel.',
        'All actors unchanged. No trading improvement or model promotion claimed.','',
        '| Seed | Scope | Labels | Pair accuracy | Mean regret | RMSE |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for r in rows:
        m=r['metrics']['held']['minimum']
        lines.append(f"| {r['seed']} | {r['scope']} | {r['labels']} | {m['pairwise_accuracy']:.6f} | {m['mean_regret']:.6f} | {m['rmse']:.6f} |")
    lines+=['',f'Recommendation: `{decision}`. This is not an automatic actor update.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,seed in enumerate((19,43)):
        for r in rows:
            if r['seed']!=seed:continue
            for ax,key in ((axes[0,col],'pairwise_accuracy'),(axes[1,col],'mean_regret')):
                ax.plot([p['updates'] for p in r['curve']],
                    [p['metrics']['held']['minimum'][key] for p in r['curve']],
                    label=r['scope']+' '+r['labels'],marker='.')
                ax.set(title=f'Seed{seed} {key}',xlabel='Critic updates')
                ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'CRITIC_CURVES.png',dpi=140);plt.close(fig)


def run(a):
    started=time.monotonic();configure(315);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name:
        raise ValueError('REPRESENTATION_DEVICE')
    cc=read(a.coverage/'CONFIG.json');verify_done(a.coverage,cc['identity'])
    if cc['arms']!=['old113','expanded146'] or cc['seeds']!=[19,43] or cc['updates']!=4096:
        raise ValueError('REPRESENTATION_PREDECESSOR')
    parent=read(a.parent/'SPEC.json');verify_files(a.parent,parent['files'])
    source,_=load_bundle(a.bundle,'cuda');sha=file_sha(a.bundle/'model.pt')
    if sha!=cc['source_sha256']:raise ValueError('REPRESENTATION_SOURCE')
    market=market_for(a.parent,parent,source.recipe)
    if source.recipe['list']['n']!=71 or support_hash(market.inputs)!=cc['support']:
        raise ValueError('REPRESENTATION_SUPPORT')
    partition=read(a.coverage/'PARTITION.json');records=partition['records']
    if digest(records)!=cc['records_sha256']:raise ValueError('REPRESENTATION_PARTITION')
    fit=np.array([i for i,r in enumerate(records) if r['role']=='fit'])
    old_count=len(records)-len(cc['new_fit'])-len(cc['held'])
    held=old_count+np.array(cc['held'])
    masks=dict(fit=fit,held=held,old_monitor=np.array([
        i for i,r in enumerate(records[:old_count]) if r['role']=='monitor']))
    if (len(fit),len(held),old_count)!=(146,55,151) or set(fit)&set(held):
        raise ValueError('REPRESENTATION_COUNTS')
    rows=np.array([r['row'] for r in records])
    for r in records:
        if not market.inputs.valid[r['row']] or market.inputs.period[r['row']]!='TRAIN' or int(market.inputs.row_ts_ns[r['row']])!=r['timestamp']:
            raise ValueError('REPRESENTATION_ROW')
        if r['role']=='fit' and r['year'] not in (2020,2021,2022):raise ValueError('REPRESENTATION_YEAR')
    cfg=dict(schema='critic_representation_chain.v1',predecessor=cc['identity'],source_sha256=sha,
        support=cc['support'],records_sha256=cc['records_sha256'],seeds=[19,43],updates=4096,
        new_arms=[['heads','shuffled'],['critic_representation','true'],['critic_representation','shuffled']],
        reused_arms=['expanded146_19','expanded146_43'],historical_mixed_cot=True,
        code={n:file_sha(Path(__file__).resolve().parents[1]/n) for n in (
            'tools/run_critic_representation_chain.py','app/critic_representation_control.py',
            'app/selected_critic_calibration.py','app/critic_fit_probe.py','app/action_ranking_contrast.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:
        raise ValueError('REPRESENTATION_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    flags={n:p.requires_grad for n,p in source.loss.named_parameters()}
    trainrows=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    baseline=emit_actor_reference(source,market.inputs,trainrows,flags)[0][trainrows]
    original={k:v.detach().cpu().clone() for k,v in source.loss.state_dict().items()}
    x=torch.tensor(market.inputs.windows[rows],device='cuda')
    truth=np.array([[o['reward_return'] for o in r['outcomes']] for r in records])
    def measures(q):
        result={k:fit_metrics(q[:,ix],truth[ix]) for k,ix in masks.items()}
        for year in range(2020,2024):
            ix=np.array([i for i in held if records[i]['year']==year])
            if len(ix):result[f'held_{year}']=fit_metrics(q[:,ix],truth[ix])
        for kind in sorted({records[i]['kind'] for i in held}):
            ix=np.array([i for i in held if records[i]['kind']==kind])
            result['held_'+kind]=fit_metrics(q[:,ix],truth[ix])
        return result
    results=[];completed=0
    for seed in cfg['seeds']:
        prior=a.coverage/f'expanded146_{seed}';verify_done(prior,cc['identity'])
        p=read(prior/'RESULT.json')
        if not p['actor_exact'] or not p['cold_exact'] or p['fit_count']!=146 or p['held_count']!=55:
            raise ValueError('REPRESENTATION_CONTROL')
        results.append(dict(p,scope='heads',labels='true',reused=True,curve=read(prior/'CURVE.json')))
        shuffled,shuffle_receipt=shuffled_fit_records(records,seed)
        for scope,labels in cfg['new_arms']:
            out=a.output/f'{scope}_{labels}_{seed}';out.mkdir(parents=True,exist_ok=True)
            if (out/'DONE.json').exists():
                verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));completed+=1;continue
            model,_=load_bundle(a.bundle,'cuda');curve=[]
            def progress(point):
                with torch.no_grad():q=probe_q(model,x).cpu().numpy()
                curve.append(dict(updates=point['updates'],metrics=measures(q),fit_target_metrics=point['metrics']['fit']))
                write(out/'CURVE.json',curve)
                write(a.output/'PROGRESS.json',dict(stage='CALIBRATION',scope=scope,labels=labels,seed=seed,
                    updates=point['updates'],arms_complete=completed,arms_total=6,at=now(),seconds=time.monotonic()-started))
            receipt=calibrate(model,x,shuffled if labels=='shuffled' else records,seed,progress,scope=scope)
            if not np.array_equal(emit_actor_reference(model,market.inputs,trainrows,flags)[0][trainrows],baseline):
                raise ValueError('REPRESENTATION_ACTOR_CHANGED')
            changed=[k for k,v in model.loss.state_dict().items() if not torch.equal(v.detach().cpu(),original[k])]
            if not changed or any(not k.startswith(('qvalue_network_params.','target_qvalue_network_params.')) for k in changed):
                raise ValueError('REPRESENTATION_FROZEN_CHANGED')
            if scope=='heads' and any('.head.' not in k for k in changed):raise ValueError('REPRESENTATION_HEAD_SCOPE')
            if scope=='critic_representation' and not any('.encoder.' in k for k in changed):
                raise ValueError('REPRESENTATION_ENCODER_NOT_MOVING')
            for n,p in model.loss.named_parameters():p.requires_grad_(flags[n])
            model.save(out/'MODEL.pt');cold,_=load_bundle(a.bundle,'cuda');cold.load(out/'MODEL.pt')
            if not np.array_equal(emit_actor_reference(cold,market.inputs,trainrows,flags)[0][trainrows],baseline):
                raise ValueError('REPRESENTATION_COLD_ACTOR')
            with torch.no_grad():
                if not torch.equal(cold.q_surfaces(x),model.q_surfaces(x)):raise ValueError('REPRESENTATION_COLD_Q')
            result=dict(identity=cfg['identity'],scope=scope,labels=labels,seed=seed,curve=curve,
                metrics=curve[-1]['metrics'],actor_exact=True,cold_exact=True,changed_tensors=changed,
                fit_count=146,held_count=55,model_sha256=file_sha(out/'MODEL.pt'))
            write(out/'CALIBRATION.json',receipt)
            if labels=='shuffled':write(out/'SHUFFLE.json',shuffle_receipt)
            write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result);completed+=1
            del model,cold
    if file_sha(a.bundle/'model.pt')!=sha:raise ValueError('REPRESENTATION_SOURCE_MUTATED')
    report(a.output,results,cfg['identity']);seal(a.output,cfg['identity'])
    write(a.output/'PROGRESS.json',dict(stage='COMPLETE',arms_complete=6,arms_total=6,
        at=now(),seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','bundle','coverage','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
