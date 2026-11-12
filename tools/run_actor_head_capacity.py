"""Finite paired fresh actor-head8/16 comparison on cached frozen features."""
import argparse
import fcntl
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from app.actor_head_capacity import make_candidate, install_head, verify_fixed, full_schedule, is_actor_head
from app.actor_learning_diagnostics import actor_terms
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import digest, support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, export_bundle, verify_files
from app.three_signal_learning import environment, evaluate, emit
from tools.diagnose_head_spillover import collect
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_refit_mask_pass import account_equal

POINTS = (0,64,256,1024,4096,10000,20000)


def plot_curve(curve, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(8,6),sharex=True)
    for key in ('loss','value','entropy_term'):
        axes[0].plot([x['updates'] for x in curve],[x[key] for x in curve],label=key)
    axes[0].legend(); axes[0].set_ylabel('TRAIN frozen-critic objective')
    axes[1].plot([x['updates'] for x in curve],[x['q_greedy_agreement'] for x in curve])
    axes[1].set_ylabel('TRAIN Q-greedy agreement'); axes[1].set_xlabel('Head updates')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def run(a):
    started=time.monotonic(); configure(310)
    calibration=read(a.calibration/'CONFIG.json'); verify_done(a.calibration,calibration['identity'])
    source_result=read(a.bundle.parent/'RESULT.json'); verify_done(a.bundle.parent,source_result['identity'])
    spec=read(a.parent/'SPEC.json'); verify_files(a.parent,spec['files'])
    sha=file_sha(a.bundle/'model.pt')
    if sha!=calibration['source_sha256'] or calibration['parent_identity']!=spec['identity']:
        raise ValueError('HEAD_CAPACITY_LINEAGE')
    source,_=load_bundle(a.bundle,'cpu');market=market_for(a.parent,spec,source.recipe)
    if support_hash(market.inputs)!=calibration['support_sha256']:
        raise ValueError('HEAD_CAPACITY_SUPPORT')
    periods=source.recipe['split']['periods']
    if periods!={'TRAIN':['2020-01-01','2024-01-01'],'VALIDATION':['2024-01-01','2025-01-01']}:
        raise ValueError('HEAD_CAPACITY_PERIODS')
    cfg=dict(schema='actor_head_capacity.v1',source_sha256=sha,
        source_identity=source_result['identity'],calibration_identity=calibration['identity'],
        support_sha256=support_hash(market.inputs),seeds=[19,43],widths=[8,16],batch=128,
        updates=20000,points=list(POINTS),learning_rate=source.effective['learning_rate'],
        weight_decay=source.effective['weight_decay'],fit_scope=periods['TRAIN'],
        objective='discrete_SAC_actor_frozen_min_twin_Q_and_alpha',early_stopping=False,
        validation_fit=False,test_read=False,device='cpu',
        code={p:file_sha(Path(__file__).resolve().parents[1]/p) for p in (
            'app/actor_head_capacity.py','tools/run_actor_head_capacity.py',
            'agent_plugins/three_signal_sac.py','app/actor_learning_diagnostics.py',
            'tools/diagnose_head_spillover.py','app/three_signal_learning.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:
        raise ValueError('HEAD_CAPACITY_COLLISION')
    write(a.output/'CONFIG.json',cfg)
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    def progress(stage,**kw):
        report=dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw)
        write(a.output/'PROGRESS.json',report);print(report,flush=True)
    rows=np.flatnonzero(market.inputs.valid)
    fit=np.flatnonzero(market.inputs.period[rows]=='TRAIN')
    risk=read(a.parent/'RISK.json')
    def measure(z,path):
        prepared=prepared_signals(z)
        return {p:evaluate(environment(market,risk,spec['design']['horizon'],period,source.effective['gamma']),
                    z,save=path/p,prepared=prepared) for p,period in periods.items()}
    cache=a.output/'cache'
    if (cache/'DONE.json').exists():
        verify_done(cache,cfg['identity'])
        with np.load(cache/'FEATURES.npz',allow_pickle=False) as saved:
            if not np.array_equal(saved['rows'],rows):raise ValueError('HEAD_CAPACITY_CACHE_ROWS')
            features=saved['features'].copy(); baseline=saved['logits'].copy()
        reference=read(cache/'REFERENCE.json')
    else:
        progress('CACHE_FROZEN_REPRESENTATION',train_rows=len(fit),all_rows=len(rows))
        compact,activations=collect(source,market.inputs.windows,rows)
        features=activations['core'];baseline=np.full((len(market.inputs.valid),3),np.nan,np.float32)
        baseline[rows]=compact
        reference=measure(baseline,cache/'reference')
        for p in periods:account_equal(reference[p],source_result['metrics'][p])
        np.savez_compressed(cache/'FEATURES.npz',features=features,logits=baseline,rows=rows)
        write(cache/'REFERENCE.json',reference);seal(cache,cfg['identity'])
    x=torch.from_numpy(features); arms=[]
    source_state={k:v.clone() for k,v in source.loss.state_dict().items()}
    for seed in cfg['seeds']:
        verify_done(a.calibration/f'critic_{seed}',calibration['identity'])
        calibrated,_=load_bundle(a.bundle,'cpu');calibrated.load(a.calibration/f'critic_{seed}/MODEL.pt')
        # Only critics may differ from the source. Their original calibration
        # changed no actor weights or temperature.
        for k,v in source_state.items():
            if not (k.startswith('qvalue_') or k.startswith('target_qvalue_')):
                if not torch.equal(v,calibrated.loss.state_dict()[k]):
                    raise ValueError('HEAD_CAPACITY_CALIBRATION_ACTOR_CHANGED')
        qcache=a.output/f'targets_{seed}'
        if (qcache/'DONE.json').exists():
            verify_done(qcache,cfg['identity']); q=np.load(qcache/'Q.npy',allow_pickle=False)
        else:
            progress('CACHE_FROZEN_CRITICS',seed=seed)
            chunks=[]
            with torch.inference_mode():
                for start in range(0,len(fit),256):
                    rr=rows[fit[start:start+256]]
                    chunks.append(calibrated.q_surfaces(torch.tensor(market.inputs.windows[rr])).min(0).values.numpy())
            q=np.concatenate(chunks);qcache.mkdir(parents=True,exist_ok=True)
            np.save(qcache/'Q.npy',q)
            write(qcache/'RECEIPT.json',dict(identity=cfg['identity'],
                critic_sha256=file_sha(a.calibration/f'critic_{seed}/MODEL.pt'),
                train_rows=rows[fit].tolist(),alpha=float(calibrated.loss.log_alpha.detach().exp())))
            seal(qcache,cfg['identity'])
        if q.shape!=(len(fit),3) or not np.isfinite(q).all():raise ValueError('HEAD_CAPACITY_Q')
        y=torch.from_numpy(q);alpha=calibrated.loss.log_alpha.detach().exp()
        schedule=full_schedule(len(fit),cfg['updates'],cfg['batch'],seed)
        fixed={k:v.clone() for k,v in calibrated.loss.state_dict().items()}
        for width in cfg['widths']:
            out=a.output/f'seed_{seed}_head_{width}'
            if (out/'DONE.json').exists():
                verify_done(out,cfg['identity']);arms.append(read(out/'RESULT.json'));continue
            begin=time.monotonic();progress('TRAIN_HEAD',seed=seed,width=width,updates=0)
            model,head=make_candidate(calibrated,width,seed)
            opt=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            curve=[]
            for step in range(cfg['updates']+1):
                if step:
                    ix=schedule[step-1]; opt.zero_grad(set_to_none=True)
                    value,entropy,_=actor_terms(head(x[fit[ix]]),y[ix],alpha)
                    loss=value+entropy;loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(),10.,error_if_nonfinite=True);opt.step()
                if step in POINTS:
                    sums=np.zeros(4)
                    with torch.inference_mode():
                        for start in range(0,len(fit),1024):
                            ix=np.arange(start,min(start+1024,len(fit)))
                            z=head(x[fit[ix]]);v,e,_=actor_terms(z,y[ix],alpha)
                            sums+=np.array([float(v+e),float(v),float(e),float((z.argmax(-1)==y[ix].argmax(-1)).float().mean())])*len(ix)
                    values=sums/len(fit)
                    curve.append(dict(updates=step,loss=float(values[0]),value=float(values[1]),
                        entropy_term=float(values[2]),q_greedy_agreement=float(values[3])))
                    write(out/'CURVE.json',curve);progress('TRAIN_HEAD',seed=seed,width=width,updates=step)
            install_head(model,head);verify_fixed(model,fixed)
            progress('NATIVE_PARITY_AND_ACCOUNTS',seed=seed,width=width)
            native,_=emit(model,market.inputs)
            with torch.inference_mode():
                compact=np.concatenate([head(x[start:start+256]).numpy() for start in range(0,len(rows),256)])
            maximum=float(np.abs(compact-native[rows]).max())
            if not np.allclose(compact,native[rows],atol=2e-5,rtol=2e-5):raise ValueError('HEAD_CAPACITY_CACHE_LOGITS')
            if not np.array_equal(prepared_signals(compact)[0],prepared_signals(native[rows])[0]):
                raise ValueError('HEAD_CAPACITY_CACHE_ACTIONS')
            metrics=measure(native,out/'accounts')
            bundle=out/'bundle'
            if not bundle.exists():
                probe=rows[np.linspace(0,len(rows)-1,32,dtype=int)]
                export_bundle(bundle,model,market.inputs.windows[probe],dict(identity=cfg['identity'],width=width,seed=seed))
            cold,_=load_bundle(bundle,'cpu');cold_logits,_=emit(cold,market.inputs)
            if not np.array_equal(native[rows],cold_logits[rows]):raise ValueError('HEAD_CAPACITY_COLD')
            verify_fixed(cold,fixed);verify_fixed(model,fixed)
            torch.save(dict(head=head.state_dict(),optimizer=opt.state_dict(),updates=cfg['updates']),out/'LEARNING.pt')
            plot_curve(curve,out/'CURVE.png')
            r=dict(identity=cfg['identity'],seed=seed,width=width,metrics=metrics,curve=curve,
                head_parameters=sum(p.numel() for p in head.parameters()),updates=cfg['updates'],
                train_rows=len(fit),validation_rows=int((market.inputs.period[rows]=='VALIDATION').sum()),
                schedule_sha256=hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest(),critic_sha256=file_sha(a.calibration/f'critic_{seed}/MODEL.pt'),
                frozen_tensors=sum(not is_actor_head(k) for k in fixed),frozen_exact=True,
                cache_native_max_abs=maximum,cached_actions_exact=True,cold_logits_exact=True,
                eligible=all(m['orders_per_week']>2 for m in metrics.values()),
                source_nondecreasing=all(metrics[p]['return_pct']>=reference[p]['return_pct']-1e-8 for p in periods),
                seconds=time.monotonic()-begin,model_promoted=False,test_read=False)
            write(out/'RESULT.json',r);seal(out,cfg['identity']);arms.append(r)
    if file_sha(a.bundle/'model.pt')!=sha:raise ValueError('HEAD_CAPACITY_SOURCE_MUTATED')
    table=['# Actor head capacity results','','Fixed endpoint; no automatic promotion.','',
        '| Seed | Width | Parameters | TRAIN return % | VAL return % | VAL Sharpe | VAL orders | Eligible |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in arms:
        tr,va=r['metrics']['TRAIN'],r['metrics']['VALIDATION']
        table.append(f"| {r['seed']} | {r['width']} | {r['head_parameters']} | {tr['return_pct']:.6f} | {va['return_pct']:.6f} | {va['sharpe']:.6f} | {va['trades']} | {r['eligible']} |")
    (a.output/'RESULTS.md').write_text('\n'.join(table)+'\n')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],arms=arms,reference=reference,
        seconds=time.monotonic()-started,model_promoted=False,test_read=False,selection_started=False))
    progress('COMPLETE',arms=len(arms));seal(a.output,cfg['identity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','calibration','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
