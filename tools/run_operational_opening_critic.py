"""Main-lane bounded critic correction on flat-state paired native outcomes."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app.action_ranking_contrast import advantage_loss
from app.actor_learning_diagnostics import clear_optimizer_memory
from app.critic_fit_probe import fit_metrics,set_scope,probe_q
from app.decision_attribution import continuation,state_view
from app.frozen_recipe_inputs import file_sha
from app.operational_opening_critic import validate_labels,BOUNDARY
from app.policy_return_calibration import policy_signals
from app.rl_subset import digest,project_inputs,support_hash
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import emit,environment
from tools.audit_fixed_actor_critics import covering_schedule
from tools.diagnose_warmstart_drift import frozen_state,verify_frozen_state
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write,configure,now
from tools.run_structural_sac_refit import market_for

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=args.expected_device_name:raise ValueError('OPENING_CRITIC_GPU')
    parent=read(args.parent/'SPEC.json');verify_files(args.parent,parent['files'])
    prior=read(args.diagnostic/'report/SUMMARY.json');verify_done(args.diagnostic/'report',prior['identity'])
    control=read(args.control/'CONFIG.json');verify_done(args.control,control['identity'])
    model,manifest=load_bundle(args.bundle,'cuda');market=market_for(args.parent,parent)
    market.inputs=project_inputs(market.inputs,model.recipe);market.recipe=model.recipe
    if support_hash(market.inputs)!=control['support_sha256']:raise ValueError('OPENING_CRITIC_SUPPORT')
    source_sha=file_sha(args.bundle/'model.pt')
    if source_sha!=control['source_sha256']:raise ValueError('OPENING_CRITIC_SOURCE')
    cfg=dict(source_sha256=source_sha,diagnostic=prior['identity'],control=control['identity'],
        seeds=[19,43],updates=4096,batch=128,losses=['absolute','pairwise'],scopes=['heads','critic_representation'],
        actor_updates=0,sources={n:file_sha(ROOT/n) for n in ('tools/run_operational_opening_critic.py',
            'app/operational_opening_critic.py','app/critic_fit_probe.py','app/action_ranking_contrast.py')})
    cfg['identity']=digest(cfg)
    if (args.output/'CONFIG.json').exists() and read(args.output/'CONFIG.json')!=cfg:raise ValueError('OPENING_CRITIC_CONFIG')
    write(args.output/'CONFIG.json',cfg)
    if (args.output/'DONE.json').exists():verify_done(args.output,cfg['identity']);return
    def progress(stage,**kw):
        write(args.output/'PROGRESS.json',dict(stage=stage,seconds=time.monotonic()-started,at=now(),**kw));print(stage,kw,flush=True)
    train_ix=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
    logits,_=emit(model,market.inputs,indices=train_ix);baseline=logits[train_ix].copy()
    policies={s:policy_signals(logits,s) for s in (7,19,43)};signals=policy_signals(logits,None)[0][0]
    alpha=float(model.loss.log_alpha.detach().exp());del model
    if (args.output/'panel/DONE.json').exists():
        verify_done(args.output/'panel',cfg['identity']);records=read(args.output/'panel/RECORDS.json')
    else:
        desired={r['timestamp']:r for r in read(args.diagnostic/'report/RESULTS.json')};records=[];excluded=[]
        risk=read(args.parent/'RISK.json')
        for role,interval in [('fit',['2020-01-01','2023-01-01']),('monitor',['2023-01-01','2024-01-01'])]:
            env=environment(market,risk,parent['design']['horizon'],interval,parent['effective']['gamma']);env.reset()
            terminal=int(env.engine.v[env.engine.last])+3600*10**9
            while not env.finished:
                row=int(env.rows[env.bar]);t=int(market.inputs.row_ts_ns[row])
                if t in desired:
                    start=int(market.inputs.slot_ts_ns[row,0])
                    if env.engine.pos is not None or (role=='monitor' and start<BOUNDARY):
                        excluded.append(dict(timestamp=t,role=role,reason='position_or_context_not_flat_admissible'))
                    else:
                        path=args.output/'panel/cells'/f'{t}.json'
                        if path.exists():
                            record=read(path)
                            if record['identity']!=cfg['identity'] or record['state']!=state_view(env):raise ValueError('OPENING_CRITIC_CELL_REPLAY')
                        else:
                            suffixes=[dict(seed=s,actions=[continuation(env,policies[s][0][0],a,policies[s][1],alpha)
                                for a in (0,1,2)]) for s in (7,19,43)]
                            forbidden=all(suffixes[0]['actions'][a]['first_effect']['entries']==0 for a in (1,2))
                            record=dict(identity=cfg['identity'],timestamp=t,row=row,role=role,terminal=terminal,
                                window_start=start,position=0,state=state_view(env),forbidden=forbidden,
                                panel_kind=desired[t]['kind'],suffixes=suffixes)
                            write(path,record)
                        records.append(record);progress('PANEL',role=role,anchors=len(records))
                env.step(int(signals[row]))
        validate_labels(records);write(args.output/'panel/RECORDS.json',records)
        write(args.output/'panel/EXCLUDED.json',excluded);seal(args.output/'panel',cfg['identity'])
    validate_labels(records)
    features=torch.tensor(market.inputs.windows[[r['row'] for r in records]],device='cuda')
    targets=np.array([np.mean([[a['soft_return'] for a in s['actions']] for s in r['suffixes']],axis=0) for r in records])
    y=torch.tensor(targets,dtype=torch.float32,device='cuda')
    masks={k:np.array([i for i,r in enumerate(records) if r['role']==k]) for k in ('fit','monitor')}
    if min(map(len,masks.values()))<8:raise ValueError('OPENING_CRITIC_TOO_FEW_ANCHORS')
    all_results=[]
    for seed in cfg['seeds']:
        for scope in cfg['scopes']:
            for objective in cfg['losses']:
                folder=args.output/f'{scope}_{objective}_{seed}'
                if (folder/'DONE.json').exists():verify_done(folder,cfg['identity']);all_results.append(read(folder/'RESULT.json'));continue
                configure(seed);model,_=load_bundle(args.bundle,'cuda');model.load(args.control/'common/CALIBRATED.pt')
                set_scope(model,scope);clear_optimizer_memory(model.optimizer);fixed=frozen_state(model)
                schedule=covering_schedule(len(masks['fit']),4096,128,seed);curve=[]
                for step in range(4097):
                    if step:
                        ix=masks['fit'][schedule[step-1]];model.optimizer.zero_grad(set_to_none=True)
                        q=probe_q(model,features[ix]);loss=advantage_loss(q,y[ix]) if objective=='pairwise' else F.smooth_l1_loss(q,y[ix][None].expand_as(q))
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.loss.parameters(),10.,error_if_nonfinite=True);model.optimizer.step()
                    if step in (0,64,256,1024,2048,4096):
                        verify_frozen_state(model,fixed)
                        with torch.no_grad():q=probe_q(model,features).cpu().numpy()
                        metrics={k:fit_metrics(q[:,ix],targets[ix]) for k,ix in masks.items()}
                        curve.append(dict(updates=step,metrics=metrics));write(folder/'CURVE.json',curve)
                        progress('CRITIC_FIT',seed=seed,scope=scope,loss=objective,updates=step)
                after,_=emit(model,market.inputs,indices=train_ix)
                if not np.array_equal(after[train_ix],baseline):raise ValueError('OPENING_CRITIC_ACTOR_CHANGED')
                model.save(folder/'DIAGNOSTIC.pt')
                result=dict(seed=seed,scope=scope,loss=objective,curve=curve,actor_unchanged=True,model_promoted=False)
                write(folder/'RESULT.json',result);seal(folder,cfg['identity']);all_results.append(result);del model
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for r in all_results:
        for ax,role in zip(axes,('fit','monitor')):
            ax.plot([v['updates'] for v in r['curve']],[v['metrics'][role]['minimum']['mean_regret'] for v in r['curve']],
                label=f"{r['scope']} {r['loss']} {r['seed']}");ax.set(title=role,xlabel='critic updates',ylabel='mean decision regret')
    axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(args.output/'CURVES.png');plt.close(fig)
    if file_sha(args.bundle/'model.pt')!=source_sha:raise ValueError('OPENING_CRITIC_SOURCE_CHANGED')
    write(args.output/'RESULT.json',dict(identity=cfg['identity'],arms=all_results,anchors={k:len(v) for k,v in masks.items()},
        actor_updates=0,validation_used=False,model_promoted=False,finished_at=now()))
    seal(args.output,cfg['identity']);progress('COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parent','bundle','diagnostic','control','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise


if __name__=='__main__':main()
