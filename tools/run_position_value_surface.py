"""Frozen-encoder continuation-value fields with external barrier queries."""
import argparse
from copy import deepcopy
import csv
import fcntl
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from app.decision_attribution import continuation, fork_env, state_view
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.position_value_surface import (NODES,geometry,fit_axes,query_weights,GapSurfaceHeads,
    evaluate_surface,oracle_fit,decision_metrics)
from app.rl_subset import digest,support_hash
from app.role_routed_critic import delayed_histories,matched_schedule,same_serialized_state
from app.structural_refit import load_bundle
from app.three_signal_learning import emit,environment
from env_plugins.discrete_curve_native import HOUR
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_action_ranking_contrast import ensure_config,seal
from tools.run_actor_preservation import collect
from tools.run_critic_fit_probe import train_logits
from tools.run_discrete_curve_probe import configure,now,read,write
from tools.run_role_routed_critic import cache_embeddings
from tools.run_stratified_critic import POINTS
from tools.run_structural_sac_refit import authenticate,market_for

ROOT=Path(__file__).resolve().parents[1]
BOUNDARY=int(np.datetime64('2023-01-01','ns').astype(np.int64))


def original_position_outcome(snapshot,signals,action):
    before=state_view(snapshot);e=fork_env(snapshot);original=e.engine.pos
    if original is None:raise ValueError('SURFACE_LOCAL_POSITION')
    count=len(e.engine.trades);first=True
    while not e.finished and e.engine.pos is not None:
        e.step(int(action) if first else int(signals[int(e.rows[e.bar])]))
        first=False
    if len(e.engine.trades)!=count+1:raise ValueError('SURFACE_LOCAL_TRADE_COUNT')
    trade=e.engine.trades[-1]
    if trade['entry_bar_abs']!=original.entry_bar or trade['direction']!=original.direction:
        raise ValueError('SURFACE_LOCAL_WRONG_POSITION')
    if state_view(snapshot)!=before:raise ValueError('SURFACE_LOCAL_PARENT_CHANGED')
    return dict(net_usd=trade['net_usd'],exit_reason=trade['close_reason'],exit_bar=trade['exit_bar_abs'])


def build_panel(args,spec,market,signals,config,progress):
    root=args.output/'panel';root.mkdir(exist_ok=True)
    if (root/'DONE.json').exists():
        if verify_collection(root)['identity']!=config['identity']:raise ValueError('SURFACE_PANEL_IDENTITY')
        return read(root/'RESULT.json')
    started=time.monotonic();archives=read(args.role_root/'residual/RESULT.json')['banks']
    lookup={(b['origin'],b['lag'],b['direction']):b for b in archives if b['compatibility'] is not None}
    quarters=[]
    for p in sorted((args.compatibility/'quarters').glob('*/RESULT.json')):
        result_path=root/(p.parent.name+'.json')
        if result_path.exists():
            stored=read(result_path)
            if stored['identity']!=config['identity'] or digest(stored['data'])!=stored['sha256']:
                raise ValueError('SURFACE_PANEL_CACHE')
            quarters.append(stored['data']);continue
        prior=read(p);chosen={o['anchor']['trace_index']:o for o in prior['origins']}
        env=environment(market,read(args.source/'RISK.json'),spec['design']['horizon'],prior['interval'],spec['effective']['gamma'])
        env.reset();index=0;trace=[];records=[];banks=[];repeats=[]
        while not env.finished:
            row=int(env.rows[env.bar]);action=int(signals[row]);trace.append(dict(row=row,signal=action,state=state_view(env)))
            if index in chosen:
                origin=chosen[index];anchor=origin['anchor']
                if not same_serialized_state(state_view(env),origin['initial_state']):raise ValueError('SURFACE_ORIGIN')
                for lag,states,info in delayed_histories(env):
                    for direction in ('long','short'):
                        bank=lookup.get((anchor['id'],lag,direction))
                        if bank is None:continue
                        if not same_serialized_state(info,bank['history']):raise ValueError('SURFACE_NATIVE_HISTORY')
                        ids=[]
                        for key in bank['states']:
                            snapshot=states[key];e=snapshot.engine;b=snapshot.bar;pos=e.pos
                            r=int(snapshot.rows[b]);ts=state_view(snapshot)['timestamp'];window=market.inputs.windows[r]
                            if (market.inputs.period[r]!='TRAIN' or not market.inputs.valid[r]
                                    or not np.array_equal(window,snapshot.last_observation['features'])
                                    or anchor['role']!=('fit' if ts<BOUNDARY else 'monitor')
                                    or (anchor['role']=='fit' and anchor['terminal']>=BOUNDARY)):
                                raise ValueError('SURFACE_TEMPORAL_INPUT')
                            liquidation=e.quote_price(b,e.c[b],-pos.direction,int(e.v[b])+HOUR)
                            coords=geometry(pos.direction,liquidation,pos.sl,pos.tp,e.decision_sigma(b))
                            values=bank['outcomes'][key];close_action=2 if pos.direction==1 else 1
                            gap=values[close_action]['reward_return']-values[0]['reward_return']
                            local_close=original_position_outcome(snapshot,signals,close_action)
                            local_hold=original_position_outcome(snapshot,signals,0)
                            if not repeats:
                                actual=[continuation(snapshot,signals,a) for a in range(3)]
                                error=max(abs(v[k]-w[k]) for v,w in zip(actual,values) for k in ('reward_return','net_usd','soft_return'))
                                if error>1e-8:raise ValueError('SURFACE_NATIVE_LABEL_PARITY')
                                repeats.append(error)
                            ids.append(len(records));records.append(dict(row=r,timestamp=ts,role=anchor['role'],
                                direction=direction,origin=anchor['id'],lag=lag,state=key,coords=coords.tolist(),
                                gap=float(gap),native_values=[v['reward_return'] for v in values],
                                local_delta=local_close['net_usd']-local_hold['net_usd'],
                                local_close=local_close,local_hold=local_hold,
                                effective_close=values[close_action]['first_effect']['count_changes'].get('close_AGENT_CLOSE',0)>0))
                        banks.append(ids)
            env.step(action);index+=1
        design=read(args.stratified/'panel'/p.parent.name/'DESIGN.json')
        if digest(trace)!=design['reference_trace_sha256'] or abs(env.engine.balance-env.engine.cfg.capital-design['reference_cash'])>1e-8:
            raise ValueError('SURFACE_REFERENCE_TRACE')
        data=dict(records=records,banks=banks,repeats=repeats)
        write(result_path,dict(identity=config['identity'],data=data,sha256=digest(data)));quarters.append(data)
        progress('PANEL',quarter=p.parent.name,completed=len(quarters),total=16)
    all_records=[];all_banks=[]
    for q in quarters:
        offset=len(all_records);all_records.extend(q['records']);all_banks.extend([[offset+i for i in b] for b in q['banks']])
    if len(all_banks)!=866:raise ValueError('SURFACE_BANK_COVERAGE')
    result=dict(records=all_records,banks=all_banks,repeats=[v for q in quarters for v in q['repeats']])
    write(root/'RESULT.json',result);seal(root,config['identity'],started);return result


def describe(prediction,records,banks,outside):
    target=np.array([r['gap'] for r in records]);local=np.array([r['local_delta'] for r in records])
    effective=np.array([r['effective_close'] for r in records]);results={}
    for role in ('fit','monitor'):
        results[role]={}
        for group in ('all','long','short','inside','outside'):
            ix=np.array([i for i,r in enumerate(records) if r['role']==role and
                (group=='all' or r['direction']==group or group=='inside' and not outside[i] or group=='outside' and outside[i])])
            if not len(ix):results[role][group]=None;continue
            index={int(v):j for j,v in enumerate(ix)}
            selected=[[index[i] for i in b if i in index] for b in banks];selected=[b for b in selected if b]
            results[role][group]=decision_metrics(prediction[ix],target[ix],local[ix],effective[ix],selected)
    return results


def oracle_audit(records,banks,weights):
    result={};target=np.array([r['gap'] for r in records])
    for mode,w in weights.items():
        rows=[]
        for bank in banks:
            r=records[bank[0]];v=oracle_fit(w[bank],target[bank],device='cuda')
            rows.append(dict(role=r['role'],origin=r['origin'],lag=r['lag'],direction=r['direction'],**v))
        result[mode]=dict(rows=rows,groups={role:dict(banks=sum(r['role']==role for r in rows),
            banks_with_action_errors=sum(r['role']==role and r['wrong_actions']>0 for r in rows),
            float32_banks_with_action_errors=sum(r['role']==role and r['float32_wrong_actions']>0 for r in rows),
            max_error=max(r['max_error'] for r in rows if r['role']==role),
            max_node_magnitude=max(r['max_node_magnitude'] for r in rows if r['role']==role)) for role in ('fit','monitor')})
    return result


def predictions(heads,z,directions,weights):
    with torch.no_grad():
        return torch.cat([evaluate_surface(heads(z[:,i:i+256]),directions[i:i+256],weights[i:i+256])
                          for i in range(0,z.shape[1],256)],dim=1).cpu().numpy()


def run(args):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':raise ValueError('SURFACE_GPU')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    previous=verify_collection(args.role_root);pc=read(args.role_root/'CONFIG.json')
    comp=verify_collection(args.compatibility);strat=verify_collection(args.stratified);probe=verify_collection(args.probe)
    cc=read(args.compatibility/'CONFIG.json');bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    if (pc['source']!=spec['identity'] or pc['predecessor']!=comp['identity'] or pc['probe']!=probe['identity']
            or cc['stratified']!=strat['identity'] or pc['source_sha256']!=file_sha(bundle/'model.pt')
            or pc['calibrated_sha256']!=file_sha(args.probe/'fresh/CALIBRATED.pt')
            or support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']):raise ValueError('SURFACE_PARENT_CHAIN')
    for n,sha in pc['sources'].items():
        if file_sha(ROOT/n)!=sha:raise ValueError('SURFACE_SOURCE_CODE:'+n)
    config=dict(source=spec['identity'],predecessor=previous['identity'],probe=probe['identity'],
        source_sha256=pc['source_sha256'],calibrated_sha256=pc['calibrated_sha256'],
        arms=list(NODES),seeds=[19,43],updates=4096,batch_size=128,learning_rate=pc['learning_rate'],
        weight_decay=pc['weight_decay'],actor_updates=0,encoder_updates=0,objective='SmoothL1_gap_mean_twins',
        axes='directional_fit_minmax_9',continuation='archived_frozen_actor_to_quarter_end',
        sources={n:file_sha(ROOT/n) for n in ('tools/run_position_value_surface.py','app/position_value_surface.py',
            'tools/run_role_routed_critic.py','app/role_routed_critic.py','app/decision_attribution.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    model,_=load_bundle(bundle,'cuda');model.load(args.probe/'fresh/CALIBRATED.pt')
    initial={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    train=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
    reference=train_logits(model,market.inputs.windows[train])
    logits,_=emit(model,market.inputs,indices=train);signals=prepared_signals(logits)[0]
    panel=build_panel(args,spec,market,signals,config,progress);records=panel['records'];banks=panel['banks']
    coords=np.array([r['coords'] for r in records]);dirs=np.array([0 if r['direction']=='long' else 1 for r in records])
    fit=np.array([r['role']=='fit' for r in records]);axes=fit_axes(coords,dirs,fit);write(args.output/'AXES.json',axes)
    weights={};outside={}
    for mode in NODES:weights[mode],outside[mode]=query_weights(coords,dirs,axes,mode)
    oracle=oracle_audit(records,banks,weights);write(args.output/'ORACLE_INTERFACE_ONLY.json',oracle)
    rows=sorted({r['row'] for r in records});lookup={r:i for i,r in enumerate(rows)}
    features=torch.tensor(market.inputs.windows[rows],device='cuda');embeddings,_,errors=cache_embeddings(model,features)
    z=embeddings[:,[lookup[r['row']] for r in records]]
    directions=torch.tensor(dirs,device='cuda');target=torch.tensor([r['gap'] for r in records],device='cuda')
    fit_banks=[b for b in banks if records[b[0]]['role']=='fit'];arms={}
    for seed in config['seeds']:
        schedule=matched_schedule(fit_banks,4096,128,seed)
        for mode,nodes in NODES.items():
            lane=args.output/f'r{seed}'/mode;lane.mkdir(parents=True,exist_ok=True)
            if (lane/'DONE.json').exists():
                if verify_collection(lane)['identity']!=config['identity']:raise ValueError('SURFACE_ARM_IDENTITY')
                arms[f'{seed}:{mode}']=read(lane/'RESULT.json');continue
            start=time.monotonic();heads=GapSurfaceHeads(z.shape[-1],nodes,seed,device='cuda')
            saved=deepcopy(heads.state_dict());torch.save(saved,lane/'INITIAL.pt');np.save(lane/'SCHEDULE.npy',schedule)
            w=torch.tensor(weights[mode],dtype=torch.float32,device='cuda')
            effective=dict(identity=config['identity'],seed=seed,mode=mode,nodes=nodes,axes_sha256=digest(axes),
                initialization_sha256=digest({k:v.cpu().tolist() for k,v in saved.items()}),
                schedule_sha256=digest(schedule.tolist()),parameters=sum(p.numel() for p in heads.parameters()))
            write(lane/'EFFECTIVE.json',effective)
            opt=torch.optim.AdamW(heads.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']);points=[]
            for step in range(4097):
                if step:
                    ix=schedule[step-1];opt.zero_grad(set_to_none=True)
                    gap=evaluate_surface(heads(z[:,ix]),directions[ix],w[ix])
                    loss=F.smooth_l1_loss(gap,target[ix][None].expand_as(gap))
                    if not torch.isfinite(loss):raise ValueError('SURFACE_NONFINITE_LOSS')
                    loss.backward();torch.nn.utils.clip_grad_norm_(heads.parameters(),10.,error_if_nonfinite=True);opt.step()
                if step in POINTS:
                    pred=predictions(heads,z,directions,w)
                    points.append(dict(updates=step,metrics=describe(pred.mean(0),records,banks,outside[mode])))
                    write(lane/'CURVE.json',points);progress('FIT',seed=seed,mode=mode,updates=step)
            np.save(lane/'PREDICTIONS.npy',pred)
            torch.save(dict(state=heads.state_dict(),effective=effective),lane/'HEADS.pt')
            reloaded=GapSurfaceHeads(z.shape[-1],nodes,seed,device='cuda')
            reloaded.load_state_dict(torch.load(lane/'HEADS.pt',weights_only=False)['state'])
            if not np.array_equal(pred,predictions(reloaded,z,directions,w)):raise ValueError('SURFACE_RELOAD')
            if not any(not torch.equal(v,saved[k]) for k,v in heads.state_dict().items()):raise ValueError('SURFACE_NO_UPDATE')
            record=dict(effective=effective,points=points,reload_exact=True,
                twins=[describe(v,records,banks,outside[mode]) for v in pred],
                twin_decision_disagreements=int(((pred[0]>1e-8)!=(pred[1]>1e-8)).sum()))
            write(lane/'RESULT.json',record);seal(lane,config['identity'],start);arms[f'{seed}:{mode}']=record
    for seed in config['seeds']:
        a=arms[f'{seed}:surface81']['effective'];b=arms[f'{seed}:mean81']['effective']
        if any(a[k]!=b[k] for k in ('initialization_sha256','schedule_sha256','parameters')):raise ValueError('SURFACE_UNPAIRED')
    if (any(not torch.equal(v,initial[k]) for k,v in model.loss.state_dict().items())
            or not np.array_equal(reference,train_logits(model,market.inputs.windows[train]))
            or file_sha(bundle/'model.pt')!=config['source_sha256']
            or file_sha(args.probe/'fresh/CALIBRATED.pt')!=config['calibrated_sha256']):raise ValueError('SURFACE_SOURCE_CHANGED')
    witnesses=[]
    for bank in banks:
        r=records[bank[0]]
        if (r['origin'],r['lag'],r['direction'])==('2023Q3:71',4,'short'):
            witnesses.append(dict(records=[records[i] for i in bank],
                learned={name:np.load(args.output/f'r{arm["effective"]["seed"]}'/arm['effective']['mode']/'PREDICTIONS.npy')[:,bank].tolist()
                         for name,arm in arms.items()},
                oracle={mode:oracle_fit(w[bank],[records[i]['gap'] for i in bank],device='cuda') for mode,w in weights.items()}))
    if len(witnesses)!=1:raise ValueError('SURFACE_WITNESS_MISSING')
    write(args.output/'WITNESS.json',witnesses[0])
    controls={name:describe(np.full(len(records),v),records,banks,np.zeros(len(records),bool))
              for name,v in (('always_hold',0.),('always_close',1.))}
    result=dict(identity=config['identity'],arms=arms,controls=controls,actor_exact=True,source_exact=True,
        embedding_max_error=errors,oracle_summary={m:o['groups'] for m,o in oracle.items()},
        records=len(records),banks=len(banks),negative_controls=len(panel['repeats']),
        negative_control_max_error=max(panel['repeats']),model_promoted=False)
    write(args.output/'RESULT.json',result);export(args.output,arms);seal(args.output,config['identity'],started);progress('COMPLETE')


def export(root,arms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,arm in arms.items():
        for ax,role in zip(axes,('fit','monitor')):
            ax.plot([p['updates'] for p in arm['points']],[p['metrics'][role]['all']['bank_regret'] for p in arm['points']],label=name)
        for point in arm['points']:
            for role in ('fit','monitor'):
                for group,m in point['metrics'][role].items():
                    if m is not None:rows.append(dict(arm=name,updates=point['updates'],role=role,group=group,**m))
    for ax,role in zip(axes,('fit','monitor')):
        ax.set_title(role);ax.set_xlabel('Updates');ax.set_ylabel('Bank-balanced decision regret');ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(root/'CURVES.png',dpi=140);plt.close(fig)
    with (root/'METRICS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('run','collect'))
    for name in ('source','source-runtime','role-root','compatibility','stratified','probe'):p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-host')
    args=p.parse_args()
    if args.stage=='collect':collect(args);return
    if not all((args.source,args.source_runtime,args.role_root,args.compatibility,args.stratified,args.probe)):p.error('source paths required')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
