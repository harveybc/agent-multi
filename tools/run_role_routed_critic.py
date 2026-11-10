"""Paired frozen-encoder role-head fit and native same-direction conflict audit."""
import argparse
from copy import deepcopy
import csv
import fcntl
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict

from agent_plugins.sac_critic_regularization import module_mode
from app.action_ranking_contrast import advantage_loss
from app.critic_fit_probe import classify_anchor, fit_metrics, probe_q
from app.decision_attribution import continuation, state_view
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import digest, support_hash
from app.role_routed_critic import RoleHeads, select_role, matched_schedule, delayed_histories
from app.state_compatibility import compatibility
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_action_ranking_contrast import ensure_config, seal
from tools.run_actor_preservation import collect
from tools.run_critic_fit_probe import train_logits
from tools.run_discrete_curve_probe import configure, now, read, write
from tools.run_stratified_critic import POINTS
from tools.run_structural_sac_refit import authenticate, market_for

ROOT=Path(__file__).resolve().parents[1]
ROLES=('flat','long','short')


def panel(root,market):
    banks=[];records=[];fit_banks=[]
    for p in sorted((root/'quarters').glob('*/RESULT.json')):
        for origin in read(p)['origins']:
            for bank in origin['banks']:
                if bank['compatibility'] is None:continue
                row=bank['row'];window=market.inputs.windows[row]
                if (market.inputs.period[row]!='TRAIN' or not market.inputs.valid[row]
                        or hashlib.sha256(np.ascontiguousarray(window).tobytes()).hexdigest()!=bank['feature_sha256']):
                    raise ValueError('ROLE_PANEL_FEATURE_PARITY')
                expected='fit' if bank['timestamp']<int(np.datetime64('2023-01-01','ns').astype(np.int64)) else 'monitor'
                if bank['role']!=expected:raise ValueError('ROLE_PANEL_SPLIT')
                anchor=origin['anchor']
                if bank['role']=='fit' and anchor['terminal']>=int(np.datetime64('2023-01-01','ns').astype(np.int64)):
                    raise ValueError('ROLE_LABEL_CROSSES_SPLIT')
                ids=[]
                for name,values in bank['outcomes'].items():
                    ids.append(len(records))
                    records.append(dict(row=row,role=bank['role'],state=name,bank=len(banks),
                        targets=[v['reward_return'] for v in values]))
                banks.append(ids)
                if bank['role']=='fit':fit_banks.append(ids)
    if len(banks)!=844:raise ValueError('ROLE_PANEL_COVERAGE')
    return banks,records,fit_banks


def cache_embeddings(model,features):
    net=model.loss.qvalue_network
    embeddings=[];templates=[];errors=[]
    with torch.no_grad(),module_mode(net,False):
        native=torch.cat([probe_q(model,features[i:i+128]) for i in range(0,len(features),128)],dim=1)
        for twin in range(2):
            with model.loss.qvalue_network_params[twin].to_module(net):
                head=net.module.head;captured=[];values=[]
                hook=head.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
                try:
                    for i in range(0,len(features),128):
                        x=features[i:i+128]
                        values.append(net(TensorDict(dict(features=x),[len(x)]))['action_value'].clone())
                    template=deepcopy(head)
                finally:hook.remove()
                z=torch.cat(captured);q=torch.cat(values)
                error=max(float((q-native[twin]).abs().max()),float((template(z)-q).abs().max()))
                if error>1e-6 or not torch.isfinite(z).all():raise ValueError('ROLE_CACHED_EMBEDDING_PARITY')
                embeddings.append(z);templates.append(template);errors.append(error)
    return torch.stack(embeddings),templates[0],errors


def predicted(heads,z,roles,mode):
    out=heads(z)
    return select_role(out,roles) if mode=='routed' else out.mean(2)


def metrics(heads,z,roles,mode,target,records,banks):
    with torch.no_grad():
        q=torch.cat([predicted(heads,z[:,i:i+256],roles[i:i+256],mode)
                     for i in range(0,len(records),256)],dim=1).cpu().numpy()
    result={}
    for role in ('fit','monitor'):
        result[role]={}
        for state in ('all',*ROLES):
            ix=[i for i,r in enumerate(records) if r['role']==role and (state=='all' or r['state']==state)]
            result[role][state]=fit_metrics(q[:,ix],target[ix]) if ix else None
        groups=[b for b in banks if records[b[0]]['role']==role]
        regret=[]
        for b in groups:
            action=q[:,b].min(0).argmax(1);y=target[b]
            regret.append(float((y.max(1)-y[np.arange(len(b)),action]).mean()))
        result[role]['bank_balanced_regret']=float(np.mean(regret))
    return result


def residual_audit(args,spec,market,signals,config,progress):
    root=args.output/'residual';root.mkdir(exist_ok=True)
    if (root/'DONE.json').exists():
        if verify_collection(root)['identity']!=config['identity']:raise ValueError('ROLE_RESIDUAL_IDENTITY')
        return read(root/'RESULT.json')
    start=time.monotonic();all_banks=[];exclusions=0;suffixes=0;negative_errors=[]
    for p in sorted((args.predecessor/'quarters').glob('*/RESULT.json')):
        source=read(p);chosen={o['anchor']['trace_index']:o for o in source['origins']}
        env=environment(market,read(args.source/'RISK.json'),spec['design']['horizon'],source['interval'],spec['effective']['gamma'])
        env.reset();index=0;reference=[];negative_done=False
        while not env.finished:
            row=int(env.rows[env.bar]);action=int(signals[row])
            reference.append(dict(row=row,signal=action,state=state_view(env)))
            if index in chosen:
                o=chosen[index]
                if state_view(env)!=o['initial_state']:raise ValueError('ROLE_RESIDUAL_REFERENCE')
                path=root/p.parent.name/f'{index:05d}.json'
                if path.exists():
                    item=read(path)
                    if item['identity']!=config['identity'] or digest(item['data'])!=item['sha256']:
                        raise ValueError('ROLE_RESIDUAL_CACHE')
                    data=item['data']
                else:
                    data=dict(banks=[],suffixes=0,excluded=0,negative_error=None);visited=[]
                    for lag,states,info in delayed_histories(env):
                        visited.append(lag)
                        for direction in ('long','short'):
                            selected={};seen=set()
                            for key,e in states.items():
                                if not key.startswith(direction+':'):continue
                                state=state_view(e)
                                # Counters/trade history are not sufficient reasons to call
                                # otherwise identical current native positions distinct.
                                signature=digest({k:state[k] for k in ('timestamp','balance','position','entries_day','next_decision')})
                                if signature not in seen:selected[key]=e;seen.add(signature)
                            values={}
                            if len(selected)>=2:
                                values={k:[continuation(e,signals,a) for a in range(3)] for k,e in selected.items()}
                                data['suffixes']+=3*len(selected)
                                for key,vv in values.items():
                                    classify_anchor(dict(anchor=dict(o['anchor'],position=1 if direction=='long' else -1),
                                        results=dict(greedy=vv)))
                                if not negative_done and data['negative_error'] is None:
                                    first=next(iter(selected));repeat=[continuation(selected[first],signals,a) for a in range(3)]
                                    data['suffixes']+=3
                                    data['negative_error']=max(abs(v[k]-w[k]) for v,w in zip(repeat,values[first])
                                        for k in ('reward_return','net_usd','soft_return'))
                                    if data['negative_error']>1e-8:raise ValueError('ROLE_RESIDUAL_NEGATIVE_CONTROL')
                            data['banks'].append(dict(origin=o['anchor']['id'],role=o['anchor']['role'],lag=lag,
                                direction=direction,history=info,states=list(selected),outcomes=values,
                                compatibility=compatibility([[v['reward_return'] for v in vv] for vv in values.values()]) if values else None))
                    data['excluded']=2*(2-len(visited))
                    write(path,dict(identity=config['identity'],data=data,sha256=digest(data)))
                all_banks.extend(data['banks']);exclusions+=data['excluded'];suffixes+=data['suffixes']
                if data['negative_error'] is not None:negative_errors.append(data['negative_error']);negative_done=True
            env.step(action);index+=1
        original=read(args.stratified/'panel'/p.parent.name/'DESIGN.json')
        if (digest(reference)!=original['reference_trace_sha256']
                or abs(env.engine.balance-env.engine.cfg.capital-original['reference_cash'])>1e-8):
            raise ValueError('ROLE_RESIDUAL_TRACE')
        progress('RESIDUAL',quarter=p.parent.name,banks=len(all_banks),suffixes=suffixes)
    groups={}
    for role in ('fit','monitor'):
        for direction in ('long','short'):
            rr=[b for b in all_banks if b['role']==role and b['direction']==direction]
            cc=[b['compatibility'] for b in rr if b['compatibility'] is not None]
            groups[f'{role}:{direction}']=dict(attempted=len(rr),comparable=len(cc),
                conflicts=sum(not c['common_optimal_actions'] for c in cc),
                mean_regret_floor=float(np.mean([c['best_common_regret'] for c in cc])) if cc else None)
    result=dict(groups=groups,banks=all_banks,terminal_exclusions=exclusions,suffixes=suffixes,
        negative_controls=len(negative_errors),negative_control_max_error=max(negative_errors) if negative_errors else None)
    write(root/'RESULT.json',result);seal(root,config['identity'],start);return result


def run(args):
    start=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':
        raise ValueError('ROLE_GPU_IDENTITY')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    prior=verify_collection(args.predecessor);pc=read(args.predecessor/'CONFIG.json')
    sc=verify_collection(args.stratified);cal=verify_collection(args.probe)
    bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    if (pc['source']!=spec['identity'] or pc['stratified']!=sc['identity'] or pc['probe']!=cal['identity']
            or support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']):
        raise ValueError('ROLE_PARENT_CHAIN')
    for name,sha in pc['sources'].items():
        if file_sha(ROOT/name)!=sha:raise ValueError('ROLE_SOURCE_CODE:'+name)
    model,_=load_bundle(bundle,'cuda');model.load(args.probe/'fresh/CALIBRATED.pt')
    frozen={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    train=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
    parent=train_logits(model,market.inputs.windows[train])
    logits,_=emit(model,market.inputs,indices=train);signals=prepared_signals(logits)[0]
    banks,records,fit_banks=panel(args.predecessor,market)
    rows=sorted({r['row'] for r in records});lookup={r:i for i,r in enumerate(rows)}
    features=torch.tensor(market.inputs.windows[rows],device='cuda')
    cached,template,errors=cache_embeddings(model,features)
    z=cached[:,[lookup[r['row']] for r in records]]
    roles=torch.tensor([ROLES.index(r['state']) for r in records],device='cuda')
    targets=np.array([r['targets'] for r in records]);target=torch.tensor(targets,dtype=torch.float32,device='cuda')
    config=dict(predecessor=prior['identity'],source=spec['identity'],probe=cal['identity'],
        source_sha256=file_sha(bundle/'model.pt'),calibrated_sha256=file_sha(args.probe/'fresh/CALIBRATED.pt'),
        seeds=[19,43],updates=4096,batch_size=128,arms=['shared','shared_mean3','routed'],
        learning_rate=model.effective['learning_rate'],weight_decay=model.effective['weight_decay'],
        records_sha256=digest(records),actor_updates=0,encoder_updates=0,model_promoted=False,
        sources={n:file_sha(ROOT/n) for n in ('tools/run_role_routed_critic.py','app/role_routed_critic.py',
            'app/state_compatibility.py','app/decision_attribution.py','app/critic_fit_probe.py','app/action_ranking_contrast.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-start,**kw))
    write(args.output/'PANEL.json',dict(banks=banks,records=records,embedding_max_error=errors))
    residual=residual_audit(args,spec,market,signals,config,progress)
    arms={}
    for seed in config['seeds']:
        schedule=matched_schedule(fit_banks,4096,128,seed)
        for mode in config['arms']:
            lane=args.output/f'r{seed}'/mode;lane.mkdir(parents=True,exist_ok=True)
            if (lane/'DONE.json').exists():
                if verify_collection(lane)['identity']!=config['identity']:raise ValueError('ROLE_ARM_IDENTITY')
                arms[f'{seed}:{mode}']=read(lane/'RESULT.json');continue
            begun=time.monotonic();heads=RoleHeads(template,1 if mode=='shared' else 3,seed).to('cuda')
            initialization=deepcopy(heads.state_dict());torch.save(initialization,lane/'INITIAL.pt')
            init_digest=digest({k:v.detach().cpu().tolist() for k,v in initialization.items()})
            optimizer=torch.optim.AdamW(heads.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            np.save(lane/'SCHEDULE.npy',schedule)
            effective=dict(identity=config['identity'],seed=seed,mode=mode,initialization_sha256=init_digest,
                schedule_sha256=digest(schedule.tolist()),parameters=sum(p.numel() for p in heads.parameters()),
                learning_rate=config['learning_rate'],weight_decay=config['weight_decay'])
            write(lane/'EFFECTIVE.json',effective);points=[]
            for step in range(4097):
                if step:
                    ix=schedule[step-1];optimizer.zero_grad(set_to_none=True)
                    loss=advantage_loss(predicted(heads,z[:,ix],roles[ix],mode),target[ix]);loss.backward()
                    torch.nn.utils.clip_grad_norm_(heads.parameters(),10.,error_if_nonfinite=True);optimizer.step()
                if step in POINTS:
                    value=metrics(heads,z,roles,mode,targets,records,banks)
                    points.append(dict(updates=step,metrics=value));write(lane/'CURVE.json',points)
                    progress('FIT',seed=seed,mode=mode,updates=step)
            torch.save(dict(state=heads.state_dict(),effective=effective),lane/'HEADS.pt')
            reloaded=RoleHeads(template,1 if mode=='shared' else 3,seed).to('cuda')
            reloaded.load_state_dict(torch.load(lane/'HEADS.pt',weights_only=False)['state'])
            with torch.no_grad():
                if not torch.equal(heads(z[:,:32]),reloaded(z[:,:32])):raise ValueError('ROLE_RELOAD_PARITY')
            if not any(not torch.equal(v,initialization[k]) for k,v in heads.state_dict().items()):
                raise ValueError('ROLE_NO_UPDATE')
            record=dict(effective=effective,points=points,reload_exact=True)
            write(lane/'RESULT.json',record);seal(lane,config['identity'],begun);arms[f'{seed}:{mode}']=record
    for seed in config['seeds']:
        a=arms[f'{seed}:routed']['effective'];b=arms[f'{seed}:shared_mean3']['effective']
        if any(a[k]!=b[k] for k in ('initialization_sha256','schedule_sha256','parameters')):
            raise ValueError('ROLE_UNPAIRED_ARMS')
    if (any(not torch.equal(v,frozen[k]) for k,v in model.loss.state_dict().items())
            or not np.array_equal(parent,train_logits(model,market.inputs.windows[train]))
            or file_sha(bundle/'model.pt')!=config['source_sha256']
            or file_sha(args.probe/'fresh/CALIBRATED.pt')!=config['calibrated_sha256']):
        raise ValueError('ROLE_SOURCE_MUTATION')
    summary=dict(identity=config['identity'],arms=arms,embedding_max_error=errors,
        residual={k:v for k,v in residual.items() if k!='banks'},actor_exact=True,source_exact=True,model_promoted=False)
    write(args.output/'RESULT.json',summary);export(args.output,arms)
    seal(args.output,config['identity'],start);progress('COMPLETE')


def export(root,arms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,arm in arms.items():
        for ax,role in zip(axes,('fit','monitor')):
            ax.plot([p['updates'] for p in arm['points']],
                [p['metrics'][role]['bank_balanced_regret'] for p in arm['points']],label=name)
        for point in arm['points']:
            for role in ('fit','monitor'):
                m=point['metrics'][role]
                rows.append(dict(arm=name,updates=point['updates'],role=role,
                    bank_regret=m['bank_balanced_regret'],pair_accuracy=m['all']['minimum']['pairwise_accuracy'],
                    gap_rmse=m['all']['minimum']['gap_rmse']))
    for ax,role in zip(axes,('fit','monitor')):
        ax.set_title(role);ax.set_xlabel('Updates');ax.set_ylabel('Bank-balanced regret');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'CURVES.png',dpi=140);plt.close(fig)
    with (root/'METRICS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('run','collect'))
    for n in ('source','source-runtime','predecessor','probe','stratified'):p.add_argument('--'+n,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-host')
    args=p.parse_args()
    if args.stage=='collect':collect(args);return
    if not all((args.source,args.source_runtime,args.predecessor,args.probe,args.stratified)):p.error('source paths required')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
