"""TRAIN-only native account-state and mean-head gradient diagnosis; no fitting."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd

from app.decision_attribution import continuation,state_view
from app.flat_opening_diagnostic import select_panel,sac_gradients,compare_pair
from app.frozen_recipe_inputs import file_sha
from app.policy_return_calibration import policy_signals
from app.rl_subset import digest,project_inputs,support_hash
from app.structural_refit import verify_files
from app.three_signal_learning import environment
from tools.attribute_changed_decisions import trace_check
from tools.attribute_ranking_improvement import account_check
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write,now
from tools.run_structural_sac_refit import market_for

ROOT=Path(__file__).resolve().parents[1]
OBJECTIVES=('smooth_l1','mean_var_1','mean_var_05')
SEEDS=(7,19,43)


def prepare(args):
    report=read(args.audit/'report/DONE.json');verify_done(args.audit/'report',report['identity'])
    parent=read(args.parent/'SPEC.json');verify_files(args.parent,parent['files'])
    config=read(args.control/'CONFIG.json');verify_done(args.control,config['identity'])
    manifest=read(args.bundle/'MANIFEST.json');verify_files(args.bundle,manifest['files'])
    # Load the recipe alone: frozen actor/Q surfaces already passed checkpoint replay.
    recipe=read(args.bundle/'RECIPE.json')
    market=market_for(args.parent,parent);market.inputs=project_inputs(market.inputs,recipe);market.recipe=recipe
    if support_hash(market.inputs)!=config['support_sha256']:raise ValueError('OPENING_SUPPORT')
    if file_sha(args.bundle/'model.pt')!=config['source_sha256']:raise ValueError('OPENING_SOURCE')
    traces=read(args.control/'reference/TRAIN/ACTIONS.json')
    calibration=read(args.control/'train_replay/ACTIONS.json')
    if [x['timestamp'] for x in traces]!=[x['timestamp'] for x in calibration]:raise ValueError('OPENING_TRACE_SUPPORT')
    for x in traces:x['year']=pd.Timestamp(x['timestamp'],tz='UTC').year
    events=[x for x in read(args.audit/'report/EVENTS.json') if x['period']=='TRAIN']
    panel=select_panel(traces,events,{x['timestamp'] for x in events})
    timestamp=market.inputs.row_ts_ns;rows=np.searchsorted(timestamp,[x['timestamp'] for x in traces])
    surfaces={};grads={};alphas={};identities={}
    for objective in OBJECTIVES:
        folder=args.audit/objective;cfg=read(folder/'CONFIG.json');verify_done(folder,cfg['identity'])
        if cfg['source_sha256']!=config['source_sha256'] or cfg['support_sha256']!=config['support_sha256']:
            raise ValueError('OPENING_SURFACE_IDENTITY')
        identities[objective]=cfg['identity']
        with np.load(folder/'SURFACES.npz') as f:surfaces[objective]={k:f[k].copy() for k in f.files}
        surface=surfaces[objective];ix=surface['indices'];train_ix=np.searchsorted(ix,rows)
        if not np.array_equal(ix[train_ix],rows):raise ValueError('OPENING_SURFACE_ROWS')
        alpha={float(x['alpha']) for x in events if x['objective']==objective}
        if len(alpha)!=1:raise ValueError('OPENING_ALPHA')
        alphas[objective]=alpha.pop()
        g=sac_gradients(surface['parent'][train_ix],surface['twins'][:,train_ix].min(0),alphas[objective])
        h=surface['hidden'][train_ix].astype(float)
        grads[objective]={k:dict(local=v,weight=v.T@h/len(h),bias=v.mean(0)) for k,v in g.items()}
    first=surfaces[OBJECTIVES[0]]
    for s in surfaces.values():
        for k in ('indices','parent','hidden'):
            if not np.allclose(s[k],first[k],atol=2e-5,rtol=2e-5):raise ValueError('OPENING_SHARED_SURFACE')
    raw=np.full((len(timestamp),3),np.nan);raw[first['indices']]=first['parent']
    policies={seed:policy_signals(raw,seed) for seed in SEEDS}
    if not np.array_equal(policies[7][0][0][rows],[r['signal'] for r in calibration]):
        raise ValueError('OPENING_CALIBRATION_TAPE')
    if not np.array_equal(raw[rows].argmax(1),[r['signal'] for r in traces]):raise ValueError('OPENING_GREEDY_TAPE')
    sources={n:file_sha(ROOT/n) for n in ('tools/diagnose_flat_openings.py','app/flat_opening_diagnostic.py',
        'app/decision_attribution.py','app/three_signal_learning.py','env_plugins/three_signal_native.py')}
    cfg=dict(schema='flat_opening_diagnostic.v1',parent=parent['identity'],audit=report['identity'],
        audit_identities=identities,source_sha256=config['source_sha256'],support_sha256=config['support_sha256'],
        panel=panel,panel_sha256=digest(panel),sources=sources,seeds=list(SEEDS),period='TRAIN',
        optimizer_updates=0,shards=args.shards)
    cfg['identity']=digest(cfg)
    return cfg,parent,market,traces,calibration,events,surfaces,grads,alphas,policies


def run(args):
    started=time.monotonic()
    cfg,parent,market,traces,calibration,events,surfaces,grads,alphas,policies=prepare(args)
    output=args.output/f'shard_{args.shard}'
    if (output/'CONFIG.json').exists() and read(output/'CONFIG.json')!=cfg:raise ValueError('OPENING_CONFIG_CHANGED')
    write(output/'CONFIG.json',cfg)
    if (output/'DONE.json').exists():verify_done(output,cfg['identity']);return
    selected={r['timestamp']:r for i,r in enumerate(cfg['panel']) if i%args.shards==args.shard}
    def progress(stage,**kw):
        write(output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
        print(stage,kw,flush=True)
    progress('REPLAY',anchors=len(selected))
    risk=read(args.parent/'RISK.json');interval=market.recipe['split']['periods']['TRAIN']
    def make():return environment(market,risk,parent['design']['horizon'],interval,parent['effective']['gamma'])
    flat,cal=make(),make();flat.reset();cal.reset();completed=[]
    for i,(old,other) in enumerate(zip(traces,calibration)):
        t=old['timestamp']
        if t in selected:
            cell=output/f'cells/{t}'
            if (cell/'DONE.json').exists():verify_done(cell,cfg['identity']);result=read(cell/'RESULT.json')
            else:
                if flat.engine.pos is not None:raise ValueError('OPENING_REPLAY_NOT_FLAT')
                states={k:state_view(e) for k,e in (('flat',flat),('calibration',cal))}
                suffixes={}
                # Replay reward once; soft return is affine in alpha. All current objectives share alpha.
                if len(set(alphas.values()))!=1:raise ValueError('OPENING_SHARED_ALPHA')
                for label,env in (('flat',flat),('calibration',cal)):
                    suffixes[label]=[dict(seed=s,actions=[continuation(env,policies[s][0][0],a,
                        policies[s][1],alphas[OBJECTIVES[0]]) for a in (0,1,2)]) for s in SEEDS]
                f=np.array([[a['soft_return'] for a in s['actions']] for s in suffixes['flat']])
                c=np.array([[a['soft_return'] for a in s['actions']] for s in suffixes['calibration']])
                pairs=[];witnesses=[e for e in events if e['timestamp']==t and e['isolated_position']==0]
                for e in witnesses:
                    if not np.allclose(f[:,e['new_signal']]-f[:,e['old_signal']],e['soft_gap_samples'],atol=1e-9,rtol=1e-9):
                        raise ValueError('OPENING_PREVIOUS_SUFFIX_PARITY')
                for objective,surface in surfaces.items():
                    row=int(np.searchsorted(market.inputs.row_ts_ns,t));ix=int(np.searchsorted(surface['indices'],row))
                    q=surface['twins'][:,ix].min(0);h=surface['hidden'][ix].astype(float)
                    for action in (0,1,2):
                        if action==old['signal']:continue
                        gradient={}
                        for name,g in grads[objective].items():
                            local=g['local'][i]
                            gradient[name]=dict(local=float(-(local[action]-local[old['signal']])*(h@h+1)),
                                shared=float(-((g['weight'][action]-g['weight'][old['signal']])@h+
                                                g['bias'][action]-g['bias'][old['signal']])))
                        observed=[dict(seed=e['seed'],outcome=e['outcome'],contribution_usd=e['contribution_usd'],
                                       boundary=e['boundary']) for e in witnesses
                                  if e['objective']==objective and e['new_signal']==action]
                        pairs.append(dict(objective=objective,old=old['signal'],new=action,
                            **compare_pair(q,f,c,old['signal'],action),gradient=gradient,observed=observed))
                result=dict(**selected[t],utc=pd.Timestamp(t,tz='UTC').isoformat(),states=states,
                    state_fields_different=[k for k in states['flat'] if states['flat'][k]!=states['calibration'][k]],
                    suffixes=suffixes,pairs=pairs)
                write(cell/'RESULT.json',result);seal(cell,cfg['identity'])
            completed.append(result)
            progress('ANCHORS',completed=len(completed),total=len(selected))
        trace_check(flat,old,old['signal']);trace_check(cal,other,other['signal'])
    account_check(flat,read(args.control/'reference/TRAIN/METRICS.json'))
    account_check(cal,read(args.control/'train_replay/METRICS.json'))
    if file_sha(args.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('OPENING_SOURCE_CHANGED')
    write(output/'RESULTS.json',completed)
    write(output/'SUMMARY.json',dict(identity=cfg['identity'],shard=args.shard,anchors=len(completed),
        suffix_simulations=18*len(completed),both_native_accounts_exact=True,source_unchanged=True,
        seconds=time.monotonic()-started,finished_at=now(),optimizer_updates=0,model_promoted=False))
    seal(output,cfg['identity']);progress('COMPLETE',anchors=len(completed))


def aggregate(args):
    rows=[];summaries=[];first=None
    for i in range(args.shards):
        folder=args.output/f'shard_{i}';cfg=read(folder/'CONFIG.json');verify_done(folder,cfg['identity'])
        if first is None:first=cfg
        elif cfg!=first:raise ValueError('OPENING_SHARD_IDENTITY')
        part=read(folder/'RESULTS.json')
        expected={r['timestamp'] for j,r in enumerate(cfg['panel']) if j%args.shards==i}
        if {r['timestamp'] for r in part}!=expected or len(part)!=len(expected):raise ValueError('OPENING_SHARD_COVERAGE')
        rows.extend(part);summaries.append(read(folder/'SUMMARY.json'))
    rows.sort(key=lambda r:r['timestamp']);stats=[]
    for objective in OBJECTIVES:
        for kind in ('changed','unchanged_control'):
            pairs=[(r,p) for r in rows if r['kind']==kind for p in r['pairs'] if p['objective']==objective]
            stats.append(dict(objective=objective,kind=kind,pairs=len(pairs),
                flat_q_conflicts=sum(p['flat_q_conflict'] for _,p in pairs),
                calibration_q_conflicts=sum(p['calibration_q_conflict'] for _,p in pairs),
                state_order_flips=sum(p['state_order_flip'] for _,p in pairs),
                conflict_only_flat=sum(p['flat_q_conflict'] and not p['calibration_q_conflict'] for _,p in pairs)))
    harmful=[]
    for r in rows:
        for p in r['pairs']:
            for e in p['observed']:
                if e['outcome']=='harmful':harmful.append(dict(timestamp=r['timestamp'],objective=p['objective'],seed=e['seed'],
                    flat_q_conflict=p['flat_q_conflict'],calibration_q_conflict=p['calibration_q_conflict'],
                    state_order_flip=p['state_order_flip'],q_gap=p['q_gap'],gradient=p['gradient'],
                    full_state_equal=not r['state_fields_different'],contribution_usd=e['contribution_usd']))
    summary=dict(identity=first['identity'],panel_sha256=first['panel_sha256'],anchors=len(rows),
        changed_anchors=sum(r['kind']=='changed' for r in rows),suffix_simulations=18*len(rows),
        shards=summaries,pair_stats=stats,harmful_occurrences=harmful,optimizer_updates=0,model_promoted=False,
        limitations=['three stochastic draws are not exact expectations','actual histories differ in multiple account fields',
                      'gradient is deterministic mean-head first-order SAC, not an AdamW trajectory replay',
                      'event and seed counts overlap; TRAIN-only diagnostic panel is not independent evidence'])
    output=args.output/'report';write(output/'SUMMARY.json',summary);write(output/'RESULTS.json',rows)
    lines=['# Flat-opening state and gradient diagnostic','',f"{len(rows)} TRAIN anchors; {18*len(rows)} native suffixes; no training.",'',
           '| Objective | Panel | Pairs | Flat Q conflicts | Calibration Q conflicts | State-order flips |',
           '|---|---|---:|---:|---:|---:|']
    for r in stats:lines.append(f"|{r['objective']}|{r['kind']}|{r['pairs']}|{r['flat_q_conflicts']}|{r['calibration_q_conflicts']}|{r['state_order_flips']}|")
    lines+=['','Full state views, first-command effects and gradient components are retained in RESULTS.json.',*summary['limitations']]
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n');seal(output,first['identity'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('run','aggregate'))
    for n in ('parent','bundle','control','audit'):p.add_argument('--'+n,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=3);a=p.parse_args()
    if not 0<=a.shard<a.shards:p.error('invalid shard')
    if a.command=='aggregate':aggregate(a);return
    if any(getattr(a,k) is None for k in ('parent','bundle','control','audit')):p.error('missing run input')
    folder=a.output/f'shard_{a.shard}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(folder/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
