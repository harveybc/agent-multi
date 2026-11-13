"""Finite CPU baselines plus reachable same-market state forks; no optimization."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np

from app.action_ranking_contrast import ranking_metrics
from app.critic_identifiability import baseline_predictions, select_origins, forked_state_test
from app.critic_representation_control import state_kind
from app.decision_attribution import state_view
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import environment
from tools.diagnose_critic_coverage_drift import INTERVALS
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now


def run(a):
    started=time.monotonic();configure(316)
    cc=read(a.coverage/'CONFIG.json');verify_done(a.coverage,cc['identity'])
    rc=read(a.representation/'CONFIG.json');verify_done(a.representation,rc['identity'])
    dc=read(a.diagnostic/'CONFIG.json');verify_done(a.diagnostic,dc['identity'])
    if rc['predecessor']!=cc['identity'] or cc['diagnostic']!=dc['identity']:
        raise ValueError('IDENTIFIABILITY_LINEAGE')
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    sha=file_sha(a.bundle/'model.pt')
    if sha!=cc['source_sha256'] or sha!=rc['source_sha256'] or sha!=dc['source_sha256']:
        raise ValueError('IDENTIFIABILITY_SOURCE')
    model,_=load_bundle(a.bundle,'cpu');market=market_for(a.parent,spec,model.recipe)
    if support_hash(market.inputs)!=cc['support'] or model.recipe['list']['n']!=71:
        raise ValueError('IDENTIFIABILITY_SUPPORT')
    partition=read(a.coverage/'PARTITION.json');records=partition['records']
    if digest(records)!=cc['records_sha256']:raise ValueError('IDENTIFIABILITY_RECORDS')
    old_count=len(records)-len(cc['new_fit'])-len(cc['held'])
    held=old_count+np.array(cc['held']);fit=np.array([i for i,r in enumerate(records) if r['role']=='fit'])
    if (old_count,len(fit),len(held))!=(151,146,55):raise ValueError('IDENTIFIABILITY_COUNTS')
    rows=np.array([r['row'] for r in records])
    for r in records:
        if (not market.inputs.valid[r['row']] or market.inputs.period[r['row']]!='TRAIN'
                or int(market.inputs.row_ts_ns[r['row']])!=r['timestamp']):raise ValueError('IDENTIFIABILITY_ROWS')
    selected,counts=select_origins(records,held)
    cfg=dict(schema='critic_identifiability.v1',coverage=cc['identity'],representation=rc['identity'],
        diagnostic=dc['identity'],source_sha256=sha,support=cc['support'],seed=316,per_year=4,
        selected=selected,counts=counts,optimizer_steps=0,validation_read=False,test_read=False,
        historical_mixed_cot=True,code={n:file_sha(Path(__file__).resolve().parents[1]/n) for n in (
            'tools/diagnose_critic_identifiability.py','app/critic_identifiability.py',
            'app/decision_attribution.py','env_plugins/three_signal_native.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('IDENTIFIABILITY_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**values):
        write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**values))
    progress('BASELINES')
    verify_done(a.diagnostic/'cache',dc['identity'])
    with np.load(a.diagnostic/'cache/SIGNALS.npz',allow_pickle=False) as data:signals=data['source'].copy()
    pred,distances=baseline_predictions(market.inputs.windows[rows],records,fit,held)
    truth=np.array([[o['reward_return'] for o in records[i]['outcomes']] for i in held])
    groups={'all':np.arange(len(held))}
    for kind in sorted({state_kind(records[i]) for i in held}):
        groups[kind]=np.array([j for j,i in enumerate(held) if state_kind(records[i])==kind])
    for year in range(2020,2024):groups[str(year)]=np.array([j for j,i in enumerate(held) if records[i]['year']==year])
    metrics={}
    for name,q in pred.items():
        metrics[name]={}
        for kind,ix in groups.items():
            use=ix[np.isfinite(q[ix]).all(1)]
            metrics[name][kind]=dict(available=len(use),requested=len(ix),
                metrics=ranking_metrics(q[use],truth[use]) if len(use) else None)
    actions={'source_actor':signals[rows[held]]}
    actions.update({name:np.full(len(held),action,dtype=int) for action,name in enumerate(('always_nop','always_buy','always_sell'))})
    action_metrics={}
    for name,choice in actions.items():
        action_metrics[name]={}
        for kind,ix in groups.items():
            if not len(ix):continue
            m=ranking_metrics(np.eye(3)[choice[ix]],truth[ix])
            action_metrics[name][kind]={k:m[k] for k in ('n','mean_regret','maximum_regret','q_greedy_counts')}
    # Exact input duplicates and support overlap are diagnostic, not independent observations.
    windows=market.inputs.windows[rows]
    overlaps=[]
    for i in held:
        overlaps.append(sum(records[j]['timestamp']>=records[i]['window_start']
            and records[j]['window_start']<=records[i]['timestamp'] for j in fit))
    baseline=dict(metrics=metrics,actions=action_metrics,distances=distances,
        held_rows=rows[held].tolist(),fit_rows=rows[fit].tolist(),
        held_window_overlaps_fit=overlaps,held_with_overlap=sum(n>0 for n in overlaps),
        exact_duplicate_held_fit=sum(any(np.array_equal(windows[i],windows[j]) for j in fit) for i in held))
    write(a.output/'BASELINES.json',baseline)
    wanted={records[i]['timestamp']:records[i] for i in selected};results=[];durations=[]
    panel=read(a.diagnostic/'design/PANEL.json')
    risk=read(a.parent/'RISK.json')
    for segment,interval in INTERVALS:
        env=environment(market,risk,spec['design']['horizon'],interval,model.effective['gamma']);env.reset()
        while not env.finished:
            row=int(env.rows[env.bar]);stamp=int(market.inputs.row_ts_ns[row])
            if stamp in wanted:
                anchor=wanted[stamp];out=a.output/'origins'/str(row)
                if digest(state_view(env))!=anchor['state_sha256']:raise ValueError('IDENTIFIABILITY_STATE')
                if (out/'DONE.json').exists():
                    verify_done(out,cfg['identity']);result=read(out/'RESULT.json')
                    if result['anchor']!=anchor:raise ValueError('IDENTIFIABILITY_ANCHOR')
                else:
                    begin=time.monotonic()
                    progress('NATIVE_FORKS',completed=len(results),total=len(selected),row=row,
                        eta_seconds=float(np.median(durations)*(len(selected)-len(results))) if durations else None)
                    value=forked_state_test(env,signals,repeat=not results)
                    result=dict(identity=cfg['identity'],anchor=anchor,**value)
                    write(out/'RESULT.json',result);seal(out,cfg['identity']);durations.append(time.monotonic()-begin)
                results.append(result)
            env.step(int(signals[row]))
        if abs(env.engine.balance-env.engine.cfg.capital-panel['reference_cash'][segment])>1e-6:
            raise ValueError('IDENTIFIABILITY_SOURCE_ACCOUNT')
    if len(results)!=len(selected):raise ValueError('IDENTIFIABILITY_MISSING_ORIGINS')
    if file_sha(a.bundle/'model.pt')!=sha:raise ValueError('IDENTIFIABILITY_SOURCE_MUTATED')
    complete=[r for r in results if r['status']=='COMPLETE']
    conflicts=[r for r in complete if not r['ambiguity']['common_optimal_actions']]
    summary=dict(identity=cfg['identity'],baseline=baseline,origins=results,
        completed_origins=len(complete),unavailable_origins=len(results)-len(complete),
        conflicting_optimal_sets=len(conflicts),
        conflicts_all_three_positions=sum(r['distinct_positions']==[-1,0,1] for r in conflicts),
        mean_minimum_shared_regret=float(np.mean([r['ambiguity']['minimum_shared_mean_regret'] for r in complete])) if complete else None,
        optimizer_steps=0,model_promoted=False,actor_changed=False,validation_read=False,test_read=False,
        stl_included=False,historical_mixed_cot=True,seconds=time.monotonic()-started)
    write(a.output/'RESULT.json',summary)
    lines=['# Critic identifiability diagnostic','',
        'Same55 previously inspected TRAIN anchors. No optimizer or actor modification.','',
        '| FIT-only baseline | Pair accuracy | Mean regret | RMSE |','| --- | ---: | ---: | ---: |']
    for name,value in metrics.items():
        m=value['all']['metrics']
        if m:lines.append(f"| {name} | {m['pairwise_accuracy']} | {m['mean_regret']:.6f} | {m['rmse']:.6f} |")
    lines+=['',f'Native origins complete: {len(complete)}/{len(results)}.',
        f'No common optimal action across reachable states: {len(conflicts)}.',
        f'Of these, actual flat/long/short all present: {summary["conflicts_all_three_positions"]}.',
        f'Mean minimum shared-action regret: {summary["mean_minimum_shared_regret"]}.',
        'This is a finite counterfactual diagnostic, not market-wide performance or a trading model release.']
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    ax.bar(range(len(complete)),[r['ambiguity']['minimum_shared_mean_regret'] for r in complete])
    ax.set(xlabel='Native origin',ylabel='Minimum shared-action mean regret',title='Same market, reachable account states')
    fig.tight_layout();fig.savefig(a.output/'STATE_AMBIGUITY.png',dpi=140);plt.close(fig)
    seal(a.output,cfg['identity']);progress('COMPLETE',completed=len(results),total=len(selected))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parent','bundle','coverage','representation','diagnostic','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
