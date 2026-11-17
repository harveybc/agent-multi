"""Resumable CPU panel, paired critic calibration and gated actor trust probes."""
import argparse
from collections import Counter
import csv
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from app.action_ranking_contrast import differentiable_q, target_loss, ranking_metrics
from app.actor_learning_diagnostics import initialize_critic_heads, clear_optimizer_memory
from app.critic_fit_probe import set_scope
from app.critic_identifiability import forked_state_test
from app.critic_refinement import sync_targets
from app.decision_attribution import continuation, state_view
from app.frozen_critic_repair import choose, passed, constrained_step, mean_kl
from app.frozen_recipe_inputs import file_sha
from app.policy_return_calibration import policy_signals
from app.rl_subset import digest, support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import emit, environment, evaluate
from app.training_forensics import tree_digest
from tools.diagnose_warmstart_drift import frozen_state, verify_frozen_state, set_arm
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_long_sac_continuation import SOURCE, SUPPORT
from tools.run_sac_training_reliability import code_identity
from tools.run_stratified_critic import state_kind


INTERVALS = (('fit', ['2020-01-01', '2023-01-01']),
             ('monitor', ['2023-01-01', '2024-01-01']))
TAPES = (101, 202, 303, 404)
POINTS = (0, 64, 256, 1024, 2048, 4096)


def cached(path, identity):
    if not path.exists():
        return None
    d = read(path)
    if d['identity'] != identity or d['sha256'] != digest(d['value']):
        raise ValueError('REPAIR_CACHE_IDENTITY')
    return d['value']


def save(path, identity, value):
    write(path, dict(identity=identity, sha256=digest(value), value=value))


def report(root, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for name,result in results.items():
        target=result['target']
        for role in ('fit','monitor'):
            for point in result['points']:
                m=point['metrics'][target][role]['all']
                rows.append(dict(arm=name,role=role,updates=point['updates'],**m))
            for ax,metric in zip(axes[0 if role=='fit' else 1],('rmse','mean_regret')):
                ax.plot([p['updates'] for p in result['points']],
                    [p['metrics'][target][role]['all'][metric] for p in result['points']],label=name)
                ax.set(title=f'{role}: {metric}',xlabel='Critic updates')
                ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=6);fig.tight_layout()
    fig.savefig(root/'CRITIC_FIT_MONITOR.png',dpi=140);plt.close(fig)
    with (root/'CALIBRATION.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(a):
    configure(369)
    if torch.cuda.is_available():
        raise ValueError('REPAIR_FORCE_CPU')
    start = time.monotonic()
    parent = read(a.parent/'SPEC.json')
    verify_files(a.parent, {k:v for k,v in parent['files'].items()
        if k.startswith(('data/', 'engine/')) or k in ('SUPPORT_RECIPE.json', 'RISK.json')})
    if file_sha(a.bundle/'model.pt') != SOURCE:
        raise ValueError('REPAIR_SOURCE')
    source, _ = load_bundle(a.bundle, 'cpu')
    original = tree_digest(source.loss.state_dict())
    market = market_for(a.parent, parent, source.recipe)
    if support_hash(market.inputs) != SUPPORT or source.recipe['list']['n'] != 71:
        raise ValueError('REPAIR_SUPPORT')
    cfg = dict(schema='frozen_critic_repair369.v1', source=SOURCE, support=SUPPORT,
        bundle_manifest=file_sha(a.bundle/'MANIFEST.json'), parent=file_sha(a.parent/'SPEC.json'),
        code=code_identity(), quota=4, tapes=list(TAPES), seeds=[7,19], updates=4096,
        batch_size=128, calibration_lr=source.effective['learning_rate'], actor_lr=6e-6,
        actor_attempts=32, kl_limit=1e-3, intervals=[[k,v] for k,v in INTERVALS],
        validation_inference=False, test_read=False, promoted=False)
    identity = digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json') != cfg:
        raise ValueError('REPAIR_RESUME_CONFIGURATION')
    write(a.output/'CONFIG.json', cfg)
    if cached(a.output/'DONE.json', identity) is not None:
        return

    def progress(stage, **kw):
        write(a.output/'PROGRESS.json', dict(stage=stage, at=now(), seconds=time.monotonic()-start, **kw))

    progress('TRAIN_INFERENCE')
    ix = np.flatnonzero(market.inputs.valid & (market.inputs.period == 'TRAIN'))
    logits, _ = emit(source, market.inputs, indices=ix)
    (prepared, probabilities) = policy_signals(logits, None)
    signals = prepared[0]
    tapes = {s:policy_signals(logits, s)[0][0] for s in TAPES}
    alpha = float(source.loss.log_alpha.detach().exp())
    risk = read(a.parent/'RISK.json')

    def make(interval):
        env = environment(market, risk, parent['design']['horizon'], interval, source.effective['gamma'])
        env.reset()
        return env

    panel = cached(a.output/'PANEL.json', identity)
    if panel is None:
        progress('FREEZE_PANEL')
        trace, cash = [], {}
        for role, interval in INTERVALS:
            env = make(interval)
            while not env.finished:
                row = int(env.rows[env.bar]);stamp = int(market.inputs.row_ts_ns[row])
                if market.inputs.period[row] != 'TRAIN':
                    raise ValueError('REPAIR_NONTRAIN')
                trace.append(dict(row=row, timestamp=stamp, year=pd.Timestamp(stamp,tz='UTC').year,
                    kind=state_kind(env), role=role, segment=role,
                    window_start=int(market.inputs.slot_ts_ns[row,0]), state_sha256=digest(state_view(env))))
                env.step(int(signals[row]))
            cash[role] = float(env.engine.balance-env.engine.cfg.capital)
        selected = choose(trace)
        if {r['year'] for r in selected} != set(range(2020,2024)):
            raise ValueError('REPAIR_YEAR_COVERAGE')
        counts = Counter(f"{r['year']}:{r['kind']}" for r in trace)
        panel = dict(anchors=selected, available=dict(counts), reference_cash=cash, trace_sha256=digest(trace))
        save(a.output/'PANEL.json', identity, panel)
    selected = panel['anchors'];wanted = {r['row']:r for r in selected}
    records, forks, durations = [], [], []
    fork_rows = {next(r['row'] for r in selected if r['year']==y and r['kind']=='flat_both_allowed')
                 for y in range(2020,2024) if any(r['year']==y and r['kind']=='flat_both_allowed' for r in selected)}
    for role, interval in INTERVALS:
        env = make(interval)
        while not env.finished:
            row = int(env.rows[env.bar])
            if row in wanted:
                anchor = wanted[row]
                if digest(state_view(env)) != anchor['state_sha256']:
                    raise ValueError('REPAIR_REFERENCE_STATE')
                path = a.output/'rows'/f'{row}.json'
                record = cached(path, identity)
                if record is None:
                    begin = time.monotonic()
                    progress('REAL_CONTINUATIONS', completed=len(records), total=len(selected), row=row,
                        eta_seconds=float(np.median(durations)*(len(selected)-len(records))) if durations else None)
                    greedy = [continuation(env,signals,j) for j in range(3)]
                    stochastic = {str(s):[continuation(env,tapes[s],j,probabilities,alpha) for j in range(3)] for s in TAPES}
                    if abs(greedy[int(signals[row])]['net_usd']-panel['reference_cash'][role]) > 1e-6:
                        raise ValueError('REPAIR_REFERENCE_ACCOUNT')
                    if not records and greedy != [continuation(env,signals,j) for j in range(3)]:
                        raise ValueError('REPAIR_REPEAT')
                    record = dict(anchor=anchor, greedy=greedy, stochastic=stochastic)
                    save(path, identity, record);durations.append(time.monotonic()-begin)
                if record['anchor'] != anchor:
                    raise ValueError('REPAIR_CACHED_ANCHOR')
                records.append(record)
                if row in fork_rows:
                    path = a.output/'forks'/f'{row}.json';value = cached(path,identity)
                    if value is None:
                        value = forked_state_test(env,signals,repeat=not forks)
                        save(path,identity,value)
                    forks.append(value)
            env.step(int(signals[row]))
        if abs(float(env.engine.balance-env.engine.cfg.capital)-panel['reference_cash'][role]) > 1e-6:
            raise ValueError('REPAIR_TRACE_CASH')
    if [r['anchor'] for r in records] != selected:
        raise ValueError('REPAIR_MISSING_ANCHORS')
    rows = np.array([r['row'] for r in selected])
    x = torch.as_tensor(market.inputs.windows[rows].copy())
    fit = np.array([i for i,r in enumerate(selected) if r['role']=='fit'])
    held = np.array([i for i,r in enumerate(selected) if r['role']=='monitor'])
    greedy = np.array([[o['reward_return'] for o in r['greedy']] for r in records])
    soft = np.array([[[o['soft_return'] for o in r['stochastic'][str(s)]] for s in TAPES] for r in records])
    targets = dict(greedy=greedy, soft=soft.mean(1))
    if not len(fit) or not len(held) or not all(np.isfinite(v).all() for v in targets.values()):
        raise ValueError('REPAIR_TARGETS')
    with torch.no_grad():
        initial_logits = source.surfaces(x).clone()

    def metrics(model):
        with torch.no_grad():
            q = model.q_surfaces(x).min(0).values.numpy()
        out = {}
        for target,y in targets.items():
            out[target] = {}
            for role,indices in (('fit',fit),('monitor',held)):
                groups = dict(all=indices)
                for kind in sorted({selected[i]['kind'] for i in indices}):
                    groups[kind] = np.array([i for i in indices if selected[i]['kind']==kind])
                out[target][role] = {k:ranking_metrics(q[v],y[v]) for k,v in groups.items()}
        return out

    baseline = metrics(source)
    controls = {k:dict(actor=ranking_metrics(np.eye(3)[signals[rows[held]]],y[held]),
        nop=ranking_metrics(np.tile([1.,0.,0.],(len(held),1)),y[held])) for k,y in targets.items()}
    save(a.output/'BASELINE.json', identity, dict(metrics=baseline, controls=controls,
        stochastic_soft_std_mean=float(soft.std(1,ddof=1).mean()), fork_results=forks,
        actor_exact=True, fit_count=len(fit), monitor_count=len(held)))
    results = {}
    for target,y in targets.items():
        for seed in cfg['seeds']:
            for initialization in ('retained','fresh'):
                name=f'{target}_{initialization}_{seed}';root=a.output/'calibration'/name
                result = cached(root/'RESULT.json',identity)
                if result is not None:
                    if file_sha(root/'MODEL.pt') != result['model_sha256']:
                        raise ValueError('REPAIR_CALIBRATED_MODEL_HASH')
                    results[name] = result;continue
                progress('CALIBRATE', arm=name, updates=0)
                m,_ = load_bundle(a.bundle,'cpu')
                if initialization=='fresh':initialize_critic_heads(m,seed)
                set_scope(m,'heads');clear_optimizer_memory(m.optimizer)
                frozen=frozen_state(m);schedule=np.random.default_rng(seed).choice(fit,(4096,128),replace=True)
                truth=torch.as_tensor(y,dtype=torch.float32);points=[]
                for step in range(4097):
                    if step:
                        batch=schedule[step-1];m.optimizer.zero_grad(set_to_none=True)
                        loss=target_loss(differentiable_q(m,x[batch]),truth[batch]);loss.backward()
                        torch.nn.utils.clip_grad_norm_(m.loss.parameters(),10.,error_if_nonfinite=True)
                        m.optimizer.step()
                    if step in POINTS:
                        verify_frozen_state(m,frozen)
                        if not torch.equal(m.surfaces(x),initial_logits):raise ValueError('REPAIR_ACTOR_CHANGED')
                        points.append(dict(updates=step,metrics=metrics(m)))
                        write(root/'CURVE.json',points);progress('CALIBRATE',arm=name,updates=step)
                sync_targets(m)
                final=points[-1]['metrics'][target]['monitor']['all']
                qualified=passed(final,baseline[target]['monitor']['all'],controls[target]['actor'],controls[target]['nop'])
                m.save(root/'MODEL.pt')
                result=dict(name=name,initialization=initialization,seed=seed,target=target,points=points,
                    qualified=qualified,model_sha256=file_sha(root/'MODEL.pt'),actor_exact=True)
                save(root/'RESULT.json',identity,result);results[name]=result
    choices=[init for init in ('retained','fresh')
             if all(results[f'soft_{init}_{s}']['qualified'] for s in cfg['seeds'])]
    # A finite diagnostic gate, not a model promotion or a statement of significance.
    selected_init=min(choices,key=lambda init:np.mean([results[f'soft_{init}_{s}']['points'][-1]['metrics']['soft']['monitor']['all']['mean_regret'] for s in cfg['seeds']])) if choices else None
    save(a.output/'DECISION.json',identity,dict(selected_initialization=selected_init,
        qualified=bool(choices),status='ACTOR_PROBE_ALLOWED' if choices else 'REPAIR_NOT_QUALIFIED'))
    actor_results=[]
    if selected_init:
        for seed in cfg['seeds']:
            for mode in ('free','kl'):
                root=a.output/'actor'/f'{selected_init}_{seed}_{mode}'
                result=cached(root/'RESULT.json',identity)
                if result is not None:actor_results.append(result);continue
                m,_=load_bundle(a.bundle,'cpu');m.load(a.output/'calibration'/f'soft_{selected_init}_{seed}'/'MODEL.pt')
                set_arm(m,'actor_only');clear_optimizer_memory(m.optimizer)
                for group in m.optimizer.param_groups:group['lr']=cfg['actor_lr']
                frozen=frozen_state(m);params=[p for p in m.loss.parameters() if p.requires_grad]
                schedule=np.random.default_rng(seed).choice(fit,(32,128),replace=True)
                events=[];accounts=[]
                for step,batch in enumerate(schedule,1):
                    z=m.surfaces(x[batch]);lp=z.log_softmax(-1)
                    with torch.no_grad():q=m.q_surfaces(x[batch]).min(0).values
                    loss=(lp.exp()*(alpha*lp-q)).sum(-1).mean()
                    event=constrained_step(m.optimizer,params,loss,lambda:m.surfaces(x[fit]),
                        initial_logits[fit],cfg['kl_limit'] if mode=='kl' else None)
                    events.append(event);verify_frozen_state(m,frozen)
                    if step in (1,8,32):
                        progress('ACTOR_TRAIN_PROBE',seed=seed,mode=mode,attempt=step)
                        z,_=emit(m,market.inputs,indices=ix)
                        acc=evaluate(make(source.recipe['split']['periods']['TRAIN']),z)
                        with torch.no_grad():kl=mean_kl(initial_logits[held],m.surfaces(x[held]))
                        accounts.append(dict(attempt=step,train_account=acc,monitor_kl=kl))
                        write(root/'ACCOUNTS.json',accounts)
                result=dict(seed=seed,mode=mode,events=events,accounts=accounts,promoted=False)
                save(root/'RESULT.json',identity,result);actor_results.append(result)
    if tree_digest(source.loss.state_dict())!=original or file_sha(a.bundle/'model.pt')!=SOURCE:
        raise ValueError('REPAIR_SOURCE_MUTATED')
    report(a.output,results)
    result=dict(identity=identity,status='COMPLETE',selected_initialization=selected_init,
        calibration={n:dict(qualified=r['qualified'],metrics=r['points'][-1]['metrics']) for n,r in results.items()},
        actor_results=actor_results,source_exact=True,validation_inference=False,test_read=False,
        promoted=False,seconds=time.monotonic()-start)
    save(a.output/'DONE.json',identity,result);progress('COMPLETE',qualified=bool(choices))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('parent','bundle','output'):parser.add_argument('--'+n,type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:
            write(a.output/'FAILURE.json',dict(at=now(),error=repr(exc)));raise
