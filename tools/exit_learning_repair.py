"""Versioned, sequential synthetic learning diagnosis. Never reads financial TEST."""
import argparse
from collections import defaultdict
import fcntl
from pathlib import Path
import time

import numpy as np

from app.frozen_recipe_inputs import file_sha, require
from app.optimal_exit import targets
from app.rl_subset import digest
from tools.recipe_sac_queue import read, write, source_pins, software, now, tensor_hash
from tools.optimal_exit_study import evaluate
from tools.audit_optimal_exit_outcomes import expectation

ROOT = Path(__file__).resolve().parents[1]


def load_corpus(source, split, manifest):
    cases, arrays = [], []
    def checked(path):
        require(manifest['artifacts'].get(str(path.relative_to(source))) == file_sha(path), 'EXIT_SOURCE_HASH')
        return path
    for path in sorted((source / 'corpus').glob(split + '_*.json')):
        cases.append(read(checked(path)))
        with np.load(checked(path.with_suffix('.npz')), allow_pickle=False) as data:
            arrays.append(data['observations'])
    require(bool(cases), 'EXIT_CORPUS_EMPTY')
    recipe = read(checked(source / 'DIAGNOSTIC_RECIPE.json'))
    return cases, arrays, recipe


def tensor_dataset(cases, arrays):
    import torch
    masks, labels = [], []
    for case, x in zip(cases, arrays):
        command = 3 if case['direction']==1 else 4
        masks.append(np.tile([1,0,0,int(command==3),int(command==4)], (len(x),1)))
        labels.extend(np.where(targets(case['close_net'],case['terminal_net'],case['risk_usd'])['oracle_close'],command,0))
    return dict(features=torch.from_numpy(np.concatenate(arrays)),
                action_mask=torch.tensor(np.concatenate(masks),dtype=torch.float32)), torch.tensor(labels)


def alias_audit(cases, arrays):
    """Exact observation collisions bound what a memoryless policy can distinguish."""
    groups = defaultdict(set)
    for case,x in zip(cases,arrays):
        labels=targets(case['close_net'],case['terminal_net'],case['risk_usd'])['oracle_close']
        for row,label in zip(x,labels):
            groups[(case['direction'],row.tobytes())].add(bool(label))
    return dict(unique_observations=len(groups), contradictory_groups=sum(len(g)>1 for g in groups.values()),
                scope='Exact byte collisions only; does not prove full observation sufficiency')


def policy_report(model,cases,arrays,constants=None):
    import torch
    from env_plugins.optimal_exit_diagnostic import OptimalExitDiagnostic
    data,_=tensor_dataset(cases,arrays)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        p=model.policy.get_distribution(data).cat.probs
        probability=(p[:,3]+p[:,4]).cpu().numpy()
    if constants is None:
        constants={str(d):float(probability[data['action_mask'][:,3 if d==1 else 4].bool().numpy()].mean()) for d in (1,-1)}
    sampled,blind=[],[];at=0
    for case,x in zip(cases,arrays):
        sampled.append(dict(direction=case['direction'],**expectation(case,probability[at:at+len(x)])))
        blind.append(dict(direction=case['direction'],**expectation(case,np.full(len(x),constants[str(case['direction'])]))))
        at+=len(x)
    def means(rows):
        return {str(d):{k:float(np.mean([r[k] for r in rows if r['direction']==d]))
                       for k in ('regret_risk','mean_net_usd','early_close_probability')} for d in (1,-1)}
    return dict(deterministic=evaluate(model,OptimalExitDiagnostic(cases,arrays)),
                constants_from_train=constants,sampled=means(sampled),constant_hazard=means(blind),
                probability_min=float(probability.min()),probability_max=float(probability.max()))


def replay_all_decisions(model,loaded,cases,arrays):
    import torch
    data,_=tensor_dataset(cases,arrays)
    model.policy.set_training_mode(False);loaded.policy.set_training_mode(False)
    checked=0
    with torch.no_grad():
        for start in range(0,len(data['features']),128):
            obs={k:v[start:start+128] for k,v in data.items()}
            da=model.policy.get_distribution(obs);db=loaded.policy.get_distribution(obs)
            require(torch.equal(da.cat.probs,db.cat.probs),'EXIT_REPLAY_ALL_PROBABILITIES')
            require(torch.equal(da.get_actions(True),db.get_actions(True)),'EXIT_REPLAY_ALL_ACTIONS')
            require(torch.equal(model.policy.predict_values(obs),loaded.policy.predict_values(obs)),
                    'EXIT_REPLAY_ALL_VALUES')
            checked+=len(obs['features'])
    return checked


def qualifies(report, cases):
    from app.optimal_exit import outcome
    for d in (1,-1):
        selected=[case for case in cases if case['direction']==d]
        controls=[]
        for mode in ('passive','immediate'):
            rows=[]
            for case in selected:
                actions=np.zeros(len(case['close_net']))
                if mode=='immediate':actions[0]=1
                rows.append(outcome(case,actions))
            controls.append(float(np.mean([r['regret_risk'] for r in rows])))
        det=report['deterministic']['summary']['by_direction'][str(d)]['mean_regret_risk']
        if not det < min(*controls,report['constant_hazard'][str(d)]['regret_risk']):return False
    return True


def optimize_exits(root,design,train,confirmation,register,status):
    from app.optimal_exit_search import panel,score,search,objective_scales
    from app.exit_comparison import paired_comparison
    cfg=design['deap'];controls=cfg['controls']
    calibration=panel(train,20261028);heldout=panel(confirmation,20261029)
    scales=objective_scales([score(g,calibration) for g in controls])
    searches=[]
    for objective in cfg['objectives']:
        for seed in design['seeds']:
            path=root/'deap'/f'{objective}_{seed}.json'
            # Recompute cheap searches after a crash; only identical registered bytes are accepted.
            status('DEAP_EXIT_SUBPROBLEM',objective=objective,seed=seed)
            result=search(calibration,objective,seed,scales=scales,bounds=cfg['bounds'],
                **{k:cfg[k] for k in ('population','generations','patience','tolerance')})
            result.update(scales_from_calibration_controls=scales,bounds=cfg['bounds'],
                          unit=f'{objective}_{seed}',scope='two close biases only; not full strategy optimization')
            if path.exists():require(read(path)==result,'EXIT_SEARCH_REPLAY')
            else:write(path,result)
            register(path);searches.append(result)
    frozen=root/'deap'/'FINALISTS_BEFORE_CONFIRMATION.json'
    chosen=[max((r for r in searches if r['objective']==objective),key=lambda r:r['history'][-1]['fitness'])
            for objective in cfg['objectives']]
    decision=[dict(unit=r['unit'],genes=r['genes'],objective=r['objective']) for r in chosen]
    if frozen.exists():require(read(frozen)==decision,'EXIT_SEARCH_SELECTION_CHANGED')
    else:write(frozen,decision)
    register(frozen)
    reports=[]
    for r in chosen:
        reports.append(dict(unit=r['unit'],genes=r['genes'],boundary_warning=any(r['bound_hits']),
            confirmation=score(r['genes'],heldout),
            paired=paired_comparison(r['genes'],controls,heldout,replicates=cfg['bootstrap_replicates'])))
    path=root/'deap'/'COMPARISON.json'
    write(path,dict(reports=reports,objective_winner=None,full_strategy_optimized=False,
                   financial_admission=False));register(path)


def run(a):
    import torch
    from stable_baselines3 import PPO
    from env_plugins.optimal_exit_diagnostic import OptimalExitDiagnostic
    from app.exit_learning_trace import ExitLearningTrace
    from tools.rl_subset_bridge import build_model
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available(),'EXIT_REPAIR_CPU_ONLY')
    design=read(a.design);source_close=read(a.source/'CLOSE.json')
    require(design['schema']=='exit_learning_repair.v1','EXIT_REPAIR_DESIGN')
    require(not a.output.resolve().is_relative_to(a.source.resolve()),'EXIT_REPAIR_OUTPUT_SOURCE')
    identity=dict(design=design,source_close_sha=file_sha(a.source/'CLOSE.json'),pilot=a.pilot,
        code=source_pins(ROOT,('app','agent_plugins','env_plugins','tools')),software=software())
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'IDENTITY.json').exists():require(read(a.output/'IDENTITY.json')==identity,'EXIT_REPAIR_IDENTITY')
        else:write(a.output/'IDENTITY.json',identity)
        receipts=read(a.output/'ARTIFACTS.json') if (a.output/'ARTIFACTS.json').exists() else {}
        for name,sha in receipts.items():require(file_sha(a.output/name)==sha,'EXIT_REPAIR_RECEIPT')
        if (a.output/'CLOSE.json').exists():
            closed=read(a.output/'CLOSE.json')
            require(closed['identity']==digest(identity) and closed['artifacts']==receipts,'EXIT_REPAIR_CLOSE')
            terminal={k:v for k,v in closed.items() if k!='artifacts'}
            write(a.output/'STATUS.json',terminal);write(a.output/'PROGRESS.json',terminal)
            return
        def register(*paths):
            for path in paths:receipts[str(path.relative_to(a.output))]=file_sha(path)
            write(a.output/'ARTIFACTS.json',receipts)
        def status(state,**kw):
            value=dict(state=state,at=now(),financial_admission=False,**kw)
            write(a.output/'STATUS.json',value);write(a.output/'PROGRESS.json',value)
        cases,arrays,recipe=load_corpus(a.source,'train',source_close)
        alias=alias_audit(cases,arrays)
        write(a.output/'OBSERVATION_AUDIT.json',alias);register(a.output/'OBSERVATION_AUDIT.json')
        budget=design['learning']
        seeds=design['seeds'][:1] if a.pilot else design['seeds']
        units=[dict(name='capacity',kind='supervised_capacity',start='first',steps=0)]
        units+=design['arms']
        results=[]
        for unit in units:
            for seed in seeds[:1] if unit['kind']=='supervised_capacity' else seeds:
                name=f"{unit['name']}_{seed}";folder=a.output/'fits'/name;result_path=folder/'RESULT.json'
                if str(result_path.relative_to(a.output)) in receipts:
                    results.append(read(result_path));continue
                status('TRAINING',unit=name)
                folder.mkdir(parents=True,exist_ok=True)
                env=OptimalExitDiagnostic(cases,arrays,start_mode=unit['start'])
                model=build_model(env,recipe,seed,'cpu',budget)
                model.gamma=model.rollout_buffer.gamma=1.
                initial=tensor_hash(model);started=time.monotonic();trace=None;losses=[]
                if unit['kind']=='supervised_capacity':
                    data,y=tensor_dataset(cases,arrays)
                    rng=np.random.default_rng(seed)
                    for epoch in range(design['capacity_epochs']):
                        order=rng.permutation(len(y));total=0.
                        for offset in range(0,len(y),budget['batch_size']):
                            ix=order[offset:offset+budget['batch_size']]
                            obs={k:v[ix] for k,v in data.items()}
                            model.policy.set_training_mode(True)
                            loss=-model.policy.get_distribution(obs).cat.log_prob(y[ix]).mean()
                            model.policy.optimizer.zero_grad();loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5)
                            model.policy.optimizer.step();total+=float(loss.detach())*len(ix)
                        losses.append(total/len(y))
                        status('CAPACITY_DIAGNOSTIC',unit=name,epoch=epoch+1,budget=design['capacity_epochs'])
                else:
                    trace=ExitLearningTrace(a.output,name,unit['steps'])
                    model.learn(unit['steps'],callback=trace)
                require(initial!=tensor_hash(model),'EXIT_REPAIR_NO_WEIGHT_UPDATE')
                weights=folder/'model.zip';model.save(weights)
                loaded=PPO.load(weights,device='cpu')
                require(tensor_hash(model)==tensor_hash(loaded),'EXIT_REPAIR_RELOAD')
                obs,_=env.reset(options={'case_index':0,'start_index':0})
                require(np.array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0]),'EXIT_REPAIR_REPLAY')
                replay_rows=replay_all_decisions(model,loaded,cases,arrays)
                report=policy_report(loaded,cases,arrays)
                result=dict(unit=name,definition=unit,seed=seed,seconds=time.monotonic()-started,
                    initial_weights=initial,final_weights=tensor_hash(model),train=report,
                    exact_reload_decision_rows=replay_rows,
                    trace=trace.report() if trace else None,capacity_losses=losses,
                    train_qualified=qualifies(report,cases),financial_admission=False)
                write(result_path,result);register(weights,result_path);results.append(result)
                del model,loaded,env
        # All prescribed training finishes before the separate confirmation is read.
        confirmation=[]
        if not a.pilot:
            cc,xx,_=load_corpus(a.source,'confirmation',source_close)
            for result in results:
                status('CONFIRMATION',unit=result['unit'])
                model=PPO.load(a.output/'fits'/result['unit']/'model.zip',device='cpu')
                report=policy_report(model,cc,xx,result['train']['constants_from_train'])
                confirmation.append(dict(unit=result['unit'],report=report,
                    qualified=result['train_qualified'] and qualifies(report,cc),
                    scope=result['definition']['kind']))
                del model
            write(a.output/'CONFIRMATION.json',confirmation);register(a.output/'CONFIRMATION.json')
        qualified=[r['unit'] for r in confirmation if r['qualified'] and r['scope']=='pure_rl']
        optimization='NOT_RUN_TRAIN_ONLY_PILOT' if a.pilot else 'NOT_RUN_LEARNED_CONTROLS_FAILED'
        if qualified:
            optimize_exits(a.output,design,cases,cc,register,status)
            optimization='EXIT_SUBPROBLEM_COMPLETE_NOT_FULL_STRATEGY'
        lines=['# Synthetic exit-learning repair results','',
            'Not financial validation. Pure RL and supervised capacity are distinct arms.',
            'Regret is the net exit shortfall divided by initial SL cash risk; lower is better.',
            'Each evaluation starts at the original first decision and uses deterministic argmax.',
            '', '| Unit | TRAIN regret | Early-close share | TRAIN qualified | Seconds |',
            '|---|---:|---:|---|---:|']
        for r in results:
            m=r['train']['deterministic']['summary']
            lines.append(f"| {r['unit']} | {m['mean_regret_risk']:.8f} | {m['early_close_share']:.4f} | {r['train_qualified']} | {r['seconds']:.1f} |")
        lines+=['','## Reused synthetic confirmation','',
                '| Unit | Regret | Early-close share | Qualified on both splits and directions |',
                '|---|---:|---:|---|']
        for r in confirmation:
            m=r['report']['deterministic']['summary']
            lines.append(f"| {r['unit']} | {m['mean_regret_risk']:.8f} | {m['early_close_share']:.4f} | {r['qualified']} |")
        lines+=['','Optimization: '+optimization+'.',
                'No full-strategy or financial release is conferred by this diagnostic.','']
        report_path=a.output/'RESULTS.md';report_path.write_text('\n'.join(lines));register(report_path)
        state='TRAIN_ONLY_PILOT_COMPLETE' if a.pilot else 'REPAIR_CONTROL_COMPLETE'
        close=dict(state=state,identity=digest(identity),units=len(results),qualified_pure_rl=qualified,
            optimization=optimization,financial_admission=False,artifacts=receipts,at=now())
        write(a.output/'CLOSE.json',close);status(state,units=len(results),qualified_pure_rl=qualified,optimization=optimization)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','design','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--pilot',action='store_true');a=p.parse_args()
    try:run(a)
    except Exception as exc:
        write(a.output/'FAILED.json',dict(error=type(exc).__name__,detail=str(exc),at=now()))
        write(a.output/'STATUS.json',dict(state='FAILED',error=type(exc).__name__,at=now()))
        write(a.output/'PROGRESS.json',read(a.output/'STATUS.json'))
        raise
