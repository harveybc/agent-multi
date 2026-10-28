"""Sequential native exit-menu proof, DEAP comparisons and synthetic PPO controls."""
import argparse
import copy
import fcntl
from pathlib import Path
import time

import numpy as np

from app.frozen_recipe_inputs import file_sha,require
from app.rl_subset import digest,seal_recipe
from app.optimal_exit import targets,outcome
from app.optimal_exit_corpus import specs,build_case
from app.optimal_exit_search import panel,score,search
from tools.recipe_sac_queue import read,write,source_pins,software,now,tensor_hash

ROOT=Path(__file__).resolve().parents[1]


def summarize(rows):
    return dict(cases=len(rows),mean_regret_risk=float(np.mean([r['regret_risk'] for r in rows])),
        median_regret_risk=float(np.median([r['regret_risk'] for r in rows])),
        p90_regret_risk=float(np.quantile([r['regret_risk'] for r in rows],.9)),
        early_close_share=float(np.mean([r['early_close'] for r in rows])),
        mean_net_usd=float(np.mean([r['net_usd'] for r in rows])),
        exact_optimal_share=float(np.mean([abs(r['regret_usd'])<1e-7 for r in rows])),
        by_direction={str(d):dict(cases=sum(r['direction']==d for r in rows),
            mean_regret_risk=float(np.mean([r['regret_risk'] for r in rows if r['direction']==d]))) for d in (1,-1)})


def evaluate(model,env):
    rows=[]
    for i,case in enumerate(env.cases):
        obs,_=env.reset(options={'case_index':i});done=False
        while not done:
            action=model.predict(obs,deterministic=True)[0]
            obs,_,done,_,info=env.step(action)
        rows.append(dict(case_id=case['id'],**info['outcome']))
    return dict(summary=summarize(rows),cases=rows)


def run(a):
    import torch
    from tools.systematic_strategy_matrix import setup
    from tools.rl_subset_bridge import build_model,progress_callback
    from env_plugins.optimal_exit_diagnostic import OptimalExitDiagnostic
    from stable_baselines3 import PPO
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available(),'CPU_ONLY_DIAGNOSTIC')
    design=read(a.design);require(design['schema']=='optimal_exit_study.v1','EXIT_STUDY_SCHEMA')
    require(design['head_bias_bounds']==[-2.,2.] and design['learning']['gamma']==1.,'FROZEN_EXIT_BOUNDS_GAMMA')
    identity=dict(design=design,recipe_sha=file_sha(a.recipe),smoke=a.smoke,
        code=source_pins(ROOT,('app','agent_plugins','env_plugins','tools')),
        engine=source_pins(a.engine_root,('app',)),software=software(),financial_admission=False)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'IDENTITY.json').exists():require(read(a.output/'IDENTITY.json')==identity,'EXIT_STUDY_IDENTITY')
        else:write(a.output/'IDENTITY.json',identity)
        manifest=read(a.output/'ARTIFACTS.json') if (a.output/'ARTIFACTS.json').exists() else {}
        for name,sha in manifest.items():require(file_sha(a.output/name)==sha,'EXIT_STUDY_ARTIFACT')
        def register(*paths):
            for path in paths:manifest[str(path.relative_to(a.output))]=file_sha(path)
            write(a.output/'ARTIFACTS.json',manifest)
        if (a.output/'CLOSE.json').exists():return
        def status(stage,**kw):write(a.output/'STATUS.json',dict(stage=stage,at=now(),financial_admission=False,**kw))
        recipe,cfg=setup(a.recipe,a.engine_root);shape=(13,recipe['list']['n'])
        recipe=copy.deepcopy(recipe)
        recipe['labels']['optimal_exit_diagnostic']='synthetic causal price channels, not selected financial inputs'
        seal_recipe(recipe);write(a.output/'DIAGNOSTIC_RECIPE.json',recipe)
        register(a.output/'DIAGNOSTIC_RECIPE.json')
        datasets={};observations={}
        for split in ('train','confirmation'):
            datasets[split]=[];observations[split]=[]
            selected=list(specs(split))
            if a.smoke:selected=[selected[0],next(s for s in selected if s['direction']==-1)]
            for i,spec in enumerate(selected):
                path=a.output/'corpus'/(spec['id']+'.json');array=path.with_suffix('.npz')
                if str(path.relative_to(a.output)) in manifest:
                    case=read(path)
                    with np.load(array,allow_pickle=False) as data:x=data['observations']
                else:
                    status('BUILDING_NATIVE_CORPUS',split=split,completed=i,total=len(selected))
                    case,x=build_case(spec,cfg,shape);write(path,case)
                    np.savez_compressed(array,observations=x);register(path,array)
                datasets[split].append(case);observations[split].append(x)
        baselines={}
        for split,cases in datasets.items():
            baselines[split]={}
            for mode in ('immediate','passive','oracle'):
                rows=[]
                for case in cases:
                    action=np.zeros(len(case['close_net']))
                    if mode=='immediate':action[0]=1
                    if mode=='oracle':action=targets(case['close_net'],case['terminal_net'],case['risk_usd'])['oracle_close']
                    rows.append(outcome(case,action))
                baselines[split][mode]=summarize(rows)
        write(a.output/'BASELINES.json',baselines);register(a.output/'BASELINES.json')
        calibration=panel(datasets['train'],20261027);confirmation=panel(datasets['confirmation'],20261028)
        deap_reports=[]
        for objective in design['objectives']:
            for seed in design['seeds'][:1] if a.smoke else design['seeds']:
                path=a.output/'deap'/f'{objective}_{seed}.json'
                if str(path.relative_to(a.output)) not in manifest:
                    status('DEAP',objective=objective,seed=seed)
                    params=dict(design['deap'])
                    if a.smoke:params.update(population=4,generations=1,patience=1)
                    report=search(calibration,objective,seed,**params)
                    report['confirmation']=score(report['genes'],confirmation)
                    report['confirmation_control']=score([0.,0.],confirmation)
                    write(path,report);register(path)
                deap_reports.append(read(path))
        fits=[]
        for reward in design['reward_modes']:
            for seed in design['seeds'][:1] if a.smoke else design['seeds']:
                unit=f'{reward}_{seed}';path=a.output/'fits'/unit/'RESULT.json'
                if str(path.relative_to(a.output)) not in manifest:
                    status('LEARNING',unit=unit)
                    env=OptimalExitDiagnostic(datasets['train'],observations['train'],reward)
                    budget=dict(design['learning'])
                    if a.smoke:budget.update(total_timesteps=512,n_steps=64,batch_size=64,n_epochs=1)
                    model=build_model(env,recipe,seed,'cpu',budget)
                    model.gamma=model.rollout_buffer.gamma=1.
                    initial=tensor_hash(model);started=time.monotonic()
                    model.learn(budget['total_timesteps'],callback=progress_callback(a.output,unit,budget['total_timesteps']))
                    require(initial!=tensor_hash(model),'EXIT_WEIGHTS_NOT_UPDATED')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    weights=path.parent/'diagnostic_model.zip';model.save(weights)
                    loaded=PPO.load(weights,device='cpu')
                    require(tensor_hash(model)==tensor_hash(loaded),'EXIT_RELOAD_WEIGHTS')
                    obs,_=env.reset(options={'case_index':0})
                    require(np.array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0]),'EXIT_RELOAD_ACTION')
                    reports={}
                    for split in datasets:
                        status('EVALUATING',unit=unit,split=split)
                        reports[split]=evaluate(loaded,OptimalExitDiagnostic(datasets[split],observations[split],reward))
                    verdict=all(reports[split]['summary']['by_direction'][str(d)]['mean_regret_risk']<min(
                        baselines[split][b]['by_direction'][str(d)]['mean_regret_risk'] for b in ('immediate','passive'))
                        for split in datasets for d in (1,-1))
                    report=dict(unit=unit,reward=reward,seed=seed,steps=budget['total_timesteps'],gamma=1.,
                        seconds=time.monotonic()-started,initial_weights=initial,final_weights=tensor_hash(model),
                        reports=reports,better_than_both_blind_controls_both_sides=verdict,financial_admission=False)
                    write(path,report);register(path,weights)
                    del loaded,model,env
                fits.append(read(path))
        state='SMOKE_COMPLETE' if a.smoke else 'SYNTHETIC_STUDY_COMPLETE_NOT_FINANCIAL_RELEASE'
        final=dict(state=state,identity=digest(identity),deap_runs=len(deap_reports),learning_runs=len(fits),
            learned_controls_passed=sum(r['better_than_both_blind_controls_both_sides'] for r in fits),
            financial_admission=False,artifacts=manifest,at=now())
        write(a.output/'CLOSE.json',final);status(state,**{k:v for k,v in final.items() if k not in ('state','at','artifacts','financial_admission')})
        write(a.output/'PROGRESS.json',read(a.output/'STATUS.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('recipe','engine-root','design','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:run(a)
    except Exception as exc:
        write(a.output/'FAILED.json',dict(error=type(exc).__name__,detail=str(exc),at=now()))
        raise
