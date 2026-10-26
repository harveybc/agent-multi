"""Sequential, restartable DEAP comparison of three synthetic trading objectives."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import random
import time

from deap import base, creator, tools

from app.frozen_recipe_inputs import require, file_sha
from app.rl_subset import digest
from app.directional_risk_calibration import DirectionalRisk, DirectionalOrders, fitness_components, objective
from tools.recipe_sac_queue import read, write, now, software
from tools.synthetic_output_calibration import assess


def seal(path,payload,identity):
    write(path,dict(identity_sha256=digest(identity),payload_sha256=digest(payload),payload=payload))


def unseal(path,identity):
    r=read(path)
    require(r['identity_sha256']==digest(identity),'ARTIFACT_IDENTITY')
    require(r['payload_sha256']==digest(r['payload']),'ARTIFACT_PAYLOAD')
    return r['payload']


def tuple_tree(x):
    return tuple(tuple_tree(v) for v in x) if isinstance(x,list) else x


def evolve(folder,seed,evaluator,cfg,identity,stop_after=None,objective_name='test_fixture'):
    if not hasattr(creator,'SyntheticOutputFitness'):
        creator.create('SyntheticOutputFitness',base.Fitness,weights=(1.,))
        creator.create('SyntheticOutputIndividual',list,fitness=creator.SyntheticOutputFitness)
    individual=creator.SyntheticOutputIndividual
    path=folder/'STATE.json'
    state_identity=dict(parent=identity,search=cfg,seed=seed,objective=objective_name)
    if path.exists():
        state=unseal(path,state_identity)
        if state['stop_reason']:return state
        random.setstate(tuple_tree(state['rng']))
        pop=[individual(x) for x in state['population']]
        for ind,value in zip(pop,state['scores']):ind.fitness.values=(value,)
    else:
        random.seed(seed)
        initial=[[.5,.5,1]*2,[.75,.25,.2]*2,[.75,.25,1]*2]
        pop=[individual(x) for x in initial]
        pop.extend(individual(random.random() for _ in range(6)) for _ in range(cfg['population']-len(pop)))
        state=dict(generation=-1,anchor=None,patience=0,evaluations=0,history=[],stop_reason=None)
    for generation in range(state['generation']+1,cfg['generations']+1):
        if generation>0:
            elite=copy.deepcopy(tools.selBest(pop,1)[0])
            offspring=[copy.deepcopy(x) for x in tools.selTournament(pop,len(pop)-1,tournsize=3)]
            for i in range(1,len(offspring),2):
                if random.random()<.8:
                    tools.cxSimulatedBinaryBounded(offspring[i-1],offspring[i],eta=20,low=0.,up=1.)
                    del offspring[i-1].fitness.values;del offspring[i].fitness.values
            for ind in offspring:
                if random.random()<.2:
                    tools.mutPolynomialBounded(ind,eta=20,low=0.,up=1.,indpb=1/6)
                    del ind.fitness.values
            pop=[elite]+offspring
        for ind in pop:
            if not ind.fitness.valid:
                DirectionalRisk(tuple(ind))
                ind.fitness.values=(float(evaluator(list(ind))),)
                state['evaluations']+=1
        best=tools.selBest(pop,1)[0];score=best.fitness.values[0]
        if state['anchor'] is None or score-state['anchor']>cfg['tolerance']*max(abs(state['anchor']),1e-12):
            state['anchor']=score;state['patience']=0
        else:state['patience']+=1
        reason=('PATIENCE' if state['patience']>=cfg['patience'] else
                'MAX_GENERATIONS' if generation==cfg['generations'] else None)
        state.update(generation=generation,population=[list(x) for x in pop],
            scores=[x.fitness.values[0] for x in pop],best=list(best),best_score=score,
            rng=json.loads(json.dumps(random.getstate())),stop_reason=reason)
        state['history'].append(dict(generation=generation,best_score=score,patience=state['patience'],
                                     evaluations=state['evaluations']))
        seal(path,state,state_identity)
        if reason or (stop_after is not None and generation>=stop_after):return state
    raise RuntimeError('SEARCH_WITHOUT_TERMINAL')


def run(a):
    from app.recipe_runtime import attach_engine
    attach_engine(a.engine_root)
    from app.short_horizon.expert_guided.env import EnvConfig
    from app.short_horizon.execution import Sizing,RealisticProfile,OrderRules
    from tools.native_strategy_contract_study import oracle
    import deap

    spec=read(a.root/'SPEC.private.json');here=Path(__file__).resolve().parents[1]
    require(file_sha(a.root/'SPEC.private.json')==read(a.root/'IDENTITY.json')['spec_sha256'],'SPEC_PIN')
    require(spec['software']==software(),'SOFTWARE_PIN')
    for role,root in [('agent',here),('engine',a.engine_root)]:
        for name,sha in spec['code'][role].items():require(file_sha(root/name)==sha,'SOURCE_PIN:'+name)
    recipe=read(a.root/'RECIPE.json');c=dict(recipe['business']['evaluation_and_policy'])
    c.update(sizing=Sizing(**c['sizing']),profile=RealisticProfile(**c['profile']),rules=OrderRules(**c['rules']))
    config=EnvConfig(**c);shape=(13,recipe['list']['n'])
    cfg=dict(population=20,generations=100,patience=10,tolerance=1e-8)
    names=['sharpe','sharpe_return','sharpe_return_activity']
    identity=dict(schema='deap_output_calibration.v2',deap=deap.__version__,software=software(),
        recipe_sha256=file_sha(a.root/'RECIPE.json'),spec_sha256=file_sha(a.root/'SPEC.private.json'),
        source_hashes={n:file_sha(here/n) for n in ('app/directional_risk_calibration.py',
            'tools/deap_output_calibration.py','tools/native_strategy_contract_study.py',
            'tools/synthetic_output_calibration.py','tools/native_wrapper_audit.py',
            'app/native_strategy_contract.py','tools/recipe_sac_queue.py')},
        search=cfg,seeds=[7,19,43],objectives=names,coefficients=[1,1,1],
        calibration=[[.002,8,0],[.0015,6,3]],confirmation=[[.0012,9,2],[.0024,7,11]],
        capital=config.capital,financial_data_read=False,training=False)
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.output/'IDENTITY.json').exists():require(read(a.output/'IDENTITY.json')==identity,'RUN_IDENTITY')
    else:write(a.output/'IDENTITY.json',identity)
    if (a.output/'CLOSE.json').exists():
        close=read(a.output/'CLOSE.json');require(close['identity_sha256']==digest(identity),'CLOSE_IDENTITY')
        for name,sha in close['artifacts'].items():require(file_sha(a.output/name)==sha,'CLOSE_HASH')
        print('COMPLETE_VERIFIED_NO_REEXECUTION',flush=True);return
    started=time.monotonic();completed=[];new_evals=0;cache={}

    def calibration(genes):
        nonlocal new_evals
        key=digest(genes);path=a.output/'cache'/(key+'.json')
        if key in cache:return cache[key]
        if path.exists():payload=unseal(path,identity);require(payload['genes']==genes,'GENOME_PIN')
        else:
            waves=[]
            for params in identity['calibration']:
                waves.extend(oracle(config,shape,*params,DirectionalRisk(tuple(genes)),DirectionalOrders,False))
            report=assess(dict(genes=genes),waves)
            payload=dict(genes=genes,report=report,components=fitness_components(waves,config.capital),waves=waves)
            seal(path,payload,identity);new_evals+=1
            if new_evals%10==0:
                write(a.output/'STATUS.json',dict(state='SEARCHING',completed_searches=len(completed),total_searches=9,
                    current=current,new_evaluations=new_evals,seconds_per_new_evaluation=(time.monotonic()-started)/new_evals,
                    heartbeat=now()))
        # Keep only metrics in RAM; the full authenticated trade ledgers stay on disk.
        brief={k:v for k,v in payload.items() if k!='waves'};cache[key]=brief
        return brief

    for seed in identity['seeds']:
        for name in names:
            current=f'{name}_s{seed}';folder=a.output/'searches'/current
            def evaluator(genes):
                value=calibration(genes)
                return objective(name,value['components'],value['report']['eligible'],value['report']['reasons'])
            print('START',current,flush=True)
            state=evolve(folder,seed,evaluator,cfg,identity,objective_name=name)
            best=calibration(state['best'])
            result=dict(id=current,objective=name,seed=seed,genes=state['best'],fitness=state['best_score'],
                generation=state['generation'],stop_reason=state['stop_reason'],evaluations=state['evaluations'],
                components=best['components'],report=best['report'])
            seal(folder/'RESULT.json',result,identity);completed.append(result)
            print('FINISHED',current,result['generation'],result['stop_reason'],result['components'],flush=True)
            write(a.output/'SEARCH_RESULTS.json',completed)
    confirmation=[]
    controls=[dict(id='midpoint',genes=[.5,.5,1]*2),dict(id='grid_control',genes=[.75,.25,.2]*2)]
    for row in completed+controls:
        path=a.output/'confirmation'/(row['id']+'.json')
        if path.exists():payload=unseal(path,identity)
        else:
            waves=[]
            for params in identity['confirmation']:
                waves.extend(oracle(config,shape,*params,DirectionalRisk(tuple(row['genes'])),DirectionalOrders,False))
            payload=dict(id=row['id'],genes=row['genes'],report=assess(dict(genes=row['genes']),waves),
                         components=fitness_components(waves,config.capital),waves=waves)
            seal(path,payload,identity)
        confirmation.append({k:v for k,v in payload.items() if k!='waves'})
    write(a.output/'CONFIRMATION.json',confirmation)
    unique=len(list((a.output/'cache').glob('*.json')))
    table=[]
    for name in names:
        selected=[r for r in completed if r['objective']==name]
        observed=[r for r in confirmation if r['id'] in {s['id'] for s in selected}]
        table.append(dict(objective=name,seeds=3,
            calibration={k:sum(r['components'][k] for r in selected)/3 for k in ('sharpe','return_fraction','orders_per_day')},
            confirmation={k:sum(r['components'][k] for r in observed)/3 for k in ('sharpe','return_fraction','orders_per_day')},
            confirmation_eligible=sum(r['report']['eligible'] for r in observed)))
    write(a.output/'COMPARISON.json',dict(rows=table,searches=9,unique_genomes=unique,
        calibration_simulations=unique*6,confirmation_simulations=66,financial_admission=False))
    lines=['# DEAP synthetic objective comparison','',
        'Synthetic diagnostic outputs only. No trained-model or real-market performance claim.',
        'Coefficients are1; S=Sharpe, R=net profit/initial balance, A=orders/business day.',
        'Fresh D/E were evaluated after every optimization finished. No winner is chosen on D/E.','',
        '| Objective | Calibration S | Calibration R | Calibration A | D/E S | D/E R | D/E A | D/E eligible seeds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        values=[r[phase][k] for phase in ('calibration','confirmation')
                for k in ('sharpe','return_fraction','orders_per_day')]
        lines.append('| '+r['objective']+' | '+' | '.join(f'{v:.8f}' for v in values)+f" | {r['confirmation_eligible']}/3 |")
    lines.extend(['','| Search | Generations | Stop | Evaluation requests | Genes: TP-L,SL-L,q-L,TP-S,SL-S,q-S |',
                  '|---|---:|---|---:|---|'])
    for r in completed:
        lines.append(f"| {r['id']} | {r['generation']} | {r['stop_reason']} | {r['evaluations']} | "+
                     ', '.join(f'{x:.8f}' for x in r['genes'])+' |')
    lines.extend(['',f'Unique evaluated genomes: {unique}; calibration simulations: {unique*6}; confirmation simulations:66.',
        'Full per-trade costs, buy/sell counts, win rates and exit reasons are authenticated in cache/ and confirmation/.',
        'A high synthetic Sharpe does not establish learned signal. All failed eligibility cases are retained.',''])
    (a.output/'RESULTS.md').write_text('\n'.join(lines))
    artifacts={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*.json')
               if p.name not in ('CLOSE.json','STATUS.json')}
    artifacts['RESULTS.md']=file_sha(a.output/'RESULTS.md')
    for name,sha in artifacts.items():require(file_sha(a.output/name)==sha,'FINAL_READBACK')
    write(a.output/'CLOSE.json',dict(state='COMPLETE',identity_sha256=digest(identity),artifacts=artifacts,at=now()))
    write(a.output/'STATUS.json',dict(state='COMPLETE',completed_searches=9,unique_genomes=unique,heartbeat=now()))
    print('COMPLETE',json.dumps(table),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','engine-root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.output.resolve().is_relative_to(a.root.resolve()),'OUTPUT_ISOLATION')
    run(a)


if __name__=='__main__':main()
