"""Read-only source audit, objective recomputation and exact cached DEAP replay."""
import argparse
from pathlib import Path
import math
import tempfile

from app.frozen_recipe_inputs import require, file_sha
from app.rl_subset import digest
from tools.recipe_sac_queue import read, write, now
from tools.native_wrapper_audit import ledger_checks
from tools.synthetic_output_calibration import assess
from tools.deap_output_calibration import unseal, evolve


def expected_score(name,row):
    require(name in ('sharpe','sharpe_return','sharpe_return_activity'),'OBJECTIVE_NAME')
    report=row['report'];c=row['components']
    if not report['eligible']:return -1e9-len(report['reasons'])
    score=c['sharpe']
    if name in ('sharpe_return','sharpe_return_activity'):score+=c['return_fraction']
    if name=='sharpe_return_activity':score+=c['orders_per_day']
    return score


def validate_state(state,name,cfg,lookup):
    require(len(state['population'])==len(state['scores'])==cfg['population'],'POPULATION')
    for genes,score in zip(state['population'],state['scores']):
        require(len(genes)==6 and all(math.isfinite(v) and 0<=v<=1 for v in genes),'GENOME_DOMAIN')
        require(score==expected_score(name,lookup(genes)),'OBJECTIVE_SCORE')
    require(state['best_score']==max(state['scores']) and
            state['best_score']==expected_score(name,lookup(state['best'])),'BEST_SCORE')
    require(len(state['history'])==state['generation']+1,'HISTORY_LENGTH')
    anchor=None;patience=0;previous=-math.inf
    for i,h in enumerate(state['history']):
        score=h['best_score']
        require(h['generation']==i and math.isfinite(score) and score>=previous,'ELITIST_HISTORY')
        previous=score
        if anchor is None or score-anchor>cfg['tolerance']*max(abs(anchor),1e-12):anchor=score;patience=0
        else:patience+=1
        require(h['patience']==patience,'HISTORY_PATIENCE')
    require(state['patience']==patience and state['anchor']==anchor,'PATIENCE')
    expected=('PATIENCE' if patience>=cfg['patience'] else
              'MAX_GENERATIONS' if state['generation']==cfg['generations'] else None)
    require(state['stop_reason']==expected,'STOP_REASON')
    require(previous==state['best_score'],'FINAL_HISTORY_SCORE')
    require(cfg['population']<=state['evaluations']<=cfg['population']+(cfg['population']-1)*state['generation'],
            'EVALUATION_BUDGET')


def audit(a):
    identity=read(a.run/'IDENTITY.json');spec=read(a.campaign/'SPEC.private.json')
    require(file_sha(a.campaign/'SPEC.private.json')==identity['spec_sha256'],'SPEC_PIN')
    require(file_sha(a.campaign/'RECIPE.json')==identity['recipe_sha256'],'RECIPE_PIN')
    for name,sha in identity['source_hashes'].items():require(file_sha(a.runtime/name)==sha,'PRODUCER_SOURCE_PIN:'+name)
    for role,root in [('agent',a.runtime),('engine',a.engine_root)]:
        for name,sha in spec['code'][role].items():require(file_sha(root/name)==sha,'BASE_SOURCE_PIN:'+name)
    require(identity['financial_data_read'] is False and identity['training'] is False,'SYNTHETIC_SCOPE')
    schema=identity['schema'];require(schema in ('deap_output_calibration.v1','deap_output_calibration.v2'),'SCHEMA')
    memo={};trades_checked=0

    def lookup(genes):
        nonlocal trades_checked
        key=digest(genes)
        if key in memo:return memo[key]
        value=unseal(a.run/'cache'/(key+'.json'),identity)
        require(value['genes']==genes,'CACHE_GENOME')
        waves=value['waves']
        require(len(waves)==6,'SIX_WAVES')
        for i,r in enumerate(waves):
            require([r['amplitude'],r['weeks'],r['phase']]==identity['calibration'][i//3],'CALIBRATION_TAPE')
            ledger_checks(r['metrics'],r['trades']);trades_checked+=len(r['trades'])
            for side,name in ((1,'n_long'),(-1,'n_short')):
                require(r['metrics'][name]==sum(t['direction']==side for t in r['trades']),'DIRECTION_COUNT')
            for t in r['trades']:
                q=genes[2 if t['direction']==1 else 5]
                require(t['percentual_order_size']==q,'DIRECTIONAL_Q')
        ideals=waves[::3]
        comp=dict(sharpe=sum(r['metrics']['sharpe_daily_sqrt365'] for r in ideals)/2,
            return_fraction=sum(sum(t['net_usd'] for t in r['trades'])/identity['capital'] for r in ideals)/2,
            orders_per_day=sum(len(r['trades'])/(5*r['weeks']) for r in ideals)/2)
        for key2,v in comp.items():require(math.isclose(v,value['components'][key2],rel_tol=1e-10,abs_tol=1e-10),'FITNESS_COMPONENT')
        require(assess(dict(genes=genes),waves)==value['report'],'ELIGIBILITY')
        memo[key]={k:v for k,v in value.items() if k!='waves'}
        return memo[key]

    cache_files=sorted((a.run/'cache').glob('*.json'))
    for path in cache_files:
        r=read(path);genes=r['payload']['genes']
        require(path.stem==digest(genes),'CACHE_FILENAME');lookup(genes)
    snapshots=[]
    allowed={f'{name}_s{seed}':(name,seed) for seed in identity['seeds'] for name in identity['objectives']}
    for path in sorted((a.run/'searches').glob('*/STATE.json')):
        run_id=path.parent.name;require(run_id in allowed,'SEARCH_ID')
        name,seed=allowed[run_id]
        state_identity=dict(parent=identity,search=identity['search'],seed=seed)
        if schema=='deap_output_calibration.v2':state_identity['objective']=name
        state=unseal(path,state_identity)
        validate_state(state,name,identity['search'],lookup)
        with tempfile.TemporaryDirectory(prefix='deap-replay-') as tmp:
            replay=evolve(Path(tmp),seed,lambda genes:expected_score(name,lookup(genes)),identity['search'],
                          identity,stop_after=state['generation'],objective_name=name)
        require(replay==state,'EXACT_SEARCH_REPLAY')
        result_path=path.parent/'RESULT.json'
        if result_path.exists():
            result=unseal(result_path,identity)
            require(result['objective']==name and result['seed']==seed and result['id']==run_id,'RESULT_OBJECTIVE')
            require(result['fitness']==expected_score(name,lookup(result['genes'])),'RESULT_SCORE')
        snapshots.append(dict(id=run_id,generation=state['generation'],stop_reason=state['stop_reason'],
                              state_payload_sha256=digest(state),objective=name,replay_exact=True))
    closed=(a.run/'CLOSE.json').exists()
    require(not a.require_complete or closed,'COMPLETION_REQUIRED')
    if closed:
        close=read(a.run/'CLOSE.json');require(close['identity_sha256']==digest(identity),'CLOSE_IDENTITY')
        for name,sha in close['artifacts'].items():require(file_sha(a.run/name)==sha,'CLOSE_ARTIFACT')
        require(len(snapshots)==9 and all(r['stop_reason'] for r in snapshots),'NINE_COMPLETE_SEARCHES')
        from app.recipe_runtime import attach_engine
        attach_engine(a.engine_root)
        from app.short_horizon.expert_guided.env import EnvConfig
        from app.short_horizon.execution import Sizing,RealisticProfile,OrderRules
        from app.directional_risk_calibration import DirectionalRisk,DirectionalOrders,fitness_components
        from tools.native_strategy_contract_study import oracle
        here=Path(__file__).resolve().parents[1]
        for name in ('tools/native_strategy_contract_study.py','app/directional_risk_calibration.py',
                     'app/native_strategy_contract.py'):
            require(file_sha(here/name)==identity['source_hashes'][name],'REPLAY_SOURCE_PIN')
        recipe=read(a.campaign/'RECIPE.json');c=dict(recipe['business']['evaluation_and_policy'])
        c.update(sizing=Sizing(**c['sizing']),profile=RealisticProfile(**c['profile']),rules=OrderRules(**c['rules']))
        config=EnvConfig(**c);shape=(13,recipe['list']['n'])
        results=read(a.run/'SEARCH_RESULTS.json')
        require({r['id'] for r in results}==set(allowed) and len(results)==9,'RESULT_SET')
        for r in results:
            require(unseal(a.run/'searches'/r['id']/'RESULT.json',identity)==r,'RESULT_SUMMARY')
        confirmation=read(a.run/'CONFIRMATION.json');confirmation_map={r['id']:r for r in confirmation}
        require(set(confirmation_map)==set(allowed)|{'midpoint','grid_control'} and len(confirmation)==11,'CONFIRMATION_SET')
        replayed=0
        for r in results+[dict(id='midpoint',genes=[.5,.5,1]*2),dict(id='grid_control',genes=[.75,.25,.2]*2)]:
            value=unseal(a.run/'confirmation'/(r['id']+'.json'),identity)
            require(value['genes']==r['genes'],'CONFIRMATION_GENOME')
            waves=[]
            for params in identity['confirmation']:
                waves.extend(oracle(config,shape,*params,DirectionalRisk(tuple(r['genes'])),DirectionalOrders,False))
            require(waves==value['waves'],'EXACT_CONFIRMATION_REPLAY');replayed+=len(waves)
            require(fitness_components(waves,identity['capital'])==value['components'],'CONFIRMATION_COMPONENTS')
            require(assess(dict(genes=r['genes']),waves)==value['report'],'CONFIRMATION_ELIGIBILITY')
            require({k:v for k,v in value.items() if k!='waves'}==confirmation_map[r['id']],'CONFIRMATION_SUMMARY')
        comparison=read(a.run/'COMPARISON.json')
        require(comparison['unique_genomes']==len(cache_files) and comparison['calibration_simulations']==len(cache_files)*6
                and comparison['confirmation_simulations']==66,'SIMULATION_COUNTS')
        require(len(comparison['rows'])==3,'COMPARISON_ROWS')
        for row in comparison['rows']:
            selected=[r for r in results if r['objective']==row['objective']]
            require(sorted(r['seed'] for r in selected)==identity['seeds'],'PAIRED_SEEDS')
            obs=[confirmation_map[r['id']] for r in selected]
            for key,source in [('calibration',selected),('confirmation',obs)]:
                require(row[key]=={k:sum(r['components'][k] for r in source)/3
                    for k in ('sharpe','return_fraction','orders_per_day')},'COMPARISON_AGGREGATE')
            require(row['confirmation_eligible']==sum(r['report']['eligible'] for r in obs),'ELIGIBILITY_COUNT')
    result=dict(state='PASS',closed=closed,identity_sha256=digest(identity),auditor_sha256=file_sha(Path(__file__)),
        cache_files_in_snapshot=len(cache_files),genomes_verified=len(memo),trades_checked=trades_checked,
        searches=snapshots,legacy_objective_verified_by_recomputation=schema.endswith('v1'),
        confirmation_simulations_replayed=66 if closed else 0,
        activity_levels_in_cache=sorted({r['components']['orders_per_day'] for r in memo.values()}),
        source_pins_verified=True,source_mutations=False,at=now())
    write(a.output/'AUDIT.json',result)
    print({k:v for k,v in result.items() if k!='searches'},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','runtime','campaign','engine-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();require(not a.output.resolve().is_relative_to(a.run.resolve()),'AUDIT_OUTPUT_ISOLATION')
    try:audit(a)
    except Exception as exc:
        write(a.output/'AUDIT.json',dict(state='FAIL',reason=str(exc),at=now()))
        raise


if __name__=='__main__':main()
