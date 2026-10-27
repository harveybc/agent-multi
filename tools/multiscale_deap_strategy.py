"""Ordered DEAP, controls, confirmation, full finalist matrices and final audit packet."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from app.frozen_recipe_inputs import require,file_sha
from app.rl_subset import digest
from app.multiscale_strategy_search import panel,initialize,episode,aggregate,score,finalists,require_final_evidence
from app.strategy_matrix_verification import verify
from tools.deap_output_calibration import evolve,seal,unseal
from tools.recipe_sac_queue import read,write,source_pins,now,software

ROOT=Path(__file__).resolve().parents[1]


def matrix_command(a,config,out):
    return [sys.executable,'-m','tools.systematic_strategy_matrix','--recipe',str(a.recipe),
        '--source-spec',str(a.source_spec),'--engine-root',str(a.engine_root),
        '--strategy-config',str(config),'--design',str(a.design),'--workers',str(a.workers),
        '--output',str(out)]


def parameter_json(genes):
    keys=('tp_fraction','sl_fraction','order_fraction')
    return dict(schema='directional_strategy_parameters.v1',
        long=dict(zip(keys,genes[:3])),short=dict(zip(keys,genes[3:])))


def emit_final(out,identity,searches,confirmation,matrices,controls,sensitivity,scales):
    require_final_evidence(searches,confirmation,matrices)
    selected=finalists(searches)
    passed={m['genome_sha256'] for m in matrices if m['state']=='VERIFIED'}
    accepted=[r for r in selected if digest(r['genes']) in passed and r['report']['eligible']]
    # Confirmation is reported but never chooses the strategy.
    recommendation=min(accepted,key=lambda r:(-r['report']['components']['sharpe'],
        -r['report']['components']['return_fraction'],r['report']['mean_drawdown_pct'],r['genes'])) if accepted else None
    decision=dict(schema='synthetic_strategy_final_decision.v1',identity_sha256=digest(identity),
        state='FINAL_SYNTHETIC_STRATEGY_READY_FOR_AUDIT' if recommendation else 'STUDY_COMPLETE_NO_ACCEPTABLE_FINALIST',
        recommended= recommendation,financial_admission=False,learned_policy_admission=False,
        independent_audit_approved=False,finalists=selected,matrices=matrices,scales=scales)
    write(out/'DECISION.json',decision)
    lines=['# Final synthetic strategy optimization report','',
        'DEAP and full finalist execution are complete. This is not learned-market performance.',
        'Commands, oracle tapes and business rules were fixed before search; only six risk fractions were genes.',
        'Tapes depend on the synthetic future only, never on a candidate genome. No financial observations were read.',
        'S = pooled daily-return Sharpe (sqrt(365)); R = mean net return / initial balance; A = orders/business day.',
        'Each component is divided by its frozen control-derived scale: '+json.dumps(scales,sort_keys=True)+'.',
        'Population 20, 100 maximum generations, patience 10, relative tolerance 1e-8, seeds 7/19/43.',
        'Initial controls enter each population; eligible-control elitism is checked on closure.',
        'Calibration chooses one finalist per objective, before confirmation. Confirmation never selects or retunes.',
        '', '| Search | Generations | Stop | Calibration S | R | Orders/week | Mean profit USD | Long/short | Win rate | Worst DD % |',
        '|---|---:|---|---:|---:|---:|---:|---|---:|---:|']
    def row(label,r,generation='-',stop='-'):
        c=r['components'];s=r['sides'];number=lambda x:'undefined' if x is None else f'{x:.8f}'
        return '| '+label+' | '+str(generation)+' | '+stop+' | '+' | '.join([
            number(c['sharpe']),number(c['return_fraction']),number(r['orders_per_week']),
            number(r['mean_profit_usd']),str(s['1']['trades'])+'/'+str(s['-1']['trades']),
            number(r['win_rate']),number(r['worst_drawdown_pct'])])+' |'
    lines.extend(row(r['id'],r['report'],r['generation'],r['stop_reason']) for r in searches)
    lines+=['','## Independent confirmation','',
        '| Arm | Generations | Stop | S | R | Orders/week | Mean profit USD | Long/short | Win rate | Worst DD % |',
        '|---|---:|---|---:|---:|---:|---:|---|---:|---:|']
    lines.extend(row(r['id'],r['report']) for r in confirmation)
    lines+=['','## Controls and parameter effects','']
    lines.extend('- '+r['id']+': '+json.dumps(r['report'],sort_keys=True) for r in controls)
    for i in range(6):
        pair=[r for r in sensitivity if r['gene']==i]
        lines.append('- Gene '+str(i)+': '+json.dumps(pair,sort_keys=True))
    lines+=['','## Finalists and all-scenario verification','']
    for r in selected:
        m=next(x for x in matrices if x['genome_sha256']==digest(r['genes']))
        lines.append('- '+r['id']+': genes='+json.dumps(r['genes'])+'; matrix='+json.dumps(m,sort_keys=True))
    lines+=['','## Decision','',decision['state'],
        'Recommended calibration arm: '+(recommendation['id'] if recommendation else 'none'),
        'Risk/reward, barrier break-even, realized costs, funding, directional win rates and exit reasons are in the JSON reports.',
        'The tapes provide ideal terminal-direction guidance, not clairvoyant optimal control with every intrabar path known.',
        'A constant activity term or unchanged stop sensitivity is reported as unidentifiable, never as evidence of optimization.',
        'Complete calibrated ledgers are sealed in cache; finalist matrices retain every case and independent coverage checks.','']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines))
    request=['# Final audit request for Satoshi','',
        'Audit FINAL_REPORT.md, DECISION.json, all searches, controls, sensitivity, confirmation and complete finalist matrices.',
        'This replaces the prematurely issued request 100. Do not substitute midpoint evidence for optimized-finalist evidence.',
        'Recompute each objective from ledgers and frozen scales; verify exact DEAP resume and seeded replay.',
        'Check unchanged tapes across genes, control elitism, SL and activity identifiability, RR/cost break-even and both directions.',
        'Verify each finalist JSON against the matrix identity, all 47628 cells and independent closure verification.',
        'Attack output routing, adverse gaps, stops, early closes, calendar, sizing and ledger accounting on disposable copies.',
        'Confirm choice precedes confirmation and no confirmation performance was used for selection.',
        'Return separate execution/optimization/confirmation verdicts and minimal executable counterexamples.',
        'No live trading, TEST, financial fitting, active worker changes or source evidence mutations are requested.',
        'The presence of these documents means all planned stages have terminal evidence, not that their results must be positive.','']
    (out/'FINAL_AUDIT_REQUEST.md').write_text('\n'.join(request))


def run(a):
    import deap
    cfg=read(a.search_config);parent=read(a.source_spec)
    require(cfg['schema']=='multiscale_deap_search.v1','SEARCH_CONFIG')
    require(cfg['population']>=20 and cfg['generations']>=100 and cfg['patience']>=10,'SEARCH_BUDGET')
    require(cfg['seeds']==[7,19,43] and len(cfg['objectives'])==3,'SEARCH_REPLICATION')
    require(1<=a.workers<=8,'WORKER_COUNT')
    require(file_sha(a.recipe)==parent['loading']['recipe_sha256'],'RECIPE_PIN')
    for name,sha in parent['code']['engine'].items():require(file_sha(a.engine_root/name)==sha,'ENGINE_PIN')
    design=read(a.design);require(design['expected_cells']==47628,'FINAL_MATRIX_DESIGN')
    identity=dict(schema='multiscale_deap_strategy.v1',recipe_sha256=file_sha(a.recipe),
        source_spec_sha256=file_sha(a.source_spec),search=cfg,software=software(),deap=deap.__version__,
        sources=source_pins(ROOT,('app','agent_plugins','env_plugins','tools')),
        engine_sources=parent['code']['engine'],design_sha256=file_sha(a.design),
        panels={p:panel(p) for p in ('calibration','confirmation')},smoke=a.smoke,
        financial_data_read=False,training_steps=0)
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.output/'IDENTITY.json').exists():require(read(a.output/'IDENTITY.json')==identity,'IDENTITY')
    else:write(a.output/'IDENTITY.json',identity)
    if (a.output/'CLOSE.json').exists():
        close=read(a.output/'CLOSE.json');require(close['identity_sha256']==digest(identity),'CLOSE_IDENTITY')
        for name,sha in close['artifacts'].items():require(file_sha(a.output/name)==sha,'CLOSE_HASH')
        print(close['state'],flush=True);return
    start=time.monotonic();cache={};count=0;current='CONTROLS'
    pool=ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
        initializer=initialize,initargs=(a.recipe,a.engine_root))
    def evaluate(genes,split='calibration',mode='ideal'):
        nonlocal count
        key=digest(dict(genes=genes,split=split,mode=mode));path=a.output/'cache'/(key+'.json')
        if key in cache:return cache[key]
        if path.exists():payload=unseal(path,identity)
        else:
            rows=list(pool.map(episode,[(cell,genes,mode) for cell in panel(split)],chunksize=1))
            payload=dict(genes=genes,split=split,mode=mode,report=aggregate(rows),episodes=rows)
            seal(path,payload,identity);count+=1
        require(payload['genes']==genes and payload['split']==split and payload['mode']==mode,'CACHE_BINDING')
        # All candidates must consume exactly the same immutable synthetic tape.
        tape_map={digest(r['cell']):r['tape_sha256'] for r in payload['episodes']}
        tape_path=a.output/('TAPES_'+split+'_'+mode+'.json')
        if tape_path.exists():require(read(tape_path)==tape_map,'GENOME_CHANGED_TAPE')
        else:write(tape_path,tape_map)
        result=payload['report'];cache[key]=result
        if count%5==0:
            write(a.output/'STATUS.json',dict(state='SEARCHING',current=current,new_evaluations=count,
                seconds=time.monotonic()-start,seconds_per_evaluation=(time.monotonic()-start)/max(count,1),at=now()))
        return result
    try:
        controls=[dict(id='control_'+str(i),genes=g,report=evaluate(g)) for i,g in enumerate(cfg['controls'])]
        scales={k:max(floor,float(np.median([abs(r['report']['components'][k]) for r in controls
                    if r['report']['components'][k] is not None]))) for k,floor in cfg['normalizer_floors'].items()}
        write(a.output/'CONTROLS.json',controls);write(a.output/'SCALES.json',scales)
        sensitivity=[]
        for i in range(6):
            for end in (0.,1.):
                genes=list(cfg['controls'][0]);genes[i]=end
                sensitivity.append(dict(gene=i,value=end,report=evaluate(genes)))
        write(a.output/'SENSITIVITY.json',sensitivity)
        if a.smoke:
            write(a.output/'SMOKE.json',dict(state='ENGINEERING_ONLY',controls=controls,scales=scales,
                evaluations=count,seconds=time.monotonic()-start));return
        searches=[]
        search_cfg={k:cfg[k] for k in ('population','generations','patience','tolerance')}
        search_cfg['initial_controls']=cfg['controls']
        for seed in cfg['seeds']:
            for name in cfg['objectives']:
                current=f'{name}_s{seed}'
                def evaluator(genes):
                    r=evaluate(genes)
                    return score(name,r['components'],scales) if r['eligible'] else -1e9-len(r['reasons'])
                state=evolve(a.output/'searches'/current,seed,evaluator,search_cfg,identity,objective_name=name)
                r=evaluate(state['best'])
                baseline=max(score(name,c['report']['components'],scales) for c in controls if c['report']['eligible'])
                require(state['best_score']>=baseline-1e-12,'CONTROL_ELITISM')
                item=dict(id=current,objective=name,seed=seed,genes=state['best'],fitness=state['best_score'],
                    generation=state['generation'],stop_reason=state['stop_reason'],report=r)
                seal(a.output/'searches'/current/'RESULT.json',item,identity);searches.append(item)
                write(a.output/'SEARCH_RESULTS.json',searches);print('SEARCH_COMPLETE',current,state['generation'],flush=True)
        chosen=finalists(searches)
        frozen=a.output/'FINALISTS_BEFORE_CONFIRMATION.json'
        if frozen.exists():require(read(frozen)==chosen,'FINALISTS_UNCHANGED')
        else:write(frozen,chosen)
        confirmation=[];current='CONFIRMATION'
        for r in searches+controls:
            confirmation.append(dict(id=r['id'],genes=r['genes'],report=evaluate(r['genes'],'confirmation')))
        for mode in ('inverted','nop'):
            confirmation.append(dict(id='midpoint_'+mode,genes=cfg['controls'][0],
                report=evaluate(cfg['controls'][0],'confirmation',mode)))
        write(a.output/'CONFIRMATION.json',confirmation)
    finally:pool.shutdown(wait=True,cancel_futures=True)
    matrices=[]
    for r in chosen:
        key=digest(r['genes'])
        if any(x['genome_sha256']==key for x in matrices):continue
        folder=a.output/'finalists'/key;config=folder/'STRATEGY.json'
        value=parameter_json(r['genes'])
        if config.exists():require(read(config)==value,'FINALIST_CONFIG')
        else:write(config,value)
        out=folder/'matrix'
        write(a.output/'STATUS.json',dict(state='VERIFYING_FINALIST',current=r['id'],
            matrices_complete=len(matrices),total=len({digest(x['genes']) for x in chosen}),at=now()))
        code=subprocess.run(matrix_command(a,config,out),cwd=ROOT,check=False).returncode
        require((out/'CLOSE.json').exists(),'MATRIX_TERMINAL_RECEIPT')
        summary=read(out/'SUMMARY.json')
        require(summary['cells']==47628,'FULL_MATRIX_COMPLETION')
        require(read(out/'IDENTITY.json')['strategy_genes']==r['genes'],'FINALIST_GENES_CONSUMED')
        if code==0:
            verification=verify(out);write(folder/'VERIFICATION.json',verification)
            state='VERIFIED'
        else:
            require(code==2 and summary['state']=='FAIL','MATRIX_INFRASTRUCTURE_FAILURE')
            state='REJECTED_CONTRACT'
        matrices.append(dict(genome_sha256=key,state=state,terminal=True,cells=summary['cells'],
            failures=summary['failures'],close_sha256=file_sha(out/'CLOSE.json')))
        write(a.output/'MATRICES.json',matrices)
    emit_final(a.output,identity,searches,confirmation,matrices,controls,sensitivity,scales)
    artifacts={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()
        and p.name not in ('CLOSE.json','STATUS.json','RUN.lock')}
    # Bind nested closures too, not just their children.
    artifacts.update({str(p.relative_to(a.output)):file_sha(p) for p in (a.output/'finalists').rglob('CLOSE.json')})
    state=read(a.output/'DECISION.json')['state']
    write(a.output/'CLOSE.json',dict(state=state,identity_sha256=digest(identity),artifacts=artifacts,at=now()))
    write(a.output/'STATUS.json',dict(state='COMPLETE',decision=state,searches=9,matrices=len(matrices),at=now()))
    print('FINAL_PACKET_COMPLETE',state,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','source-spec','engine-root','design','search-config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
