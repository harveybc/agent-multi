"""Resumable CPU grid on simulated heads; no financial data or model training."""
import argparse
from dataclasses import asdict
import fcntl
import itertools
from pathlib import Path

import numpy as np

from app.frozen_recipe_inputs import file_sha, require
from app.rl_subset import digest
from tools.recipe_sac_queue import read, write, now, software


def assess(parameters,waves):
    require(len(waves)==6 and [r['mode'] for r in waves]==['ideal','inverted','nop']*2,
            'WAVE_SET')
    reasons=[]
    ideals=waves[::3]
    for i in (0,3):
        m=waves[i]['metrics']; inv=waves[i+1]['metrics']; nop=waves[i+2]['metrics']
        if not m['valid']: reasons.append(f'INVALID_{i}')
        if m['orders_per_week']<2: reasons.append(f'LOW_ACTIVITY_{i}')
        if not(m['n_long']>0 and m['n_short']>0): reasons.append(f'DIRECTION_{i}')
        if m['net_profit_usd']<=0: reasons.append(f'NONPOSITIVE_IDEAL_{i}')
        if not m['net_profit_usd']>inv['net_profit_usd']: reasons.append(f'INVERSION_{i}')
        if nop['trades']!=0 or nop['net_profit_usd']!=0: reasons.append(f'NOP_{i}')
        require(all(np.isfinite(m[k]) for k in ('net_profit_usd','sharpe_daily_sqrt365',
                    'max_drawdown_pct','orders_per_week')),'FINITE_METRICS')
    for side,name in ((1,'LONG'),(-1,'SHORT')):
        if not any(t['direction']==side and t['close_reason']=='AGENT_CLOSE'
                   for r in ideals for t in r['trades']):
            reasons.append('NO_EXECUTED_CLOSE_'+name)
    return dict(parameters=parameters,eligible=not reasons,reasons=reasons,
        mean_sharpe=float(np.mean([r['metrics']['sharpe_daily_sqrt365'] for r in ideals])),
        mean_drawdown_pct=float(np.mean([r['metrics']['max_drawdown_pct'] for r in ideals])),
        mean_profit_usd=float(np.mean([r['metrics']['net_profit_usd'] for r in ideals])))


def choose(rows):
    allowed=[r for r in rows if r['eligible']]
    return min(allowed,key=lambda r:(-r['mean_sharpe'],r['mean_drawdown_pct'],
        r['parameters']['order_fraction'],r['parameters']['tp_fraction'],
        r['parameters']['sl_fraction'])) if allowed else None


def checked(path,identity):
    value=read(path)
    require(value['identity_sha256']==digest(identity),'CELL_IDENTITY')
    require(value['payload_sha256']==digest(value['payload']),'CELL_PAYLOAD')
    return value['payload']


def run(a):
    from app.recipe_runtime import attach_engine
    attach_engine(a.engine_root)
    from app.short_horizon.expert_guided.env import EnvConfig
    from app.short_horizon.execution import Sizing,RealisticProfile,OrderRules
    from app.native_strategy_contract import RiskParameters
    from tools.native_strategy_contract_study import oracle

    spec=read(a.root/'SPEC.private.json')
    require(file_sha(a.root/'SPEC.private.json')==read(a.root/'IDENTITY.json')['spec_sha256'],'SPEC_PIN')
    require(spec['software']==software(),'SOFTWARE_PIN')
    here=Path(__file__).resolve().parents[1]
    for role,base in [('agent',here),('engine',a.engine_root)]:
        for name,sha in spec['code'][role].items():require(file_sha(base/name)==sha,'SOURCE_PIN:'+name)
    r=read(a.root/'RECIPE.json')
    cfg=dict(r['business']['evaluation_and_policy'])
    cfg.update(sizing=Sizing(**cfg['sizing']),profile=RealisticProfile(**cfg['profile']),rules=OrderRules(**cfg['rules']))
    config=EnvConfig(**cfg); shape=(13,r['list']['n'])
    grid=[RiskParameters(*v) for v in itertools.product((.25,.5,.75),(.25,.5,.75),(.2,.5,1.))]
    grid.sort(key=lambda p:p!=RiskParameters())
    identity=dict(schema='synthetic_output_calibration.v1',recipe_sha256=file_sha(a.root/'RECIPE.json'),
        spec_sha256=file_sha(a.root/'SPEC.private.json'),software=software(),
        source_hashes={name:file_sha(here/name) for name in
          ('tools/synthetic_output_calibration.py','tools/native_strategy_contract_study.py',
           'app/native_strategy_contract.py','tools/native_wrapper_audit.py','tools/recipe_sac_queue.py')},
        grid=[asdict(p) for p in grid],calibration_waves=[[.002,8,0],[.0015,6,3]],
        confirmation_wave=[.0018,7,7],selection='eligible; mean Sharpe desc, DD asc, q asc, TP asc, SL asc',
        parent_waves_sha256=file_sha(a.previous/'IDEAL_WAVES.json'),financial_data_read=False,fits=0)
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.output/'IDENTITY.json').exists():require(read(a.output/'IDENTITY.json')==identity,'STUDY_IDENTITY')
    else:write(a.output/'IDENTITY.json',identity)
    if (a.output/'CLOSE.json').exists():
        close=read(a.output/'CLOSE.json')
        require(close['identity_sha256']==digest(identity),'CLOSE_IDENTITY')
        for name,sha in close['artifacts'].items():require(file_sha(a.output/name)==sha,'CLOSE_ARTIFACT')
        print('COMPLETE_VERIFIED_NO_REEXECUTION',flush=True);return
    rows=[]
    for parameters in grid:
        p=asdict(parameters);path=a.output/('cell_'+digest(p)[:16]+'.json')
        if path.exists():
            waves=checked(path,identity)
        else:
            waves=[]
            for wave in identity['calibration_waves']:waves.extend(oracle(config,shape,*wave,parameters))
            write(path,dict(identity_sha256=digest(identity),payload_sha256=digest(waves),payload=waves))
        if parameters==RiskParameters():require(waves==read(a.previous/'IDEAL_WAVES.json'),'EXACT_OLD_CONTROLS')
        row=assess(p,waves);row['artifact']=path.name;rows.append(row)
        write(a.output/'STATUS.json',dict(state='CALIBRATING',cells=len(rows),total=27,at=now()))
        print(len(rows),p,row['eligible'],row['mean_sharpe'],flush=True)
    best=choose(rows)
    write(a.output/'GRID.json',rows)
    write(a.output/'DECISION.json',dict(state='SYNTHETIC_ONLY_SELECTED' if best else 'NO_FEASIBLE_CANDIDATE',
        selected=best,eligible=sum(r['eligible'] for r in rows),candidates=27,calibration_simulations=162,
        financial_admission=False))
    confirmation=[]
    for role,p in [('chosen_or_control',best['parameters'] if best else asdict(RiskParameters())),
                   ('midpoint_control',asdict(RiskParameters()))]:
        confirmation.append(dict(role=role,parameters=p,
            waves=oracle(config,shape,*identity['confirmation_wave'],RiskParameters(**p))))
    write(a.output/'CONFIRMATION.json',confirmation)
    artifacts={p.name:file_sha(p) for p in a.output.glob('*.json') if p.name not in ('CLOSE.json','STATUS.json')}
    for name,sha in artifacts.items():require(file_sha(a.output/name)==sha,'READBACK')
    write(a.output/'CLOSE.json',dict(state='COMPLETE',identity_sha256=digest(identity),artifacts=artifacts,
        calibration_simulations=162,confirmation_simulations=6,at=now()))
    write(a.output/'STATUS.json',dict(state='COMPLETE',cells=27,simulations=168,at=now()))
    print('COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','engine-root','previous','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    require(not a.output.resolve().is_relative_to(a.root.resolve()),'OUTPUT_ISOLATION')
    require(not a.output.resolve().is_relative_to(a.previous.resolve()),'PREVIOUS_ISOLATION')
    run(a)


if __name__=='__main__':main()
