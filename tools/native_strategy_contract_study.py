"""Finite CPU output-contract witnesses and exhaustive TRAIN-only risk calibration."""
import argparse
import copy
from dataclasses import asdict, replace
import fcntl
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.frozen_recipe_inputs import require, file_sha
from app.native_strategy_contract import RiskParameters, CalibratedOrders, executed_behavior
from app.rl_subset import digest
from tools.recipe_sac_queue import read, write, now, clean, evaluate, tensor_hash, software
from tools.native_wrapper_audit import local_artifact, ledger_checks


def synthetic_inputs(ts,shape):
    windows=np.zeros((len(ts),*shape),np.float32)
    return SimpleNamespace(row_ts_ns=(ts+__import__('pandas').Timedelta(hours=1)).as_unit('ns').asi8,
        valid=np.ones(len(ts),bool),windows=windows,
        observation=lambda row:dict(features=windows[row].copy()))


def summary_env(env):
    from app.short_horizon.expert_guided.env import summary
    report=clean(summary(env.engine.result()))
    report.update(n_long=sum(t['direction']==1 for t in env.engine.trades),
                  n_short=sum(t['direction']==-1 for t in env.engine.trades))
    ledger_checks(report,env.engine.trades)
    return dict(metrics=report,commands=env.order_counts,trades=clean(env.engine.trades))


def witnesses(config,shape):
    import pandas as pd
    from app.short_horizon.expert_guided.env import BarEnv
    from app.short_horizon.expert_guided.data import calendar_from_closures
    from env_plugins.hybrid_orders import HybridOrderEnv
    ts=pd.date_range('2020-01-05T22:00Z',periods=120,freq='h')
    cal=calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],
        ['2020-01-10T22:00Z','2020-01-12T22:00Z']],delay=12,flat=12,cutoff=0,calendar_id='SYNTHETIC')
    cases=[]
    for side in (1,-1):
        for scenario in ('close','SL','TP','both','gap','forced','daily_cap','invalid',
                         'missing_window','close_before_SL','close_before_TP'):
            bars=pd.DataFrame(dict(OPEN=1.1,HIGH=1.10001,LOW=1.09999,CLOSE=1.1),index=ts)
            engine=BarEnv(bars,cal,np.full(120,.001),np.full(120,.002),ts[0],
                          ts[-1]+pd.Timedelta(hours=1),config)
            env=HybridOrderEnv(engine,synthetic_inputs(ts,shape),gamma=.99,learned_risk=False)
            obs,_=env.reset()
            require(not obs['action_mask'][1],'REOPEN_DELAY')
            def step(cmd):return env.step([cmd,.5,.5,.5])
            while not obs['action_mask'][1]:obs,*_=step(0)
            before=env.bar
            obs,*_=step(1 if side==1 else 2)
            pos=engine.pos
            require(pos is not None and pos.entry_bar==before+1,'NEXT_OPEN')
            require(pos.units==config.sizing.order_units(config.capital,1.1,1.),'ACTUAL_SIZING')
            require(not obs['action_mask'][1] and not obs['action_mask'][2],'NO_PYRAMID_OR_REVERSE')
            k=env.bar+1
            if scenario in ('SL','TP','both','gap','close_before_SL','close_before_TP'):
                kind=scenario.removeprefix('close_before_')
                lo,hi=1.09999,1.10001
                for level in ([pos.sl,pos.tp] if kind=='both' else [pos.tp if kind=='TP' else pos.sl]):
                    through=2*config.profile.limit_through+.00001
                    lo=min(lo,level-through);hi=max(hi,level+through)
                # Only extend the adverse/favorable side, never cross both accidentally.
                if kind!='both':
                    level=pos.tp if kind=='TP' else pos.sl
                    lo=level-through if level<1.1 else 1.09999
                    hi=level+through if level>1.1 else 1.10001
                o=c=1.1
                if kind=='gap':o=c=pos.sl-side*.001;lo=hi=o
                for key,value in zip(('OPEN','HIGH','LOW','CLOSE'),(o,hi,lo,c)):
                    engine.bars.iloc[k,engine.bars.columns.get_loc(key)]=value
                engine.o,engine.h,engine.l,engine.c=(engine.bars[x].to_numpy(float) for x in ('OPEN','HIGH','LOW','CLOSE'))
            close=3 if side==1 else 4
            if scenario=='invalid':
                for cmd in (1.2,2 if side==1 else 1,4 if side==1 else 3):
                    old=len(engine.trades)
                    try:step(cmd)
                    except ValueError:pass
                    else:raise ValueError('INVALID_COMMAND_ACCEPTED')
                    require(len(engine.trades)==old,'INVALID_COMMAND_MUTATED_TRADES')
            if scenario=='missing_window':env.valid[env.bar+1:env.bar+4]=False
            if scenario in ('close','daily_cap','close_before_SL','close_before_TP'):
                obs,*_=step(close)
                require(engine.trades[-1]['close_reason']=='AGENT_CLOSE','DISCRETIONARY_CLOSE')
                if scenario=='daily_cap':require(obs['action_mask'].tolist()==[1,0,0,0,0],'DAILY_CAP')
            else:
                _,_,_,_,info=step(0)
                if scenario=='missing_window':require(info['elapsed_bars']==4,'MISSING_WINDOW_HOLD')
            while not env.finished:step(0)
            require(engine.pos is None and len(engine.trades)==1,'TERMINAL_FLAT')
            t=engine.trades[0]
            expected=('AGENT_CLOSE' if scenario in ('close','daily_cap','close_before_SL','close_before_TP')
                      else 'SL' if scenario in ('SL','both','gap') else 'TP' if scenario=='TP' else 'FORCED_FLAT')
            require(t['close_reason']==expected,'SCENARIO_EXIT:'+scenario)
            if scenario=='both':require(t['ambiguous'],'SL_FIRST')
            if scenario=='gap':require(t['exit_price']==engine.o[k],'GAP_AT_OPEN')
            if expected=='FORCED_FLAT':require(t['exit_time'].value<=engine.ff[pos.seg],'FORCED_DEADLINE')
            cases.append(dict(direction=side,scenario=scenario,**summary_env(env)))
    for side in (1,-1):
        by={r['scenario']:r['metrics']['net_profit_usd'] for r in cases if r['direction']==side}
        require(by['close_before_SL']>by['SL'] and by['TP']>by['close_before_TP'],'CLOSE_HOLD_ORDERING')
    return cases


def oracle(config,shape,amplitude,weeks,phase,parameters=RiskParameters(),
           env_type=CalibratedOrders,verify_controls=True):
    """Known synthetic future drives only the diagnostic action generator."""
    import pandas as pd
    from app.short_horizon.expert_guided.env import BarEnv
    from app.short_horizon.expert_guided.data import calendar_from_closures
    from app.short_horizon.zigzag import causal_sigma,causal_range_sigma
    ts=pd.DatetimeIndex([t for w in range(weeks) for t in
        pd.date_range(pd.Timestamp('2020-01-05T22:00Z')+pd.Timedelta(weeks=w),periods=120,freq='h')])
    closures=[[str(pd.Timestamp('2020-01-03T22:00Z')+pd.Timedelta(weeks=w)),
               str(pd.Timestamp('2020-01-05T22:00Z')+pd.Timedelta(weeks=w))] for w in range(weeks+1)]
    cal=calendar_from_closures(closures,delay=12,flat=12,cutoff=0,calendar_id='SYNTHETIC_WAVE')
    n=np.arange(len(ts)); curve=1.1+amplitude*np.sin(2*np.pi*(n+phase)/24)+.1*amplitude*np.sin(2*np.pi*(n+phase)/6)
    o=curve;c=1.1+amplitude*np.sin(2*np.pi*(n+1+phase)/24)+.1*amplitude*np.sin(2*np.pi*(n+1+phase)/6)
    bars=pd.DataFrame(dict(OPEN=o,HIGH=np.maximum(o,c)+.00001,LOW=np.minimum(o,c)-.00001,CLOSE=c),index=ts)
    reports=[]
    for mode in ('ideal','inverted','nop'):
        engine=BarEnv(bars,cal,causal_sigma(bars),causal_range_sigma(bars),ts[0],ts[-1]+pd.Timedelta(hours=1),config)
        env=env_type(engine,synthetic_inputs(ts,shape),parameters)
        obs,_=env.reset()
        while not env.finished:
            b=env.bar;k=b+1;future=min(k+4,engine.last)
            delta=engine.c[future]-engine.o[k]
            if mode=='inverted':delta=-delta
            cmd=0
            if mode!='nop':
                if engine.pos is not None:
                    if engine.pos.direction*delta<0:cmd=3 if engine.pos.direction>0 else 4
                elif abs(delta)>2*config.profile.round_trip_price():
                    desired=1 if delta>0 else 2
                    if obs['action_mask'][desired]:cmd=desired
            obs,*_=env.step([cmd,.5,.5,.5])
        reports.append(dict(mode=mode,amplitude=amplitude,weeks=weeks,phase=phase,**summary_env(env)))
    require(reports[2]['metrics']['trades']==0 and reports[2]['metrics']['net_profit_usd']==0,'NOP_CASH')
    if verify_controls:
        require(reports[0]['metrics']['net_profit_usd']>reports[1]['metrics']['net_profit_usd'],'ORACLE_CONTROL_ORDERING')
    return reports


class TapePolicy:
    def __init__(self,model,market,logits):
        self.policy=model.policy;self.market=market;self.logits=logits

    def predict(self,obs,deterministic=True):
        row=self.market.active.rows[self.market.active.bar]
        z=self.logits[row]
        require(np.isfinite(z).all(),'LOGIT_SUPPORT')
        cmd=int(np.argmax(np.where(obs['action_mask'].astype(bool),z,-np.inf)))
        return np.array([cmd,.5,.5,.5],np.float32),None


class CalibratedMarket:
    def __init__(self,source,parameters):
        self.source=source;self.recipe=source.recipe;self.parameters=parameters;self.active=None

    def env(self,period):
        original=self.source.env(period)
        self.active=CalibratedOrders(original.engine,self.source.inputs,self.parameters)
        return self.active


def run(a):
    import torch
    from stable_baselines3 import PPO
    from feature_selector.evaluators import RLModelEvaluator
    from tools.rl_subset_bridge import SubsetMarket
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=read(a.root/'SPEC.private.json');sha=file_sha(a.root/'SPEC.private.json')
    require(sha==read(a.root/'IDENTITY.json')['spec_sha256'],'SPEC_PIN')
    require(spec['software']==software(),'SOFTWARE_PIN')
    runtime=Path(__file__).resolve().parents[1]
    for role,base in [('agent',runtime),('engine',a.engine_root),('selector',a.selector_root)]:
        for name,value in spec['code'][role].items():require(file_sha(base/name)==value,'SOURCE_PIN:'+role+':'+name)
    original_root=Path(spec['inputs']['recipe']).parent
    s=copy.deepcopy(spec);s['engine_root']=str(a.engine_root)
    s['inputs'].update(recipe=str(a.root/'RECIPE.json'),store=str(a.data_root/'full_pool_s_a1/full_pool_eurusd_h1.parquet'),
        manifest=str(a.data_root/'full_pool_s_a1/MANIFEST.json'),bars=str(a.data_root/'admitted/eurusd.parquet'),
        closures=str(a.data_root/'admitted/CLOSURE_INTERVALS.json'))
    market=SubsetMarket(s,spec['dataset']['features'],'configured')
    grid=[RiskParameters(*v) for v in itertools.product((.25,.5,.75),(.25,.5,.75),(.2,.5,1.))]
    grid.sort(key=lambda p:p!=RiskParameters())
    identity=dict(parent_spec_sha256=sha,tool_sha256=file_sha(Path(__file__)),
        contract_sha256=file_sha(runtime/'app/native_strategy_contract.py'),
        grid=[asdict(p) for p in grid],seeds=[7,19,43],period='TRAIN',fits=0,test_access=False,
        selection='all three seeds executed behavior and native valid; mean TRAIN Sharpe, then smaller DD and size')
    if (a.output/'IDENTITY.json').exists():require(read(a.output/'IDENTITY.json')==identity,'STUDY_IDENTITY')
    else:write(a.output/'IDENTITY.json',identity)
    shape=market.inputs.windows.shape[1:]
    cases=witnesses(market.base.config,shape)
    write(a.output/'SCENARIOS.json',cases)
    waves=oracle(market.base.config,shape,.002,8,0)+oracle(market.base.config,shape,.0015,6,3)
    write(a.output/'IDEAL_WAVES.json',waves)
    completed=[]
    for seed in identity['seeds']:
        matches=[]
        for p in (a.root/'tasks').glob('*-attempt-*/RESULT.json'):
            value=read(p)
            if value['seed']==seed and value['model_id']=='tcn' and value['cost_mode']=='configured' and value['features']==spec['dataset']['features']:
                matches.append(p)
        require(len(matches)==1,'UNIQUE_PARENT_CONTROL')
        parent=matches[0];native=read(parent)
        task=read(a.root/'tasks'/(parent.parent.name.split('-attempt-')[0]+'.json'))
        translated=copy.deepcopy(native)
        for artifact in translated['artifacts']:artifact['path']=str(local_artifact(artifact['path'],original_root,a.root))
        RLModelEvaluator().verify(translated,task['request'],task['params'],a.root)
        model=PPO.load(parent.parent/'model.zip',device='cpu');model.policy.set_training_mode(False)
        before=tensor_hash(model);require(before==read(parent.parent/'TRAINING.json')['final_weights'],'WEIGHTS')
        idx=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
        logits=np.full((len(market.inputs.valid),5),np.nan,np.float32)
        with torch.no_grad():
            for start in range(0,len(idx),128):
                rows=idx[start:start+128]
                obs={'features':torch.as_tensor(market.inputs.windows[rows].copy())}
                latent=model.policy.features_extractor(obs)
                pi,_=model.policy.mlp_extractor(latent)
                logits[rows]=model.policy.commands(pi).numpy()
        for parameters in grid:
            key=f's{seed}_'+digest(asdict(parameters))[:16];folder=a.output/key;receipt=folder/'RESULT.json'
            if receipt.exists():
                r=read(receipt);require(r['identity_sha256']==digest(identity) and r['parent_sha256']==file_sha(parent),'CELL_PIN')
                for name,value in r['artifacts'].items():require(file_sha(local_artifact(folder/name,folder,folder))==value,'CELL_ARTIFACT')
                completed.append(r);continue
            write(a.output/'STATUS.json',dict(state='CALIBRATING',completed=len(completed),total=81,unit=key,heartbeat_at=now()))
            m=CalibratedMarket(market,parameters);policy=TapePolicy(model,m,logits)
            report=evaluate(policy,m,'TRAIN',folder)
            trades=read(folder/'TRAIN.trades.json');counts=read(folder/'TRAIN.orders.json')['counts']
            ledger_checks(report,trades)
            if parameters==RiskParameters():
                require(file_sha(folder/'TRAIN.trades.json')==file_sha(parent.parent/'TRAIN.trades.json'),'CONTROL_TRADES_EXACT')
                require(file_sha(folder/'TRAIN.daily_equity.csv')==file_sha(parent.parent/'TRAIN.daily_equity.csv'),'CONTROL_EQUITY_EXACT')
            require(tensor_hash(model)==before,'NO_WEIGHT_UPDATE')
            r=dict(seed=seed,parameters=asdict(parameters),identity_sha256=digest(identity),parent_sha256=file_sha(parent),
                metrics=report,behavior=executed_behavior(trades,counts,report),
                artifacts={p.name:file_sha(p) for p in folder.iterdir() if p.is_file()},finished_at=now())
            write(receipt,r);completed.append(r)
            print(seed,parameters,report['net_profit_usd'],r['behavior'],flush=True)
        del model
    grouped=[]
    for p in grid:
        rows=[r for r in completed if r['parameters']==asdict(p)]
        require(sorted(r['seed'] for r in rows)==[7,19,43],'ALL_SEEDS')
        grouped.append(dict(parameters=asdict(p),eligible=all(r['behavior']['eligible'] for r in rows),
            mean_sharpe=float(np.mean([r['metrics']['sharpe_daily_sqrt365'] for r in rows])),
            mean_drawdown_pct=float(np.mean([r['metrics']['max_drawdown_pct'] for r in rows])),
            mean_profit_usd=float(np.mean([r['metrics']['net_profit_usd'] for r in rows])),
            seeds=[dict(seed=r['seed'],behavior=r['behavior'],metrics=r['metrics']) for r in rows]))
    allowed=[r for r in grouped if r['eligible']]
    best=max(allowed,key=lambda r:(r['mean_sharpe'],-r['mean_drawdown_pct'],-r['parameters']['order_fraction'])) if allowed else None
    write(a.output/'GRID.json',grouped)
    write(a.output/'DECISION.json',dict(state='SELECTED_FOR_SEPARATE_CONFIRMATION' if best else 'NO_FEASIBLE_CANDIDATE',
        selected=best,total_candidates=len(grouped),completed_replays=len(completed),test_access=False,financial_validation_read=False))
    write(a.output/'CLOSE.json',dict(state='COMPLETE',identity_sha256=digest(identity),finished_at=now(),
        artifacts={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*.json') if p.name not in ('CLOSE.json','STATUS.json')},
        scientific_fits=0,financial_test_access=False))
    write(a.output/'STATUS.json',dict(state='COMPLETE',completed=len(completed),finished_at=now()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','data-root','engine-root','selector-root'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.output.resolve().is_relative_to(a.root.resolve()),'OUTPUT_ISOLATION')
    run(a)


if __name__=='__main__':main()
