"""Read-only C1-C3/M2 follow-up; writes isolated evidence, never fits a model."""
import argparse
import copy
from pathlib import Path
from collections import Counter

import numpy as np

from app.frozen_recipe_inputs import require, file_sha
from app.rl_subset import digest, support_hash
from app.native_strategy_contract import executed_behavior
from tools.recipe_sac_queue import read, write, now, evaluate
from tools.native_wrapper_audit import local_artifact, ledger_checks
from tools.recipe_hybrid_queue import business_behavior


def trade_economics(trades):
    if not trades:
        return dict(trades=0,binary_break_even_reference=None)
    pip=.0001
    tp=np.array([abs(t['take_profit']-t['entry_price'])/pip for t in trades])
    sl=np.array([abs(t['stop_loss']-t['entry_price'])/pip for t in trades])
    costs=np.array([sum(t[k+'_usd'] for k in ('spread','slippage','commission','financing'))
                    /(t['units']*pip) for t in trades])
    other=sum(t['close_reason'] not in ('TP','SL') for t in trades)
    return dict(trades=len(trades),mean_tp_pips=float(tp.mean()),mean_sl_pips=float(sl.mean()),
        ratio_of_mean_distances=float(tp.mean()/sl.mean()),mean_cost_pips=float(costs.mean()),
        binary_break_even_reference=float((sl.mean()+costs.mean())/(tp.mean()+sl.mean())),
        binary_model_applies_to_all_exits=other==0,non_barrier_exits=other,
        realized_win_rate=sum(t['net_usd']>0 for t in trades)/len(trades),
        net_profit_usd=sum(t['net_usd'] for t in trades),
        financing_usd=sum(t['financing_usd'] for t in trades),
        other_costs_usd=sum(sum(t[k+'_usd'] for k in ('spread','slippage','commission')) for t in trades),
        exit_counts=dict(Counter(t['close_reason'] for t in trades)))


class BlindPolicy:
    def __init__(self,model,rule,seed=0):
        require(rule in ('LONG_HOLD','SHORT_HOLD','COIN_HOLD'),'BLIND_RULE')
        self.policy=model.policy
        self.rule=rule
        self.rng=np.random.default_rng(seed)

    def predict(self,obs,deterministic=True):
        mask=obs['action_mask']
        cmd=0
        if mask[1] or mask[2]:
            cmd=(int(self.rng.integers(1,3)) if self.rule=='COIN_HOLD' else
                 1 if self.rule=='LONG_HOLD' else 2)
            if not mask[cmd]:cmd=0
        return np.array([cmd,.5,.5,.5],np.float32),None


def require_gene_effects(effects):
    missing=[name for name,changed in effects.items() if not changed]
    require(not missing,'UNIDENTIFIED_GENES:'+','.join(missing))


def realized_signature(wave):
    # A different unhit SL changes its recorded level, not execution or reward.
    trades=[{k:v for k,v in t.items() if k!='stop_loss'} for t in wave['trades']]
    return dict(metrics=wave['metrics'],commands=wave['commands'],trades=trades)


def verify_close(root):
    close=read(root/'CLOSE.json')
    for name,sha in close['artifacts'].items():
        path=local_artifact(root/name,root,root)
        require(file_sha(path)==sha,'SOURCE_ARTIFACT:'+name)
    return file_sha(root/'CLOSE.json')


def run(a):
    import torch
    from stable_baselines3 import PPO
    from feature_selector.evaluators import RLModelEvaluator
    from tools.rl_subset_bridge import SubsetMarket
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for source in (a.root,a.study,a.deap,a.engine_root,a.data_root):
        require(not a.output.resolve().is_relative_to(source.resolve()),'OUTPUT_ISOLATION')
    require(not a.output.exists(),'FRESH_AUDIT_OUTPUT')
    a.output.mkdir(parents=True)
    spec=read(a.root/'SPEC.private.json')
    require(file_sha(a.root/'SPEC.private.json')==read(a.root/'IDENTITY.json')['spec_sha256'],'SPEC_PIN')
    for name,sha in spec['code']['engine'].items():
        require(file_sha(a.engine_root/name)==sha,'ENGINE_PIN')
    sources=dict(study_close=verify_close(a.study),deap_close=verify_close(a.deap))
    # Freeze the receipt list; the collector may add later completed units.
    paths=sorted((a.root/'tasks').glob('*-attempt-*/RESULT.json'))
    snapshot={str(p.relative_to(a.root)):file_sha(p) for p in paths}
    write(a.output/'SNAPSHOT.json',snapshot)
    economics=[]
    for path in sorted(a.study.glob('s*/RESULT.json')):
        row=read(path);trades=read(path.parent/'TRAIN.trades.json')
        ledger_checks(row['metrics'],trades)
        economics.append(dict(cell=path.parent.name,seed=row['seed'],parameters=row['parameters'],
                              **trade_economics(trades)))
    write(a.output/'GRID.json',dict(source_close=sources['study_close'],rows=economics,
        scope='derived TRAIN ledger diagnostics, no change to original eligibility',
        caveat='Binary break-even is a reference, not an exact model of variable size, early exits, gaps or financing.'))

    s=copy.deepcopy(spec);s['engine_root']=str(a.engine_root)
    s['inputs'].update(recipe=str(a.root/'RECIPE.json'),
        store=str(a.data_root/'full_pool_s_a1/full_pool_eurusd_h1.parquet'),
        manifest=str(a.data_root/'full_pool_s_a1/MANIFEST.json'),
        bars=str(a.data_root/'admitted/eurusd.parquet'),
        closures=str(a.data_root/'admitted/CLOSURE_INTERVALS.json'))
    market=SubsetMarket(s,spec['dataset']['features'],'configured')
    require(support_hash(market.inputs)==spec['dataset']['support_hash'],'PAIRED_SUPPORT')
    model=PPO.load(paths[0].parent/'model.zip',device='cpu')
    baselines={}
    for mode in ('configured','zero_cost'):
        market.mode=mode
        for period in ('TRAIN','INNER_A'):
            for rule,seed in [('LONG_HOLD',0),('SHORT_HOLD',0)]+[('COIN_HOLD',n) for n in range(5)]:
                key=f'{mode}_{period}_{rule}_{seed}';folder=a.output/'baselines'/key
                report=evaluate(BlindPolicy(model,rule,seed),market,period,folder)
                trades=read(folder/(period+'.trades.json'))
                counts=read(folder/(period+'.orders.json'))['counts']
                ledger_checks(report,trades)
                baselines[key]=dict(metrics=report,economics=trade_economics(trades),
                    behavior=executed_behavior(trades,counts,report),
                    trades_sha256=file_sha(folder/(period+'.trades.json')),
                    equity_sha256=file_sha(folder/(period+'.daily_equity.csv')))
            print(mode,period,'baselines complete',flush=True)
    write(a.output/'BASELINES.json',baselines)
    original_root=Path(spec['inputs']['recipe']).parent
    receipts=[]
    for path in paths:
        native=read(path);task=read(a.root/'tasks'/(path.parent.name.split('-attempt-')[0]+'.json'))
        translated=copy.deepcopy(native)
        for artifact in translated['artifacts']:
            artifact['path']=str(local_artifact(artifact['path'],original_root,a.root))
        RLModelEvaluator().verify(translated,task['request'],task['params'],a.root)
        row=dict(unit=path.parent.name,model=native['model_id'],mode=native['cost_mode'],
                 seed=native['seed'],features=len(native['features']),periods={})
        for period in ('TRAIN','INNER_A'):
            report=read(path.parent/(period+'.json'));trades=read(path.parent/(period+'.trades.json'))
            counts=read(path.parent/(period+'.orders.json'))['counts']
            ledger_checks(report,trades)
            # Reconstruct the historical requested-command rule from its pinned evidence.
            requested=business_behavior(report,counts)['business_behavior_eligible']
            strict=executed_behavior(trades,counts,report)
            comparisons={}
            for rule,seed in [('LONG_HOLD',0),('SHORT_HOLD',0)]+[('COIN_HOLD',n) for n in range(5)]:
                b=baselines[f'{native["cost_mode"]}_{period}_{rule}_{seed}']
                comparisons[f'{rule}_{seed}']=dict(
                    trades_exact=file_sha(path.parent/(period+'.trades.json'))==b['trades_sha256'],
                    equity_exact=file_sha(path.parent/(period+'.daily_equity.csv'))==b['equity_sha256'])
            row['periods'][period]=dict(requested_command_behavior=requested,
                requested_with_native_activity=requested and report['valid'] and report['orders_per_week']>=2,
                strict=strict,comparisons=comparisons)
        receipts.append(row)
    write(a.output/'RECEIPTS.json',receipts)

    confirmation=read(a.deap/'CONFIRMATION.json')
    result=read(a.deap/'SEARCH_RESULTS.json')
    comparison=read(a.deap/'COMPARISON.json')
    comparison['controls']=[r for r in confirmation if r['id'] in ('midpoint','grid_control')]
    comparison['scientific_disposition']='NOT_ADMITTED_FOR_FITNESS_OR_RISK_SELECTION'
    write(a.output/'COMPARISON.json',comparison)
    touches=0
    for p in (a.deap/'cache').glob('*.json'):
        waves=read(p)['payload']['waves']
        touches+=sum(w['metrics']['n_close_SL'] for w in waves if w['mode']=='ideal')
    from app.directional_risk_calibration import DirectionalRisk,DirectionalOrders
    from tools.native_strategy_contract_study import oracle
    chosen=next(r for r in result if r['id']=='sharpe_return_s7')
    base=read(a.deap/'cache'/(digest(chosen['genes'])+'.json'))['payload']['waves']
    sensitivity=[]
    for index,name in ((1,'sl_long'),(4,'sl_short')):
        for endpoint in (0.,1.):
            genes=chosen['genes'].copy();genes[index]=endpoint;waves=[]
            for wave in read(a.deap/'IDENTITY.json')['calibration']:
                waves.extend(oracle(market.base.config,market.inputs.windows.shape[1:],*wave,
                    DirectionalRisk(tuple(genes)),DirectionalOrders,False))
            # The identified objective uses ideal tapes; inverted controls remain recorded separately.
            equal=all(realized_signature(a)==realized_signature(b)
                      for a,b in zip(waves[::3],base[::3]))
            sensitivity.append(dict(gene=name,endpoint=endpoint,ideal_trajectory_exact=equal))
    write(a.output/'SENSITIVITY.json',dict(ideal_sl_touches=touches,probes=sensitivity,
          calibrated_sl=False,activity_identified=False))
    for name,sha in snapshot.items():require(file_sha(a.root/name)==sha,'SNAPSHOT_MUTATED')
    tcn=[r for r in receipts if r['model']=='tcn' and r['mode']=='configured']
    summary=dict(finished_at=now(),source_closures=sources,receipts=len(receipts),
        tcn_configured=len(tcn),
        long_hold_exact={period:sum(r['periods'][period]['comparisons']['LONG_HOLD_0']['trades_exact']
                        for r in tcn) for period in ('TRAIN','INNER_A')},
        regate_changes={period:sum(r['periods'][period]['requested_with_native_activity']!=
                          r['periods'][period]['strict']['eligible'] for r in receipts)
                          for period in ('TRAIN','INNER_A')},
        ideal_sl_touches=touches,sl_sensitivity=sensitivity,financial_admission=False,new_fits=0,
        positive_learning_control='NOT_EXECUTED',selection_scope='INNER_A is reused selection, not independent confirmation')
    write(a.output/'SUMMARY.json',summary)
    write(a.output/'CLOSE.json',dict(state='COMPLETE_EVIDENCE_CORRECTIONS_NOT_FINANCIAL_RELEASE',
        artifacts={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}))
    print(summary,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','study','deap','engine-root','data-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
