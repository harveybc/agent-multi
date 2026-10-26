"""Deliberate-lookahead diagnostic inputs. Never a financial feature recipe."""
import copy
from dataclasses import replace

import numpy as np

from app.frozen_recipe_inputs import require
from app.rl_subset import seal_recipe, support_hash


def future_move(opens,closes,periods,horizon=4):
    o,c,p=np.asarray(opens),np.asarray(closes),np.asarray(periods)
    require(o.shape==c.shape==p.shape and o.ndim==1,'C4_BAR_SHAPES')
    require(type(horizon) is int and horizon>0,'C4_HORIZON')
    out=np.full(len(o),np.nan)
    for b in range(len(o)-horizon):
        if p[b] in ('TRAIN','INNER_A') and np.all(p[b:b+horizon+1]==p[b]):
            out[b]=c[b+horizon]-o[b+1]
    return out


def control_inputs(inputs,signal,arm,permutation_seed=20261026):
    require(arm in ('original','planted','permuted'),'C4_ARM')
    signal=np.asarray(signal,dtype=float)
    require(signal.shape==inputs.valid.shape,'C4_SIGNAL_SHAPE')
    train=(inputs.period=='TRAIN') & inputs.valid & np.isfinite(signal)
    require(train.sum()>1,'C4_TRAIN_SIGNAL')
    mean=float(signal[train].mean());std=float(signal[train].std())
    require(np.isfinite(std) and std>0,'C4_SIGNAL_VARIANCE')
    z=(signal-mean)/std
    if arm=='permuted':
        z=z.copy()
        for offset,period in enumerate(('TRAIN','INNER_A')):
            rows=np.flatnonzero((inputs.period==period)&np.isfinite(z))
            z[rows]=np.random.default_rng(permutation_seed+offset).permutation(z[rows])
    slots=np.searchsorted(inputs.row_ts_ns,inputs.slot_ts_ns)
    require((slots<len(inputs.row_ts_ns)).all() and
            np.array_equal(inputs.row_ts_ns[slots],inputs.slot_ts_ns),'C4_SLOT_ALIGNMENT')
    common=inputs.valid & np.isfinite(signal[slots]).all(axis=1) & np.isin(inputs.period,['TRAIN','INNER_A'])
    windows=inputs.windows if arm=='original' else inputs.windows.copy()
    rec=copy.deepcopy(inputs.recipe)
    if arm!='original':
        windows[:,:,0]=z[slots]
        name='DIAGNOSTIC_FUTURE_MOVE_NOT_FINANCIAL'
        rec['tensor']['columns'][0].update(store_column=name,tensor_name=name,consumer_ops=[])
        rec['list']['columns'][0]=name
        rec['scaler']['features'][0]=name;rec['scaler']['mean'][0]=mean;rec['scaler']['std'][0]=std
    rec['labels']['c4_diagnostic']=dict(arm=arm,deliberate_lookahead=True,financial_admission=False)
    seal_recipe(rec)
    for a in (windows,common):a.flags.writeable=False
    result=replace(inputs,recipe=rec,windows=windows,valid=common)
    return result,dict(arm=arm,mean=mean,std=std,train_scaler_rows=int(train.sum()),
        common_support_hash=support_hash(result),
        rows={p:int(np.sum(common&(inputs.period==p))) for p in ('TRAIN','INNER_A')},
        signal='close[b+4]-open[b+1]; four observed bars, same split',
        permutation_seed=permutation_seed,financial_admission=False)


def qualify(rows):
    expected={(a,s) for s in (7,19,43) for a in ('original','planted','permuted')}
    require(len(rows)==9 and {(r['arm'],r['seed']) for r in rows}==expected,'NINE_CONTROL_UNITS')
    failures=[]
    planted=[r for r in rows if r['arm']=='planted']
    for row in planted:
        for period,r in row['periods'].items():
            score=r['metrics']['sharpe_daily_sqrt365']
            if not r['strict']['eligible'] or score is None or not np.isfinite(score) or score<=0:
                failures.append(f'{row["seed"]}:{period}:POSITIVE_LEARNING_NOT_DEMONSTRATED')
    scores={a:[r['periods']['INNER_A']['metrics']['sharpe_daily_sqrt365'] for r in rows if r['arm']==a]
            for a in ('original','planted','permuted')}
    if all(v is not None and np.isfinite(v) for values in scores.values() for v in values):
        means={a:float(np.mean(v)) for a,v in scores.items()}
        if not means['planted']>max(means['original'],means['permuted']):
            failures.append('PLANTED_NOT_ABOVE_BOTH_CONTROLS')
    else:
        means=None;failures.append('UNMEASURED_COMPARATOR')
    return dict(state='POSITIVE_CONTROL_PASSED' if not failures else 'LEARNING_QUALIFICATION_FAILED',
        failures=failures,mean_selection_sharpe=means,financial_admission=False,
        scope='diagnostic future information, not market forecasting or an admitted feature ranking')
