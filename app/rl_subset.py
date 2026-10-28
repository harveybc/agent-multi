"""Structural recipes and paired observations for the actual RL subset evaluator."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd

from app.frozen_recipe_inputs import FrozenRecipeInputs, IDENTITY_KEYS, require, validate_recipe


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal_recipe(rec):
    rec['identity_covers']=list(IDENTITY_KEYS)
    rec['identity']='recipe_'+hashlib.sha256(json.dumps(
        {k:rec[k] for k in IDENTITY_KEYS},sort_keys=True).encode()).hexdigest()[:16]
    validate_recipe(rec)
    return rec


def subset_recipe(parent, columns):
    validate_recipe(parent)
    names=parent['list']['columns']
    require(columns and len(set(columns))==len(columns)
            and columns==[n for n in names if n in columns], 'SUBSET_ORDER')
    take=[names.index(c) for c in columns]
    remap={old:new for new,old in enumerate(take)}
    rec=copy.deepcopy(parent)
    rec['list'].update(columns=list(columns),n=len(columns))
    rec['list']['sha256']=digest(columns)
    rec['tensor']['columns']=[copy.deepcopy(parent['tensor']['columns'][i]) for i in take]
    for key in ('features','mean','std'):
        rec['scaler'][key]=[parent['scaler'][key][i] for i in take]
    rec['scaler'].pop('artifact_sha256',None)
    rec['scaler']['derivation']={'parent_scaler_sha256':digest(parent['scaler']), 'fit':'projected_without_refit'}
    groups=parent['representation']['resolved']
    out={k:[] for k in ('names','indices','units','kinds')}
    for name,idx,units,kind in zip(*(groups[k] for k in out)):
        active=[remap[i] for i in idx if i in remap]
        if active:
            for k,v in zip(out,(name,active,units,kind)): out[k].append(v)
    rec['representation']['resolved']=out
    # Remove stale descriptive maps as well as executable indices.
    rec['representation'].pop('params_per_module',None)
    rec['representation'].pop('groups_file_sha256',None)
    rec['representation']['groups']={
        'name':'structural_subset',
        'groups':{n:[rec['tensor']['columns'][i]['tensor_name'] for i in idx]
                  for n,idx in zip(out['names'],out['indices'])},
        'units':dict(zip(out['names'],out['units'])), 'kinds':dict(zip(out['names'],out['kinds']))}
    rec['provenance']={'parent_recipe_identity':parent['identity'],
                       'derivation':'structural_subset; paired full-parent support; fresh policy required'}
    return seal_recipe(rec)


def project_inputs(parent, recipe):
    take=[parent.recipe['list']['columns'].index(c) for c in recipe['list']['columns']]
    windows=parent.windows[:,:,take].copy()
    windows.flags.writeable=False
    return FrozenRecipeInputs(copy.deepcopy(recipe),parent.row_ts_ns,parent.slot_ts_ns,
                              windows,parent.valid,parent.period)


def refit_train_scaler(parent, frame):
    """Only consumer DIFF and TRAIN constants; no new clipping or rolling transform."""
    rec=copy.deepcopy(parent)
    ts=pd.DatetimeIndex(pd.to_datetime(frame.timestamp,utc=True)) + pd.Timedelta(hours=1)
    a,b=(pd.Timestamp(t,tz='UTC') for t in rec['split']['periods']['TRAIN'])
    mask=(ts>=a)&(ts<b)
    values=frame[rec['list']['columns']].to_numpy(float,copy=True)
    for j,c in enumerate(rec['tensor']['columns']):
        require(c['consumer_ops'] in ([],['DIFF']),'CONSUMER_OP')
        if c['consumer_ops']: values[:,j]=np.r_[np.nan,np.diff(values[:,j])]
    fit=values[mask]
    require(len(fit)>1 and np.isfinite(fit).sum(axis=0).min()>1,'SCALER_TRAIN_SUPPORT')
    mean=np.nanmean(np.where(np.isfinite(fit),fit,np.nan),axis=0)
    std=np.nanstd(np.where(np.isfinite(fit),fit,np.nan),axis=0)
    require(np.isfinite(mean).all() and np.isfinite(std).all(),'SCALER_FINITE')
    rec['scaler']={'policy':'fixed_train','features':[c['tensor_name'] for c in rec['tensor']['columns']],
                   'mean':mean.tolist(),'std':np.where(std>0,std,1).tolist(),
                   'fit_period':rec['split']['periods']['TRAIN'], 'ddof':0,
                   'constant_columns':[c for c,s in zip(rec['list']['columns'],std) if s==0]}
    return seal_recipe(rec)


def support_hash(inputs):
    h=hashlib.sha256()
    for a in (inputs.row_ts_ns,inputs.slot_ts_ns,inputs.valid):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(json.dumps(inputs.period.tolist()).encode())
    return h.hexdigest()


def cost_profile(p):
    return dict(
        spread=dict(rates=[x[2] for x in p['spread_schedule']]+[p['weekly_open_spread']],
                    settings=dict(unit='price',intervals=[x[:2] for x in p['spread_schedule']],
                                  clock='hours_since_session_rollover',last_rate='weekly_open',charge='half_per_fill')),
        slippage=dict(rates=[p['slippage_market'],p['slippage_stop']],settings=dict(unit='price',order=['market','stop'])),
        commission=dict(rates=[p['commission_per_lot_side']],settings=dict(unit='USD_per_lot_per_side',lot_units=p['lot_units'])),
        financing=dict(rates=[p['swap_long_per_lot'],p['swap_short_per_lot']],settings=dict(unit='USD_per_lot_per_rollover',
            lot_units=p['lot_units'],order=['long','short'],triple_index=p['triple_swap_rollover_index'])))


def groups_for_selector(rec):
    g=rec['representation']['resolved']; cols=rec['list']['columns']
    return [dict(name=n,columns=[cols[i] for i in idx],indices=list(idx),extractor=k,units=u)
            for n,idx,u,k in zip(g['names'],g['indices'],g['units'],g['kinds'])]


def coherent_blocks(columns):
    remaining=set(columns);blocks=[]
    for c in columns:
        if c not in remaining: continue
        paired=c[:-4]+'_cos' if c.endswith('_sin') else c[:-4]+'_sin' if c.endswith('_cos') else None
        block=[n for n in columns if n==c or n==paired] if paired in remaining else [c]
        blocks.append(block);remaining.difference_update(block)
    return blocks
