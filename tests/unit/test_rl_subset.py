import copy

import numpy as np
import pandas as pd
import pytest
import torch
from gymnasium import spaces

from tests.unit.test_frozen_recipe_inputs import recipe, seal, frame
from app.frozen_recipe_inputs import FrozenRecipeInputs
from app.rl_subset import subset_recipe, project_inputs, refit_train_scaler, cost_profile
from agent_plugins.grouped_tcn_orders import GroupedTCNExtractor


def split_recipe():
    r=recipe()
    r['representation']['resolved']=dict(names=['a','b'],indices=[[0],[1]],units=[2,2],kinds=['conv','pointwise'])
    r['split']['periods']={'TRAIN':['2020-01-01','2020-01-04'], 'INNER_A':['2020-01-04','2024-01-01']}
    return seal(r)


def test_structural_pruning_and_reintroduction():
    r=split_recipe()
    s=subset_recipe(r,['y'])
    assert s['list']['columns']==['y'] and s['scaler']['mean']==[3.]
    assert s['representation']['resolved']==dict(names=['b'],indices=[[0]],units=[2],kinds=['pointwise'])
    assert subset_recipe(r,['x','y'])['representation']['resolved']==r['representation']['resolved']
    assert r['list']['columns']==['x','y']
    with pytest.raises(ValueError): subset_recipe(r,['y','x'])


def test_projection_keeps_support_even_after_removing_nan_column():
    r=split_recipe();f=frame();f.loc[8,'x']=np.nan
    full=FrozenRecipeInputs.from_frame(r,f)
    projected=project_inputs(full,subset_recipe(r,['y']))
    np.testing.assert_array_equal(projected.valid,full.valid)
    np.testing.assert_array_equal(projected.row_ts_ns,full.row_ts_ns)
    np.testing.assert_array_equal(projected.windows,full.windows[:,:,[1]])
    assert not projected.valid[8]


def test_scaler_fits_train_only_and_leaves_future_unseen():
    r=split_recipe();f=frame()
    a=refit_train_scaler(r,f)
    later=f.copy();later.loc[later.timestamp>=pd.Timestamp('2020-01-04',tz='UTC'),['x','y']]=1e9
    b=refit_train_scaler(r,later)
    assert a['scaler']==b['scaler']
    assert a['scaler']['mean']==[2.5,12.5]
    assert r['scaler']['mean']==[2.,3.]


def test_tcn_causal_prefix_oldest_current_and_pruned_shape():
    r=split_recipe();r['tensor']['window']['observations']=13;seal(r)
    obs=spaces.Dict({'features':spaces.Box(-np.inf,np.inf,(13,2),np.float32),
                     'action_mask':spaces.Box(0,1,(5,),np.float32)})
    net=GroupedTCNExtractor(obs,r)
    for p in net.parameters(): torch.nn.init.constant_(p,.1)
    x=torch.ones(1,13,2,requires_grad=True)
    y=net.sequence({'features':x})
    changed=x.detach().clone();changed[:,8:]=100
    torch.testing.assert_close(y[:,:,:8],net.sequence({'features':changed})[:,:,:8],rtol=0,atol=0)
    y[:,:,-1].sum().backward()
    assert torch.all(x.grad[:,0]!=0) and torch.all(x.grad[:,-1]!=0)
    s=subset_recipe(r,['y'])
    reduced=GroupedTCNExtractor(spaces.Dict({'features':spaces.Box(-np.inf,np.inf,(13,1),np.float32)}),s)
    assert len(reduced.branches)==1 and reduced.fusion.conv.in_channels==2
    assert reduced({'features':torch.ones(1,13,1)}).shape==(1,104)


def test_profile_retains_schedule_and_signed_financing():
    p=dict(spread_schedule=[[0,24,.001]],weekly_open_spread=.002,slippage_market=.0001,
           slippage_stop=.0002,commission_per_lot_side=3.5,lot_units=100000,
           swap_long_per_lot=-7,swap_short_per_lot=2.5,triple_swap_rollover_index=3)
    c=cost_profile(p)
    assert c['spread']['rates']==[.001,.002]
    assert c['spread']['settings']['intervals']==[[0,24]]
    assert c['financing']['rates']==[-7,2.5]
