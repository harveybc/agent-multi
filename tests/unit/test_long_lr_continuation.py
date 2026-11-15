from copy import deepcopy
import numpy as np
import pytest
import torch

from app.optimizer_layout import optimizer_layout, restore_optimizer_layout
from app.sac_training_reliability import split_optimizer
from app.structural_refit import export_bundle, load_bundle
from app.training_forensics import rng_state, restore_rng, tree_digest
from agent_plugins.three_signal_sac import ThreeSignalSAC
from tools.run_structural_sac_refit import patience_can_stop, fit_epochs
from tests.unit.test_sac_training_reliability import model, batch
from tests.unit.test_double_tcn_probe import probe_recipe
from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401


def test_grouped_bundle_retains_rates_moments_and_next_update(tmp_path):
    a=model();b=batch();a.update(b.clone())
    split_optimizer(a,dict(actor=.1,critic=1.,temperature=1.))
    a.update(b.clone());x=b['features'].numpy()
    export_bundle(tmp_path/'bundle',a,x,dict(epoch=0))
    c,_=load_bundle(tmp_path/'bundle')
    assert optimizer_layout(a)==optimizer_layout(c)
    assert tree_digest(a.optimizer.state_dict())==tree_digest(c.optimizer.state_dict())
    rng=rng_state();va=a.update(b.clone());restore_rng(rng);vc=c.update(b.clone())
    assert va==vc and tree_digest(a.loss.state_dict())==tree_digest(c.loss.state_dict())


@pytest.mark.parametrize('kind',['duplicate','missing','unknown','length'])
def test_bad_named_optimizer_layout_rejected(kind):
    a=model();layout=optimizer_layout(a);state=deepcopy(a.optimizer.state_dict())
    if kind=='duplicate':layout[0][-1]=layout[0][0]
    elif kind=='missing':layout[0].pop()
    elif kind=='unknown':layout[0][-1]='not_a_parameter'
    else:state['param_groups'][0]['params'].pop()
    with pytest.raises(ValueError,match='OPTIMIZER_LAYOUT'):restore_optimizer_layout(a,layout,state)


def test_patience_observed_but_stop_deferred_to_minimum():
    d=dict(max_epochs=2000,minimum_epochs=100)
    assert not patience_can_stop(d,40,20,20)
    assert patience_can_stop(d,100,80,20)
    assert not patience_can_stop(d,100,0,20)
    assert patience_can_stop(dict(max_epochs=2000),40,20,20)
    with pytest.raises(ValueError,match='MINIMUM'):patience_can_stop(dict(max_epochs=2,minimum_epochs=3),0,0,20)


def test_long_design_preserves_data_and_optimizer_budget():
    from tools.run_long_sac_continuation import long_design,ORDERS,checked_baseline
    parent=dict(batch_size=128,warmup=4096,replay_size=65536,update_every=4,horizon=6,warmup_policy=True)
    d=long_design(parent)
    assert all(d[k]==v for k,v in parent.items()) and d['minimum_epochs']==100 and d['max_epochs']==2000
    assert d['monitor_mode']=='validation' and not d['head_only'] and d['patience']==20
    assert all(set(order)=={'control','actor_lr01','all_lr01'} for order in ORDERS.values())
    good=dict(TRAIN=dict(return_pct=16.05822355000215,trades=745,orders_per_week=3.5),
              VALIDATION=dict(return_pct=18.369945750000117,trades=207,orders_per_week=3.8))
    checked_baseline(good)
    good['VALIDATION']['trades']=0
    with pytest.raises(ValueError,match='BASELINE_PARITY'):checked_baseline(good)


def test_collector_waits_for_all_arms_and_rejects_corruption(tmp_path):
    from tools.collect_long_sac_continuation import collect
    from tools.run_discrete_curve_probe import write
    from app.frozen_recipe_inputs import file_sha
    fleet=dict(workers=[])
    for seed in (7,19,43):
        root=tmp_path/str(seed);root.mkdir()
        spec=dict(seed=seed,identity=str(seed),source='s',manifest='m',parent='p',code='c',design={},effective={},arms={})
        write(root/'SPEC.json',spec);fleet['workers'].append(dict(seed=seed,root=str(root)))
        results=[]
        for name in ('control','actor_lr01','all_lr01'):
            p=root/name;p.mkdir();write(p/'w.json',dict(value=seed))
            r=dict(parent_identity=str(seed),cold_exact=True,seed=seed,arm=name,files={'w.json':file_sha(p/'w.json')})
            results.append(r);write(p/'RESULT.json',r)
        write(root/'RESULT.json',dict(identity=str(seed),arms=results))
    assert collect(fleet,tmp_path/'out')
    write(tmp_path/'7'/'control'/'w.json',dict(value=-1))
    assert not collect(fleet,tmp_path/'out')


def test_epoch_zero_is_visible_in_full_fit_diagnostics(tmp_path):
    from tests.unit.test_training_diagnostics import history
    from tools.run_discrete_curve_probe import write,read
    from tools.run_structural_sac_refit import render_fit_diagnostics
    rows=history();write(tmp_path/'EPOCH_ZERO.json',dict(metrics=rows[0]['metrics']))
    state=dict(history=rows,best_bundle=dict(path='epoch_0000'))
    render_fit_diagnostics(state,tmp_path,{})
    receipt=read(tmp_path/'diagnostics'/'DIAGNOSTICS.json')
    assert receipt['best_epoch']==0 and receipt['epochs']==3 and receipt['evaluation_points']==4


@pytest.mark.parametrize('joint',[False,True])
def test_grouped_native_epoch_restart_and_hook_exact(tmp_path,request,joint):
    from app.three_signal_learning import synthetic_env,emit,evaluate
    torch.set_num_threads(1);r=probe_recipe();base=request.getfixturevalue('curve_env')
    factory=lambda:synthetic_env(r,base.risk_config,101,1,.99)[0]
    def assess(m):
        out={}
        for period,seed in [('TRAIN',101),('VALIDATION',102)]:
            e=synthetic_env(r,base.risk_config,seed,1,.99)[0]
            logits,_=emit(m,e.inputs);out[period]=evaluate(e,logits)
        return out
    def make():
        m=ThreeSignalSAC(r,seed=7);split_optimizer(m,dict(actor=.1,critic=1.,temperature=1.));return m
    d=dict(replay_size=64,batch_size=16,warmup=16,update_every=4,max_epochs=2,
        patience=20,min_delta=1e-8,monitor_mode='validation',monitor_start_epoch=1)
    from app.g3_replay import JointReplay
    kwargs=dict(replay_factory=lambda real:JointReplay(real,None,16,207)) if joint else {}
    if joint:d['replay_backend']='memmap'
    probe=factory().inputs.windows[30:34].copy();seen=[]
    def hook(m,row):seen.append(row['epoch']);row['weight_sha']=tree_digest(m.loss.state_dict())
    a=make();sa=fit_epochs(a,factory,assess,d,7,tmp_path/'a','id',probe,epoch_hook=hook,**kwargs)
    b=make();fit_epochs(b,factory,assess,dict(d,max_epochs=1),7,tmp_path/'b','id',probe,epoch_hook=hook,**kwargs)
    c=make();sc=fit_epochs(c,factory,assess,d,7,tmp_path/'b','id',probe,epoch_hook=hook,**kwargs)
    assert seen==[1,2,1,2]
    assert tree_digest(a.loss.state_dict())==tree_digest(c.loss.state_dict())
    assert tree_digest(a.optimizer.state_dict())==tree_digest(c.optimizer.state_dict())
    assert [h['weight_sha'] for h in sa['history']]==[h['weight_sha'] for h in sc['history']]
    assert [h['mean_update'] for h in sa['history']]==[h['mean_update'] for h in sc['history']]
