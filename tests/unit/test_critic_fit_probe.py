import numpy as np
import pytest
import torch

from app.critic_fit_probe import classify_anchor, choose_small_set, fit_metrics, fit_passed, set_scope


def record(year=2020,role='fit',position=0,allowed=True,idx=0):
    returns=[0.,.1,-.2] if position==0 else [0.,0.,-.2] if position==1 else [0.,.1,0.]
    if not allowed:returns=[0.,0.,0.]
    outcomes=[]
    for a in range(3):
        outcomes.append(dict(reward_return=returns[a],soft_return=returns[a],parent_unchanged=True,
            net_usd=returns[a],first_effect=dict(duration=1,command=a,
                entries=int(position==0 and allowed and a!=0),count_changes={})))
    return dict(anchor=dict(id=idx,year=year,role=role,position=position),results=dict(greedy=outcomes))


def test_classification_and_hold_equivalence():
    assert classify_anchor(record())['kind']=='flat_both_allowed'
    assert classify_anchor(record(allowed=False))['kind']=='flat_restricted'
    assert classify_anchor(record(position=1))['kind']=='open_long'
    assert classify_anchor(record(position=-1))['kind']=='open_short'
    bad=record(position=1);bad['results']['greedy'][1]['reward_return']=1
    bad['results']['greedy'][1]['soft_return']=1
    with pytest.raises(ValueError,match='HOLD'):classify_anchor(bad)
    bad=record();bad['results']['greedy'][0]['reward_return']=float('nan')
    with pytest.raises(ValueError):classify_anchor(bad)


def test_selection_is_fixed_fit_only_and_covers_every_year():
    rows=[record(y,idx=i*10+j) for i,y in enumerate(range(2020,2024)) for j in range(10)]
    rows+=[record(2020,role='monitor',idx=999)]
    chosen=choose_small_set(rows,19)
    assert chosen==choose_small_set(rows,19) and len(chosen)==32
    assert all(rows[i]['anchor']['role']=='fit' for i in chosen)
    assert all(sum(rows[i]['anchor']['year']==y for i in chosen)==8 for y in range(2020,2024))
    with pytest.raises(ValueError,match='COVERAGE'):choose_small_set(rows[:20],19)


def test_fit_metrics_identify_wrong_order_and_common_level():
    target=np.array([[0.,1.,2.],[1.,1.,-1.]])
    q=np.stack((target,target))
    assert fit_passed(fit_metrics(q,target))
    shifted=fit_metrics(q+4,target)
    assert shifted['minimum']['level_rmse']==pytest.approx(4.)
    assert shifted['minimum']['gap_rmse']==0
    wrong=fit_metrics(q[...,::-1].copy(),target)
    assert not fit_passed(wrong)
    with pytest.raises(ValueError):fit_metrics(q[:1],target)


def test_trainability_only_online_critic():
    class Loss:
        def __init__(self):
            self.params={n:torch.nn.Parameter(torch.ones(1)) for n in
                ('qvalue_network_params.module.head.weight','qvalue_network_params.module.encoder.weight',
                 'actor_network_params.module.head.weight','log_alpha','target_qvalue_network_params.module.head.weight')}
        def named_parameters(self):return self.params.items()
    from types import SimpleNamespace
    model=SimpleNamespace(loss=Loss(),effective={'learning_rate':3e-5},
                          optimizer=SimpleNamespace(param_groups=[{'lr':1.}]))
    set_scope(model,'heads')
    assert [n for n,p in model.loss.named_parameters() if p.requires_grad]==['qvalue_network_params.module.head.weight']
    set_scope(model,'critic_representation')
    assert sum(p.requires_grad for n,p in model.loss.named_parameters())==2
    with pytest.raises(ValueError):set_scope(model,'actor')
