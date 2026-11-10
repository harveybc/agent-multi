import numpy as np
import pytest
import torch

from app.position_decision_loss import fit_loss_constants, partner_schedule, decision_loss


def test_fit_constants_ignore_monitor_and_balance_banks():
    y=np.array([1.,-3.,2.,999.]); banks=[[0,1],[2]]
    c=fit_loss_constants(y,banks)
    assert c==pytest.approx({'sign':2.,'pair':1.})
    y[-1]=-99999
    assert fit_loss_constants(y,banks)==c
    with pytest.raises(ValueError):fit_loss_constants(np.array([1.,2.]),[[0,1]])


def test_partner_schedule_stays_in_fit_bank_and_is_repeatable():
    schedule=np.array([[0,1,2],[2,0,1]])
    banks=[[0,1],[2]]; p=partner_schedule(schedule,banks,19)
    assert np.array_equal(p,partner_schedule(schedule,banks,19))
    assert np.isin(p[schedule!=2],[0,1]).all() and (p[schedule==2]==2).all()
    with pytest.raises(ValueError):partner_schedule(np.array([[3]]),banks,19)
    with pytest.raises(ValueError):partner_schedule(schedule,[[0,1],[1,2]],19)


def test_sign_gradient_matches_cost_and_neutral_has_zero_gradient():
    z=torch.zeros(2,3,requires_grad=True); y=torch.tensor([2.,-4.,0.])
    loss=decision_loss(z,y,{'sign':2.,'pair':1.},'sign'); loss.backward()
    torch.testing.assert_close(z.grad,torch.tensor([[-1/12,1/6,0.],[-1/12,1/6,0.]]))


def test_pair_pushes_opposed_scores_apart_and_ignores_same_sign():
    z=torch.zeros(2,2,requires_grad=True); q=torch.zeros(2,2,requires_grad=True)
    y=torch.tensor([1.,1.]); other=torch.tensor([-1.,2.]); c={'sign':1.,'pair':1.}
    loss=decision_loss(z,y,c,'sign_pair',q,other); loss.backward()
    assert (q.grad[:,0]>0).all() and (q.grad[:,1]==0).all()
    assert (z.grad[:,0]<z.grad[:,1]).all()


def test_sign_inversion_and_extreme_scores_are_finite():
    z=torch.tensor([[1000.,-1000.],[900.,-900.]],requires_grad=True)
    y=torch.tensor([-2.,3.]); c={'sign':2.5,'pair':5.}
    a=decision_loss(z,y,c,'sign_pair',-z,-y)
    b=decision_loss(-z,-y,c,'sign_pair',z,y)
    torch.testing.assert_close(a,b); a.backward()
    assert torch.isfinite(a) and torch.isfinite(z.grad).all()


def test_zero_targets_do_not_affect_sign_or_pair_losses():
    z=torch.ones(2,2,requires_grad=True); y=torch.zeros(2)
    loss=decision_loss(z,y,{'sign':1.,'pair':1.},'sign_pair',-z,y)
    loss.backward(); assert loss.item()==0 and (z.grad==0).all()


def test_decision_report_does_not_mislabel_score_as_q_gap():
    from tools.run_position_decision_loss import decision_description
    records=[dict(role=role,direction='long',gap=1.,local_delta=-1.,effective_close=True)
             for role in ('fit','monitor')]
    report=decision_description(np.array([2.,2.]),records,[[0],[1]],np.zeros(2,dtype=bool))
    for role in ('fit','monitor'):
        m=report[role]['all']
        assert 'gap_rmse' not in m and m['bank_regret']==0
        assert m['harmful_effective_closes']==1
