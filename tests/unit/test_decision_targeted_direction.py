import numpy as np
import pytest
from app.role_routed_critic import RoleHeads
import torch
from app.decision_targeted_direction import correction_candidates,strict_correction


def fixture():
    head=RoleHeads(torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.SiLU(),torch.nn.Linear(2,3)).double(),1,7)
    z=np.ones((2,4,2));y=np.array([[0.,1.,1.],[0.,1.,2.],[1.,1.,1.],[1.,0.,0.]])
    q=np.array([[[2.,1.,0.],[2.,1.,0.],[2.,1.,0.],[2.,1.,0.]]]*2)
    return head,z,y,q


def test_only_wrong_consensus_nonanchors_and_ties():
    h,z,y,q=fixture()
    rows,counts=correction_candidates(h.state_dict(),z,np.arange(4),y,{'a':q,'b':q},[1],100.,8)
    assert [r['row'] for r in rows]==[0]
    assert rows[0]['pairs']==[(0,1,0)]
    assert counts['eligible_wrong']==1
    other=q.copy();other[:,0]=[0.,2.,1.]
    assert correction_candidates(h.state_dict(),z,np.arange(4),y,{'a':q,'b':other},[1],100.,8)[0]==[]


def test_radius_and_count_cap():
    h,z,y,q=fixture()
    rows,_=correction_candidates(h.state_dict(),z,np.arange(4),y,{'a':q},[],100.,1)
    assert len(rows)==1
    assert rows[0]['lower_bound']>=0 and rows[0]['margin']>0
    with pytest.raises(ValueError):correction_candidates(h.state_dict(),z,np.arange(4),y,{'a':q},[],0.,1)


def test_strict_gate_requires_actual_target_and_regret_gain():
    baseline={'a':dict(regret=1.)};assessment=dict(passed=True,metrics={'a':dict(regret=.5)})
    q=np.array([[[0.,2.,1.]]]*2)
    assert strict_correction(assessment,baseline,{'a':q},0,np.array([False,True,False]))
    assert not strict_correction(assessment,baseline,{'a':q},0,np.array([True,False,False]))
    assessment['metrics']['a']['regret']=1.
    assert not strict_correction(assessment,baseline,{'a':q},0,np.array([False,True,False]))


def test_strict_gate_cannot_drop_a_geometry():
    q=np.zeros((2,1,3));assessment=dict(passed=True,metrics={'a':dict(regret=.5)})
    assert not strict_correction(assessment,{'a':dict(regret=1.),'b':dict(regret=1.)},{'a':q},0,np.array([True,False,False]))


def test_deterministic_rank_and_single_halfspace_bound():
    from app.optimal_fit_protection import complete_pair_jacobian
    h,z,y,q=fixture()
    with torch.no_grad():
        for p in h.parameters():p.zero_()
        for twin in h.twins:twin[0][2].bias.copy_(torch.tensor([2.,1.,0.]))
    rows,_=correction_candidates(h.state_dict(),z,np.arange(4),y,{'a':q},[],100.,8)
    assert [r['row'] for r in rows]==[0,1]
    for r in rows:
        gaps,jac=complete_pair_jacobian(h.state_dict(),z,r['pairs'])
        expected=np.max(np.maximum(r['margin']-gaps,0)/np.linalg.norm(jac,axis=1))
        assert r['lower_bound']==expected
    assert correction_candidates(h.state_dict(),z,np.arange(4),y,{'a':q},[],1e-8,8)[0]==[]
