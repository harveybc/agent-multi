import pytest
import torch

from app.optimal_action_sets import SetGuard, error_targets, correction_loss, requested_margins, set_margin, error_progress


@pytest.mark.parametrize('native,source,bad', [([1.,1.,0.],[2.,1.,0.],[0.,0.,3.]),
                                            ([1.,0.,1.],[2.,0.,1.],[0.,3.,0.])])
def test_both_hold_aliases_protected_but_bad_close_rejected(native,source,bad):
    z=torch.tensor([source]);y=torch.tensor([native]);g=SetGuard(z,y,[{'role':'fit'}])
    assert len(g.rows)==1 and len(g.floor)==1
    assert not g.check(torch.tensor([bad]))['passed']
    alternative=z.clone();alternative[0,y[0]==1]+=3
    alternative[0,0]-=2
    assert g.check(alternative)['passed']
    assert g.check(alternative)['action_changes']==1


def test_counterexample_and_fit_only_all_ties_excluded():
    z=torch.tensor([[0.,3.,0.],[2.,1.,0.],[0.,0.,3.],[0.,3.,0.]])
    y=torch.tensor([[0.,2.,1.],[1.,1.,0.],[1.,1.,1.],[0.,2.,1.]])
    records=[{'role':r} for r in ['fit','fit','fit','monitor']]
    g=SetGuard(z,y,records)
    assert g.rows.tolist()==[0,1] and len(g.floor)==3
    after=z.clone();after[1]=torch.tensor([0.,0.,3.])
    assert not g.check(after)['passed']
    after=z.clone();after[3]=torch.tensor([100.,0.,0.])
    assert g.check(after)['passed']
    assert (g.buffer>0).all()


def test_two_good_holds_receive_mass_not_arbitrary_onehot():
    z=torch.tensor([[0.,0.,3.],[0.,3.,0.],[3.,0.,0.],[1.,2.,3.]],dtype=torch.double,requires_grad=True)
    y=torch.tensor([[1.,1.,0.],[1.,0.,1.],[0.,2.,2.],[1.,1.,1.]],dtype=torch.double)
    records=[{'role':'fit'}]*4
    targets,w=error_targets(z,y,records)
    assert targets.tolist()==[[True,True,False],[True,False,True],[False,True,True],[True,True,True]]
    assert w.tolist()==[1.,1.,2.,0.]
    indices=torch.tensor([0,1,2,0,3]);loss=correction_loss(z,targets,w,indices)
    grad=torch.autograd.grad(loss,z)[0]
    ids,requested=requested_margins(z,targets,w,indices)
    actual=torch.autograd.functional.jvp(lambda v:set_margin(v,ids,targets[ids]),z,-grad)[1]
    assert torch.allclose(requested,actual,atol=1e-12)
    assert (requested>0).all()
    assert torch.all(grad[targets & (w[:,None]>0)]<0)
    assert torch.all(grad[~targets]>0)
    progress=error_progress(z.detach(),y,y,records)
    assert progress['fit']['rows']==3 and progress['fit']['corrected']==3


def test_monitor_never_enters_targets_or_gradient_and_bad_inputs_refuse():
    z=torch.zeros(2,3,dtype=torch.double,requires_grad=True)
    y=torch.tensor([[0.,1.,1.],[0.,9.,10.]])
    records=[{'role':'fit'},{'role':'monitor'}]
    mask,w=error_targets(z,y,records)
    assert w.tolist()==[1.,0.]
    loss=correction_loss(z,mask,w,torch.tensor([0,1]));grad=torch.autograd.grad(loss,z)[0]
    assert torch.equal(grad[1],torch.zeros(3,dtype=torch.double))
    with pytest.raises(ValueError):correction_loss(z,mask,w,torch.tensor([1]))
    with pytest.raises(ValueError):SetGuard(z,torch.full_like(z,float('nan')),records)


def test_optimal_mass_alone_is_not_greedy_acceptance():
    z=torch.tensor([[2.,1.,0.]])
    g=SetGuard(z,torch.tensor([[1.,1.,0.]]),[{'role':'fit'}])
    after=torch.tensor([[0.,0.,.1]])
    assert after.softmax(-1)[0,:2].sum()>.5
    assert not g.check(after)['passed']


def test_measured_remaining_gap_error_can_switch_to_worse_with_lower_loss():
    # Protecting correct rows is not a source-value floor for incorrect rows.
    source=torch.tensor([[.5,.4,.1],[.1,.8,.1]],dtype=torch.double).log()
    after=torch.tensor([[.4,.48,.12],[.1,.8,.1]],dtype=torch.double).log()
    native=torch.tensor([[.9,.6,1.16],[0.,1.,0.]])
    records=[{'role':'fit'}]*2
    guard=SetGuard(source,native,records)
    targets,weights=error_targets(source,native,records)
    indices=torch.tensor([0,1])
    assert correction_loss(after,targets,weights,indices)<correction_loss(source,targets,weights,indices)
    assert guard.check(after)['passed']
    assert native[0,after[0].argmax()]<native[0,source[0].argmax()]
