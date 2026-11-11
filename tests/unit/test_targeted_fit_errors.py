import pytest
import torch

from app.targeted_fit_errors import error_targets, correction_loss, requested_margins


def fixture():
    z=torch.tensor([[3.,0.,0.],[0.,3.,0.],[3.,0.,0.],[3.,0.,0.]],dtype=torch.double)
    y=torch.tensor([[0.,2.,1.],[0.,2.,0.],[0.,1.,1.],[0.,1.,2.]],dtype=torch.double)
    records=[{'role':r} for r in ('fit','fit','fit','monitor')]
    return z,y,records


def test_fixed_error_set_excludes_correct_tied_and_monitor():
    z,y,records=fixture();targets,weights=error_targets(z,y,records)
    assert targets.tolist()==[1,-1,-1,-1]
    assert weights.tolist()==[2.,0.,0.,0.]
    y[3]=torch.tensor([1e12,-1e12,0.])
    t,w=error_targets(z,y,records)
    assert torch.equal(t,targets) and torch.equal(w,weights)


def test_weighted_loss_duplicate_multiplicity_and_scale_invariance():
    z=torch.tensor([[2.,0.,-1.],[0.,1.,2.],[1.,0.,0.]],dtype=torch.double,requires_grad=True)
    targets=torch.tensor([1,0,-1]);weights=torch.tensor([2.,3.,0.],dtype=torch.double)
    ix=torch.tensor([0,0,1,2])
    got=correction_loss(z,targets,weights,ix)
    expected=-(4*z.log_softmax(-1)[0,1]+3*z.log_softmax(-1)[1,0])/7
    torch.testing.assert_close(got,expected)
    torch.testing.assert_close(got,correction_loss(z,targets,100*weights,ix))
    gradient=torch.autograd.grad(got,z)[0]
    assert torch.equal(gradient[2],torch.zeros(3,dtype=torch.double))


def test_requested_direction_matches_autograd_and_favors_target():
    z=torch.tensor([[2.,0.,-1.],[0.,1.,2.]],dtype=torch.double,requires_grad=True)
    targets=torch.tensor([1,0]);weights=torch.tensor([2.,3.],dtype=torch.double)
    ix=torch.tensor([0,0,1])
    loss=correction_loss(z,targets,weights,ix)
    g=torch.autograd.grad(loss,z)[0]
    unique,desired=requested_margins(z.detach(),targets,weights,ix)
    expected=torch.stack([-(g[i,targets[i]]-g[i,torch.arange(3)!=targets[i]]) for i in unique])
    torch.testing.assert_close(desired,expected)
    assert (desired>0).all()


def test_empty_error_batch_refuses():
    z,y,records=fixture();targets,weights=error_targets(z,y,records)
    with pytest.raises(ValueError):correction_loss(z,targets,weights,torch.tensor([1,2,3]))


def test_invalid_target_weights_refuse():
    with pytest.raises(ValueError):
        correction_loss(torch.zeros(1,3),torch.tensor([-1]),torch.ones(1),torch.tensor([0]))
