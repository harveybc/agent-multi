import pytest
import torch
from tensordict import TensorDict

from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.critic_moment_loss import residual_moment_loss, moment_critic_objective
from tests.unit.test_double_tcn_probe import probe_recipe


@pytest.mark.parametrize('batch',[1,32,128])
def test_population_moments_equal_mse_and_gradient(batch):
    torch.manual_seed(8)
    x=torch.randn(2,batch,dtype=torch.float64,requires_grad=True)
    y=torch.randn_like(x)
    actual=residual_moment_loss(x,y,1.)
    expected=(x-y).square().sum(0).mean()
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],
                               torch.autograd.grad(expected,x)[0])


def test_bias_cannot_cancel_between_twins():
    x=torch.tensor([[2.,2.],[-2.,-2.]])
    assert residual_moment_loss(x,torch.zeros_like(x),.5)==8


def test_variance_weight_changes_loss_and_gradient_without_target_gradient():
    x=torch.tensor([[1.,-1.],[2.,-2.]],requires_grad=True)
    y=torch.zeros_like(x,requires_grad=True)
    one=residual_moment_loss(x,y,1.)
    half=residual_moment_loss(x,y,.5)
    assert half==one*.5
    g1=torch.autograd.grad(one,x,retain_graph=True)[0]
    half.backward()
    torch.testing.assert_close(x.grad,g1*.5)
    assert y.grad is None


def batch_fixture():
    torch.set_num_threads(1)
    model=ThreeSignalSAC(probe_recipe())
    x=torch.randn(8,13,2);mask=torch.ones(8,3,dtype=torch.bool)
    batch=TensorDict(dict(features=x,mask=mask,action=torch.arange(8)%3,
        steps_to_next_obs=torch.ones(8,1,dtype=torch.long),
        next=TensorDict(dict(features=x,mask=mask,reward=torch.arange(8).float().reshape(-1,1),
            done=torch.ones(8,1,dtype=torch.bool),terminated=torch.ones(8,1,dtype=torch.bool)),[8])),[8])
    return model,batch


def test_real_torchrl_mse_parity_and_active_override():
    model,batch=batch_fixture()
    original=model.loss.qvalue_loss
    model.loss.loss_function='l2'
    expected,_=original(batch)
    params=tuple(model.loss.qvalue_network_params.parameters())
    grad=torch.autograd.grad(expected,params,allow_unused=True)
    with moment_critic_objective(model,1.):
        actual,meta=model.loss.qvalue_loss(batch)
        torch.testing.assert_close(actual,expected)
        actual_grad=torch.autograd.grad(actual,params,allow_unused=True)
        for a,b in zip(grad,actual_grad):
            if a is not None:torch.testing.assert_close(a,b)
        assert meta['td_error'].shape==(8,)
    assert model.loss.qvalue_loss==original
    with moment_critic_objective(model,.5):
        half,_=model.loss.qvalue_loss(batch)
        assert half<actual
    model.loss.loss_function='smooth_l1'
    huber,_=original(batch)
    assert not torch.isclose(huber,actual)


def test_restore_on_exception_and_real_update():
    model,batch=batch_fixture();original=model.loss.qvalue_loss
    with pytest.raises(RuntimeError):
        with moment_critic_objective(model,.5):raise RuntimeError('witness')
    assert model.loss.qvalue_loss==original
    with moment_critic_objective(model,.5):
        value=model.update(batch)
        assert value['critic']>0
    assert model.loss.qvalue_loss==original


@pytest.mark.parametrize('weight',[0,-1,float('nan'),float('inf')])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError):residual_moment_loss(torch.ones(2,8),torch.zeros(2,8),weight)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA worker canary')
def test_cuda_moments_have_cuda_gradients():
    x=torch.randn(2,128,device='cuda',requires_grad=True)
    y=torch.randn_like(x)
    loss=residual_moment_loss(x,y,.5)
    loss.backward()
    assert loss.is_cuda and x.grad.is_cuda and torch.isfinite(x.grad).all()
