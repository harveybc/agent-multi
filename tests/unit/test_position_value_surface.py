import numpy as np
import pytest
import torch

from app.position_value_surface import (geometry,linear_weights,fit_axes,query_weights,
    GapSurfaceHeads,evaluate_surface,oracle_fit,decision_metrics)


def test_geometry_mirror_and_no_clipping():
    assert geometry(1,100.,98.,106.,2.)==pytest.approx([1.,3.])
    assert geometry(-1,100.,102.,94.,2.)==pytest.approx([1.,3.])
    assert geometry(1,97.,98.,106.,2.)[0]<0
    for args in ((1,100,98,106,0),(0,100,98,106,2),(1,100,106,98,2)):
        with pytest.raises(ValueError):geometry(*args)


def test_linear_and_surface_affine_exact_including_extrapolation():
    axis=np.linspace(0,4,9);x=np.array([-1.,.3,4.,5.])
    w=linear_weights(x,axis)
    np.testing.assert_allclose(w.sum(1),1)
    np.testing.assert_allclose(w@(2*axis+3),2*x+3,atol=1e-12)
    coords=np.array([[1.,2.],[2.,4.],[5.,6.]])
    axes={'long':[axis.tolist(),axis.tolist()],'short':[axis.tolist(),axis.tolist()]}
    b,out=query_weights(coords,np.zeros(3,dtype=int),axes,'surface81')
    grid=axis[:,None]*axis[None,:]+2*axis[:,None]-axis[None,:]
    np.testing.assert_allclose(b@grid.ravel(),coords[:,0]*coords[:,1]+2*coords[:,0]-coords[:,1],atol=1e-12)
    assert out.tolist()==[False,False,True]


def test_fit_axes_ignore_monitor_and_progress_loses_width():
    coords=np.array([[1.,2.],[2.,4.],[2.,1.],[4.,2.],[999.,888.]])
    dirs=np.array([0,0,1,1,0]);fit=np.array([True]*4+[False])
    axes=fit_axes(coords,dirs,fit)
    changed=coords.copy();changed[-1]=[9999,2222]
    assert axes==fit_axes(changed,dirs,fit)
    a,_=query_weights(coords[:2],dirs[:2],axes,'progress9')
    b,_=query_weights(coords[:2],dirs[:2],axes,'surface81')
    np.testing.assert_array_equal(a[0],a[1])
    assert not np.array_equal(b[0],b[1])
    assert oracle_fit(b,np.array([-1.,1.]))['max_error']<1e-10
    assert oracle_fit(a,np.array([-1.,1.]))['max_error']>.9


def test_market_only_outputs_routing_and_gradients():
    net=GapSurfaceHeads(5,81,19)
    z=torch.ones(2,2,5);out=net(z)
    assert out.shape==(2,2,2,81)
    # CPU batched GEMM differs by one float32 ulp for these identical rows.
    torch.testing.assert_close(out[:,0],out[:,1],rtol=0,atol=2e-7)
    weights=torch.zeros(2,81);weights[0,0]=1;weights[1,80]=1
    directions=torch.tensor([0,1])
    selected=evaluate_surface(out,directions,weights)
    torch.testing.assert_close(selected[:,0],out[:,0,0,0])
    torch.testing.assert_close(selected[:,1],out[:,1,1,80])
    selected.sum().backward()
    assert any(p.grad.abs().sum()>0 for p in net.parameters())
    with pytest.raises(ValueError):evaluate_surface(out,torch.tensor([0,2]),weights)


def test_close_harm_and_ties_are_separate_from_account_regret():
    # Closing row0 helps the later account but hurts the original trade.
    m=decision_metrics(np.array([1.,-1.,0.]),np.array([2.,-3.,0.]),
        np.array([-5.,4.,0.]),np.array([True,True,False]),[[0,1],[2]])
    assert m['bank_regret']==0 and m['requested_closes']==1
    assert m['effective_closes']==1 and m['harmful_effective_closes']==1
    assert m['missed_account_beneficial_closes']==0


def test_native_original_position_outcome_preserves_parent(curve_env):
    from app.decision_attribution import state_view
    from app.role_routed_critic import delayed_histories
    from app.three_signal_learning import synthetic_env
    from tests.unit.test_double_tcn_probe import probe_recipe
    from tools.run_position_value_surface import original_position_outcome
    env,regime=synthetic_env(probe_recipe(),curve_env.risk_config,17,1,.99)
    env.reset()
    while not env.engine.entry_gate(env.bar)[0]:env.step(0)
    cases=[e for _,states,_ in delayed_histories(env) for e in states.values()]
    assert cases
    signals=np.zeros(len(regime),dtype=int)
    for snapshot in cases[:4]:
        before=state_view(snapshot);action=2 if snapshot.engine.pos.direction==1 else 1
        close=original_position_outcome(snapshot,signals,action)
        hold=original_position_outcome(snapshot,signals,0)
        assert state_view(snapshot)==before
        assert np.isfinite(close['net_usd']) and np.isfinite(hold['net_usd'])
        assert close==original_position_outcome(snapshot,signals,action)


def test_surface_controller_positive_control_opposed_actions():
    coords=np.array([[1.,2.],[2.,4.]])
    axes={name:[np.linspace(0,4,9).tolist(),np.linspace(0,5,9).tolist()] for name in ('long','short')}
    basis,_=query_weights(coords,np.array([0,0]),axes,'surface81')
    nodes=np.linalg.lstsq(basis,np.array([1.,-1.]),rcond=None)[0]
    output=torch.tensor(np.broadcast_to(nodes,(2,2,2,81)).copy(),dtype=torch.float64)
    gap=evaluate_surface(output,torch.zeros(2,dtype=torch.long),torch.tensor(basis))
    torch.testing.assert_close(gap,torch.tensor([[1.,-1.],[1.,-1.]],dtype=torch.float64))


from tests.unit.test_discrete_curve_native import curve_env
