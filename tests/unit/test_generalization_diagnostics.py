import numpy as np
import pandas as pd
import pytest

from app.generalization_diagnostics import ring_timeline, quarter_metrics, context_coverage
from app.generalization_diagnostics import verify_replay_windows, frozen_probe
from app.generalization_diagnostics import soft_td_target


def test_ring_timeline_wrap_and_episode_origin():
    out=ring_timeline(23,8,5,7)
    assert out['slots'].tolist()==[7,0,1,2,3,4,5,6]
    assert out['decision'].tolist()==[0,1,2,3,4,0,1,2]
    assert out['epoch'].tolist()==[4,4,4,4,4,5,5,5]
    assert out['age'].tolist()==list(range(7,-1,-1))
    assert ring_timeline(3,8,5,3)['slots'].tolist()==[0,1,2]
    with pytest.raises(ValueError,match='REPLAY_CURSOR'):
        ring_timeline(23,8,5,0)


def test_quarters_use_entering_equity_and_exit_attribution():
    ts=pd.to_datetime(['2023-03-30','2023-03-31','2023-04-01','2023-04-02'],utc=True).as_unit('ns').asi8
    out=quarter_metrics(ts,np.array([100.,110.,99.,121.]),[
        dict(exit_time='2023-04-01',direction=1,net_usd=-11.,close_reason='SL')])
    assert out[0]['return_pct']==pytest.approx(10.)
    assert out[1]['return_pct']==pytest.approx(10.)
    assert out[1]['max_drawdown_pct']==pytest.approx(10.)
    assert out[1]['trades_closed']==1
    assert out[1]['realized_pnl_exit_attributed']==-11
    assert out[1]['equity_change_usd']==11


def test_context_coverage_does_not_compress_gaps_or_drop_base_rows():
    times=np.arange(100,dtype=np.int64)*3600*10**9
    times[50:]+=48*3600*10**9
    z=np.ones((100,2)); z[40,0]=np.nan
    base=np.ones(100,dtype=bool)
    out=context_coverage(times,z,base,13,6)
    assert out['baseline_rows']==100 and out['unavailable_rows']>0
    assert out['elapsed_hours']['max']>72
    assert base.all() and np.isnan(z[40,0])


def test_replay_dates_require_exact_window_match():
    windows=np.arange(5*3*2).reshape(5,3,2)
    timeline=ring_timeline(23,8,5,7)
    stored=np.empty((8,3,2),dtype=int)
    stored[timeline['slots']]=windows[timeline['decision']]
    assert verify_replay_windows(stored,windows,np.arange(5),timeline)==8
    stored[0,0,0]+=1
    with pytest.raises(ValueError,match='REPLAY_WINDOW_MAPPING'):
        verify_replay_windows(stored,windows,np.arange(5),timeline)


def test_bn_probe_restores_buffers_modes_and_rng_after_exception():
    import torch
    model=torch.nn.Sequential(torch.nn.BatchNorm1d(3),torch.nn.Dropout(.5))
    model.eval()
    state={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with frozen_probe(model):
            model.train()
            model(torch.randn(16,3))
            raise RuntimeError('probe')
    assert not model.training and all(not m.training for m in model.modules())
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(),rng)


def test_td_uses_duration_and_no_terminal_bootstrap():
    import torch
    p=torch.tensor([[1.,0.,0.],[1.,0.,0.]])
    q=torch.ones(2,2,3)*4
    y=soft_td_target(torch.tensor([1.,2.]),torch.tensor([False,True]),torch.tensor([2.,1.]),p,q,0.,.5)
    assert torch.equal(y,torch.tensor([2.,2.]))
