import copy

import pytest
import torch

from app.actor_preservation import policy_kl, anchored_update, arm_config


def test_kl_identity_direction_detachment_and_validation():
    reference=torch.tensor([[1.,2.,3.]],requires_grad=True)
    current=torch.tensor([[3.,1.,2.]],requires_grad=True)
    assert policy_kl(reference,reference).item()==pytest.approx(0,abs=1e-7)
    value=policy_kl(reference,current)
    assert value.item()>0
    value.backward()
    assert reference.grad is None and current.grad.abs().sum()>0
    with pytest.raises(ValueError):policy_kl(reference,torch.zeros(2,3))
    with pytest.raises(ValueError):policy_kl(reference,torch.full_like(reference,float('nan')))


def test_actual_optimizer_and_native_zero_beta_parity():
    from tests.unit.test_double_tcn_probe import probe_recipe
    from agent_plugins.three_signal_sac import ThreeSignalSAC
    from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
    from tools.run_discrete_curve_probe import transition
    from tensordict import cat
    torch.set_num_threads(1)
    a=ThreeSignalSAC(probe_recipe(),lr=3e-5); b=ThreeSignalSAC(probe_recipe(),lr=3e-5)
    for model in (a,b):set_arm(model,'actor_only')
    obs=dict(features=torch.randn(13,2).numpy(),mask=torch.ones(3,dtype=torch.bool).numpy())
    batch=cat([transition(obs,1,obs,.1,False,1) for _ in range(4)],0)
    before=frozen_state(b)
    ref=b.surfaces(batch['features']).detach().clone()
    torch.manual_seed(71);a.update(batch.clone())
    torch.manual_seed(71);anchored_update(b,batch.clone(),ref,0.)
    for name,param in a.loss.named_parameters():
        assert torch.equal(param,dict(b.loss.named_parameters())[name])
    verify_frozen_state(b,before)
    # Compare positive penalty against native from the same already-drifted state.
    c=ThreeSignalSAC(probe_recipe(),lr=3e-5)
    c.loss.load_state_dict(b.loss.state_dict());c.optimizer.load_state_dict(copy.deepcopy(b.optimizer.state_dict()))
    set_arm(c,'actor_only')
    torch.manual_seed(72);anchored_update(b,batch.clone(),ref,1.)
    torch.manual_seed(72);c.update(batch.clone())
    assert any(not torch.equal(p,dict(c.loss.named_parameters())[n]) for n,p in b.loss.named_parameters() if p.requires_grad)
    verify_frozen_state(b,before)
    conf=arm_config(b,'lr_div3')
    assert conf['learning_rate']==pytest.approx(1e-5)
    assert all(g['lr']==conf['learning_rate'] for g in b.optimizer.param_groups)
    with pytest.raises(ValueError):arm_config(b,'unknown')


def test_report_keeps_negative_inactive_accounts(tmp_path):
    import csv
    from tools.run_actor_preservation import report, ARMS
    m=dict(return_pct=-1.,net_profit_usd=-100.,sharpe=-.2,max_drawdown_pct=2.,
        trades=1,win_rate=0.,orders_per_week=.1,buy=dict(trades=0,wins=0),
        sell=dict(trades=1,wins=0),close_reasons=dict(AGENT_CLOSE=1))
    periods={p:m for p in ('TRAIN','VALIDATION')}
    point=dict(updates=256,metrics=periods,eligible=False,
        policy_drift={p:dict(rows=10,kl_mean=.1,kl_max=.2,changed_actions=2,maximum_logit_change=.3) for p in periods})
    results={r:dict(reference=dict(metrics=periods),arms={a:dict(points=[point]) for a in ARMS}) for r in ('r19','r43')}
    report(tmp_path,results)
    rows=list(csv.DictReader((tmp_path/'TRADING.csv').open()))
    assert len(rows)==12 and all(r['eligible']=='False' and r['buy_win_rate']=='' for r in rows)
    assert (tmp_path/'RETURNS.png').stat().st_size>1000
