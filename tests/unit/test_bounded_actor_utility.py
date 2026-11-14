from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from app.bounded_actor_utility import install_native_heads,choose_fold,qualify,winner


def native():
    params={};state={}
    for layer,shape in ((0,(3,4)),(2,(3,3))):
        for kind in ('weight','bias'):
            s=shape if kind=='weight' else shape[:1]
            params['module','head',str(layer),kind]=torch.zeros((2,*s))
            for t in range(2):state[f'twins.{t}.0.{layer}.{kind}']=torch.full(s,float(1+t+layer))
    return SimpleNamespace(loss=SimpleNamespace(qvalue_network_params=params)),state


def test_native_exact_twins():
    m,s=native();install_native_heads(m,s)
    for (mod,head,layer,kind),v in m.loss.qvalue_network_params.items():
        for t in range(2):assert torch.equal(v[t],s[f'twins.{t}.0.{layer}.{kind}'])


@pytest.mark.parametrize('bad',['extra','shape','nonfinite'])
def test_donor_rejected_before_any_mutation(bad):
    m,s=native()
    if bad=='extra':s['extra']=torch.zeros(1)
    elif bad=='shape':s['twins.1.0.2.bias']=torch.zeros(10)
    else:s['twins.1.0.2.bias'][0]=float('nan')
    with pytest.raises(ValueError):install_native_heads(m,s)
    assert all(torch.count_nonzero(v)==0 for v in m.loss.qvalue_network_params.values())


def metrics():return {p:dict(return_pct=1.,orders_per_week=3.,cash_error=0.) for p in ('TRAIN','VALIDATION')}


def test_only_eight_and_strict_gain_qualify():
    b=metrics();m=deepcopy(b);m['VALIDATION']['return_pct']=1.001
    assert qualify(m,b,8)['qualified']
    assert not qualify(m,b,7)['qualified']
    assert not qualify(b,b,8)['qualified']


@pytest.mark.parametrize('field,value',[('return_pct',.999),('orders_per_week',1.99),('cash_error',1e-4),('return_pct',float('nan'))])
def test_single_period_failure_cannot_be_averaged(field,value):
    b=metrics();m=deepcopy(b);m['VALIDATION']['return_pct']=100.;m['TRAIN'][field]=value
    assert not qualify(m,b,8)['qualified']


def test_fold_uses_coverage_only():
    assert choose_fold({'folds':{'A':{'expanded':[1]},'B':{'expanded':[1,2]}}})=='B'


def test_winner_no_inactive_seed_or_earlier_checkpoint():
    r=[dict(seed=s,metrics=metrics(),qualification={'qualified':False}) for s in (19,43)]
    assert winner(r) is None
    r[1]['qualification']['qualified']=True
    assert winner(r)==43
    with pytest.raises(ValueError):winner(r[:1])


def actor_fixture():
    from torch import nn
    torch.manual_seed(350)
    loss=nn.Module();loss.actor_network_params=nn.Module();loss.actor_network_params.head=nn.Sequential(nn.Linear(4,3),nn.SiLU(),nn.Linear(3,3))
    loss.log_alpha=nn.Parameter(torch.tensor(-5.));loss.frozen=nn.Parameter(torch.ones(2))
    model=SimpleNamespace(loss=loss,optimizer=torch.optim.AdamW(loss.parameters(),lr=3e-5,weight_decay=.01),surfaces=loss.actor_network_params.head)
    x=torch.randn(32,4);q=torch.randn(32,3);z=model.surfaces(x).detach()
    from tools.run_bounded_actor_utility import actor_parameters,flat
    theta=flat([p for _,p in actor_parameters(model)])
    return model,x,q,torch.arange(16),z,theta


def test_step_only_actor_and_exact_cold_replay():
    from tools.run_bounded_actor_utility import actor_step,snapshot,restore
    from app.training_forensics import tree_digest
    args=actor_fixture();m=args[0];before=snapshot(m);r=actor_step(*args);after=snapshot(m)
    assert r['accepted'] and torch.equal(m.loss.frozen.detach(),before['loss']['frozen'])
    assert torch.equal(m.loss.log_alpha.detach(),before['loss']['log_alpha'])
    assert len(m.optimizer.state)==4
    restore(m,before);rr=actor_step(*args)
    assert rr==r and tree_digest(snapshot(m))==tree_digest(after)


def test_rejected_step_restores_weights_and_memory(monkeypatch):
    import tools.run_bounded_actor_utility as worker
    from app.training_forensics import tree_digest
    args=actor_fixture();m=args[0];worker.actor_step(*args);before=worker.snapshot(m)
    monkeypatch.setattr(worker,'backtrack',lambda *args:dict(accepted=False))
    assert not worker.actor_step(*args)['accepted']
    assert tree_digest(worker.snapshot(m))==tree_digest(before)


def test_exception_is_transactional(monkeypatch):
    import tools.run_bounded_actor_utility as worker
    from app.training_forensics import tree_digest
    args=actor_fixture();m=args[0];before=worker.snapshot(m)
    def fail(*args):raise RuntimeError('witness')
    monkeypatch.setattr(worker,'backtrack',fail)
    with pytest.raises(RuntimeError,match='witness'):worker.actor_step(*args)
    assert tree_digest(worker.snapshot(m))==tree_digest(before)
