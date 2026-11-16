import pytest
import torch

from app.online_transient_forensics import functional_metrics, swap_actor, checkpoint_drift
from app.training_forensics import tree_digest


def test_common_q_offset_is_not_action_advantage():
    logits=torch.tensor([[1.,0.,-1.],[0.,2.,1.]])
    q=torch.stack((logits,logits+.1))
    a=functional_metrics(logits,q,logits,.2)
    b=functional_metrics(logits,q+100,logits,.2)
    for k in ('q_action_gap','kl_to_soft_q','q_disagrees_actor','q_centered_rms'):
        assert a[k]==pytest.approx(b[k],abs=1e-5)
    assert b['q_common_mean']-a['q_common_mean']==pytest.approx(100)
    assert a['kl_to_initial']==0 and a['action_flips_fraction']==0


def test_actor_swap_is_limited_and_nonmutating():
    current={k:torch.ones(2) for k in ('actor_network_params.a.head.weight',
        'actor_network_params.a.encoder.weight','qvalue_network_params.a.head.weight')}
    initial={k:v*2 for k,v in current.items()}
    before=tree_digest(current)
    hybrid,replaced=swap_actor(current,initial,'head')
    assert replaced==['actor_network_params.a.head.weight']
    assert hybrid[replaced[0]].tolist()==[2,2]
    assert hybrid['qvalue_network_params.a.head.weight'].tolist()==[1,1]
    assert tree_digest(current)==before
    drift=checkpoint_drift(hybrid,current)
    assert drift['actor.head']['relative_l2']==1
    assert drift['actor.encoder']['relative_l2']==0
    with pytest.raises(ValueError):swap_actor(current,initial,'arbitrary')
    with pytest.raises(ValueError):swap_actor(current,{},'head')


def test_probe_rejects_bad_input():
    z=torch.zeros(3,3);q=torch.zeros(2,3,3)
    with pytest.raises(ValueError):functional_metrics(z,q,z,0)
    with pytest.raises(ValueError):functional_metrics(z,q,z,float('nan'))
    with pytest.raises(ValueError):functional_metrics(z,q,z,float('inf'))
    z[0,0]=float('nan')
    with pytest.raises(ValueError):functional_metrics(z,q,z,1)
