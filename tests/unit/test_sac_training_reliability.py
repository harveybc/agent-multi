from copy import deepcopy

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.sac_training_reliability import role, split_optimizer, snapshot, restore, signatures
from app.training_forensics import rng_state, restore_rng
from tests.unit.test_modular_temporal_architecture import grouped_recipe


def model():
    torch.set_num_threads(1)
    return ThreeSignalSAC(grouped_recipe(), critic_regularization={'kind':'core_dropout','p':.1})


def batch():
    x = torch.randn(32,13,2)
    b = TensorDict(dict(features=x,mask=torch.ones(32,3,dtype=torch.bool),action=torch.arange(32)%3),[32])
    b['next'] = TensorDict(dict(features=x*.9, mask=torch.ones(32,3,dtype=torch.bool),
        reward=torch.linspace(-.1,.1,32)[:,None],done=torch.zeros(32,1,dtype=torch.bool),
        terminated=torch.zeros(32,1,dtype=torch.bool)),[32])
    return b


def test_grouping_preserves_moments_and_exact_original_step():
    a=model(); b=batch(); a.update(b.clone())
    c=model(); c.loss.load_state_dict(deepcopy(a.loss.state_dict()))
    c.optimizer.load_state_dict(deepcopy(a.optimizer.state_dict()))
    split_optimizer(c,dict.fromkeys(('actor','critic','temperature'),1.))
    rng=rng_state(); va=a.update(b.clone()); restore_rng(rng); vc=c.update(b.clone())
    assert va==vc
    assert all(torch.equal(v,c.loss.state_dict()[k]) for k,v in a.loss.state_dict().items())
    for pa,pc in zip(a.loss.parameters(),c.loss.parameters()):
        for key,value in a.optimizer.state[pa].items():
            assert torch.equal(value,c.optimizer.state[pc][key]) if isinstance(value,torch.Tensor) else value==c.optimizer.state[pc][key]


def test_cold_resume_sampler_dropout_optimizer_and_identity(tmp_path):
    a=model(); b=batch(); factors=dict(actor=.1,critic=1.,temperature=1.)
    receipt=split_optimizer(a,factors)
    assert receipt[0]['lr']==a.effective['learning_rate']*.1
    sampler=np.random.default_rng(19); a.update(b.clone())
    state=snapshot(a,sampler,1,'id'); torch.save(state,tmp_path/'state.pt')
    ix=sampler.integers(32,size=32); va=a.update(b[ix].clone()); end=snapshot(a,sampler,2,'id')
    c=model(); split_optimizer(c,factors); sr=np.random.default_rng(0)
    restore(c,sr,torch.load(tmp_path/'state.pt',weights_only=False),'id')
    ix2=sr.integers(32,size=32); vc=c.update(b[ix2].clone())
    assert np.array_equal(ix,ix2) and va==vc
    assert signatures(end)==signatures(snapshot(c,sr,2,'id'))
    with pytest.raises(ValueError,match='RESTART_IDENTITY'):restore(c,sr,state,'other')


def test_unknown_parameters_or_invalid_factors_refused():
    with pytest.raises(ValueError,match='UNCLASSIFIED'):role('something_new.weight')
    with pytest.raises(ValueError,match='ROLE_LR'):split_optimizer(model(),dict(actor=0,critic=1,temperature=1))
