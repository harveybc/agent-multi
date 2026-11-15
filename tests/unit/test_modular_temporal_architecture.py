import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from tensordict import TensorDict

from agent_plugins.double_tcn_probe import DoubleTCNExtractor
from agent_plugins.temporal_features import CausalConv1d
from agent_plugins.temporal_stage_config import stage_config
from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.order_qualification import OrderSignal
from tests.unit.test_double_tcn_probe import probe_recipe
from tests.unit.test_frozen_recipe_inputs import seal


def grouped_recipe():
    r = probe_recipe()
    r['representation']['resolved'] = dict(names=['a','b'], indices=[[0],[1]],
                                          units=[2,2], kinds=['conv','pointwise'])
    return seal(r)


class LegacyBlock(nn.Module):
    """Frozen pre356 formula; do not use the new builder as its own oracle."""
    def __init__(self, width, dilation, kernel):
        super().__init__()
        self.first = CausalConv1d(width,width,kernel,dilation)
        self.second = CausalConv1d(width,width,kernel,dilation)
        self.dropout1, self.dropout2 = nn.Dropout(0.), nn.Dropout(0.)

    def forward(self, x):
        return F.silu(x+self.dropout2(self.second(self.dropout1(F.silu(self.first(x))))))


class LegacyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList()
        for i,kernel in enumerate((3,1)):
            self.register_buffer(f'indices_{i}',torch.tensor([i]))
            self.branches.append(nn.Sequential(CausalConv1d(1,8,1),LegacyBlock(8,1,kernel),LegacyBlock(8,2,kernel)))
        self.fusion = CausalConv1d(16,16,1)
        self.core = nn.Sequential(LegacyBlock(16,1,3),LegacyBlock(16,2,3))

    def forward(self,x):
        x=x.transpose(1,2)
        grouped=[b(x.index_select(1,getattr(self,f'indices_{i}'))) for i,b in enumerate(self.branches)]
        return self.core(F.silu(self.fusion(torch.cat(grouped,1)))).flatten(1)


def test_legacy_parameters_order_and_outputs_exact():
    torch.set_num_threads(1)
    torch.manual_seed(31); old=LegacyEncoder()
    torch.manual_seed(31); new=DoubleTCNExtractor(OrderSignal((13,2)).observation_space,grouped_recipe())
    assert list(old.state_dict())==list(new.state_dict())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    x=torch.randn(3,13,2)
    assert torch.equal(old(x),new({'features':x}))


def modular():
    return dict(group_configs=[dict(width=4,blocks=1,kernel=5,dilations=[3],activation='relu'),
                               dict(width=12,blocks=3,kind='pointwise',activation='tanh')],
                core_config=dict(width=24,blocks=3,kernel=3,dilations=[1,3,5],activation='gelu'),
                fusion_activation='tanh',head_width=12,head_depth=3,head_activation='relu',
                actor_head_width=4,actor_head_depth=2,actor_head_activation='tanh',
                temporal_attention=dict(kind='causal_patch',patch_len=3,embedding=8,heads=2,feedforward=16,gate_init=.1))


def test_consumed_dimensions_heads_causality_and_immutable_config():
    cfg=modular(); before=copy.deepcopy(cfg)
    m=ThreeSignalSAC(grouped_recipe(),architecture=cfg,critic_regularization={'kind':'core_dropout','p':.1})
    assert cfg==before
    e=m.actor_net.encoder
    assert [b[0].conv.out_channels for b in e.branches]==[4,12]
    assert [len(b)-1 for b in e.branches]==[1,3]
    assert (e.fusion.conv.in_channels,e.fusion.conv.out_channels)==(16,24)
    assert len(e.core)==3 and e.core[-1].first.conv.dilation==(5,)
    assert e.resolved_architecture['token_receptive_fields_without_attention']==[61,37]
    assert len(m.actor_net.head)==5 and len(m.q_net.head)==7
    assert isinstance(m.actor_net.head[1],nn.Tanh) and isinstance(m.q_net.head[1],nn.ReLU)
    x=torch.randn(2,13,2); y=x.clone(); y[:,7:]+=100
    e.eval()
    with torch.no_grad():
        assert torch.equal(e.sequence({'features':x})[:,:,:7],e.sequence({'features':y})[:,:,:7])
        assert m.surfaces(x).shape==(2,3)
        assert m.q_surfaces(x).shape==(2,2,3)
        branch=e.branches[1]
        assert torch.equal(branch(x.transpose(1,2)[:,1:])[:,:,:7],branch(y.transpose(1,2)[:,1:])[:,:,:7])


@pytest.mark.parametrize('raw', [dict(width=True),dict(blocks=0),dict(kernel=0),dict(activation='oops'),
    dict(dilations=[1]),dict(dilations=[1,False]),dict(kind='pointwise',kernel=3),
    dict(kind='transformer'),dict(ignored=8),[]])
def test_invalid_stages_refused(raw):
    with pytest.raises(ValueError):stage_config(raw,width=8,blocks=2)


def test_group_count_and_head_depth_refused():
    with pytest.raises(ValueError,match='GROUP_CONFIG_COUNT'):
        ThreeSignalSAC(grouped_recipe(),architecture={'group_configs':[{}]})
    with pytest.raises(ValueError,match='HEAD_DEPTH'):
        ThreeSignalSAC(grouped_recipe(),architecture={'actor_head_depth':False})


@pytest.mark.parametrize('fraction',[.3,.9])
def test_batch32_update_entropy_gradient_and_cold_reload(tmp_path,fraction):
    torch.set_num_threads(1)
    cfg=modular(); reg={'kind':'core_dropout','p':.1}
    m=ThreeSignalSAC(grouped_recipe(),architecture=cfg,entropy_fraction=fraction,critic_regularization=reg)
    x=torch.randn(32,13,2)
    batch=TensorDict(dict(features=x,mask=torch.ones(32,3,dtype=torch.bool),action=torch.arange(32)%3),[32])
    batch['next']=TensorDict(dict(features=x*.9,mask=torch.ones(32,3,dtype=torch.bool),
        reward=torch.linspace(-.1,.1,32)[:,None],done=torch.zeros(32,1,dtype=torch.bool),
        terminated=torch.zeros(32,1,dtype=torch.bool)),[32])
    before={k:p.detach().clone() for k,p in m.loss.named_parameters()}
    values=m.update(batch)
    assert all(np.isfinite(list(values.values())))
    assert values['target_entropy']==pytest.approx(fraction*np.log(3))
    for part in ('actor_network_params','qvalue_network_params'):
        assert any(not torch.equal(before[k],p) for k,p in m.loss.named_parameters() if k.startswith(part))
    with torch.no_grad(): a=m.surfaces(x).clone(); q=m.q_surfaces(x).clone()
    m.save(tmp_path/'m.pt')
    cold=ThreeSignalSAC(grouped_recipe(),architecture=cfg,entropy_fraction=fraction,critic_regularization=reg,seed=99)
    cold.load(tmp_path/'m.pt')
    with torch.no_grad():
        assert torch.equal(a,cold.surfaces(x)) and torch.equal(q,cold.q_surfaces(x))
