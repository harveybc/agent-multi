import copy
import numpy as np
import pytest
import torch
from torch import nn

from app.cumulative_readmission import cumulative_pass, expand_actor


def test_return_includes_history_and_uses_latest_baseline():
    seen=[]
    scores={('a','b'):1., ('b',):1., ('a',):0., ():0.,
            ('b','c'):2., ('b','c','d'):3., ('a','b','c','d'):3.}
    def measure(cols):
        seen.append(tuple(cols))
        return {'score':scores.get(tuple(cols),0.),'eligible':True}
    r=cumulative_pass(['a','b','c','d'],[['a'],['b'],['c'],['d']],['a','b'],measure)
    assert r['selected']==['b','c','d']
    assert r['counts']=={'initial':2,'after_deletion':1,'final':3,'universe':4}
    assert ('b','c') in seen and ('b','c','d') in seen
    assert [x['block'] for x in r['trace'] if x['phase']=='REINSERT']==[['a'],['c'],['d']]


def test_no_deletion_still_returns_historical_and_rejects_inactivity():
    def measure(cols):
        return {'score':float(len(cols)),'eligible':'c' not in cols}
    r=cumulative_pass(['a','b','c'],[['a'],['b'],['c']],['a'],measure)
    assert r['selected']==['a','b']
    assert not r['trace'][-1]['accepted']


def test_atomic_blocks_and_insertion_tie():
    r=cumulative_pass(['a','s','c'],[['a'],['s','c']],['a'],
                      lambda c:dict(score=1.,eligible=True))
    assert r['selected']==['a']
    with pytest.raises(ValueError,match='PARTIAL_BLOCK'):
        cumulative_pass(['a','s','c'],[['a'],['s','c']],['a','s'],lambda c:None)


class Conv(nn.Module):
    def __init__(self,i,o):
        super().__init__(); self.conv=nn.Conv1d(i,o,1)
    def forward(self,x): return self.conv(x)


class ToyActor(nn.Module):
    def __init__(self,recipe):
        super().__init__(); self.encoder=nn.Module()
        enc=self.encoder; enc.shape=(13,len(recipe['list']['columns'])); enc.transport_shape=enc.shape
        enc.context=None
        groups=recipe['representation']['resolved']['indices']
        enc.branches=nn.ModuleList([nn.Sequential(Conv(len(ix),2),nn.Tanh()) for ix in groups])
        for i,ix in enumerate(groups): enc.register_buffer(f'indices_{i}',torch.tensor(ix))
        enc.fusion=Conv(len(groups)*2,3); self.head=nn.Linear(39,3)
    def surface(self,x):
        e=self.encoder; x=x.transpose(1,2)
        z=torch.cat([b(x.index_select(1,getattr(e,f'indices_{i}'))) for i,b in enumerate(e.branches)],1)
        return self.head(e.fusion(z).flatten(1))


def recipe(names,groups,means=None,stds=None):
    return {'list':{'columns':names},'scaler':{'features':names,'mean':means or [0.]*len(names),
            'std':stds or [1.]*len(names)}, 'tensor':{'window':{'observations':13}},
            'representation':{'resolved':{'names':list(groups),'indices':list(groups.values()),
                                           'kinds':['conv']*len(groups)}}}


def test_expansion_preserves_active_policy_and_restores_all_nonzero_paths():
    torch.manual_seed(7)
    full=recipe(['a','b','c'],{'first':[0,1],'other':[2]},[3.,4.,5.],[2.,3.,4.])
    current=recipe(['a'],{'first':[0]},[3.],[2.])
    old=recipe(['a','b','c'],{'first':[0,1],'other':[2]},[1.,2.,3.],[3.,4.,5.])
    base=ToyActor(current); donor=ToyActor(old)
    before=copy.deepcopy(base.state_dict()); dbefore=copy.deepcopy(donor.state_dict())
    expanded,receipt=expand_actor(base,current,full,[(donor,old,'old')])
    x=torch.randn(9,13,3); masked=x.clone(); masked[:,:,1:]=0
    torch.testing.assert_close(expanded.surface(masked),base.surface(masked[:,:,:1]),rtol=1e-5,atol=1e-6)
    assert not torch.allclose(expanded.surface(x),expanded.surface(masked))
    assert receipt['restored_columns']==['b','c']
    for k,v in before.items(): assert torch.equal(v,base.state_dict()[k])
    for k,v in dbefore.items(): assert torch.equal(v,donor.state_dict()[k])
    for c in (1,2):
        probe=masked.clone();probe[:,:,c]=x[:,:,c]
        assert not torch.allclose(expanded.surface(probe),expanded.surface(masked))
    restored,_=expand_actor(base,current,full,[(donor,old,'old')])
    restored.load_state_dict(expanded.state_dict())
    assert torch.equal(restored.surface(x),expanded.surface(x))


def test_missing_donor_refuses_instead_of_random_connection():
    base=recipe(['a'],{'g':[0]}); full=recipe(['a','b'],{'g':[0,1]})
    with pytest.raises(ValueError,match='DONOR'):
        expand_actor(ToyActor(base),base,full,[])
