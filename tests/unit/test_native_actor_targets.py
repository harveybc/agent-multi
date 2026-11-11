import copy

import pytest
import torch
import numpy as np
from types import SimpleNamespace

from app.native_actor_targets import target_values, validate_anchor_identity, rebuild_panel


def records():
    return [dict(role=role, row=i, timestamp=i+10, window_start=i,
                 outcomes=[dict(reward_return=float(i+a)) for a in range(3)])
            for i,role in enumerate(('fit','monitor','fit'))]


def test_native_replaces_fit_only_and_does_not_mutate_critic():
    q=torch.full((3,3),-5.)
    rows=records()
    got=target_values(q,rows,'native_fit')
    assert torch.equal(got[0],torch.tensor([0.,1.,2.]))
    assert torch.equal(got[1],q[1])
    assert torch.equal(q,torch.full_like(q,-5.))
    changed=copy.deepcopy(rows)
    for v in changed[1]['outcomes']:v['reward_return']=1e12
    assert torch.equal(got,target_values(q,changed,'native_fit'))
    assert torch.equal(target_values(q,rows,'critic'),q)
    assert not got.requires_grad


def test_reject_bad_target_and_fit_values():
    q=torch.zeros(3,3)
    with pytest.raises(ValueError,match='TARGET_KIND'):target_values(q,records(),'other')
    rows=records();rows[0]['outcomes'][0]['reward_return']=float('nan')
    with pytest.raises(ValueError,match='TARGET_FINITE'):target_values(q,rows,'native_fit')
    with pytest.raises(ValueError,match='TARGET_SHAPE'):target_values(q[:1],records(),'critic')


def test_anchor_identity_requires_complete_exact_timestamps_and_support():
    rows=records()
    validate_anchor_identity(rows,copy.deepcopy(rows))
    for bad in [rows[:-1],rows+[rows[0]],list(reversed(rows))]:
        with pytest.raises(ValueError,match='ANCHOR_IDENTITY'):validate_anchor_identity(rows,bad)
    for key in ('timestamp','row','window_start','role'):
        bad=copy.deepcopy(rows);bad[0][key]='altered'
        with pytest.raises(ValueError,match='ANCHOR_IDENTITY'):validate_anchor_identity(rows,bad)


def test_rebuild_uses_current_positions_signals_and_returns(monkeypatch):
    import app.native_actor_targets as module
    from app.all_state_refinement import START, BOUNDARY
    times=np.array([START+30*3600*10**9,BOUNDARY+30*3600*10**9])
    inputs=SimpleNamespace(row_ts_ns=times,slot_ts_ns=times[:,None])
    anchors=[dict(row=i,timestamp=int(t),window_start=int(t),role=r,year=2020+i*3,
                  position=-1,outcomes='STALE') for i,(t,r) in enumerate(zip(times,('fit','monitor')))]
    used=[]
    class Env:
        def __init__(self,i):
            self.i=i;self.rows=[i];self.bar=0;self.finished=False
            self.engine=SimpleNamespace(v=[int(times[i])+3600*10**9],last=0,
                                        pos=None if i==0 else SimpleNamespace(direction=1))
        def reset(self):pass
        def step(self,a):used.append((self.i,a));self.finished=True
    monkeypatch.setattr(module,'state_view',lambda e:dict(row=e.i))
    def outcome(e,s,a):
        v=[0.,3.,1.][a] if e.i==0 else [2.,2.,4.][a]
        return dict(reward_return=v,soft_return=v,net_usd=v*100,parent_unchanged=True,
                    first_effect=dict(duration=1,entries=int(e.i==0 and a>0)))
    monkeypatch.setattr(module,'continuation',outcome)
    result=rebuild_panel(lambda interval:Env(int(interval[0]=='2023-01-01')),inputs,
                         np.array([2,1]),anchors,lambda **kw:None)
    assert [r['position'] for r in result]==[0,1]
    assert result[0]['outcomes'][1]['reward_return']==3.
    assert result[1]['allowed'] is False
    assert used==[(0,2),(1,1)]
    assert anchors[0]['outcomes']=='STALE'
