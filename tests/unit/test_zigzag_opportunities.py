import hashlib
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.zigzag_opportunities import HOUR, PIP, Tape, OpportunityEvaluator, contiguous_bounds, schedule
from tools.zigzag_opportunity_deap import authenticated, decode, receipt, search


def fixture(prices=(100, 101, 103, 102, 100, 99), pivots=((2,1,104),(4,-1,99))):
    o=np.array(prices,dtype=float)
    n=len(o)
    tape=Tape(np.arange(n)*HOUR,o,o+1,o-1,np.zeros(n,int),np.arange(n)<n-1,
              np.full(n,n-1),np.full(n,.1),np.zeros((2,n)),[(0,n)],{})
    z=SimpleNamespace(Params=lambda *a:a,hindsight_pivots=lambda h,l,p:list(pivots))
    return tape,z


def test_hand_computed_long_excursions_and_costs():
    t,z=fixture()
    r=OpportunityEvaluator(t,z,3).evaluate((2,5,1),True)['records']
    # Entry 100; ideal peak 104, round trip .2, three held hourly bars.
    assert r['net'][0,0] == pytest.approx(3.8)
    assert r['duration'][0,0] == 3
    assert r['mae'][0,0] == 1
    assert r['drawdown'][0,0] == 2
    assert r['score'][0,0] == pytest.approx(3.8/6)
    assert r['target'][1,0] == 99
    assert r['net'][1,0] == pytest.approx(.8)


def test_strictly_later_pivot_and_forced_flat_open_quote():
    t,z=fixture()
    t.limit[:]=3
    t.allowed=np.arange(len(t.open))<3
    r=OpportunityEvaluator(t,z,3).evaluate((2,5,1),True)['records']
    assert not r['pivot_exit'][0,2]  # pivot at entry is not a future exit
    assert r['end'][0,2] == 3
    assert r['target'][0,2] == t.open[3]
    assert r['duration'][0,2] == 1
    assert r['release'][0,2] == 3
    assert r['release'][0,0] == 3  # pivot at bar 2 releases only after bar


def test_zero_drawdown_finite_floor_nop_and_costs():
    t,z=fixture(prices=(100,100,100,100,100,100),pivots=())
    t.high=t.open.copy(); t.low=t.open.copy()
    r=OpportunityEvaluator(t,z,3).evaluate((2,5,1),True)
    assert np.isfinite(r['records']['score']).all()
    assert (r['records']['net']<0).all()
    assert r['fitness']==0 and r['nop_pct']==100


def test_mirror_swaps_sides():
    t,z=fixture()
    a=OpportunityEvaluator(t,z,3).evaluate((2,5,1),True)
    mirrored=fixture(prices=200-t.open,pivots=((2,-1,96),(4,1,101)))[0]
    mz=SimpleNamespace(Params=z.Params,hindsight_pivots=lambda h,l,p:[(2,-1,96),(4,1,101)])
    b=OpportunityEvaluator(mirrored,mz,3).evaluate((2,5,1),True)
    np.testing.assert_allclose(a['records']['net'],b['records']['net'][::-1])
    np.testing.assert_allclose(a['records']['score'],b['records']['score'][::-1])


def test_financing_direction_and_exit_open_gap():
    t,z=fixture(prices=(100,100,100,90,90,90),pivots=())
    t.limit[:]=3; t.allowed=np.arange(6)<3
    t.financing[0,3:]=.7
    r=OpportunityEvaluator(t,z,3).evaluate((2,5,1),True)['records']
    assert r['net'][0,0] == pytest.approx(-10.9)
    assert r['drawdown'][0,0]==11
    assert r['mae'][0,0]==10


def test_missing_hour_and_session_not_compressed():
    assert contiguous_bounds(np.array([0,1,3,4,5])*HOUR,np.array([0,0,0,1,1])) == [(0,2),(2,3),(3,5)]


def test_schedule_matches_exhaustive_interval_subsets():
    o=np.array([0,1,2]); rel=np.array([[2,3,4],[3,4,4]])
    p=np.array([[.0005,.0001,.0004],[.0008,-.0001,.0002]])
    actual=schedule(o,rel,p,4)
    candidates=[(int(o[i]),int(rel[d,i]),p[d,i]) for i in range(3) for d in range(2)]
    best=0
    for mask in itertools.product((0,1),repeat=6):
        trades=sorted([x for x,k in zip(candidates,mask) if k])
        if all(a[1]<=b[0] for a,b in zip(trades,trades[1:])):
            best=max(best,sum(x[2] for x in trades))
    assert actual['net_pips'] == pytest.approx(best/PIP)
    assert all(a['release']<=b['origin'] for a,b in zip(actual['trades'],actual['trades'][1:]))


def test_schedule_negative_is_nop():
    r=schedule(np.array([0]),np.array([[1],[1]]),np.array([[-1.],[-2.]]),2)
    assert r['trades']==[] and r['net_pips']==0


def test_deap_full_budget_replay_and_validity_priority():
    def evaluate(g):
        return dict(eligible=g[0]<10,fitness=-abs(g[0]-6),params=g)
    a=search(evaluate,7,15,30)
    b=search(evaluate,7,15,30)
    assert a==b and len(a['history'])==30 and a['best']['eligible']
    with pytest.raises(ValueError,match='OWNER_MINIMUM'):
        search(evaluate,7,14,30)
    with pytest.raises(ValueError,match='OWNER_MINIMUM'):
        search(evaluate,7,15,29)


def test_resume_checks_identity_and_content(tmp_path):
    p=tmp_path/'x.json'
    receipt(p,'abc',dict(done=40))
    assert authenticated(p,'abc')==dict(done=40)
    with pytest.raises(ValueError,match='RESUME_IDENTITY'):
        authenticated(p,'other')
    r=json.loads(p.read_text()); r['payload']['done']=41
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='RESUME_ARTIFACT'):
        authenticated(p,'abc')


def test_decode_bounds_and_backstep():
    assert decode([1,0,100])==(2,1,1)
    assert decode([100,200,100])==(96,120,24)


def test_train_loader_predicate_and_pins(tmp_path,monkeypatch):
    import app.zigzag_opportunities as mod
    from pathlib import Path
    p=tmp_path/'bars'; p.write_bytes(b'x')
    spec=dict(inputs={k:str(p) for k in ('bars','closures','recipe')},
              input_sha256={k:'bad' for k in ('bars','closures','recipe')})
    with pytest.raises(ValueError,match='INPUT_HASH'):
        mod.load_tape(spec)
    source=Path(mod.__file__).read_text()
    assert "filters=[('available_at', '>=', start), ('available_at', '<', end)]" in source
    assert "['2020-01-01', '2024-01-01']" in source
