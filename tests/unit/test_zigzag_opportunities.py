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


def test_strictly_later_pivot_no_forced_flat_substitute():
    t,z=fixture()
    t.limit[:]=3
    t.allowed=np.arange(len(t.open))<3
    r=OpportunityEvaluator(t,z,3).evaluate((2,5,1),True)['records']
    assert not r['pivot_exit'][0,2]  # pivot at entry is not a future exit
    assert r['end'][0,2] == -1
    assert np.isnan(r['target'][0,2])
    assert np.isnan(r['duration'][0,2])
    assert r['release'][0,2] == -1
    assert r['release'][0,0] == 3  # pivot at bar 2 releases only after bar
    assert r['end'][1,0] == 4  # hindsight target, not replaced by flat at 3


def test_zero_drawdown_finite_floor_nop_and_costs():
    t,z=fixture(prices=(100,100,100,100,100,100),pivots=())
    t.high=t.open.copy(); t.low=t.open.copy()
    r=OpportunityEvaluator(t,z,3).evaluate((2,5,1),True)
    assert np.isnan(r['records']['score']).all()
    assert r['fitness']==0 and r['nop_pct']==0
    assert r['censored_origins']==len(r['records']['origin'])


def test_mirror_swaps_sides():
    t,z=fixture()
    a=OpportunityEvaluator(t,z,3).evaluate((2,5,1),True)
    mirrored=fixture(prices=200-t.open,pivots=((2,-1,96),(4,1,101)))[0]
    mz=SimpleNamespace(Params=z.Params,hindsight_pivots=lambda h,l,p:[(2,-1,96),(4,1,101)])
    b=OpportunityEvaluator(mirrored,mz,3).evaluate((2,5,1),True)
    np.testing.assert_allclose(a['records']['net'],b['records']['net'][::-1])
    np.testing.assert_allclose(a['records']['score'],b['records']['score'][::-1])


def test_financing_direction_to_real_pivot():
    t,z=fixture(prices=(100,100,100,90,90,90),pivots=((3,1,91),(4,-1,89)))
    t.limit[:]=3; t.allowed=np.arange(6)<3
    t.financing[0,3:]=.7
    r=OpportunityEvaluator(t,z,3).evaluate((2,5,1),True)['records']
    # Both buys lose: the later endpoint has the less-negative risk/time score.
    assert r['net'][0,0] == pytest.approx(-11.9)
    assert r['drawdown'][0,0]==12
    assert r['mae'][0,0]==11


@pytest.mark.parametrize('mirror', [False, True])
def test_next_two_types_cross_first_extreme_and_old_time_cap(mirror):
    prices=np.array([100,102,101,100,98,96,94,92,90,91],float)
    pivots=[(1,1,103),(8,-1,89)]
    if mirror:
        prices=200-prices
        pivots=[(i,-kind,200-price) for i,kind,price in pivots]
    t,z=fixture(prices=prices,pivots=pivots)
    t.fill_cost[:]=0
    r=OpportunityEvaluator(t,z,1).evaluate((2,5,1),True)['records']
    side=0 if mirror else 1
    assert r['end'][side,0]==8  # greater than the removed 4*h cap
    assert r['end'][1-side,0]==1
    assert r['net'][side,0]==11
    assert r['mae'][side,0]==3  # opposite extreme is inside the held path
    assert r['duration'][side,0]==9
    assert r['pair_valid'][0]


def test_no_future_pivot_borrowed_across_unobserved_chunk():
    t,z=fixture()
    t.chunks=[(0,3),(3,6)]
    z.hindsight_pivots=lambda h,l,p:[(1,1,float(h[1]))]
    r=OpportunityEvaluator(t,z,1).evaluate((2,5,1),True)['records']
    assert r['end'][0,2]==-1
    assert not r['pair_valid'].any()


def test_three_events_select_third_but_never_fourth():
    t,z=fixture(prices=(100,101,100,102,102,102,102,102),
                pivots=((1,1,102),(2,-1,99),(3,1,103),(6,-1,101)))
    t.fill_cost[:]=0
    a=OpportunityEvaluator(t,z,1,future_pivots=3).evaluate((2,5,1),True)['records']
    np.testing.assert_array_equal(a['candidate_end'][0], [1,2,3])
    assert a['end'][0,0] in (1,2,3)
    assert a['end'][1,0] in (1,2,3)
    # Change only the fourth event and prices strictly after the third.
    t.high[4:]=1000; t.low[4:]=1
    z.hindsight_pivots=lambda h,l,p:[(1,1,102),(2,-1,99),(3,1,103),(6,-1,1)]
    b=OpportunityEvaluator(t,z,1,future_pivots=3).evaluate((2,5,1),True)['records']
    np.testing.assert_array_equal(a['score'][:,0], b['score'][:,0])


def test_third_endpoint_can_improve_best_opportunity():
    t,z=fixture(prices=(100,100,99,110,110),
                pivots=((1,1,101),(2,-1,98),(3,1,111)))
    t.fill_cost[:]=0
    two=OpportunityEvaluator(t,z,1,2).evaluate((2,5,1),True)['records']
    three=OpportunityEvaluator(t,z,1,3).evaluate((2,5,1),True)['records']
    assert three['end'][0,0]==3
    assert three['score'][0,0]>two['score'][:,0].max()
    from tools.audit_zigzag_opportunity_results import audit_event_choices
    assert audit_event_choices(t,z,(2,5,1),three,3,[0])==6
    three['candidate_end'][0,2]=4
    with pytest.raises(AssertionError):
        audit_event_choices(t,z,(2,5,1),three,3,[0])


@pytest.mark.parametrize('k', [4,5])
def test_extra_events_only_enlarge_the_allowed_exit_set(k):
    from tools.audit_zigzag_opportunity_results import audit_event_choices
    prices=(100,102,100,98,100,105,100,95,100,110,100,90)
    events=[(1,1,103),(3,-1,97),(5,1,106),(7,-1,94),(9,1,111),(11,-1,89)]
    t,z=fixture(prices=prices,pivots=events)
    t.fill_cost[:]=0
    a=OpportunityEvaluator(t,z,1,k-1).evaluate((2,5,1),True)['records']
    b=OpportunityEvaluator(t,z,1,k).evaluate((2,5,1),True)['records']
    assert b['candidate_end'][0].tolist()==[i for i,_,_ in events[:k]]
    assert np.all(b['score'][:,0]>=a['score'][:,0])
    assert audit_event_choices(t,z,(2,5,1),b,k,[0])==2*k
    last=events[k-1][0]
    t.high[last+1:]=1000; t.low[last+1:]=1
    z.hindsight_pivots=lambda h,l,p:events[:k]+[(events[k][0],-1,1)]
    c=OpportunityEvaluator(t,z,1,k).evaluate((2,5,1),True)['records']
    np.testing.assert_array_equal(b['score'][:,0],c['score'][:,0])


def test_future_event_count_outside_registered_sweep_refused():
    t,z=fixture()
    with pytest.raises(ValueError,match='FUTURE_PIVOTS_MUST_BE_2_TO_5'):
        OpportunityEvaluator(t,z,1,6)


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
