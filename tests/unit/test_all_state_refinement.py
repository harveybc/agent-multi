import numpy as np
import pytest

from app.all_state_refinement import choose_panel, validate_panel, policy_effect, qualification_identity
from app.rl_subset import digest


def test_qualification_identity_is_configuration_digest_not_embedded_field():
    config=dict(schema='one_update_actor_qualification.v1',seeds=[19,43])
    assert qualification_identity(config)==digest(config)
    with pytest.raises(ValueError):qualification_identity(dict(schema='other'))


def record(position=0, role='fit'):
    boundary = 1672531200000000000
    end = boundary if role == 'fit' else 1704067200000000000
    start = 1577836800000000000 if role == 'fit' else boundary
    vals = [1., 1., 2.] if position == 1 else [1., 2., 1.] if position == -1 else [0., 2., 1.]
    outcomes = [dict(reward_return=v, soft_return=v, net_usd=100*v,
                     parent_unchanged=True, first_effect=dict(duration=1, entries=int(position == 0 and a>0)))
                for a,v in enumerate(vals)]
    return dict(timestamp=start+30*3600*10**9, window_start=start, terminal=end,
                position=position, role=role, outcomes=outcomes)


def test_panel_deterministic_outcome_independent_and_balanced():
    rows = [dict(timestamp=1577836800000000000+i*3600*10**9,
                 year=2020, position=p, row=i) for p in (-1,0,1)
            for i in range((p+1)*20,(p+1)*20+20)]
    a=choose_panel(rows)
    assert len(a)==16
    assert [sum(r['position']==p for r in a) for p in (-1,0,1)]==[4,8,4]
    assert [r['timestamp'] for r in a]==[r['timestamp'] for r in choose_panel([dict(r,unused=999) for r in rows])]
    with pytest.raises(ValueError):choose_panel(rows+[rows[0]])
    with pytest.raises(ValueError):choose_panel([dict(rows[0],year=2024)])


def test_valid_position_labels_and_periods():
    rows=[record(p,r) for p in (-1,0,1) for r in ('fit','monitor')]
    assert validate_panel(rows)


@pytest.mark.parametrize('mutation', ['fit_terminal','context','role','hold','nan','parent','soft','position','timestamp','empty_role'])
def test_invalid_panel_refused(mutation):
    rows=[record(1),record(-1,'monitor')]
    if mutation=='fit_terminal':rows[0]['terminal']+=1
    if mutation=='context':rows[1]['window_start']-=1
    if mutation=='role':rows[0]['role']='test'
    if mutation=='hold':rows[0]['outcomes'][1]['reward_return']+=1
    if mutation=='nan':rows[0]['outcomes'][0]['net_usd']=float('nan')
    if mutation=='parent':rows[0]['outcomes'][0]['parent_unchanged']=False
    if mutation=='soft':rows[0]['outcomes'][0]['soft_return']+=1
    if mutation=='position':rows[0]['position']=2
    if mutation=='timestamp':rows[0]['timestamp']=rows[0]['terminal']
    if mutation=='empty_role':rows=rows[:1]
    with pytest.raises(ValueError):validate_panel(rows)


def test_policy_effect_distinguishes_regret_and_expected_change():
    old=np.array([[3.,0.,0.]])
    new=np.array([[0.,0.,3.]])
    target=np.array([[0.,1.,2.]])
    result=policy_effect(old,new,target)
    assert result['expected_return_delta']>0
    assert result['greedy_regret_before']==2 and result['greedy_regret_after']==0
    with pytest.raises(ValueError):policy_effect(old,new,target[:,:2])
