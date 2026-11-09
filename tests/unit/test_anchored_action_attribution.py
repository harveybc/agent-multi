import numpy as np
import pytest

from app.anchored_action_attribution import event_metrics, summarize_events


def fixture():
    return dict(old_signal=0,new_signal=2,contribution_usd=-10.,state=dict(position=None),
        isolated_old=dict(net_usd=100.,reward_return=.1),
        isolated_new=dict(net_usd=90.,reward_return=.2),
        prefix_old=dict(reward_return=.1,first_effect=dict(command=0,entries=0,closes=0)),
        prefix_new=dict(reward_return=.2,first_effect=dict(command=2,entries=1,closes=0)))


def test_margin_q_and_cash_reward_are_not_conflated():
    e=fixture();r=event_metrics(e,np.array([.01,-1.,0.]),np.array([0.,-1.,.01]),np.array([[0.,-.1,.2],[.1,-.2,.3]]),.005)
    assert r['parent_pair_margin']==pytest.approx(.01)
    assert r['candidate_pair_margin']==pytest.approx(.01)
    assert r['q_gap']==pytest.approx(.2)
    assert r['isolated_reward_gap']==pytest.approx(.1)
    assert r['isolated_cash_gap']==pytest.approx(-10.)
    assert r['cash_reward_sign_conflict'] and not r['q_reward_sign_conflict']
    assert r['effect']=='added_entry' and r['outcome']=='harmful' and r['kl']>0
    assert summarize_events([r])['cash_delta_usd']==-10.


def test_closed_position_and_malformed_input():
    e=fixture();e['state']['position']=dict(direction=1)
    e['prefix_new']['first_effect'].update(command=3,entries=0,closes=1)
    a=np.array([.01,-1.,0.]);b=np.array([0.,-1.,.01]);q=np.zeros((2,3))
    assert event_metrics(e,a,b,q,.005)['effect']=='added_close_command'
    with pytest.raises(ValueError):event_metrics(e,a,b,np.zeros((3,2)),.005)
    with pytest.raises(ValueError):event_metrics(e,a,b*float('nan'),q,.005)
    with pytest.raises(ValueError):event_metrics(e,b,a,q,.005)


def test_native_tie_and_archive_support_gate():
    from types import SimpleNamespace
    from tools.attribute_anchored_actions import check_trace
    e=fixture()
    # Native routing returns NOP when BUY and SELL tie, even above NOP's logit.
    assert event_metrics(e,np.array([0.,1.,1.]),np.array([0.,1.,2.]),np.zeros((2,3)),.005)['old_signal']==0
    inputs=SimpleNamespace(row_ts_ns=np.array([10,20]),valid=np.ones(2,dtype=bool))
    logits=np.array([[0.,1.,1.],[0.,1.,2.]])
    check_trace(inputs,logits,[dict(timestamp=10,signal=0),dict(timestamp=20,signal=2)])
    with pytest.raises(ValueError):check_trace(inputs,logits,[dict(timestamp=11,signal=0)])
    with pytest.raises(ValueError):check_trace(inputs,logits,[dict(timestamp=10,signal=1)])


def test_report_preserves_harmful_and_neutral_events(tmp_path):
    import csv
    from tools.attribute_anchored_actions import export
    e=fixture();row=event_metrics(e,np.array([.01,-1.,0.]),np.array([0.,-1.,.01]),np.zeros((2,3)),.005)
    row.update(replica='r19',period='TRAIN',utc='2020-01-01T00:00:00Z')
    neutral=dict(row,contribution_usd=0.,outcome='neutral')
    account=dict(replica='r19',period='TRAIN',**summarize_events([row,neutral]),reconciliation=dict(error_usd=0.))
    export(tmp_path,[account],[row,neutral])
    records=list(csv.DictReader((tmp_path/'EVENTS.csv').open()))
    assert len(records)==2 and records[0]['outcome']=='harmful' and records[1]['outcome']=='neutral'
