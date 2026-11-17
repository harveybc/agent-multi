import copy
import numpy as np
import pytest
from app.prioritized_training_diagnosis import data_profile, summarize_history


def example():
    m=dict(return_pct=1.,signals={'0':4,'1':1,'2':0},trades=2,orders_per_week=1.)
    return [dict(epoch=1,metrics={'TRAIN':m,'VALIDATION':copy.deepcopy(m)},
                 epoch_seconds=10.,decisions=100,updates=25)], dict(completed_epochs=1,minimum_epochs=100,
                 updates=25,stage='COMPLETE')


def test_summary_no_promotion_or_replay_claim():
    h,p=example(); r=summarize_history(h,p)
    assert r['eta_to_minimum_seconds']==990
    assert r['updates_per_completed_decision']==.25
    assert r['latest']['validation_signal_shares']['0']==.8
    assert r['selected_epoch'] is None


def test_partial_epoch_never_mixed_exposure():
    h,p=example();p['stage']='TRAIN';p['updates']=999
    assert summarize_history(h,p)['updates_per_completed_decision']==.25


def test_absent_updates_are_not_invented():
    h,p=example();h[0].pop('updates')
    assert summarize_history(h,p)['updates_per_completed_decision'] is None


@pytest.mark.parametrize('kind',['empty','duplicate','race','nan'])
def test_bad_histories_fail(kind):
    h,p=example()
    if kind=='empty':h=[]
    if kind=='duplicate':h*=2
    if kind=='race':p['completed_epochs']=0
    if kind=='nan':h[0]['metrics']['TRAIN']['return_pct']=float('nan')
    with pytest.raises(ValueError):summarize_history(h,p)


def profile_fixture():
    ts=np.array(['2020-01-01T00','2020-01-01T01','2020-01-01T04','2024-01-01T00'],dtype='datetime64[ns]').astype('int64')
    return np.array([[1.,0],[2,0],[3,0],[5,0]]),ts,np.ones(4,bool),np.array(['TRAIN']*3+['VALIDATION']),['a','constant']


def test_data_gaps_and_constants():
    r=data_profile(*profile_fixture())
    assert r['columns'][0]['years']['2020']['adjacent_pairs']==1
    assert r['columns'][0]['years']['2020']['lag1'] is None
    assert r['columns'][1]['constant_train']
    assert r['columns'][1]['validation_mean_shift_train_std'] is None


def test_no_test_period():
    x,t,v,p,c=profile_fixture();t[-1]=np.datetime64('2025-01-01','ns').astype('int64')
    with pytest.raises(ValueError,match='TEST_DATA'):data_profile(x,t,v,p,c)


@pytest.mark.parametrize('kind',['valid','wrong_identity','missing','result_wrong'])
def test_snapshot_contract(tmp_path,kind):
    import json
    from tools.run_prioritized_training_diagnosis import fetch
    docs={'SPEC.json':{'identity':'run'},'PROGRESS.json':{'identity':'run'},'HISTORY.json':[]}
    if kind=='wrong_identity':docs['PROGRESS.json']['identity']='wrong'
    if kind=='missing':docs.pop('HISTORY.json')
    if kind=='result_wrong':docs['RESULT.json']={'identity':'other'}
    for name,value in docs.items():(tmp_path/name).write_text(json.dumps(value))
    args=dict(root=str(tmp_path),identity='run')
    if kind=='valid':
        raw,actual=fetch(args)
        assert actual==docs
        assert raw['SPEC.json']==(tmp_path/'SPEC.json').read_bytes()
    else:
        with pytest.raises(ValueError):fetch(args)
