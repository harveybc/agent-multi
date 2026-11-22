import copy

from tools.verify_loss_cache397 import compare


def case():
    a=dict(epoch=1,steps=100,decisions=100,updates=25,epoch_updates=25,
           mean_update={'critic':0.25},last_update={'alpha':0.05},
           metrics={'VALIDATION':{'buy':{'trades':3},'sell':{'trades':7}}},epoch_seconds=10.)
    b=copy.deepcopy(a)
    b['epoch_seconds']=6.
    return a,b,dict(setup_seconds=1.,paired_updates=16,max_deltas={'grad':0.}),dict(speedup=10/6)


def test_equal_and_faster_including_setup():
    assert compare(*case())['pass_gate']


def test_same_count_wrong_direction_is_rejected():
    a,b,g,t=case()
    b['metrics']['VALIDATION']={'buy':{'trades':7},'sell':{'trades':3}}
    result=compare(a,b,g,t)
    assert not result['pass_gate']
    assert 'metrics.VALIDATION.buy.trades' in result['differences']


def test_exact_metrics_do_not_hide_loss_or_more_cost():
    a,b,g,t=case()
    b['last_update']['alpha']+=1e-12
    assert not compare(a,b,g,t)['pass_gate']
    a,b,g,t=case()
    g['setup_seconds']=5.
    assert not compare(a,b,g,t)['pass_gate']
