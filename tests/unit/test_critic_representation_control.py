import copy

import numpy as np
import pytest
import torch

from app.critic_representation_control import shuffled_fit_records, state_kind, recommendation
from app.selected_critic_calibration import calibrate
from agent_plugins.three_signal_sac import ThreeSignalSAC
from tests.unit.test_double_tcn_probe import probe_recipe


def records():
    return [dict(role=role,year=year,position=pos,allowed=True,
        outcomes=[dict(reward_return=float(i+a),soft_return=float(i+a)) for a in range(3)])
        for i,(role,year,pos) in enumerate(
            [('fit',2020,0)]*8+[('fit',2021,1)]*8+[('monitor',2023,0)]*4)]


def test_shuffle_preserves_strata_vectors_and_monitor_without_mutating_source():
    source=records();before=copy.deepcopy(source)
    shuffled,receipt=shuffled_fit_records(source,19)
    assert source==before and receipt['moved_rows']>0
    assert (shuffled,receipt)==shuffled_fit_records(source,19)
    for i,src in enumerate(receipt['source_indices']):
        assert source[i]['year']==source[src]['year']
        assert state_kind(source[i])==state_kind(source[src])
        assert shuffled[i]['outcomes']==source[src]['outcomes']
        if source[i]['role']=='monitor':assert i==src and shuffled[i]==source[i]
    altered=copy.deepcopy(source)
    altered[0]['outcomes'][0]['reward_return']=999
    assert shuffled_fit_records(altered,19)[1]==receipt
    source[0]['year']=2024
    with pytest.raises(ValueError,match='FIT_YEAR'):shuffled_fit_records(source,19)


@pytest.mark.parametrize('scope',['heads','critic_representation'])
def test_real_model_calibration_moves_only_requested_scope(scope):
    torch.set_num_threads(1);torch.manual_seed(19)
    model=ThreeSignalSAC(probe_recipe())
    before={k:v.clone() for k,v in model.loss.state_dict().items()}
    seen=[]
    calibrate(model,torch.randn(20,13,2),records(),19,seen.append,updates=2,scope=scope)
    changed=[k for k,v in model.loss.state_dict().items() if not torch.equal(v,before[k])]
    assert changed and all(k.startswith(('qvalue_network_params.','target_qvalue_network_params.')) for k in changed)
    assert any('.encoder.' in k for k in changed)==(scope=='critic_representation')
    assert [p['updates'] for p in seen]==[0,2]


def test_recommendation_requires_both_seeds_and_negative_controls():
    rows=[dict(seed=s,scope=k,labels=l,metrics={'held':{'minimum':dict(
        pairwise_accuracy=.6 if k=='critic_representation' and l=='true' else .5,
        mean_regret=.2 if k=='critic_representation' and l=='true' else .3)}})
        for s in (19,43) for k in ('heads','critic_representation') for l in ('true','shuffled')]
    assert recommendation(rows)=='DESIGN_BOUNDED_OWN_POLICY_ACTOR_STEP'
    rows[-2]['metrics']['held']['minimum']['mean_regret']=.4
    assert recommendation(rows)=='DIAGNOSE_PARTIAL_OBSERVATION_AND_NATIVE_COVERAGE'
    with pytest.raises(ValueError,match='INCOMPLETE'):recommendation(rows[:-1])


def test_default_head_calibration_matches_explicit_scope_and_rejects_actor():
    torch.set_num_threads(1)
    x=torch.randn(20,13,2)
    a=ThreeSignalSAC(probe_recipe(),seed=19)
    b=ThreeSignalSAC(probe_recipe(),seed=19)
    calibrate(a,x,records(),19,lambda p:None,updates=2)
    calibrate(b,x,records(),19,lambda p:None,updates=2,scope='heads')
    assert all(torch.equal(v,b.loss.state_dict()[k]) for k,v in a.loss.state_dict().items())
    with pytest.raises(ValueError,match='SCOPE'):
        calibrate(None,None,[],19,None,scope='actor')
