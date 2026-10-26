import numpy as np
import torch

from tests.unit.test_native_subset_bridge import Synthetic
from tests.unit.test_rl_subset import split_recipe
from app.rl_subset import seal_recipe
from tools.rl_subset_bridge import build_model
from tools.recipe_sac_queue import tensor_hash
from app.native_policy_sensitivity import probe_policy, FixedRule


def model():
    r=split_recipe();r['tensor']['window']['observations']=13;seal_recipe(r)
    return build_model(Synthetic(),r,7,'cpu',dict(total_timesteps=32,n_steps=16,batch_size=16,n_epochs=1))


def test_probe_detects_constant_policy_without_changing_weights():
    m=model()
    with torch.no_grad():
        m.policy.commands.weight.zero_();m.policy.commands.bias.zero_();m.policy.commands.bias[1]=1
    before=tensor_hash(m)
    x=np.random.default_rng(7).normal(size=(32,13,2)).astype('float32')
    r=probe_policy(m,x)
    flat=r['routes']['flat']
    assert flat['argmax_counts']==[0,32,0,0,0]
    assert flat['probability_span_max']==0
    assert flat['margin_input_gradient_abs_max']==0
    assert r['encoder_std_mean']>0 and tensor_hash(m)==before


def test_unsaturated_policy_has_input_response():
    m=model();x=np.random.default_rng(7).normal(size=(32,13,2)).astype('float32')
    before=tensor_hash(m);r=probe_policy(m,x)
    assert r['routes']['flat']['probability_span_max']>0
    assert r['routes']['flat']['margin_input_gradient_abs_max']>0
    assert r['perturbations']['zero']['flat_probability_abs_delta_max']>0
    assert tensor_hash(m)==before


def test_saturated_head_is_measured_separately_from_encoder():
    m=model()
    with torch.no_grad():
        m.policy.mlp_extractor.policy_net[0].weight.zero_()
        m.policy.mlp_extractor.policy_net[0].bias.fill_(20)
    x=np.random.default_rng(7).normal(size=(32,13,2)).astype('float32')
    r=probe_policy(m,x)
    assert r['actor_tanh_saturated_fraction']==1
    assert r['actor_std_mean']==0 and r['encoder_std_mean']>0


def test_fixed_rule_never_uses_features_or_illegal_commands():
    m=model()
    for rule in ('LONG_HOLD','LONG_CLOSE'):
        wrapper=FixedRule(m,rule)
        for mask,expected in [([1,1,1,0,0],1),([1,0,0,1,0],3 if rule=='LONG_CLOSE' else 0),
                              ([1,0,0,0,1],0),([1,0,0,0,0],0)]:
            a,_=wrapper.predict(dict(features=np.full((13,2),np.nan),action_mask=np.array(mask)))
            assert a[0]==expected and mask[int(a[0])]
