import numpy as np
import torch
from stable_baselines3 import PPO
from agent_plugins.qualified_orders import QualifiedOrderPolicy, GroupedSequenceExtractor
from agent_plugins.hybrid_orders import ElapsedRolloutBuffer
from app.order_qualification import OrderSignal, conditional_probe, passes
from tests.unit.test_frozen_recipe_inputs import recipe


def test_lag_signal_and_early_close_routes_are_observable():
    env = OrderSignal((3, 2), lag=2)
    env.reset(seed=7)
    for route in ('flat', 'long', 'short'):
        for s in (-1, 0, 1):
            env.route, env.signal = route, s
            o = env.observation()
            assert (o['features'][0] == s).all()
            assert not o['features'][1:].any()
            assert o['action_mask'][env.expected()] == 1
    assert not passes([])


def test_sequence_extractor_mask_independence_and_causal_prefix():
    torch.manual_seed(7)
    env = OrderSignal((3, 2))
    ext = GroupedSequenceExtractor(env.observation_space, recipe())
    x = torch.randn(1, 3, 2)
    a = ext({'features': x, 'action_mask': torch.ones(1, 5)})
    b = ext({'features': x, 'action_mask': torch.zeros(1, 5)})
    assert torch.equal(a, b)
    y = x.clone(); y[:, -1] += 10
    c = ext({'features': y}).reshape(1, 8, 3)
    assert torch.equal(a.reshape(1, 8, 3)[:, :, :-1], c[:, :, :-1])
    assert not torch.equal(a.reshape(1, 8, 3)[:, :, -1], c[:, :, -1])


def test_new_policy_serialization_and_masks(tmp_path):
    torch.set_num_threads(1)
    env = OrderSignal((3, 2))
    model = PPO(QualifiedOrderPolicy, env, policy_kwargs=dict(recipe=recipe(), learned_risk=True),
                rollout_buffer_class=ElapsedRolloutBuffer, n_steps=8, batch_size=8, device='cpu', seed=7)
    env.reset(seed=7)
    before = conditional_probe(model, env)
    model.save(tmp_path/'model')
    after = conditional_probe(PPO.load(tmp_path/'model', device='cpu'), env)
    assert before == after
    for r in after:
        env.route, env.signal = r['route'], r['signal']
        mask = env.observation()['action_mask']
        assert not np.asarray(r['probabilities'])[mask == 0].any()


def test_qualification_rejects_duplicate_routes_and_forged_expectation():
    from copy import deepcopy
    rows=[]
    env=OrderSignal((3,2))
    for route in ('flat','long','short'):
        for signal in (-1,0,1):
            env.route,env.signal=route,signal
            p=[0.]*5;p[env.expected()]=1.
            rows.append(dict(route=route,signal=signal,expected=env.expected(),
                             chosen=env.expected(),probabilities=p))
    assert passes(rows)
    duplicate=deepcopy(rows);duplicate[-1]=duplicate[0]
    assert not passes(duplicate)
    changed=deepcopy(rows);changed[0].update(expected=1,chosen=1,probabilities=[0,1,0,0,0])
    assert not passes(changed)
