import torch

from agent_plugins.hybrid_orders import SeparateHeadOrderPolicy
from tools.qualify_hybrid_learning import ObservableSignal
from tests.unit.test_frozen_recipe_inputs import recipe


def test_actor_and_value_latent_heads_are_independent():
    env=ObservableSignal()
    policy=SeparateHeadOrderPolicy(env.observation_space,env.action_space,lambda _: .0003,recipe=recipe())
    obs,_=env.reset(seed=7)
    tensor,_=policy.obs_to_tensor(obs)
    policy.zero_grad()
    _,value=policy._parts(tensor)
    value.sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in policy.mlp_extractor.policy_net.parameters())
    assert any(p.grad is not None and p.grad.any() for p in policy.mlp_extractor.value_net.parameters())
    policy.zero_grad()
    dist,_=policy._parts(tensor)
    (-dist.cat.log_prob(torch.tensor([1]))).sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in policy.mlp_extractor.value_net.parameters())
    assert any(p.grad is not None and p.grad.any() for p in policy.mlp_extractor.policy_net.parameters())
