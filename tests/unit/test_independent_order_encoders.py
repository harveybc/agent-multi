import torch
from stable_baselines3 import PPO

from agent_plugins.qualified_orders import QualifiedOrderPolicy
from agent_plugins.independent_order_encoders import IndependentOrderPolicy
from tools.qualify_hybrid_learning import ObservableSignal
from tests.unit.test_frozen_recipe_inputs import recipe


def test_equal_initial_function_disjoint_gradients_and_reload(tmp_path):
    env=ObservableSignal()
    def build(cls):
        return PPO(cls,env,policy_kwargs=dict(recipe=recipe(),learned_risk=False),
                   seed=7,device='cpu',n_steps=8,batch_size=8)
    control=build(QualifiedOrderPolicy)
    model=build(IndependentOrderPolicy)
    policy=model.policy
    obs,_=env.reset(seed=7)
    tensor,_=policy.obs_to_tensor(obs)
    d,v=policy._parts(tensor)
    ref,rv=control.policy._parts(tensor)
    torch.testing.assert_close(d.cat.probs,ref.cat.probs,rtol=0,atol=0)
    torch.testing.assert_close(v,rv,rtol=0,atol=0)
    assert not ({id(p) for p in policy.features_extractor.parameters()} &
                {id(p) for p in policy.critic_encoder.parameters()})
    policy.zero_grad();v.sum().backward()
    assert all(p.grad is None for p in policy.features_extractor.parameters())
    assert any(p.grad is not None and p.grad.any() for p in policy.critic_encoder.parameters())
    policy.zero_grad();d,_=policy._parts(tensor)
    (-d.cat.log_prob(torch.tensor([1]))).sum().backward()
    assert all(p.grad is None for p in policy.critic_encoder.parameters())
    assert any(p.grad is not None and p.grad.any() for p in policy.features_extractor.parameters())
    model.save(tmp_path/'model.zip')
    loaded=PPO.load(tmp_path/'model.zip',device='cpu')
    ld,lv=loaded.policy._parts(tensor)
    torch.testing.assert_close(d.cat.probs,ld.cat.probs,rtol=0,atol=0)
    torch.testing.assert_close(v,lv,rtol=0,atol=0)
