import numpy as np
import torch
from agent_plugins.double_tcn_probe import DoubleTCNExtractor, ResidualTCN
from app.order_qualification import OrderSignal
from agent_plugins.discrete_curve_sac import CurveSAC, route_surface, tensor_observation
from tests.unit.test_frozen_recipe_inputs import recipe, seal


def probe_recipe():
    r = recipe()
    r['tensor']['window']['observations'] = 13
    return seal(r)


def test_structure_and_no_future_or_routing_influence():
    r = probe_recipe()
    env = OrderSignal((13, 2))
    encoder = DoubleTCNExtractor(env.observation_space, r)
    assert sum(isinstance(m, ResidualTCN) for m in encoder.branches[0]) == 2
    assert sum(isinstance(m, ResidualTCN) for m in encoder.core) == 2
    a = {'features': torch.randn(2, 13, 2), 'action_mask': torch.ones(2, 5)}
    b = {**a, 'features': a['features'].clone(), 'action_mask': torch.zeros(2, 5)}
    assert torch.equal(encoder(a), encoder({**a, 'action_mask': b['action_mask']}))
    b['features'][:, 7:, :] += 100
    assert torch.equal(encoder.sequence(a)[:, :, :7], encoder.sequence(b)[:, :, :7])
    assert encoder(a).shape == (2, 13 * 16)


def test_checkpoint_codec_and_no_risk_head(tmp_path):
    torch.set_num_threads(1)
    model = CurveSAC(probe_recipe())
    obs = tensor_observation(dict(features=np.ones((13, 2), np.float32),
        route=np.array([1]), progress=np.array([.5], np.float32), mask=np.array([1, 1, 0], bool)))
    with torch.no_grad():
        before = model.distribution(obs).probs.clone()
        before_q = model.q_surfaces(obs['features']).clone()
    model.save(tmp_path / 'model.pt')
    loaded = CurveSAC(probe_recipe(), seed=19)
    loaded.load(tmp_path / 'model.pt')
    with torch.no_grad():
        assert torch.equal(before, loaded.distribution(obs).probs)
        assert torch.equal(before_q, loaded.q_surfaces(obs['features']))
    assert before[0, 2] == 0
    assert not hasattr(model.actor_net, 'risks')


def test_position_sensitive_controller_market_independent_surface():
    surface = torch.zeros(2, 39)
    curves = surface[:, 3:].reshape(2, 2, 9, 2)
    curves[:, :, 0, 1] = 10
    curves[:, :, -1, 0] = 10
    routed = route_surface(surface, torch.tensor([[1], [1]]), torch.tensor([[0.], [1.]]))
    assert routed.argmax(-1).tolist() == [1, 0]
    assert torch.equal(surface[0], surface[1])


def test_sac_update_finite_masked_and_changes_weights():
    from tensordict import TensorDict
    model = CurveSAC(probe_recipe())
    obs = TensorDict(dict(features=torch.randn(4, 13, 2), route=torch.ones(4, 1, dtype=torch.long),
        progress=torch.rand(4, 1), mask=torch.tensor([[True, True, False]] * 4)), [4])
    batch = obs.clone()
    batch['action'] = torch.zeros(4, dtype=torch.long)
    batch['next'] = obs.clone()
    batch['next', 'reward'] = torch.ones(4, 1)
    batch['next', 'done'] = torch.ones(4, 1, dtype=torch.bool)
    batch['next', 'terminated'] = torch.ones(4, 1, dtype=torch.bool)
    before = [p.clone() for p in model.loss.parameters()]
    assert all(np.isfinite(list(model.update(batch).values())))
    assert any(not torch.equal(a, b) for a, b in zip(before, model.loss.parameters()))


def test_elapsed_discount_uses_native_duration():
    from tools.run_discrete_curve_probe import transition
    obs = dict(features=np.ones((13, 2), np.float32), route=np.array([0]),
               progress=np.array([.5], np.float32), mask=np.array([True] * 3))
    td = transition(obs, 0, obs, .2, False, 4)
    model = CurveSAC(probe_recipe())
    with torch.no_grad():
        long = model.loss._compute_target(td).clone()
        td['steps_to_next_obs'].fill_(1)
        short = model.loss._compute_target(td).clone()
    assert not torch.equal(long, short)
    td['next', 'done'].fill_(True)
    td['next', 'terminated'].fill_(True)
    with torch.no_grad():
        assert torch.allclose(model.loss._compute_target(td), torch.tensor([.2]))


def test_pointwise_group_preserved():
    r = probe_recipe()
    r['representation']['resolved']['kinds'] = ['pointwise']
    encoder = DoubleTCNExtractor(OrderSignal((13, 2)).observation_space, seal(r))
    x = torch.randn(1, 2, 13)
    y = x.clone()
    y[:, :, :-1] += 100
    assert torch.equal(encoder.branches[0](x)[:, :, -1], encoder.branches[0](y)[:, :, -1])
