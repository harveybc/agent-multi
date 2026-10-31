import numpy as np
import pytest
import torch

from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401
from tests.unit.test_double_tcn_probe import probe_recipe
from env_plugins.three_signal_native import ThreeSignalEnv, command_for
from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.three_signal_learning import Patience, synthetic_env
from agent_plugins.discrete_curve_sac import tensor_observation


@pytest.mark.parametrize('position,expected', [(0,[0,1,2]), (1,[0,0,3]), (-1,[0,3,0])])
def test_complete_signal_table(position, expected):
    assert [command_for(a, position) for a in range(3)] == expected


@pytest.mark.parametrize('side', [1, -1])
def test_native_opposite_closes_only_and_reward_cash(request, side):
    base = request.getfixturevalue('curve_env')
    env = ThreeSignalEnv(base.engine, base.inputs, base.risk_config, 6, gamma=1.)
    obs = env.reset()
    signal = 1 if side == 1 else 2
    total = 0.
    while not env.engine.entry_gate(env.bar)[0]:
        _, r, _, _ = env.step(0); total += r
    _, r, _, _ = env.step(signal); total += r
    if env.engine.pos is not None:
        n = len(env.engine.trades)
        _, r, _, _ = env.step(3-signal); total += r
        assert env.engine.pos is None and len(env.engine.trades) == n+1
    while not env.finished:
        obs, r, _, _ = env.step(0); total += r
    assert set(obs) == {'features', 'mask'} and obs['mask'].all()
    assert total == pytest.approx(100*(env.engine.balance/env.engine.cfg.capital-1), abs=1e-8)
    assert all(t['direction'] == side for t in env.engine.trades)


def test_actor_twins_three_outputs_and_market_only(tmp_path):
    model = ThreeSignalSAC(probe_recipe())
    obs = tensor_observation({'features':np.zeros((13,2),np.float32), 'mask':np.ones(3,bool)})
    with torch.no_grad():
        a = model.distribution(obs).probs.clone()
        assert a.shape == (1,3)
        assert model.q_surfaces(obs['features']).shape == (2,1,3)
        other = obs.clone(); other['route'] = torch.tensor([[2]]); other['progress'] = torch.tensor([[99.]])
        assert torch.equal(a, model.distribution(other).probs)
    model.save(tmp_path/'m.pt'); model.load(tmp_path/'m.pt')
    assert torch.equal(a, model.distribution(obs).probs)


def test_patience_minimum_activity_and_improvement():
    p = Patience(minimum=100, patience=2, delta=.01)
    assert not p.observe(0., False, 50)['stop']
    assert not p.observe(-1., True, 100)['stop']
    assert not p.observe(-1.005, True, 200)['stop']
    assert p.observe(-1.006, True, 300)['stop']
    assert p.best == -1.
    p.observe(-.5, True, 400)
    assert p.bad == 0 and p.best == -.5


def test_synthetic_causal_histories_and_flat_control(request):
    base = request.getfixturevalue('curve_env')
    r = probe_recipe()
    e, signal = synthetic_env(r, base.risk_config, 17, 1, .99)
    assert np.array_equal(e.inputs.windows[30, :, 0], signal[np.arange(6,31,2)])
    flat, _ = synthetic_env(r, base.risk_config, 17, 1, .99, flat=True)
    flat.reset()
    while not flat.finished:
        flat.step(0)
    assert not flat.engine.trades and flat.engine.balance == flat.engine.cfg.capital


def test_sequential_native_replay_updates_not_terminal_labels(request):
    from tools.run_discrete_curve_probe import transition
    from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
    base=request.getfixturevalue('curve_env'); r=probe_recipe()
    env,_=synthetic_env(r,base.risk_config,99,1,.99)
    obs=env.reset(); replay=TensorDictReplayBuffer(storage=LazyTensorStorage(128),batch_size=16)
    terminal=nonterminal=0
    for i in range(110):
        action=i%3
        nxt,reward,done,duration=env.step(action)
        replay.extend(transition(obs,action,nxt,reward,done,duration))
        terminal+=done; nonterminal+=not done
        obs=env.reset() if done else nxt
    assert terminal and nonterminal>terminal
    model=ThreeSignalSAC(r)
    values=model.update(replay.sample())
    assert all(np.isfinite(list(values.values())))


def test_monitor_refuses_overlap_and_out_of_fit_scope():
    import json
    from pathlib import Path
    from tools.run_three_signal_baseline import validate_monitor_scope
    d=json.loads((Path(__file__).resolve().parents[2]/'examples/config/three_signal_sac_v1.json').read_text())
    recipe={'split':{'periods':{'TRAIN':['2020-01-01','2023-07-01'],'INNER_A':['2023-07-01','2024-01-01']}}}
    validate_monitor_scope(d,recipe)
    d['validation_blocks'][0]=['2023-01-01','2023-02-26']
    with pytest.raises(ValueError,match='MONITOR_SCOPE'): validate_monitor_scope(d,recipe)


def test_patience_reports_no_eligible_checkpoint():
    p=Patience(100,2,.01)
    assert not p.observe(0.,False,100)['stop']
    result=p.observe(0.,False,200)
    assert result['stop'] and result['stop_reason']=='no_eligible_checkpoint'
    assert p.best is None


@pytest.mark.parametrize('mirror',[1,-1])
def test_final_close_diagnostic_with_opposite_signal(request,mirror):
    from app.three_signal_learning import evaluate
    from tools.run_three_signal_baseline import known_logits
    base=request.getfixturevalue('curve_env')
    env,s=synthetic_env(probe_recipe(),base.risk_config,91001,mirror,.99)
    metrics=evaluate(env,known_logits(s),diagnose=True)
    assert metrics['early_close_diagnostic']['proposals']>0
    assert metrics['cash_error']<1e-6
