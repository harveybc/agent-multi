import numpy as np
import torch
import pytest
from gymnasium import Env, spaces
from stable_baselines3 import SAC

from app.recipe_runtime import build_sac
from tests.unit.test_frozen_recipe_inputs import recipe
from tools.recipe_sac_queue import DESIGN, tensor_hash


class Tiny(Env):
    observation_space = spaces.Dict({'features': spaces.Box(-np.inf, np.inf, (3, 2), np.float32)})
    action_space = spaces.Box(-1., 1., (3,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.n = 0
        return {'features': np.ones((3, 2), np.float32)}, {}

    def step(self, action):
        self.n += 1
        return {'features': np.full((3, 2), self.n/20, np.float32)}, float(-np.square(action).sum()), self.n == 20, False, {'discount': .99}


def build(p=None):
    design = {**DESIGN, 'learning_starts': 2, 'batch_size': 4, 'buffer_size': 32}
    return build_sac(Tiny(), recipe(), design, 'GROUPED', 'cpu', critic_dropout=p)


def test_initial_tensor_parity_and_only_critic_probability_changes():
    torch.set_num_threads(1)
    legacy, base, treatment = build(), build(.3), build(0.)
    assert tensor_hash(legacy) == tensor_hash(base) == tensor_hash(treatment)
    for model in (legacy, base, treatment):
        assert model.actor.features_extractor.drop.p == .3
        assert model.actor.optimizer.param_groups[0]['weight_decay'] == .01
    assert base.critic.features_extractor.drop.p == .3
    assert treatment.critic.features_extractor.drop.p == 0
    assert treatment.critic_target.features_extractor.drop.p == 0
    assert treatment.critic.features_extractor is not treatment.actor.features_extractor


def test_fresh_base_training_exact_and_treatment_reload(tmp_path):
    legacy = build()
    legacy.learn(total_timesteps=16)
    digest = tensor_hash(legacy)
    base = build(.3)
    base.learn(total_timesteps=16)
    assert digest == tensor_hash(base)
    model = build(0.)
    model.learn(total_timesteps=16)
    obs, _ = Tiny().reset()
    a = model.predict(obs, deterministic=True)[0]
    model.save(tmp_path/'m.zip')
    loaded = SAC.load(tmp_path/'m.zip', device='cpu')
    assert loaded.critic.features_extractor.drop.p == 0
    assert loaded.critic_target.features_extractor.drop.p == 0
    assert loaded.actor.features_extractor.drop.p == .3
    assert tensor_hash(model) == tensor_hash(loaded)
    np.testing.assert_array_equal(a, loaded.predict(obs, deterministic=True)[0])


def prepared_queue(tmp_path):
    from types import SimpleNamespace
    from tools.recipe_sac_queue import write, prepare_critic, software
    from app.frozen_recipe_inputs import file_sha
    from env_plugins.order_contract import CONTRACTS
    parent = tmp_path/'parent'
    write(parent/'recipe.json', recipe())
    spec = dict(schema='recipe_sac_order_queue.v1', recipe_sha256=file_sha(parent/'recipe.json'),
                inputs={'recipe':str(parent/'recipe.json')}, design=DESIGN, software=software(),
                engine_root=str(parent), code={'engine':{}}, order_contracts=CONTRACTS)
    write(parent/'SPEC.private.json', spec)
    sha = file_sha(parent/'SPEC.private.json')
    result = parent/'COMMANDS5/attempt-1/RESULT.json'
    write(result, dict(state='COMPLETE', spec_sha256=sha, artifacts={},
                       reports=[dict(step=s) for s in (25000,50000,75000,100000) for _ in range(3)]))
    write(parent/'CLOSE.json', dict(state='COMPLETE', spec_sha256=sha,
                                   completed={'COMMANDS5':str(result.relative_to(parent))}))
    root = tmp_path/'queue'
    prepare_critic(SimpleNamespace(root=root, critic_parent=parent, parent_spec_sha=sha))
    return root


def test_effective_spec_is_parent_matched_and_mutations_rejected(tmp_path):
    from tools.recipe_sac_queue import verify, read, write
    from app.frozen_recipe_inputs import file_sha
    root = prepared_queue(tmp_path)
    spec = verify(root)
    assert spec['arms'] == ['CRITIC_BASE','CRITIC_NODROP']
    assert spec['critic_dropout'] == {'CRITIC_BASE':.3, 'CRITIC_NODROP':0.}
    spec['inputs']['recipe'] += '.different'
    write(root/'SPEC.private.json', spec)
    write(root/'IDENTITY.json', {'spec_sha256':file_sha(root/'SPEC.private.json')})
    with pytest.raises(ValueError, match='CRITIC_PARENT_CHANGED_inputs'):
        verify(root)


def test_failed_base_does_not_dispatch_treatment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.recipe_sac_queue import run, read
    root = prepared_queue(tmp_path)
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0 if 'admit' in command else 2)
    monkeypatch.setattr('tools.recipe_sac_queue.admission', lambda root: True)
    monkeypatch.setattr('tools.recipe_sac_queue.subprocess.run', child)
    monkeypatch.setattr('tools.recipe_sac_queue.time.sleep', lambda _: None)
    assert run(root) == 2
    assert read(root/'STATUS.json')['state'] == 'CONTRAST_CONTROL_FAILED'
    assert len([c for c in calls if 'worker' in c]) == 2
    assert all('CRITIC_NODROP' not in c for c in calls)
    n = len(calls)
    assert run(root) == 2 and len(calls) == n


def test_collector_binds_critic_override(tmp_path):
    from tools.recipe_sac_collect import bind_report
    from tools.recipe_sac_queue import write
    from app.frozen_recipe_inputs import file_sha
    p = tmp_path/'CRITIC_NODROP/attempt-1/step-25000/model.zip'
    p.parent.mkdir(parents=True)
    p.write_bytes(b'model')
    write(p.parent/'TRAIN.json', dict(net_profit_usd=0.,policy_tensor_sha256='p'))
    effective = dict(actor_dropout=.3,critic_dropout=0.,target_critic_dropout=0.,weight_decay=.01)
    write(p.parent.parent/'EFFECTIVE_MODEL.json', effective)
    r = dict(source_unit='u', report=dict(experiment_key='u',tags=dict(arm='CRITIC_NODROP',
        checkpoint='25000',checkpoint_archive_sha256=file_sha(p),policy_tensor_sha256='p',
        actor_dropout='0.3',critic_dropout='0.0'),
        metrics=[dict(metric='net_profit_usd',value=0.,split='train')]))
    assert bind_report(tmp_path,r)['report']['tags']['attempt'] == 'attempt-1'
    effective['actor_dropout'] = 0.
    write(p.parent.parent/'EFFECTIVE_MODEL.json',effective)
    with pytest.raises(ValueError,match='REPORT_CRITIC_REGULARIZATION'):
        bind_report(tmp_path,r)
