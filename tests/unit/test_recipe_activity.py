import copy

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from app.recipe_activity import ENTROPY_GRID, activity, choose
from app.recipe_runtime import build_sac
from tests.unit.test_critic_dropout_contrast import Tiny
from tests.unit.test_frozen_recipe_inputs import recipe
from tools.recipe_sac_queue import DESIGN, tensor_hash


def report(n=53, weeks=26, **kwargs):
    return dict(trades=n, weeks_with_decisions=weeks, orders_per_week=n/weeks,
        max_entries_per_day=1, valid=True, ruined=False, mean_daily_log_return=.001,
        max_drawdown_pct=4., **kwargs)


@pytest.mark.parametrize('n,eligible', [(0,False),(51,False),(52,False),(53,True)])
def test_activity_is_strict_and_uses_all_calendar_weeks(n, eligible):
    assert activity(report(n))['activity_eligible'] is eligible


@pytest.mark.parametrize('change', [{'trades':float('nan')}, {'weeks_with_decisions':0},
    {'orders_per_week':9}, {'max_entries_per_day':2}, {'valid':False}, {'ruined':True},
    {'trades':None}, {'mean_daily_log_return':None}])
def test_invalid_activity_never_passes(change):
    r = report()
    r.update(change)
    assert not activity(r)['activity_eligible']


def test_selection_filters_activity_and_never_uses_2024_or_intermediate():
    rows = []
    for arm,n,score in [('A',0,10),('B',53,.001),('C',60,-.001)]:
        for period in ('TRAIN','INNER_A','OBS_2024'):
            r = report(n, arm=arm, step=100000, period=period)
            r['mean_daily_log_return'] = score
            rows.append(r)
    rows += [report(99, arm='A', step=25000, period='INNER_A')]
    assert choose(rows)['selected'] == 'B'
    changed = copy.deepcopy(rows)
    for r in changed:
        if r['period'] == 'OBS_2024':
            r.update(mean_daily_log_return=999, trades=0, orders_per_week=0)
    assert choose(changed) == choose(rows)
    assert choose([r for r in rows if r['arm']=='A'])['selected'] is None
    assert choose([r for r in rows if r['arm']=='A'])['state'] == 'ACTIVITY_NOT_MET'


def test_duplicate_or_missing_period_cannot_win():
    rows = [report(arm='A',step=100000,period='TRAIN')]*2
    assert choose(rows)['selected'] is None


@pytest.mark.parametrize('alpha', list(ENTROPY_GRID.values()))
def test_fixed_alpha_preserves_initial_tensors_and_reload(tmp_path, alpha):
    torch.set_num_threads(1)
    design = {**DESIGN, 'learning_starts':2, 'batch_size':4, 'buffer_size':32}
    base = build_sac(Tiny(),recipe(),design,'GROUPED','cpu')
    model = build_sac(Tiny(),recipe(),design,'GROUPED','cpu',entropy_coefficient=alpha)
    assert tensor_hash(model) == tensor_hash(base)
    assert model.ent_coef_optimizer is None
    assert model.actor.features_extractor.drop.p == model.critic.features_extractor.drop.p == .3
    model.learn(total_timesteps=16)
    assert float(model.ent_coef_tensor) == pytest.approx(alpha)
    model.save(tmp_path/'model.zip')
    loaded = SAC.load(tmp_path/'model.zip',device='cpu')
    assert loaded.ent_coef == alpha
    assert loaded.ent_coef_optimizer is None
    assert float(loaded.ent_coef_tensor) == pytest.approx(alpha)
    obs,_ = Tiny().reset()
    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])


def prepare_activity_queue(tmp_path):
    from types import SimpleNamespace
    from tests.unit.test_critic_dropout_contrast import prepared_queue
    from tools.recipe_sac_queue import write, prepare_activity
    from app.frozen_recipe_inputs import file_sha
    parent = prepared_queue(tmp_path)
    sha = file_sha(parent/'SPEC.private.json')
    result = parent/'CRITIC_BASE/attempt-1/RESULT.json'
    write(result,dict(state='COMPLETE',spec_sha256=sha,artifacts={},fresh_control_policy_exact=True,
        reports=[dict(step=s) for s in (25000,50000,75000,100000) for _ in range(3)]))
    write(parent/'CLOSE.json',dict(state='COMPLETE',spec_sha256=sha,
        completed={'CRITIC_BASE':str(result.relative_to(parent))}))
    root = tmp_path/'activity'
    prepare_activity(SimpleNamespace(root=root,activity_parent=parent,parent_spec_sha=sha))
    return root


def test_preparation_and_changed_parent_or_grid_refused(tmp_path):
    from tools.recipe_sac_queue import verify, read, write
    from app.frozen_recipe_inputs import file_sha
    root = prepare_activity_queue(tmp_path)
    spec = verify(root)
    assert spec['entropy_coefficients']==ENTROPY_GRID
    spec['design']['seed'] = 8
    write(root/'SPEC.private.json',spec)
    write(root/'IDENTITY.json',dict(spec_sha256=file_sha(root/'SPEC.private.json')))
    with pytest.raises(ValueError,match='DESIGN'):
        verify(root)


def test_collector_rejects_false_activity_and_entropy(tmp_path):
    from tools.recipe_sac_collect import bind_report
    from tools.recipe_sac_queue import write
    from app.frozen_recipe_inputs import file_sha
    p = tmp_path/'ACT_ALPHA001/attempt-1/step-25000/model.zip'
    p.parent.mkdir(parents=True)
    p.write_bytes(b'model')
    native = report(policy_tensor_sha256='p')
    native.update(activity(native))
    write(p.parent/'TRAIN.json',native)
    effective = dict(entropy_coefficient=.01,actor_dropout=.3,critic_dropout=.3,weight_decay=.01)
    write(p.parent.parent/'EFFECTIVE_MODEL.json',effective)
    request = dict(source_unit='u',report=dict(experiment_key='u',tags=dict(arm='ACT_ALPHA001',
        checkpoint='25000',checkpoint_archive_sha256=file_sha(p),policy_tensor_sha256='p',entropy_coefficient='0.01'),
        metrics=[dict(metric='trades',value=53,split='train')]))
    assert bind_report(tmp_path,request)['report']['tags']['attempt']=='attempt-1'
    native['activity_eligible'] = False
    write(p.parent/'TRAIN.json',native)
    with pytest.raises(ValueError,match='REPORT_ACTIVITY_GATE'):
        bind_report(tmp_path,request)
    native['activity_eligible'] = True
    write(p.parent/'TRAIN.json',native)
    effective['entropy_coefficient'] = .1
    write(p.parent.parent/'EFFECTIVE_MODEL.json',effective)
    with pytest.raises(ValueError,match='REPORT_ACTIVITY_PARAMETERS'):
        bind_report(tmp_path,request)


def test_failed_activity_arms_continue_finitely_without_winner(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tools.recipe_sac_queue import run, read
    root = prepare_activity_queue(tmp_path)
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0 if 'admit' in command else 2)
    monkeypatch.setattr('tools.recipe_sac_queue.admission',lambda root:True)
    monkeypatch.setattr('tools.recipe_sac_queue.subprocess.run',child)
    monkeypatch.setattr('tools.recipe_sac_queue.time.sleep',lambda _:None)
    assert run(root)==0
    assert len([c for c in calls if 'worker' in c])==6
    assert read(root/'ACTIVITY_DECISION.json')['selected'] is None
    assert read(root/'ACTIVITY_DECISION.json')['state']=='INCOMPLETE_GRID'
    n = len(calls)
    assert run(root)==0 and len(calls)==n
