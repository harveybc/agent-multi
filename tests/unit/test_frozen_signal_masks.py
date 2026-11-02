import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401
from tests.unit.test_double_tcn_probe import probe_recipe
from app.three_signal_learning import evaluate, synthetic_env
from app.frozen_signal_masks import masked_emit, prepared_signals, cached_measurement
from tools.run_three_signal_baseline import known_logits


def test_mask_keeps_full_window_support_and_weights():
    class Model:
        device = 'cpu'
        def surfaces(self, x):
            return torch.stack((x[:, :, 0].sum(1), x[:, :, 1].sum(1), x[:, :, 2].sum(1)), 1)
    x = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
    x.flags.writeable = False
    valid = np.array([False, True, True, False, True])
    inputs = SimpleNamespace(windows=x, valid=valid)
    before = x.copy()
    full = masked_emit(Model(), inputs, [True, True, True])
    drop = masked_emit(Model(), inputs, [True, False, True])
    assert np.array_equal(before, x) and np.array_equal(valid, inputs.valid)
    assert np.isnan(drop[~valid]).all()
    assert np.array_equal(full[valid][:, [0, 2]], drop[valid][:, [0, 2]])
    assert (drop[valid, 1] == 0).all()
    with pytest.raises(ValueError, match='MASK_WIDTH'):
        masked_emit(Model(), inputs, [True])


def test_prepared_signals_exact_scalar_parity():
    x = np.random.default_rng(4).normal(size=(100, 3)).astype(np.float32)
    x[:4] = [[1, 1, 0], [0, 0, 0], [0, 1, 1], [0, 1e-11, 0]]
    action, entropy, commands, finite = prepared_signals(x)
    assert finite.all()
    for i, raw in enumerate(x):
        ties = np.flatnonzero(np.isclose(raw, raw.max(), atol=1e-10, rtol=0))
        expected = int(ties[0]) if len(ties) == 1 else 0
        p = np.exp(raw - raw.max()); p /= p.sum()
        assert action[i] == expected
        assert entropy[i] == float(-np.sum(p * np.log(np.maximum(p, 1e-30))))
        for d in (-1, 0, 1):
            pc = p if not d else np.array([p[0]+p[1 if d == 1 else 2], p[2 if d == 1 else 1]])
            assert commands[i, d+1] == float(-np.sum(pc*np.log(np.maximum(pc, 1e-30))))


@pytest.mark.parametrize('mirror', [-1, 1])
def test_prepared_native_account_and_files_exact(curve_env, tmp_path, mirror):
    r = probe_recipe()
    a, s = synthetic_env(r, curve_env.risk_config, 91001, mirror, .99)
    b, _ = synthetic_env(r, curve_env.risk_config, 91001, mirror, .99)
    logits = known_logits(s)
    first = evaluate(a, logits, tmp_path/'old', diagnose=True)
    second = evaluate(b, logits, tmp_path/'new', diagnose=True, prepared=prepared_signals(logits))
    assert first == second
    for name in ('TRADES.json', 'ACTIONS.json', 'METRICS.json', 'CLOSE_COUNTERFACTUALS.json'):
        assert (tmp_path/'old'/name).read_bytes() == (tmp_path/'new'/name).read_bytes()


def test_cache_identity_resume_and_tamper(tmp_path):
    calls = []
    context = dict(checkpoint='abc', support='support', source='src', costs='costs')
    def run():
        calls.append(1)
        return dict(score=1.0, gap=2.0, eligible=True)
    a = cached_measurement(tmp_path, context, ['a'], run)
    assert cached_measurement(tmp_path, context, ['a'], run) == a
    assert len(calls) == 1
    for field in context:
        changed = copy.deepcopy(context); changed[field] += 'changed'
        cached_measurement(tmp_path, changed, ['a'], run)
    assert len(calls) == 5
    target = next(tmp_path.glob('*.json'))
    target.write_text(target.read_text().replace('1.0', '9.0'))
    with pytest.raises(ValueError, match='MASK_CACHE_CHANGED'):
        for field in [None, *context]:
            changed = dict(context)
            if field: changed[field] += 'changed'
            cached_measurement(tmp_path, changed, ['a'], run)


def test_probability_and_cyclic_codings_remain_atomic():
    from app.rl_subset import coherent_blocks
    assert coherent_blocks(['hod_sin', 'x', 'hod_cos', 'rg.gmm.p0', 'rg.gmm.p1', 'rg.gmm.p2', 'rg.gmm.entropy']) == [
        ['hod_sin', 'hod_cos'], ['x'], ['rg.gmm.p0', 'rg.gmm.p1', 'rg.gmm.p2'], ['rg.gmm.entropy']]


def test_parity_cannot_ignore_trade_or_return_changes():
    from tools.run_frozen_mask_selection import exact_account_check
    a = dict(trades=4, return_pct=1., mean_signal_entropy=.1)
    exact_account_check(a, dict(a, mean_signal_entropy=.10000001))
    with pytest.raises(ValueError, match='FULL_ACCOUNT_PARITY'):
        exact_account_check(a, dict(a, trades=3))


def test_cross_search_reuse_requires_same_evaluation_and_compute(tmp_path, monkeypatch):
    from tools import run_frozen_mask_selection as runner
    from app.rl_subset import digest
    from app.frozen_recipe_inputs import file_sha
    from tools.run_discrete_curve_probe import write
    name='tools/run_frozen_mask_selection.py'
    old=tmp_path/'old'; new=tmp_path/'new'; cache=tmp_path/'cache'
    for root in (old,new):
        (root/'tools').mkdir(parents=True)
        (root/name).write_text('def outer():\n    def compute(x):\n        return x + 1\n')
    cache.mkdir()
    fields=('parent','pilot','baseline','dataset','design','risk','checkpoint_rule','periods',
            'device','torch_version','numpy_version','inference_precision')
    previous={k:'same' for k in fields}
    previous['sources']={name:file_sha(old/name),'app/science.py':'samehash'}
    previous['identity']=digest(previous)
    write(cache/'SPEC.private.json',previous)
    s=copy.deepcopy(previous)
    s['reuse']=dict(root=str(cache),runtime=str(old),spec_sha256=file_sha(cache/'SPEC.private.json'))
    monkeypatch.setattr(runner,'ROOT',new)
    assert runner.check_reuse(s)['identity']==previous['identity']
    s['dataset']='changed'
    with pytest.raises(ValueError,match='EVALUATION_CONTRACT'): runner.check_reuse(s)
    s['dataset']='same'; s['sources']['app/science.py']='changed'
    with pytest.raises(ValueError,match='EVALUATOR_SOURCES'): runner.check_reuse(s)
    s['sources']['app/science.py']='samehash'
    (new/name).write_text('def outer():\n    def compute(x):\n        return x + 2\n')
    with pytest.raises(ValueError,match='COMPUTE_CHANGED'): runner.check_reuse(s)
