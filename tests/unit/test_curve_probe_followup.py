import numpy as np
import pytest
import torch

from tools.followup_curve_probe import hold_surfaces, entropy_witness, verify_closure, account_pair
from tools.run_discrete_curve_probe import write
from app.frozen_recipe_inputs import file_sha
from agent_plugins.discrete_curve_sac import route_surface
from tests.unit.test_double_tcn_probe import probe_recipe


def test_hold_ablation_only_changes_close_curves():
    x = np.random.default_rng(3).normal(size=(10, 39)).astype(np.float32)
    before = x.copy()
    held = hold_surfaces(x)
    assert np.array_equal(x, before)
    assert np.array_equal(held[:, :3], x[:, :3])
    for route in (1, 2):
        values = route_surface(torch.tensor(held), torch.full((10, 1), route),
                               torch.linspace(-.2, 1.2, 10)[:, None])
        assert values[:, :2].argmax(-1).tolist() == [0] * 10


def test_zero_economics_entropy_witness_uses_library_target():
    result = entropy_witness(probe_recipe(), alpha=.2)
    assert result['passed']
    assert result['observed_difference'] == pytest.approx(.99 * .2 * np.log(1.5), abs=1e-6)


def test_closure_hash_required(tmp_path):
    write(tmp_path / 'AGGREGATE.json', {'complete': 12})
    write(tmp_path / 'CLOSE.json', {'results_sha256': file_sha(tmp_path / 'AGGREGATE.json')})
    verify_closure(tmp_path, 'AGGREGATE.json', 'results_sha256')
    write(tmp_path / 'AGGREGATE.json', {'complete': 11})
    with pytest.raises(ValueError, match='CLOSURE_CHANGED'):
        verify_closure(tmp_path, 'AGGREGATE.json', 'results_sha256')


def test_resume_rejects_diagnostic_binding_and_mutated_artifact(tmp_path):
    archive, diagnostic, output = (tmp_path / n for n in ('archive', 'diagnostic', 'output'))
    write(archive / 'RESULT.json', dict(unit=dict(id='u', horizon=3)))
    result_sha = file_sha(archive / 'RESULT.json')
    write(diagnostic / 'u_TRAIN.json', dict(binding=dict(result_sha256=result_sha), baseline_exact=True))
    binding = dict(campaign='campaign', source='source', period='TRAIN', result_sha256=result_sha,
                   diagnosis_sha256=file_sha(diagnostic / 'u_TRAIN.json'))
    write(output / 'artifact.json', {'value': 1})
    write(output / 'RESULT.json', dict(binding=binding, artifacts={'artifact.json': file_sha(output / 'artifact.json')}))
    result = account_pair(archive, None, None, 'TRAIN', output, {'identity': 'campaign'}, diagnostic, 'source')
    assert result['binding'] == binding
    with pytest.raises(ValueError, match='FOLLOWUP_BINDING'):
        account_pair(archive, None, None, 'TRAIN', output, {'identity': 'campaign'}, diagnostic, 'changed')
    write(output / 'artifact.json', {'value': 2})
    with pytest.raises(ValueError, match='FOLLOWUP_ARTIFACT_CHANGED'):
        account_pair(archive, None, None, 'TRAIN', output, {'identity': 'campaign'}, diagnostic, 'source')
