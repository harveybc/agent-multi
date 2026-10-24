import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from tools.recipe_sac_queue import (DESIGN, authenticated_result, clean, preserve_rng, software,
                                    read, run, verify, write)
from app.frozen_recipe_inputs import file_sha


def spec_root(tmp_path):
    spec = dict(schema='recipe_sac_queue.v1', design=copy.deepcopy(DESIGN), arms=['GROUPED', 'FLAT'],
                max_attempts=2, device='cuda', agent_root=str(tmp_path), engine_root=str(tmp_path),
                software=software(), code={'agent': {}, 'engine': {}})
    write(tmp_path / 'SPEC.private.json', spec)
    write(tmp_path / 'IDENTITY.json', {'spec_sha256': file_sha(tmp_path / 'SPEC.private.json')})
    return spec


def test_spec_changed_or_design_changed_refused(tmp_path):
    spec = spec_root(tmp_path)
    assert verify(tmp_path) == spec
    spec['design']['agent_feedback'] = True
    write(tmp_path / 'SPEC.private.json', spec)
    with pytest.raises(ValueError, match='SPEC_PIN'):
        verify(tmp_path)
    write(tmp_path / 'IDENTITY.json', {'spec_sha256': file_sha(tmp_path / 'SPEC.private.json')})
    with pytest.raises(ValueError, match='DESIGN'):
        verify(tmp_path)


def test_rng_is_unchanged_by_evaluation_or_reload():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with preserve_rng():
        random.random()
        np.random.rand(20)
        torch.rand(20)
        torch.manual_seed(99)
    assert random.getstate() == before[0]
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_undefined_metrics_are_null_not_zero():
    assert clean({'loss': np.float64('nan'), 'pf': float('inf'), 'value': -1.}) == {
        'loss': None, 'pf': None, 'value': -1.}


def test_completed_result_requires_all_artifacts(tmp_path):
    p = tmp_path / 'model.zip'
    p.write_bytes(b'checkpoint')
    result = dict(state='COMPLETE', spec_sha256='abc', artifacts={'model.zip':file_sha(p)},
                  reports=[{'step':step} for step in (25000,50000,75000,100000) for _ in range(3)])
    write(tmp_path / 'RESULT.json', result)
    authenticated_result(tmp_path / 'RESULT.json', 'abc')
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='RESULT_ARTIFACT'):
        authenticated_result(tmp_path / 'RESULT.json', 'abc')


def test_resource_wait_does_not_consume_attempt(tmp_path, monkeypatch):
    spec_root(tmp_path)
    monkeypatch.setattr('tools.recipe_sac_queue.admission', lambda root:False)
    assert run(tmp_path) == 75
    status = read(tmp_path / 'STATUS.json')
    assert status['state'] == 'WAIT_RESOURCES' and status['attempts'] == {}


def test_failures_bounded_and_next_arm_runs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec = spec_root(tmp_path)
    spec['task_timeout_seconds'] = 10
    write(tmp_path / 'SPEC.private.json', spec)
    write(tmp_path / 'IDENTITY.json', {'spec_sha256':file_sha(tmp_path / 'SPEC.private.json')})
    calls = []
    def failure(cmd, **kwargs):
        if 'admit' in cmd:
            return SimpleNamespace(returncode=0)
        calls.append(cmd)
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr('tools.recipe_sac_queue.admission', lambda root:True)
    monkeypatch.setattr('tools.recipe_sac_queue.subprocess.run', failure)
    monkeypatch.setattr('tools.recipe_sac_queue.time.sleep', lambda n:None)
    assert run(tmp_path) == 0
    assert len(calls) == 4
    status = read(tmp_path / 'CLOSE.json')
    assert status['state'] == 'COMPLETE_WITH_FAILURES'
    assert len(status['attempts']['FLAT']) == 2
    assert run(tmp_path) == 0 and len(calls) == 4
