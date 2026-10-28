import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools.verify_rl_static_engineering import run


@pytest.mark.parametrize('outcome', ['pass', 'failure', 'timeout', 'missing', 'empty'])
def test_checks_require_real_reports_and_persist_terminal_state(tmp_path, monkeypatch, outcome):
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs['timeout'] == 10
        if outcome == 'timeout':
            raise subprocess.TimeoutExpired(command, 10)
        if outcome != 'missing':
            path = Path(command[-1].split('=', 1)[1])
            path.write_text('<testsuites><testsuite tests="{}" failures="{}" errors="0" skipped="0"/></testsuites>'.format(
                0 if outcome == 'empty' else 3, 1 if outcome == 'failure' else 0))
        return SimpleNamespace(returncode=1 if outcome == 'failure' else 0)
    monkeypatch.setattr(subprocess, 'run', execute)
    output = tmp_path / 'output'
    rc = run(tmp_path / 'gym', tmp_path / 'agent', output, 10)
    state = json.loads((output / 'STATUS.json').read_text())
    assert rc == (0 if outcome == 'pass' else 2)
    assert len(calls) == (2 if outcome == 'pass' else 1)
    assert state['scientific_admission'] is False
    assert state['financial_fits'] == 0
    assert state['state'] != 'RUNNING'


def test_existing_run_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path / 'gym', tmp_path / 'agent', tmp_path, 10)


def test_repository_output_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path / 'gym', tmp_path / 'agent', tmp_path / 'agent' / 'out', 10)
