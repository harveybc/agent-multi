from types import SimpleNamespace

import pytest

from app.frozen_recipe_inputs import file_sha
from tools.run_discrete_curve_probe import read, write
from tools import run_monitor_policy_queue as queue


def config(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'model.pt').write_bytes(b'authenticated source')
    predecessor = tmp_path/'predecessor'
    predecessor.mkdir()
    write(predecessor/'SPEC.json', dict(identity='previous'))
    return dict(schema='monitor_queue367.v1', seed=7,
        policies=['balanced_min','champion_nonregression'], poll_seconds=10,
        output=str(tmp_path/'runs'), predecessor=str(predecessor), bundle=str(source),
        predecessor_service='prior.service', predecessor_identity='previous',
        parent=str(tmp_path/'parent'), source_model_sha256=file_sha(source/'model.pt'),
        code_identity='code', minimum_available_bytes=1024)


def completion(path, identity):
    (path/'evidence.txt').write_text('verified')
    write(path/'RESULT.json', dict(identity=identity, status='COMPLETE',
        files={'evidence.txt':file_sha(path/'evidence.txt')}))


def test_result_tampering_rejected(tmp_path):
    completion(tmp_path, 'one')
    queue.verify_completion(tmp_path, 'one')
    with pytest.raises(ValueError):
        queue.verify_completion(tmp_path, 'other')
    (tmp_path/'evidence.txt').write_text('changed')
    with pytest.raises(ValueError):
        queue.verify_completion(tmp_path, 'one')


def test_policies_and_source_overlap_rejected(tmp_path):
    c = config(tmp_path)
    queue.validate_config(c)
    with pytest.raises(ValueError):
        queue.validate_config(dict(c, policies=['balanced_min']*2))
    with pytest.raises(ValueError):
        queue.validate_config(dict(c, output=c['predecessor']))


def test_wait_execute_sequentially_and_restart_without_duplicates(tmp_path, monkeypatch):
    from pathlib import Path
    c = config(tmp_path)
    path = tmp_path/'QUEUE.json'
    write(path, c)
    monkeypatch.setattr(queue, 'code_identity', lambda: 'code')
    sleeps, calls = [], []
    def sleep(seconds):
        sleeps.append(seconds)
        completion(Path(c['predecessor']), 'previous')
    monkeypatch.setattr(queue.time, 'sleep', sleep)
    reasons = iter(['GPU_IN_USE', None, None])
    monkeypatch.setattr(queue, 'admission_reason', lambda _: next(reasons))
    def child(command, **kw):
        assert kw['check']
        target = Path(command[command.index('--output')+1])
        policy = command[command.index('--monitor-policy')+1]
        calls.append(policy)
        write(target/'SPEC.json', dict(identity=policy, monitor_policy=policy, seed=7,
            code='code', source=c['source_model_sha256']))
        completion(target, policy)
    monkeypatch.setattr(queue.subprocess, 'run', child)
    queue.run(SimpleNamespace(config=path))
    assert calls == c['policies'] and len(sleeps) == 2
    queue.run(SimpleNamespace(config=path))
    assert calls == c['policies']
    assert read(Path(c['output'])/'QUEUE_STATUS.json')['stage'] == 'COMPLETE'
