import copy
from pathlib import Path

import pytest

from app.rl_subset import digest
from tools.recipe_sac_queue import write, read
from tools.rl_selection_run import aggregate, execute_task


def test_aggregate_never_turns_no_orders_into_a_score():
    row=dict(score=None,gap=None,train_score=None,eligible=False,
             native=dict(seed=7,metrics={},request_sha256='a'*64))
    result=aggregate([row])
    assert result['score'] is None and result['gap'] is None and not result['eligible']


def task():
    return dict(request=dict(unit='u',features=['x'],seed=7),
                params=dict(cost_mode='configured',model_id='tcn'),cross=False)


def test_completed_result_requires_authentication(tmp_path,monkeypatch):
    import tools.rl_selection_run as q
    t=task(); folder=tmp_path/'tasks'/(digest(t)+'-attempt-1')
    write(folder/'RESULT.json',{'tampered':True})
    def reject(*args): raise ValueError('HASH_MISMATCH')
    monkeypatch.setattr(q,'authenticate',reject)
    with pytest.raises(ValueError,match='HASH_MISMATCH'):
        execute_task(tmp_path,dict(max_attempts=2),t)


def test_interrupted_attempt_is_not_overwritten_and_retry_is_bounded(tmp_path,monkeypatch):
    import tools.rl_selection_run as q
    t=task(); name=digest(t)
    for attempt in (1,2):
        write(tmp_path/'tasks'/(name+f'-attempt-{attempt}.json'),dict(state='STARTED',attempt=attempt))
    before={p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    with pytest.raises(RuntimeError,match='BOUNDED_RETRIES'):
        execute_task(tmp_path,dict(max_attempts=2),t)
    for p,b in before.items(): assert p.read_bytes()==b


def test_all_seeds_must_pass_activity():
    rows=[dict(score=1.,gap=2.,train_score=3.,eligible=e,
               native=dict(seed=s,metrics={},request_sha256=str(s)*64)) for s,e in [(7,True),(9,False)]]
    result=aggregate(rows)
    assert result['score']==1 and not result['eligible']
