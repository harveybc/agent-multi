from copy import deepcopy
import json

import pytest

from tools import collect_sac_training_reliability as module


def fleet():
    return dict(workers=[dict(seed=s,root='/unused') for s in (7,19,43)])


def result(entry):
    metric=dict(return_pct=1.,sharpe=.2,trades=100,orders_per_week=3.)
    spec={k:'same' for k in ('source','manifest','parent','code','design','effective','arms',
                           'updates_per_arm','trainability')}
    return dict(verified=True,spec=spec,result=dict(arms=[dict(name='control',cold_exact=True,
        final_metrics=dict(TRAIN=metric,VALIDATION=metric))]))


def test_missing_worker_never_complete(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'fetch',lambda e: result(e) if e['seed']!=43 else dict(verified=False))
    assert not module.collect(fleet(),tmp_path)
    assert not (tmp_path/'RESULT.json').exists()
    assert not json.loads((tmp_path/'STATUS.json').read_text())['complete']


def test_mismatched_contract_never_publishes_complete(tmp_path,monkeypatch):
    def fetch(e):
        r=deepcopy(result(e))
        if e['seed']==43:r['spec']['code']='other'
        return r
    monkeypatch.setattr(module,'fetch',fetch)
    with pytest.raises(ValueError,match='CONTRACT_MISMATCH'):module.collect(fleet(),tmp_path)
    assert not json.loads((tmp_path/'STATUS.json').read_text())['complete']
    assert not (tmp_path/'RESULT.json').exists()


def test_completed_still_does_not_promote(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'fetch',result)
    assert module.collect(fleet(),tmp_path)
    r=json.loads((tmp_path/'RESULT.json').read_text())
    assert len(r['pairs'])==3 and not r['promoted'] and not r['phase2_closed'] and not r['deap_released']


def test_duplicate_seed_refused(tmp_path):
    with pytest.raises(ValueError,match='DISTINCT_SEEDS'):
        module.collect(dict(workers=[dict(seed=7)]*3),tmp_path)
