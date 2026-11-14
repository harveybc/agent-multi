from copy import deepcopy
import pytest
import torch
from app.hybrid_local_selection import choose_local,continuation_decision


def pair():
    metrics={k:{'regret':1.} for k in ('cpu','gpu','numpy','gpu_single','cpu_single')}
    return dict(passed=True,metrics=deepcopy(metrics)),metrics,{k:1. for k in metrics},{k:2. for k in metrics}


def test_local_and_exact_fallback():
    control={'w':torch.tensor([1.])};local={'w':torch.tensor([2.])};a,m,d,c=pair()
    selected,r=choose_local(control,local,a,m,d,c)
    assert selected is local and r['qualified'] and r['selected_state']==r['local_state']
    a['metrics']['cpu_single']['regret']=1.001
    selected,r=choose_local(control,local,a,m,d,c)
    assert selected is control and not r['qualified'] and r['selected_state']==r['control_state']


def test_failed_local_solver_preserves_original_object():
    control={'w':torch.tensor([1.])};_,m,_,c=pair()
    selected,r=choose_local(control,None,None,m,None,c)
    assert selected is control and r['local_state'] is None


def units():
    _,m,_,_=pair()
    return [dict(control={'fit':deepcopy(m)},curve=[{'metrics':{'fit':deepcopy(m)}}]) for _ in range(4)]


def test_continuation_ties_require_actual_locality_gain():
    assert continuation_decision(units(),0)['selected']=='original345'
    assert continuation_decision(units(),1)['selected']=='hybrid349'


def test_one_geometry_loss_rejects_despite_three_wins():
    u=units()
    for item in u[:3]:
        for v in item['curve'][-1]['metrics']['fit'].values():v['regret']=.1
    u[-1]['curve'][-1]['metrics']['fit']['cpu_single']['regret']=1.0000001
    assert continuation_decision(u,8)['selected']=='original345'


def test_no_held_metric_used():
    u=units()
    for item in u:item['curve'][-1]['metrics']['held_internal_train']={'regret':float('nan')}
    assert continuation_decision(u,1)['selected']=='hybrid349'


def test_invalid_case_count():
    with pytest.raises(ValueError,match='CASE_COUNT'):continuation_decision(units()[:3],1)


def test_ordinary_path_never_runs_locality(monkeypatch):
    import tools.run_hybrid_local_continuation as worker
    sentinel=object();record={'arm':'ordinary'};budget=[4]
    monkeypatch.setattr(worker,'rescue',lambda *args:(sentinel,record))
    monkeypatch.setattr(worker,'views',lambda *args:pytest.fail('ordinary path must not attempt locality'))
    state,receipt=worker.hybrid(None,None,None,budget)
    assert state is sentinel and receipt is record and budget==[4]
