import pytest

from tools.run_causal_history_queue import search_gate, proposals, source_identity, close_behavior
from app.causal_history_workers import activity_eligible


def test_gate_depends_on_runtime_tests_not_control_performance():
    assert search_gate({'passed':200,'skipped':0})['eligible']
    assert not search_gate({'passed':199,'skipped':0})['eligible']
    assert not search_gate({'passed':200,'skipped':1})['eligible']
    assert not search_gate({'passed':200,'skipped':0,'failed':1})['eligible']
    assert not search_gate({})['eligible']


@pytest.mark.parametrize('orders,accepted',[(0,False),(2,False),(2.01,True),(2.69,True),(3,True)])
def test_activity_is_strictly_more_than_two(orders,accepted):
    assert activity_eligible(orders) is accepted


@pytest.mark.parametrize('orders',[-1,float('nan'),float('inf')])
def test_invalid_activity_is_not_an_eligible_candidate(orders):
    with pytest.raises(ValueError): activity_eligible(orders)


def test_absent_closes_are_not_reported_as_close_success():
    assert close_behavior({'orders':10,'early_closes':0})=='NO_EARLY_CLOSE_OBSERVED'
    assert close_behavior({'orders':0,'early_closes':0})=='NO_ORDERS'
    assert close_behavior({'orders':10,'early_closes':0},False)=='EXPLICITLY_DISABLED'
    assert close_behavior({'orders':10,'early_closes':2})=='EARLY_CLOSE_OBSERVED_NOT_QUALIFIED'


def test_controls_are_four_independent_history_arms():
    c=proposals()
    assert {(tuple(g[11:13])) for _,g,e,v in c if e is None and v}=={(0,0),(0,1),(1,0),(1,1)}
    assert all(g[8]==1 for _,g,e,v in c)


def test_manifest_identity_refuses_mutation(tmp_path):
    from tools.run_causal_history_queue import freeze_manifest
    path=tmp_path/'MANIFEST.json'
    identity=freeze_manifest(path,{'a':1})
    assert freeze_manifest(path,{'a':1})==identity
    with pytest.raises(ValueError,match='RESUME'):
        freeze_manifest(path,{'a':2})


def test_source_identity_covers_controller_history_and_queue():
    files=source_identity()
    assert 'app/close_hold_controller.py' in files
    assert 'app/causal_history_account.py' in files
    assert 'tools/run_causal_history_queue.py' in files
