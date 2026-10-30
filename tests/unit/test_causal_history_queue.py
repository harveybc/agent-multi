import pytest

from tools.run_causal_history_queue import search_gate, proposals, source_identity


def row(ret,dd,orders):
    return dict(objectives=[ret,-dd],mean_orders=orders,eligible=orders>=3)


def test_gate_cannot_pass_by_abstention_or_looking_at_confirmation():
    c={'neither':row(-2,3,4),'entry':row(0,0,0),'close':row(-3,4,8),'both':row(-2,3,4)}
    assert not search_gate(c)['eligible']
    c['both']=row(-1,2,4)
    assert search_gate(c)['eligible']
    c['both']=row(-1,4,4)
    assert not search_gate(c)['eligible']


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
