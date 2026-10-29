import json

import pytest

from tools.optimize_dynamic_account import Receipts, digest, controls, search


def test_durable_receipts_reject_alteration_and_identity(tmp_path):
    path=tmp_path/'receipts.jsonl';r=Receipts(path,'a')
    r.append([1.,2.],dict(objectives=[2.,-1.],eligible=True))
    assert Receipts(path,'a').rows==r.rows
    with pytest.raises(ValueError): Receipts(path,'b')
    record=json.loads(path.read_text());record['result']['objectives'][0]=9.
    path.write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError): Receipts(path,'a')


def test_duplicate_truncated_receipts_and_controls(tmp_path):
    path=tmp_path/'receipts.jsonl';r=Receipts(path,'a');r.append([1.],{})
    with pytest.raises(ValueError): r.append([1.],{})
    assert digest([1.]) in r.rows
    path.write_text(path.read_text()+'{"unfinished":')
    with pytest.raises(ValueError): Receipts(path,'a')
    rows=controls()
    assert rows[0][0]=='nop' and rows[0][2]==set()
    assert len(rows)==7 and len({x[0] for x in rows})==7


def test_sequential_search_restarts_exactly_from_authenticated_cache(tmp_path,monkeypatch):
    import tools.optimize_dynamic_account as module
    calls=[]
    def fake(worlds,g):
        calls.append(tuple(g))
        return dict(objectives=[sum(g),-g[0]],eligible=True,mean_orders=3.,worlds=[])
    monkeypatch.setattr(module,'evaluate',fake)
    receipts=Receipts(tmp_path/'r.jsonl','fixture')
    a=search([],receipts,7,20,2,tmp_path,'fixture',lambda:None)
    before=len(calls)
    b=search([],Receipts(tmp_path/'r.jsonl','fixture'),7,20,2,tmp_path,'fixture',lambda:None)
    assert len(calls)==before and a['history']==b['history']
    assert a['new_evaluations']>20 and b['new_evaluations']==0
