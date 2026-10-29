import json
import pytest

from tools.run_crossed_account_queue import load, save, representatives, summary_table


def test_queue_identity_and_tamper_refusal(tmp_path):
    p=tmp_path/'r.json'
    assert load(p,'one') is None
    save(p,'one',{'count':2})
    assert load(p,'one')=={'count':2}
    with pytest.raises(ValueError): load(p,'two')
    r=json.loads(p.read_text());r['data']['count']=9;p.write_text(json.dumps(r))
    with pytest.raises(ValueError): load(p,'one')


def test_representatives_do_not_overweight_duplicate_vectors():
    vectors={'a':[3,-3],'b':[3,-3],'c':[2,-2],'d':[1,-1]}
    genes={'a':[2],'b':[1],'c':[1],'d':[1]}
    selected,reps,n=representatives(vectors,genes)
    assert len(selected)==4 and n==3 and reps==['b','c','d']
    assert representatives({}, {})==([],[],0)


def test_summary_does_not_invent_win_rate_without_orders(tmp_path):
    path=tmp_path/'summary.md'
    row=dict(quality='missing',name='nop',objectives=[0,0],orders=0,buys=0,sells=0,
             wins=0,buy_wins=0,sell_wins=0,early_closes=0)
    summary_table(path,[row],'fixture','Control')
    assert 'no financial' in path.read_text() and '| - | - | - |' in path.read_text()
