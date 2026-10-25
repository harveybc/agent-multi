import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from app.frozen_recipe_inputs import file_sha
from tools import recipe_hybrid_queue as q


def result(root,unit,sha):
    folder=root/unit/'attempt-1'; folder.mkdir(parents=True,exist_ok=True)
    reports=[]
    for period in ('TRAIN','INNER_A','OBS_2024'):
        r=dict(period=period,arm=unit.rsplit('_s',1)[0],seed=7,step=102400,
               trades=0,weeks_with_decisions=4,orders_per_week=0,valid=False,ruined=False,max_entries_per_day=0,
               mean_daily_log_return=0,max_drawdown_pct=0,initial_capital=10000,net_profit_usd=0,
               sharpe_daily_sqrt365=0,average_entries_per_business_day=0)
        q.write(folder/(period+'.json'),r);reports.append(r)
    q.write(folder/'EFFECTIVE_MODEL.json',{})
    (folder/'model-102400.zip').write_bytes(b'test-only')
    value=dict(state='COMPLETE',unit=unit,arm=unit.rsplit('_s',1)[0],seed=7,spec_sha256=sha,reports=reports,
               artifacts={p.name:file_sha(p) for p in folder.iterdir() if p.is_file()})
    q.write(folder/'RESULT.json',value)
    return folder/'RESULT.json'


def test_authenticated_refuses_changed_native_and_missing_checkpoint(tmp_path):
    p=result(tmp_path,'CAT_FIXED_s7','pin')
    q.authenticated(p,'pin')
    q.write(p.parent/'TRAIN.json',{'tampered':1})
    with pytest.raises(ValueError): q.authenticated(p,'pin')


def test_direction_and_mandatory_exit_are_required_for_behavior_eligibility():
    counts={n:1 for n in ('NOP','CLOSE_LONG','CLOSE_SHORT','HOLD_LONG','HOLD_SHORT')}
    assert q.business_behavior(dict(n_long=1,n_short=1),counts)['business_behavior_eligible']
    assert not q.business_behavior(dict(n_long=100,n_short=0),counts)['business_behavior_eligible']
    counts['CLOSE_SHORT']=0
    assert not q.business_behavior(dict(n_long=1,n_short=1),counts)['business_behavior_eligible']


def test_failed_learning_qualification_does_not_launch_financial_worker(tmp_path,monkeypatch):
    q.write(tmp_path/'SPEC.private.json',{'test':True})
    monkeypatch.setattr(q,'verify',lambda root:dict(qualification={'test':True},units=['CAT_FIXED_s7']))
    monkeypatch.setattr(q,'admission',lambda root:True)
    monkeypatch.setattr(q,'qualify',lambda *a:False)
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**kw:pytest.fail('financial worker called'))
    assert q.run(tmp_path)==0
    assert q.read(tmp_path/'CLOSE.json')['state']=='QUALIFICATION_FAILED'


def test_sequential_recovery_adopts_completed_child_without_refit(tmp_path,monkeypatch):
    q.write(tmp_path/'SPEC.private.json',{'test':True});sha=file_sha(tmp_path/'SPEC.private.json')
    q.write(tmp_path/'GPU_ADMISSION.json',{'state':'PASS','spec_sha256':sha})
    first='CAT_FIXED_s7'; second='HYBRID_LOG_s7'
    result(tmp_path,first,sha)
    q.write(tmp_path/'STATUS.json',dict(spec_sha256=sha,state='RUNNING',completed={},
        attempts={first:[{'attempt':1,'state':'RUNNING'}]}))
    monkeypatch.setattr(q,'verify',lambda root:dict(units=[first,second],task_timeout_seconds=1))
    monkeypatch.setattr(q,'admission',lambda root:True)
    called=[]
    def child(cmd,**kwargs):
        unit=cmd[cmd.index('--unit')+1];called.append(unit);result(tmp_path,unit,sha)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(q.subprocess,'run',child)
    assert q.run(tmp_path)==0
    assert called==[second]
    assert q.read(tmp_path/'CLOSE.json')['state']=='COMPLETE'
    assert q.run(tmp_path)==0 and called==[second]
