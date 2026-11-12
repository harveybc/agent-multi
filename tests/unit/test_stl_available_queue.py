from types import SimpleNamespace

import pytest

from tools import run_available_stl_subset as runner


def test_queue_runs_fixed_variants_in_order(tmp_path,monkeypatch):
    calls=[];writes=[]
    monkeypatch.setattr(runner,'validate_training',lambda root:{'identity':'fixed'})
    monkeypatch.setattr(runner.lane,'train',lambda a:calls.append(a.variant))
    monkeypatch.setattr(runner,'read',lambda p:{'identity':'run'})
    monkeypatch.setattr(runner,'verify_done',lambda *a:None)
    monkeypatch.setattr(runner,'write',lambda p,v:writes.append((p,v)))
    args=SimpleNamespace(root=tmp_path,variants=['raw','binary'])
    runner.queue(args)
    assert calls==['raw','binary']
    assert writes[-1][1]['status']=='COMPLETE'


def test_failed_arm_does_not_report_success_or_skip_to_next(tmp_path,monkeypatch):
    calls=[];writes=[]
    monkeypatch.setattr(runner,'validate_training',lambda root:{'identity':'fixed'})
    def fail(a):
        calls.append(a.variant)
        raise ValueError('failed-real-worker')
    monkeypatch.setattr(runner.lane,'train',fail)
    monkeypatch.setattr(runner,'write',lambda p,v:writes.append((p,v)))
    with pytest.raises(ValueError,match='failed-real-worker'):
        runner.queue(SimpleNamespace(root=tmp_path,variants=['raw','binary']))
    assert calls==['raw']
    assert all(v.get('status')!='COMPLETE' for _,v in writes)


def test_training_gate_rejects_lost_reference_support(tmp_path,monkeypatch):
    selection={'selected':['a'],'preserve_full_reference_train':True}
    cfg={'schema':'causal_stl_available_subset.v1','identity':'fixed','columns':['a'],
         'selection_sha256':runner.digest(selection),'original_support':'original'}
    result={'training_eligible':True,'activity_excluded':True,'original_support_preserved':True,
            'monthly_coverage':{'TRAIN':{'2020-01':1}},
            'variants':{k:{'support':'wrong-but-paired'} for k in runner.lane.VARIANTS}}
    monkeypatch.setattr(runner.lane,'authenticated',lambda root:cfg)
    monkeypatch.setattr(runner,'verify_done',lambda *a:None)
    monkeypatch.setattr(runner,'read',lambda p:selection if p.name=='SELECTION.json' else result)
    with pytest.raises(ValueError,match='REFERENCE_SUPPORT'):
        runner.validate_training(tmp_path)
