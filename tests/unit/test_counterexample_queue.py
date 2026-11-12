from types import SimpleNamespace

from tools import run_counterexample_selection_queue as queue


def test_failed_contrasts_do_not_repeat_unchanged_selection(monkeypatch,tmp_path):
    calls=[];writes=[]
    args=SimpleNamespace(root=tmp_path,parent=tmp_path,bundle=tmp_path,floor=tmp_path,
        probe=tmp_path,diagnostic=tmp_path,selection_template=tmp_path/'template',expected_device_name='test')
    monkeypatch.setattr(queue.subprocess,'run',lambda command,**kw:calls.append(command))
    monkeypatch.setattr(queue,'read',lambda path:dict(identity='fixed',qualified_for_selection=False,
        rounds=[{'added_constraints':[[1,0,2]]}]))
    monkeypatch.setattr(queue,'verify_done',lambda *a:None)
    monkeypatch.setattr(queue,'write',lambda path,value:writes.append((path,value)))
    queue.run(args)
    assert len(calls)==2
    assert all(c[1]=='tools/run_account_counterexamples.py' for c in calls)
    assert writes[-1][1]['status']=='COMPLETE_NO_QUALIFIED_NEW_CONTEXT'


def test_primary_success_chains_selection_report_export_without_fallback(monkeypatch,tmp_path):
    calls=[];writes=[]
    args=SimpleNamespace(root=tmp_path,parent=tmp_path,bundle=tmp_path,floor=tmp_path,
        probe=tmp_path,diagnostic=tmp_path,selection_template=tmp_path/'template',expected_device_name='test')
    def read(path):
        if path.name=='template':
            return dict(archive='/archive',archive_step=12,original_selection='/original',
                        source_files={'/archive/RESULT.json':'sha'})
        return dict(identity='fixed',qualified_for_selection=True)
    monkeypatch.setattr(queue.subprocess,'run',lambda command,**kw:calls.append(command))
    monkeypatch.setattr(queue,'read',read)
    monkeypatch.setattr(queue,'verify_done',lambda *a:None)
    monkeypatch.setattr(queue,'file_sha',lambda *a:'sha')
    monkeypatch.setattr(queue,'write',lambda path,value:writes.append((path,value)))
    queue.run(args)
    assert [c[1] for c in calls]==['tools/run_account_counterexamples.py',
        'tools/run_cumulative_mask_pass.py','tools/run_cumulative_mask_pass.py',
        'tools/report_cumulative_mask_pass.py','tools/export_cumulative_sac_actor.py']
    assert writes[-1][1]['selected_worker']=='floor'


def test_no_counterexamples_skips_identical_fallback(monkeypatch,tmp_path):
    calls=[];writes=[]
    args=SimpleNamespace(root=tmp_path,parent=tmp_path,bundle=tmp_path,floor=tmp_path,
        probe=tmp_path,diagnostic=tmp_path,selection_template=tmp_path/'template',expected_device_name='test')
    monkeypatch.setattr(queue.subprocess,'run',lambda command,**kw:calls.append(command))
    monkeypatch.setattr(queue,'read',lambda path:dict(identity='fixed',qualified_for_selection=False,
        rounds=[{'added_constraints':[]}]))
    monkeypatch.setattr(queue,'verify_done',lambda *a:None)
    monkeypatch.setattr(queue,'write',lambda path,value:writes.append((path,value)))
    queue.run(args)
    assert len(calls)==1
    assert writes[-1][1]['selection_started'] is False
