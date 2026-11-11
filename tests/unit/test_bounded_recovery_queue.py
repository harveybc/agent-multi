import sys
import pytest
from app.bounded_recovery_queue import execute, sha


def task(tmp_path, name, code, outputs):
    return dict(id=name, argv=[sys.executable,'-c',code], cwd=str(tmp_path),
                timeout_seconds=10, outputs=[str(p) for p in outputs])


def test_failure_does_not_block_independent_work(tmp_path):
    f=tmp_path/'answer'
    cfg=dict(source_hashes={},tasks=[task(tmp_path,'bad','raise ValueError()',[]),
        task(tmp_path,'good',f'from pathlib import Path;Path({str(f)!r}).write_text("ok")',[f])])
    result=execute(cfg,tmp_path/'queue')
    assert [x['status'] for x in result['tasks']]==['FAILED','COMPLETE']
    assert f.read_text()=='ok'


def test_resume_and_tamper(tmp_path):
    f=tmp_path/'answer'
    cfg=dict(source_hashes={},tasks=[task(tmp_path,'good',
        f'from pathlib import Path;p=Path({str(f)!r});assert not p.exists();p.write_text("ok")',[f])])
    assert execute(cfg,tmp_path/'q')['status']=='COMPLETE'
    assert execute(cfg,tmp_path/'q')['tasks'][0]['resumed']
    f.write_text('changed')
    with pytest.raises(ValueError,match='OUTPUT_CHANGED'):
        execute(cfg,tmp_path/'q')


def test_no_false_success_or_identity_change(tmp_path):
    cfg=dict(source_hashes={},tasks=[task(tmp_path,'empty','pass',[tmp_path/'missing'])])
    assert execute(cfg,tmp_path/'q')['status']=='COMPLETE_WITH_FAILURES'
    cfg['tasks'][0]['timeout_seconds']=11
    with pytest.raises(ValueError,match='IDENTITY_CHANGED'):
        execute(cfg,tmp_path/'q')


def test_changed_source_rejected_before_launch(tmp_path):
    source=tmp_path/'source';source.write_text('original')
    cfg=dict(source_hashes={str(source):sha(source)},tasks=[])
    source.write_text('changed')
    with pytest.raises(ValueError,match='SOURCE_CHANGED'):
        execute(cfg,tmp_path/'q')
    assert not (tmp_path/'q').exists()


def test_timeout_continues_independent_work(tmp_path):
    slow=task(tmp_path,'slow','import time;time.sleep(30)',[])
    slow['timeout_seconds']=0.1
    f=tmp_path/'answer'
    cfg=dict(source_hashes={},tasks=[slow,task(tmp_path,'good',
        f'from pathlib import Path;Path({str(f)!r}).write_text("ok")',[f])])
    result=execute(cfg,tmp_path/'q')
    assert [x['status'] for x in result['tasks']]==['FAILED','COMPLETE']
    assert 'timed out' in result['tasks'][0]['error']
