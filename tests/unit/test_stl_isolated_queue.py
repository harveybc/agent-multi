import hashlib
import json
from types import SimpleNamespace
import subprocess

import pytest
from tools import run_stl_isolated_queue as queue


def test_each_variant_is_an_independent_process_and_prerequisite_is_verified(tmp_path,monkeypatch):
    a=SimpleNamespace(root=tmp_path,runtime=tmp_path,python=tmp_path/'python',parent=tmp_path,
        bundle=tmp_path,expected_device_name='TEST',after=['continuous'],variants=['ordinal','binary'])
    events=[]
    monkeypatch.setattr(queue,'verify_complete',lambda root,v:events.append(('verify',v)))
    monkeypatch.setattr(queue.subprocess,'run',lambda cmd,**kw:events.append(('process',cmd[cmd.index('--variants')+1])))
    queue.run(a)
    assert events==[('verify','continuous'),('process','ordinal'),('verify','ordinal'),('process','binary'),('verify','binary')]


def test_receipt_rejects_corruption_and_checks_existing_variant(tmp_path,monkeypatch):
    folder=tmp_path/'training/raw';folder.mkdir(parents=True)
    result=dict(status='COMPLETE',variant='raw',identity='x')
    (folder/'RESULT.json').write_text(json.dumps(result))
    h=hashlib.sha256((folder/'RESULT.json').read_bytes()).hexdigest()
    (folder/'DONE.json').write_text(json.dumps(dict(identity='x',files={'RESULT.json':h})))
    queue.verify_complete(tmp_path,'raw')
    (folder/'RESULT.json').write_text(json.dumps(result)+' ')
    with pytest.raises(ValueError,match='HASH'):queue.verify_complete(tmp_path,'raw')


def test_process_error_stops_before_next_variant(tmp_path,monkeypatch):
    a=SimpleNamespace(root=tmp_path,runtime=tmp_path,python=tmp_path/'python',parent=tmp_path,
        bundle=tmp_path,expected_device_name='TEST',after=[],variants=['ordinal','binary'])
    calls=[]
    def fail(cmd,**kw):
        calls.append(cmd);raise subprocess.CalledProcessError(1,cmd)
    monkeypatch.setattr(queue.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):queue.run(a)
    assert len(calls)==1
