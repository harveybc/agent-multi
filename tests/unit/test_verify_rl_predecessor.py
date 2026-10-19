import json

import pytest

from tools.verify_rl_predecessor import predecessor_ready
from tools.curated_rl_campaign import digest


def test_pending_does_not_create_or_change_predecessor(tmp_path):
    root=tmp_path/'absent'
    assert predecessor_ready(root,'a'*64)=='WAIT_PREDECESSOR'
    assert not root.exists()


def fixture(root):
    root.mkdir()
    (root/'CONTRACT.json').write_text(json.dumps(dict(seeds=[7],arms=['a'])))
    pin=digest(root/'CONTRACT.json')
    (root/'metrics-backup.sqlite').write_bytes(b'fixture-closed-copy')
    tasks=[dict(arm='a',seed=7,split=role,state='ENQUEUED',artifacts={})
           for role in ('train','validation')]
    (root/'STATUS.json').write_text(json.dumps(dict(contract_sha256=pin,tasks=tasks)))
    close=dict(state='COMPLETED_AND_WAREHOUSE_READBACK_VERIFIED',contract_sha256=pin,
        completed_tasks=2,warehouse=dict(expected=2,delivered=2,pending=0,mismatch=0),
        physical_copy='metrics-backup.sqlite',physical_sha256=digest(root/'metrics-backup.sqlite'))
    (root/'CLOSE.json').write_text(json.dumps(close))
    return pin,close


def test_verified_predecessor_is_read_only(tmp_path):
    root=tmp_path/'old'; pin,_=fixture(root)
    before={p.name:p.read_bytes() for p in root.iterdir()}
    assert predecessor_ready(root,pin)=='READY'
    assert before=={p.name:p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize('change',['contract','database','pending','tasks'])
def test_broken_close_does_not_dispatch(tmp_path,change):
    root=tmp_path/'old'; pin,close=fixture(root)
    if change=='contract':(root/'CONTRACT.json').write_text('{}')
    if change=='database':(root/'metrics-backup.sqlite').write_bytes(b'changed')
    if change=='pending':close['warehouse']['pending']=1
    if change=='tasks':close['completed_tasks']=1
    (root/'CLOSE.json').write_text(json.dumps(close))
    with pytest.raises(ValueError):predecessor_ready(root,pin)
