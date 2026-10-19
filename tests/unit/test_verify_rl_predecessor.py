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


@pytest.mark.parametrize('tamper',[False,True])
def test_paired_predecessor_counts_and_authenticates_monitor_files(tmp_path,tamper):
    from agent_plugins.paired_early_stopping import SETTINGS
    root=tmp_path/'old'; _,close=fixture(root)
    body=dict(seeds=[7],arms=['SAC_FLAT'],common_config=dict(paired_early_stopping=SETTINGS))
    (root/'CONTRACT.json').write_text(json.dumps(body));pin=digest(root/'CONTRACT.json')
    state=json.loads((root/'STATUS.json').read_text());state['contract_sha256']=pin
    for t in state['tasks']:t['arm']='SAC_FLAT'
    (root/'STATUS.json').write_text(json.dumps(state))
    pair=root/'task/attempt-1/paired-monitor/step-005000/PAIR.json'
    pair.parent.mkdir(parents=True);pair.write_text('{"test":"pair"}')
    close.update(contract_sha256=pin,monitor_reports=2,
        monitor_files={str(pair.relative_to(root)):digest(pair)},
        warehouse=dict(expected=4,delivered=4,pending=0,mismatch=0))
    (root/'CLOSE.json').write_text(json.dumps(close))
    if tamper:pair.write_text('changed')
    if tamper:
        with pytest.raises(ValueError):predecessor_ready(root,pin)
    else:assert predecessor_ready(root,pin)=='READY'
