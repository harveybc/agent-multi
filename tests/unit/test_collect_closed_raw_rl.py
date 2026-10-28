import json
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess

import pytest

from tools import collect_closed_raw_rl as m

REMOTE = '/srv/example/raw control'
HOST = 'source.example.invalid'


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source = tmp_path / 'fixture'
    source.mkdir()
    (source / 'CONTRACT.json').write_text('{"schema":"curated_rl_campaign.v1"}')
    tasks = []
    for i in range(12):
        task = dict(id=str(i), state='ENQUEUED', artifacts={})
        for key in ('config', 'summary', 'checkpoint', 'effective_config'):
            path = source / f'task-{i}' / f'{key}.bin'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f'fixture {i} {key}'.encode())
            task[key] = REMOTE + '/' + str(path.relative_to(source))
            task['artifacts'][key] = m.digest(path)
        tasks.append(task)
    pin = m.digest(source / 'CONTRACT.json')
    (source / 'STATUS.json').write_text(json.dumps(dict(contract_sha256=pin, tasks=tasks)))
    with sqlite3.connect(source / 'metrics-backup.sqlite') as db:
        db.execute('CREATE TABLE evidence(value INTEGER)')
        db.execute('INSERT INTO evidence VALUES(12)')
    db.close()
    for name in ('warehouse-outbox.sqlite', 'warehouse-outbox.sqlite-wal', 'other.sqlite-shm', '.publish.lock'):
        (source / name).write_bytes(b'LIVE MUST NOT COPY')
    close = dict(schema='raw_rl_close.v1', state='COMPLETED_AND_WAREHOUSE_READBACK_VERIFIED',
                 completed_tasks=12, contract_sha256=pin, physical_copy='metrics-backup.sqlite',
                 physical_sha256=m.digest(source / 'metrics-backup.sqlite'),
                 warehouse=dict(expected=12, delivered=12, pending=0, mismatch=0))
    (source / 'CLOSE.json').write_text(json.dumps(close))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == 'ssh':
            return subprocess.CompletedProcess(command, 0, (source / 'CLOSE.json').read_bytes(), b'')
        assert command[0] == 'rsync'
        assert '--protect-args' in command and '--safe-links' in command
        assert '--exclude=warehouse-outbox.sqlite*' in command
        shutil.copytree(source, tmp_path / 'archive', dirs_exist_ok=True, symlinks=True,
                        ignore=shutil.ignore_patterns('warehouse-outbox.sqlite*', '*-wal', '*-shm', '*.lock*', 'locks'))
        return subprocess.CompletedProcess(command, 0, b'', b'')

    monkeypatch.setattr(m.subprocess, 'run', run)
    return source, tmp_path / 'archive', tmp_path / 'state', calls


def test_pending_one_ssh_no_archive_or_poll(campaign, monkeypatch):
    _, archive, state, calls = campaign

    def pending(command, **kw):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, b'PENDING', b'')

    monkeypatch.setattr(m.subprocess, 'run', pending)
    assert m.collect(HOST, REMOTE, archive, state) == 0
    assert len(calls) == 1 and not archive.exists()
    assert json.loads((state / 'COLLECTION_STATUS.json').read_text())['state'] == 'PENDING'
    assert not (state / 'COLLECTION_COMPLETE.json').exists()


def test_exact_collection_closed_sqlite_exclusions_and_readonly_next_tick(campaign, monkeypatch):
    source, archive, state, calls = campaign
    assert m.collect(HOST, REMOTE, archive, state) == 0
    assert [c[0] for c in calls] == ['ssh', 'rsync']
    assert not any(p.name.startswith('warehouse-outbox.sqlite') or p.name.endswith(('-wal', '-shm', '.lock'))
                   for p in archive.rglob('*'))
    proof = json.loads((state / 'COLLECTION_COMPLETE.json').read_text())
    assert proof['state'] == 'COLLECTION_COMPLETE'
    assert proof['proofs']['metrics-backup.sqlite'] == m.digest(source / 'metrics-backup.sqlite')
    assert len(proof['proofs']) == 52
    before = {p.name: p.read_bytes() for p in state.iterdir()}
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: pytest.fail('complete collection contacted source'))
    assert m.collect(HOST, REMOTE, archive, state) == 0
    assert {p.name: p.read_bytes() for p in state.iterdir()} == before


@pytest.mark.parametrize('tamper',[None,'scope','binding','columns'])
def test_selected_collection_requires_same_scoped_selection_as_contract(campaign,tamper):
    source,archive,state,_=campaign
    flags=dict(feature_selection_complete=True,selection_complete='WITHIN_DECLARED_ADMITTED_POPULATION',full_inventory_complete=False)
    binding=dict(selection_sha256='a'*64,feature_columns=['selected_a','selected_b'])
    contract=dict(schema='curated_rl_campaign.v1',**flags,selected_input_binding=binding)
    (source/'CONTRACT.json').write_text(json.dumps(contract))
    pin=m.digest(source/'CONTRACT.json')
    status=json.loads((source/'STATUS.json').read_text()); status['contract_sha256']=pin
    (source/'STATUS.json').write_text(json.dumps(status))
    close=json.loads((source/'CLOSE.json').read_text())
    close.update(schema='selected_rl_close.v1',contract_sha256=pin,**flags,selected_input_binding=dict(binding))
    if tamper=='scope':close['full_inventory_complete']=True
    if tamper=='binding':close['selected_input_binding']['selection_sha256']='b'*64
    if tamper=='columns':close['selected_input_binding']['feature_columns']=['selected_b','selected_a']
    (source/'CLOSE.json').write_text(json.dumps(close))
    assert m.collect(HOST,REMOTE,archive,state)==(0 if tamper is None else 2)
    assert (state/'COLLECTION_COMPLETE.json').exists()==(tamper is None)


@pytest.mark.parametrize('tamper',[False,True])
@pytest.mark.parametrize('legacy_outbox',[False,True])
@pytest.mark.parametrize('arm',['SAC_FLAT','SAC_PER_FEATURE_CONV1D'])
@pytest.mark.parametrize('fixed',[False,True])
@pytest.mark.parametrize('wrong_task',[False,True])
def test_paired_direct_collection_counts_monitor_reports_and_checks_files(campaign,tamper,legacy_outbox,arm,fixed,wrong_task):
    from agent_plugins.paired_early_stopping import SETTINGS
    if fixed:
        from agent_plugins.fixed_budget_monitor import SETTINGS
    source,archive,state,_=campaign
    flags=dict(feature_selection_complete=True,selection_complete='WITHIN_DECLARED_ADMITTED_POPULATION',full_inventory_complete=False)
    binding=dict(selection_sha256='a'*64,feature_columns=['a','b'])
    contract=dict(schema='curated_rl_campaign.v1',arms=[arm],
        common_config=dict(paired_early_stopping=SETTINGS),**flags,selected_input_binding=binding)
    (source/'CONTRACT.json').write_text(json.dumps(contract))
    pin=m.digest(source/'CONTRACT.json')
    status=json.loads((source/'STATUS.json').read_text())
    status['tasks']=status['tasks'][:2]; status['contract_sha256']=pin
    for task,role in zip(status['tasks'],('train','validation')):
        task.update(arm=arm,split=role)
    if wrong_task:status['tasks'][1]['arm']='SAC_FLAT' if arm!='SAC_FLAT' else 'SAC_PER_FEATURE_CONV1D'
    (source/'STATUS.json').write_text(json.dumps(status))
    pair=source/'task-0/paired-monitor/step-005000/PAIR.json'
    pair.parent.mkdir(parents=True); pair.write_text('{"schema":"synthetic-pair"}')
    monitor_files={str(pair.relative_to(source)):m.digest(pair)}
    if legacy_outbox:
        live = pair.parent.parent / 'warehouse-outbox.sqlite'
        live.write_bytes(b'LIVE REDUNDANT OUTBOX')
        monitor_files[str(live.relative_to(source))] = m.digest(live)
    close=json.loads((source/'CLOSE.json').read_text())
    close.update(schema='selected_rl_close.v1',contract_sha256=pin,
        completed_tasks=2,monitor_files=monitor_files,monitor_reports=2,
        warehouse=dict(expected=4,delivered=4,pending=0,mismatch=0),
        **flags,selected_input_binding=binding)
    (source/'CLOSE.json').write_text(json.dumps(close))
    if tamper: pair.write_text('changed')
    assert m.collect(HOST,REMOTE,archive,state)==(2 if tamper or wrong_task else 0)
    if legacy_outbox and not tamper and not wrong_task:
        record=json.loads((state/'COLLECTION_COMPLETE.json').read_text())
        assert record['excluded_live_monitor_state']==[str(live.relative_to(source))]
        assert not (archive/live.relative_to(source)).exists()


@pytest.mark.parametrize('target', ['CONTRACT.json', 'metrics-backup.sqlite', 'task-0/summary.bin'])
def test_changed_bytes_never_mark_collection_complete(campaign, target):
    source, archive, state, _ = campaign
    (source / target).write_bytes(b'tampered')
    assert m.collect(HOST, REMOTE, archive, state) == 2
    assert not (state / 'COLLECTION_COMPLETE.json').exists()


@pytest.mark.parametrize('path', ['/tmp/outside', REMOTE + '/../outside', 'relative', REMOTE + '/task-0/../../outside'])
def test_artifact_paths_cannot_escape_remote_or_archive_roots(campaign, path):
    source, archive, state, _ = campaign
    status = json.loads((source / 'STATUS.json').read_text())
    status['tasks'][0]['summary'] = path
    (source / 'STATUS.json').write_text(json.dumps(status))
    assert m.collect(HOST, REMOTE, archive, state) == 2
    assert not (state / 'COLLECTION_COMPLETE.json').exists()


def test_symlink_escape_cannot_satisfy_a_digest(campaign, tmp_path):
    source, archive, state, _ = campaign
    path = source / 'task-0/summary.bin'
    outside = tmp_path / 'outside'
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    assert m.collect(HOST, REMOTE, archive, state) == 2


@pytest.mark.parametrize('host', ['-oProxyCommand=bad', 'host;echo bad', 'user@host bad', 'host:/path'])
def test_unsafe_ssh_host_refused_without_network(campaign, host):
    _, archive, state, calls = campaign
    with pytest.raises(ValueError):
        m.collect(host, REMOTE, archive, state)
    assert not calls


def test_source_root_must_be_absolute_and_is_shell_quoted(campaign):
    _, archive, state, calls = campaign
    with pytest.raises(ValueError):
        m.collect(HOST, '../relative', archive, state)
    assert not calls
    assert m.collect(HOST, REMOTE, archive, state) == 0
    assert shlex.quote(REMOTE + '/CLOSE.json') in calls[0][-1]


def test_non_terminal_status_or_unreadback_warehouse_refuses(campaign):
    source, archive, state, _ = campaign
    status = json.loads((source / 'STATUS.json').read_text())
    status['tasks'][0]['state'] = 'RUNNING'
    (source / 'STATUS.json').write_text(json.dumps(status))
    assert m.collect(HOST, REMOTE, archive, state) == 2
    assert not (state / 'COLLECTION_COMPLETE.json').exists()


def test_later_archive_tampering_detected_without_network(campaign, monkeypatch):
    _, archive, state, _ = campaign
    assert m.collect(HOST, REMOTE, archive, state) == 0
    (archive / 'task-0/summary.bin').write_bytes(b'changed after collection')
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: pytest.fail('no source polling on skip'))
    assert m.collect(HOST, REMOTE, archive, state) == 2


@pytest.mark.parametrize('change', ['pending', 'mismatch', 'delivered', 'completed_tasks'])
def test_incomplete_close_readback_never_marks_complete(campaign, change):
    source, archive, state, _ = campaign
    close = json.loads((source / 'CLOSE.json').read_text())
    if change == 'completed_tasks':
        close[change] = 11
    else:
        close['warehouse'][change] = 1
    (source / 'CLOSE.json').write_text(json.dumps(close))
    assert m.collect(HOST, REMOTE, archive, state) == 2
    assert not (state / 'COLLECTION_COMPLETE.json').exists()


def test_matching_digest_does_not_make_non_sqlite_a_closed_backup(campaign):
    source, archive, state, _ = campaign
    (source / 'metrics-backup.sqlite').write_bytes(b'not sqlite')
    close = json.loads((source / 'CLOSE.json').read_text())
    close['physical_sha256'] = m.digest(source / 'metrics-backup.sqlite')
    (source / 'CLOSE.json').write_text(json.dumps(close))
    assert m.collect(HOST, REMOTE, archive, state) == 2
    assert not (state / 'COLLECTION_COMPLETE.json').exists()


def test_close_change_between_ssh_and_rsync_fails_exact_bytes(campaign, monkeypatch):
    _, archive, state, _ = campaign
    fake = m.subprocess.run

    def changed(command, **kw):
        result = fake(command, **kw)
        if command[0] == 'rsync':
            path = archive / 'CLOSE.json'
            path.write_bytes(path.read_bytes() + b'\n')
        return result

    monkeypatch.setattr(m.subprocess, 'run', changed)
    assert m.collect(HOST, REMOTE, archive, state) == 2
    assert not (state / 'COLLECTION_COMPLETE.json').exists()


def test_shell_metacharacters_in_absolute_root_are_only_data(campaign):
    source, archive, state, calls = campaign
    remote = REMOTE + "/'; echo NOT_A_COMMAND; #"
    status = json.loads((source / 'STATUS.json').read_text())
    for task in status['tasks']:
        for key in task['artifacts']:
            task[key] = remote + task[key][len(REMOTE):]
    (source / 'STATUS.json').write_text(json.dumps(status))
    assert m.collect(HOST, remote, archive, state) == 0
    assert shlex.quote(remote + '/CLOSE.json') in calls[0][-1]
    assert calls[1][-2] == HOST + ':' + remote + '/'
