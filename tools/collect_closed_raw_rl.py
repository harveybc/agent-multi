"""One-shot closed RAW/selected RL collection; never trains or contacts a warehouse."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sqlite3
import subprocess

from app.curated_rl_input import _digest as digest
from tools.verify_rl_static_engineering import save_status


def selected_close_matches(close, contract):
    if close.get('schema') != 'selected_rl_close.v1':
        return close.get('schema') == 'raw_rl_close.v1'
    flags = dict(feature_selection_complete=True,
                 selection_complete='WITHIN_DECLARED_ADMITTED_POPULATION', full_inventory_complete=False)
    binding = close.get('selected_input_binding')
    if (any(close.get(k) != v or contract.get(k) != v for k,v in flags.items())
            or not isinstance(binding,dict) or binding != contract.get('selected_input_binding')):
        return False
    columns = binding.get('feature_columns')
    return (isinstance(columns,list) and bool(columns) and all(isinstance(c,str) for c in columns)
            and len(set(columns)) == len(columns)
            and bool(re.fullmatch('[0-9a-f]{64}',str(binding.get('selection_sha256')))))


def member(root, relative):
    name = PurePosixPath(relative)
    path = root / str(name)
    if name.is_absolute() or '..' in name.parts or not path.resolve().is_relative_to(root):
        raise ValueError('ARCHIVE_PATH_ESCAPE')
    return path


def live_monitor_state(relative):
    name = PurePosixPath(relative)
    return (name.parent.name == 'paired-monitor' and name.name in
            ('warehouse-outbox.sqlite', 'warehouse-outbox.sqlite-wal', 'warehouse-outbox.sqlite-shm'))


def verify(archive, source, close_sha):
    names = ('CLOSE.json', 'CONTRACT.json', 'STATUS.json', 'metrics-backup.sqlite')
    proofs = {name: digest(member(archive, name)) for name in names}
    close, status, contract = [json.loads((archive / name).read_bytes()) for name in ('CLOSE.json', 'STATUS.json', 'CONTRACT.json')]
    warehouse, tasks = close['warehouse'], status['tasks']
    monitor_files = close.get('monitor_files', {})
    paired = contract.get('common_config', {}).get('paired_early_stopping')
    task_count, report_count = 12, 12
    if paired:
        from agent_plugins.paired_early_stopping import validate_settings
        validate_settings(paired)
        arms=contract.get('arms')
        if (arms not in (['SAC_FLAT'],['SAC_PER_FEATURE_CONV1D']) or close.get('schema') != 'selected_rl_close.v1'
                or [(t.get('arm'),t.get('split')) for t in tasks] !=
                    [(arms[0],'train'),(arms[0],'validation')]):
            raise ValueError('PAIRED_COLLECTION_TASKS_INVALID')
        n_pairs = sum(name.endswith('/PAIR.json') for name in monitor_files)
        if not 1 <= n_pairs <= 40 or close.get('monitor_reports') != 2 * n_pairs:
            raise ValueError('PAIRED_COLLECTION_MONITORS_INVALID')
        task_count, report_count = 2, 2 + 2 * n_pairs
    if (proofs['CLOSE.json'] != close_sha or not selected_close_matches(close,contract)
            or close.get('state') != 'COMPLETED_AND_WAREHOUSE_READBACK_VERIFIED'
            or close['completed_tasks'] != task_count or len(tasks) != task_count or len({t['id'] for t in tasks}) != task_count
            or warehouse['expected'] != report_count or warehouse['delivered'] != report_count or warehouse['pending'] or warehouse['mismatch']
            or close['physical_copy'] != 'metrics-backup.sqlite' or proofs['metrics-backup.sqlite'] != close['physical_sha256']
            or proofs['CONTRACT.json'] != close['contract_sha256'] or status['contract_sha256'] != close['contract_sha256']):
        raise ValueError('CLOSED_COLLECTION_IDENTITY_MISMATCH')
    for task in tasks:
        if task['state'] != 'ENQUEUED' or not {'config', 'summary', 'checkpoint', 'effective_config'} <= task['artifacts'].keys():
            raise ValueError('TASK_NOT_CLOSED')
        for key, pin in task['artifacts'].items():
            remote = PurePosixPath(task[key])
            if not remote.is_absolute() or '..' in remote.parts:
                raise ValueError('REMOTE_ARTIFACT_PATH_ESCAPE')
            relative = str(remote.relative_to(source))
            if digest(member(archive, relative)) != pin:
                raise ValueError('ARTIFACT_DIGEST_MISMATCH')
            proofs[relative] = pin
    for name, pin in monitor_files.items():
        path = member(archive,name)
        # Older publishers indexed their redundant live outbox; the canonical
        # closed backup and readback counts above remain mandatory.
        if paired and live_monitor_state(name):
            continue
        if digest(path) != pin:
            raise ValueError('MONITOR_DIGEST_MISMATCH')
        proofs[name] = pin
    for path in archive.rglob('*'):
        if (path.name.startswith('warehouse-outbox.sqlite') or '.lock' in path.name or path.name == 'locks'
                or path.name.endswith(('-wal', '-shm')) or '.sqlite' in path.name and path != archive / 'metrics-backup.sqlite'):
            raise ValueError('LIVE_DATABASE_OR_LOCK_IN_ARCHIVE')
    db = sqlite3.connect((archive / 'metrics-backup.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
            raise ValueError('SQLITE_BACKUP_INVALID')
    finally:
        db.close()
    return proofs


def collect(host, source_root, archive_root, state_root):
    source = PurePosixPath(source_root)
    if (not re.fullmatch(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*', host)
            or not source.is_absolute() or str(source) == '/' or '..' in source.parts
            or any(c in source_root for c in '\n\r\0') or not archive_root.is_absolute() or not state_root.is_absolute()):
        raise ValueError('UNSAFE_COLLECTION_ARGUMENTS')
    archive, state = archive_root.resolve(), state_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    if (archive.is_relative_to(state) or state.is_relative_to(archive)
            or any(p.is_relative_to(repo) for p in (archive, state))):
        raise ValueError('SEPARATE_PRIVATE_COLLECTION_ROOTS_REQUIRED')
    state.mkdir(parents=True, exist_ok=True)
    complete = state / 'COLLECTION_COMPLETE.json'
    identity = dict(source_host=host, source_root=str(source), archive_root=str(archive))
    try:
        if complete.exists():
            saved = json.loads(complete.read_bytes())
            if saved['identity'] != identity or verify(archive, source, saved['close_sha256']) != saved['proofs']:
                raise ValueError('EXISTING_COLLECTION_CHANGED')
            return 0
        quoted = shlex.quote(str(source / 'CLOSE.json'))
        command = f'if test -f {quoted}; then cat -- {quoted}; else printf PENDING; fi'
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '--', host, command],
                                check=True, capture_output=True, timeout=30)
        if result.stdout.strip() == b'PENDING':
            save_status(state / 'COLLECTION_STATUS.json', dict(state='PENDING'))
            return 0
        close_sha = hashlib.sha256(result.stdout).hexdigest()
        close = json.loads(result.stdout)
        if close.get('schema') not in ('raw_rl_close.v1','selected_rl_close.v1') or close.get('state') != 'COMPLETED_AND_WAREHOUSE_READBACK_VERIFIED':
            raise ValueError('REMOTE_CLOSE_INVALID')
        archive.mkdir(parents=True, exist_ok=True)
        command = ['rsync', '-rlt', '--fsync', '--safe-links', '--protect-args', '--timeout=60',
                   '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10', '--include=/metrics-backup.sqlite',
                   '--exclude=warehouse-outbox.sqlite*', '--exclude=*.sqlite*', '--exclude=*-wal', '--exclude=*-shm',
                   '--exclude=*.lock*', '--exclude=locks/', '--exclude=*.partial*', '--', f'{host}:{source}/', str(archive) + '/']
        subprocess.run(command, check=True, capture_output=True, timeout=3600)
        proofs = verify(archive, source, close_sha)
        record = dict(schema='selected_rl_collection.v1' if close['schema']=='selected_rl_close.v1' else 'raw_rl_collection.v1', state='COLLECTION_COMPLETE',
                      identity=identity, close_sha256=close_sha, proofs=proofs)
        if close.get('monitor_files'):
            record['excluded_live_monitor_state'] = sorted(
                name for name in close['monitor_files'] if live_monitor_state(name))
        save_status(state / 'COLLECTION_STATUS.json', record)
        save_status(complete, record)
        return 0
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, subprocess.SubprocessError) as exc:
        save_status(state / 'COLLECTION_STATUS.json', dict(state='COLLECTION_REFUSED', error_type=type(exc).__name__))
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-host', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--state-root', type=Path, required=True)
    args = parser.parse_args()
    return collect(args.source_host, args.source_root, args.archive_root, args.state_root)


if __name__ == '__main__':
    raise SystemExit(main())
