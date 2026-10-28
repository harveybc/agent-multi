"""Systemd entrypoint and read-only status/results for the existing static queue."""
import argparse
import fcntl
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3

from tools import curated_rl_campaign as queue
from tools.verify_rl_static_engineering import save_status


METRICS = ('initial_cash', 'final_equity', 'net_profit', 'total_return',
           'max_drawdown_pct', 'sharpe_ratio', 'trades_closed', 'win_rate_pct',
           'profit_factor', 'trade_costs_total', 'entry_submissions',
           'filled_entries', 'max_entries_filled_in_day')


def run(root):
    return queue.run(root/'CONTRACT.json', root, resume=(root/'STATUS.json').exists())


def read_status(root):
    state = queue.status(root)
    tasks = state.get('tasks', [])
    result = dict(state=state['state'], completed_tasks=sum(t['state'] == 'ENQUEUED' for t in tasks),
        declared_tasks=len(tasks), current_training_eta_seconds=None, campaign_eta_seconds=None,
        scientific_scope='REAL_DATA_DEVELOPMENT', feature_selection_complete=False)
    current = next((t for t in tasks if t['state'] == 'RUNNING'), None)
    if current:
        result['current'] = {k: current[k] for k in ('arm', 'split', 'seed', 'attempt') if k in current}
        if current.get('config'):
            progress_path = Path(current['config']).parent/'progress.json'
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                result['training_progress'] = progress
                updated = datetime.fromisoformat(progress['updated_at_utc'].replace('Z', '+00:00'))
                age = (datetime.now(timezone.utc)-updated).total_seconds()
                result['heartbeat_age_seconds'] = age
                step, total = progress['current_step'], progress['total_timesteps']
                elapsed = progress.get('elapsed_seconds')
                if (progress.get('status') == 'training' and 0 <= age <= 180
                        and type(step) is int and type(total) is int
                        and 5000 <= step <= total
                        and type(elapsed) in (int, float)
                        and math.isfinite(elapsed) and elapsed > 0):
                    result['current_training_eta_seconds'] = elapsed*(total-step)/step
    warehouse = root/'WAREHOUSE_STATUS.json'
    if warehouse.exists():
        result['warehouse'] = json.loads(warehouse.read_text())
    return result


def results(root):
    rows = []
    for task in queue.status(root).get('tasks', []):
        if task['state'] != 'ENQUEUED':
            continue
        summary = json.loads(Path(task['summary']).read_text())
        rows.append({**{k: task[k] for k in ('arm', 'seed', 'split')},
                     **{k: summary.get(k) for k in METRICS}})
    return rows


def publish_and_close(root):
    with (root/'.publish.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        return _publish_and_close(root)


def _publish_and_close(root):
    close_path = root/'CLOSE.json'
    if close_path.exists():
        close = json.loads(close_path.read_text())
        if queue.digest(root/close['physical_copy']) != close['physical_sha256']:
            raise ValueError('CLOSED_PHYSICAL_COPY_CHANGED')
        return 0
    body = json.loads((root/'CONTRACT.json').read_text())
    if queue.pump_warehouse(body, root) != 0:
        return 2
    state = queue.status(root)
    tasks = state.get('tasks', [])
    warehouse = json.loads((root/'WAREHOUSE_STATUS.json').read_text())['warehouse']
    if (not tasks or any(t['state'] != 'ENQUEUED' for t in tasks)
            or warehouse['expected'] != len(tasks) or warehouse['delivered'] != len(tasks)
            or warehouse['pending'] or warehouse['mismatch']):
        return 0  # computation still owns continuation; publisher remains scheduled
    queue._verify_artifacts(state)
    source = sqlite3.connect(f'file:{root / "warehouse-outbox.sqlite"}?mode=ro', uri=True)
    partial, target = root/'metrics-backup.partial', root/'metrics-backup.sqlite'
    destination = sqlite3.connect(partial)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    with partial.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(partial, target)
    save_status(close_path, dict(schema='raw_rl_close.v1',
        state='COMPLETED_AND_WAREHOUSE_READBACK_VERIFIED', contract_sha256=queue.digest(root/'CONTRACT.json'),
        completed_tasks=len(tasks), physical_copy=target.name, physical_sha256=queue.digest(target),
        warehouse=warehouse, scientific_scope='REAL_DATA_DEVELOPMENT', feature_selection_complete=False))
    return 0


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=('run', 'status', 'results', 'publish'))
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'publish':
        return publish_and_close(a.root)
    if a.action != 'run':
        print(json.dumps(read_status(a.root) if a.action == 'status' else results(a.root), indent=2, allow_nan=False))
        return 0
    try:
        code = run(a.root)
        state = queue.status(a.root)
        if state['state'] == 'ATTEMPTS_EXHAUSTED':
            return 78
        if code == 0:
            publish_and_close(a.root)
        return code
    except (ValueError, FileNotFoundError) as exc:
        save_status(a.root/'OPERATION_STATUS.json', dict(state='CONTRACT_OR_INTEGRITY_REFUSAL',
                    error_type=type(exc).__name__, reason=str(exc)[:300]))
        return 78


if __name__ == '__main__':
    raise SystemExit(main())
