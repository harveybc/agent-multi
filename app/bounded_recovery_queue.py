"""Durable finite independent tasks; success requires verified artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.building')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def execute(config, output):
    output = Path(output)
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    ids = [t['id'] for t in config['tasks']]
    if len(ids) != len(set(ids)) or any(not x.replace('_','').isalnum() for x in ids):
        raise ValueError('QUEUE_TASK_IDS')
    for path, expected in config['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('QUEUE_SOURCE_CHANGED')
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'CONFIG.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('QUEUE_IDENTITY_CHANGED')
    save(manifest, config)
    outcomes = []
    for task in config['tasks']:
        folder = output/task['id']
        folder.mkdir(exist_ok=True)
        receipt = folder/'DONE.json'
        if receipt.exists():
            done = json.loads(receipt.read_text())
            if done['identity'] != identity or set(done['files']) != set(task['outputs']):
                raise ValueError('QUEUE_RECEIPT_IDENTITY')
            if any(sha(p) != h for p,h in done['files'].items()):
                raise ValueError('QUEUE_OUTPUT_CHANGED')
            outcomes.append(dict(id=task['id'], status='COMPLETE', resumed=True))
            continue
        started = time.time()
        save(output/'STATUS.json', dict(identity=identity, stage=task['id'], status='RUNNING',
                                        started_at=started, completed=outcomes))
        try:
            with (folder/'RUN.log').open('a') as log:
                proc = subprocess.run(task['argv'], cwd=task['cwd'], stdout=log,
                                      stderr=subprocess.STDOUT, timeout=task['timeout_seconds'], check=False)
            if proc.returncode:
                raise RuntimeError(f'PROCESS_EXIT_{proc.returncode}')
            files = {p:sha(p) for p in task['outputs']}
            if not files:
                raise ValueError('QUEUE_NO_ARTIFACTS')
            save(receipt, dict(identity=identity, files=files, finished_at=time.time()))
            outcomes.append(dict(id=task['id'], status='COMPLETE', seconds=time.time()-started))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            failure = dict(id=task['id'], status='FAILED', error=str(exc), seconds=time.time()-started)
            save(folder/'FAILED.json', failure)
            outcomes.append(failure)
    result = dict(identity=identity, status='COMPLETE' if all(x['status']=='COMPLETE' for x in outcomes)
                  else 'COMPLETE_WITH_FAILURES', tasks=outcomes, finished_at=time.time())
    save(output/'RESULT.json', result)
    save(output/'STATUS.json', result)
    return result
