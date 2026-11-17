"""Resume paired monitor runs after a pinned predecessor, without stopping workers."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from app.frozen_recipe_inputs import file_sha
from app.structural_refit import verify_files
from tools.run_discrete_curve_probe import read, write, now
from tools.run_sac_training_reliability import code_identity


def verify_completion(root, identity):
    result = read(root / 'RESULT.json')
    if result['identity'] != identity or result['status'] != 'COMPLETE':
        raise ValueError('QUEUE_RESULT_IDENTITY_OR_STATUS')
    verify_files(root, result['files'])
    return result


def validate_config(c):
    if c['schema'] != 'monitor_queue367.v1' or c['seed'] not in (7, 19):
        raise ValueError('QUEUE_SCHEMA_OR_SEED')
    if (len(c['policies']) != 2 or set(c['policies']) != {'balanced_min','champion_nonregression'}
            or type(c['poll_seconds']) is not int or c['poll_seconds'] < 10):
        raise ValueError('QUEUE_POLICIES_OR_POLL')
    root = Path(c['output']).resolve()
    if root == Path(c['predecessor']).resolve() or root == Path(c['bundle']).resolve():
        raise ValueError('QUEUE_OVERWRITES_SOURCE')


def admission_reason(c):
    service = subprocess.run(['systemctl','--user','is-active',c['predecessor_service']],
        capture_output=True, text=True, check=False)
    if service.stdout.strip() in ('active','activating','deactivating','reloading'):
        return 'PREDECESSOR_SERVICE_ACTIVE'
    result = subprocess.run(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-compute-apps=pid','--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True)
    if any(line.strip().isdigit() for line in result.stdout.splitlines()):
        return 'GPU_IN_USE'
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available < c['minimum_available_bytes']:
        return 'HOST_MEMORY_PRESSURE'
    return None


def run(a):
    c = read(a.config)
    validate_config(c)
    if code_identity() != c['code_identity']:
        raise ValueError('QUEUE_CODE_CHANGED')
    if file_sha(Path(c['bundle'])/'model.pt') != c['source_model_sha256']:
        raise ValueError('QUEUE_SOURCE_CHANGED')
    predecessor = Path(c['predecessor'])
    if read(predecessor/'SPEC.json')['identity'] != c['predecessor_identity']:
        raise ValueError('QUEUE_PREDECESSOR_CHANGED')
    output = Path(c['output'])
    output.mkdir(parents=True, exist_ok=True)
    config_sha = file_sha(a.config)
    def status(stage, **kw):
        write(output/'QUEUE_STATUS.json',dict(at=now(), stage=stage, seed=c['seed'],
            config_sha256=config_sha, **kw))
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (predecessor/'RESULT.json').exists():
            status('WAITING_PREDECESSOR', failure_present=(predecessor/'FAILURE.json').exists())
            time.sleep(c['poll_seconds'])
        verify_completion(predecessor, c['predecessor_identity'])
        for policy in c['policies']:
            target = output / policy
            if (target/'RESULT.json').exists():
                spec = read(target/'SPEC.json')
                if (spec.get('monitor_policy') != policy or spec['seed'] != c['seed']
                        or spec['code'] != c['code_identity'] or spec['source'] != c['source_model_sha256']):
                    raise ValueError('QUEUE_COMPLETED_UNIT_CONTRACT')
                verify_completion(target, spec['identity'])
                continue
            while True:
                reason = admission_reason(c)
                if reason is None:
                    break
                status('WAITING_RESOURCE', policy=policy, reason=reason)
                time.sleep(c['poll_seconds'])
            if code_identity() != c['code_identity']:
                raise ValueError('QUEUE_RUNTIME_MUTATED')
            command = [sys.executable, '-u', '-m', 'tools.run_branch_capacity_contrast',
                '--parent', c['parent'], '--bundle', c['bundle'], '--output', str(target),
                '--arm', 'control', '--seed', str(c['seed']), '--lr-policy', 'restore_best_halves',
                '--monitor-policy', policy]
            target.mkdir(parents=True, exist_ok=True)
            status('TRAINING', policy=policy, command=command)
            with (target/'WORKER.log').open('a') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            verify_completion(target, read(target/'SPEC.json')['identity'])
        status('COMPLETE', promoted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        c = read(args.config)
        root = Path(c['output'])
        root.mkdir(parents=True, exist_ok=True)
        write(root/'QUEUE_FAILURE.json',dict(at=now(), error=repr(exc)))
        raise
