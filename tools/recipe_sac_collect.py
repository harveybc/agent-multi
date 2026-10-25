"""Copy the finite recipe campaign and deliver its durable reports via the existing Outbox."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess

from app.frozen_recipe_inputs import file_sha, require
from tools.recipe_sac_queue import authenticated_result, now, read, write
from tools import rl_warehouse


def bind_report(mirror, request):
    """Authenticate the checkpoint and native score; retain retries as distinct attempts."""
    import copy
    request = copy.deepcopy(request)
    report = request['report']
    tags = report['tags']
    arm, step = tags['arm'], int(tags['checkpoint'])
    require(arm in ('GROUPED','FLAT','TARGET','COMMANDS5') and step in (25000,50000,75000,100000), 'REPORT_ARM_STEP')
    candidates = [p for p in (mirror / arm).glob(f'attempt-*/step-{step}/model.zip')
                  if file_sha(p) == tags['checkpoint_archive_sha256']]
    require(len(candidates) == 1, 'REPORT_CHECKPOINT_BINDING')
    checkpoint = candidates[0]
    splits = {m['split'] for m in report['metrics']}
    require(len(splits) == 1 and splits <= {'train','inner_a','obs_2024'}, 'REPORT_SPLIT')
    native = read(checkpoint.parent / (next(iter(splits)).upper() + '.json'))
    require(native['policy_tensor_sha256'] == tags['policy_tensor_sha256'], 'REPORT_POLICY')
    require(all(m['metric'] in native and native[m['metric']] == m['value'] for m in report['metrics']),
            'REPORT_NATIVE_VALUES')
    attempt = checkpoint.parent.parent.name
    tags['attempt'] = attempt
    tags['consumer_scope'] = 'DEVELOPMENT_CHECKPOINT_NOT_FINAL_PROMOTION'
    report['experiment_key'] += '-' + attempt
    request['source_unit'] += '-' + attempt
    return request


def collect(spec):
    state_root, mirror = Path(spec['state_root']), Path(spec['mirror'])
    state_root.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    lock = (state_root / 'COLLECT.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    subprocess.run(['rsync','-a','--checksum','--timeout=60','-e',
                    'ssh -o BatchMode=yes -o ConnectTimeout=15',
                    f'{spec["host"]}:{spec["remote_root"]}/',str(mirror)+'/'], check=True, timeout=240)
    identity = read(mirror / 'IDENTITY.json')['spec_sha256']
    require(identity == spec['campaign_sha256'] and file_sha(mirror / 'SPEC.private.json') == identity,
            'COLLECTION_IDENTITY')
    pub = spec['publication']
    rl_warehouse.prepare(pub, state_root)
    delivered = read(state_root / 'ENQUEUED.json') if (state_root / 'ENQUEUED.json').exists() else {}
    for p in sorted((mirror / 'outbox').glob('*.json')):
        sha = file_sha(p)
        version = p.name + '#' + sha
        if version in delivered:
            continue
        request = read(p)
        require(request['report']['config_sha256'] == identity, 'REPORT_IDENTITY')
        # A new attempt can regenerate a checkpoint after an interruption. Never
        # overwrite the old report or merge its metrics with the successful retry.
        archived = state_root / 'reports' / (sha + '.json')
        write(archived, bind_report(mirror, request))
        rl_warehouse.invoke(pub, state_root, 'enqueue', archived)
        delivered[version] = sha
        write(state_root / 'ENQUEUED.json', delivered)
    request = state_root / 'PUMP.private.json'
    write(request, {k:pub[k] for k in ('warehouse_url','token_env')})
    publication = rl_warehouse.invoke(pub, state_root, 'pump', request)
    verified = []
    for path in mirror.glob('*/attempt-*/RESULT.json'):
        authenticated_result(path, identity)
        verified.append(str(path.relative_to(mirror)))
    write(state_root / 'STATUS.json', dict(updated_at=now(), state='COPIED_AND_PUMPED',
          reports_enqueued=len(delivered), warehouse=publication,
          verified_completed_arms=verified, campaign_closed=(mirror / 'CLOSE.json').exists()))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    args = parser.parse_args()
    return collect(read(args.spec))


if __name__ == '__main__':
    raise SystemExit(main())
