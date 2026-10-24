"""Copy the finite recipe campaign and deliver its durable reports via the existing Outbox."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess

from app.frozen_recipe_inputs import file_sha, require
from tools.recipe_sac_queue import authenticated_result, now, read, write
from tools import rl_warehouse


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
        if p.name in delivered:
            require(delivered[p.name] == sha, 'REPORT_CHANGED')
            continue
        request = read(p)
        require(request['report']['config_sha256'] == identity, 'REPORT_IDENTITY')
        result = rl_warehouse.invoke(pub, state_root, 'enqueue', p)
        delivered[p.name] = sha
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
