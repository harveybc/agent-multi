"""Execute an owner-configured finite recovery queue, protected by an exclusive lock."""
import argparse
import fcntl
import json
from pathlib import Path
from app.bounded_recovery_queue import execute


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        result = execute(json.loads(a.config.read_text()), a.output)
        print(json.dumps(result))
        raise SystemExit(0 if result['status']=='COMPLETE' else 1)
