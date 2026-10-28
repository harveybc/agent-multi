"""Isolated calls to predictor's existing metrics Outbox, never a second warehouse."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

PIN_FILES = ('tools/i6dmo/outbox.py', 'tools/i6n_archive/sink.py',
             'tools/i6n_swarm/common.py', 'olap/store/src/predictor_olap_store/query.py')


def pin_sources(root, commit, paths):
    root = Path(root).resolve()
    if not re.fullmatch('[0-9a-f]{40}', str(commit)):
        raise ValueError('RUNTIME_COMMIT_REQUIRED')
    result = subprocess.run(['git', '-C', str(root), 'archive', '--format=tar', commit, *paths],
                            capture_output=True, check=True, timeout=30)
    checked = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        for item in archive.getmembers():
            if not item.isfile() or not item.name.endswith('.py'):
                continue
            path = (root / item.name).resolve()
            if not path.is_relative_to(root):
                raise ValueError('RUNTIME_PATH_ESCAPE')
            expected = hashlib.sha256(archive.extractfile(item).read()).hexdigest()
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('RUNTIME_SOURCE_NOT_PINNED')
            checked[item.name] = expected
    if not checked:
        raise ValueError('RUNTIME_SOURCES_REQUIRED')
    return checked


def require_ext4(root):
    path = Path(root).resolve()
    while not path.exists():
        path = path.parent
    result = subprocess.run(['findmnt', '-n', '-o', 'FSTYPE', '--target', str(path)],
                            capture_output=True, text=True, check=True, timeout=10)
    if result.stdout.strip() != 'ext4':
        raise ValueError('WAREHOUSE_EXT4_ROOT_REQUIRED')


def prepare(publication, root):
    pin_sources(publication['predictor_root'], publication['predictor_code_commit'], PIN_FILES)
    require_ext4(root)


# A separate interpreter prevents predictor's shared `app`/`tools` namespaces
# from replacing agent-multi modules. Only the existing Outbox owns SQLite/HTTP.
CHILD = r'''
import json, sys
from pathlib import Path
root, action, request_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
request = json.loads(request_path.read_text())
from tools.i6dmo.outbox import Outbox, HttpWarehouse
box = Outbox(root / 'warehouse-outbox.sqlite')
try:
    if action == 'enqueue':
        digest = box.enqueue(request['report'], source_unit=request['source_unit'])
        result = {'report_sha256': digest, 'warehouse': box.summary()}
    else:
        result = {'warehouse': box.pump(HttpWarehouse(request['warehouse_url'], request['token_env']))}
    print(json.dumps(result, allow_nan=False))
finally:
    box.close()
'''


def invoke(publication, root, action, request_path):
    runtime = Path(publication['predictor_root']).resolve()
    pin_sources(runtime, publication['predictor_code_commit'], PIN_FILES)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'PYTHONPATH': os.pathsep.join((str(runtime), str(runtime / 'olap/store/src')))}
    result = subprocess.run([sys.executable, '-c', CHILD, str(Path(root).resolve()), action,
                             str(Path(request_path).resolve())], cwd=runtime, env=env,
                            capture_output=True, text=True, check=True,
                            timeout=publication.get('pump_timeout_seconds', 300) if action == 'pump' else 60)
    return json.loads(result.stdout.strip().splitlines()[-1])
