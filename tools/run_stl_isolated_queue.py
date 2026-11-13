"""Run each sealed STL variant in a fresh process, without changing its runtime."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess


def verify_complete(root, variant):
    folder = root/'training'/variant
    result = json.loads((folder/'RESULT.json').read_text())
    receipt = json.loads((folder/'DONE.json').read_text())
    if result['status']!='COMPLETE' or receipt['identity']!=result['identity'] or result['variant']!=variant:
        raise ValueError('STL_ISOLATED_COMPLETION')
    if 'RESULT.json' not in receipt['files']:raise ValueError('STL_ISOLATED_RESULT_UNSEALED')
    for name, expected in receipt['files'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()):raise ValueError('STL_ISOLATED_PATH')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                raise ValueError('STL_ISOLATED_HASH:'+name)


def run(a):
    for variant in a.after: verify_complete(a.root,variant)
    for variant in a.variants:
        if (a.root/'training'/variant/'DONE.json').exists():
            verify_complete(a.root,variant)
            print({'variant':variant,'status':'VERIFIED_ALREADY_COMPLETE'},flush=True)
            continue
        command=[str(a.python),str(a.runtime/'tools/run_available_stl_subset.py'),'queue',
                 '--root',str(a.root),'--parent',str(a.parent),'--bundle',str(a.bundle),
                 '--variants',variant,'--expected-device-name',a.expected_device_name]
        print({'variant':variant,'status':'START_FRESH_PROCESS'},flush=True)
        subprocess.run(command,cwd=a.runtime,check=True)
        verify_complete(a.root,variant)
    print({'variants':a.variants,'status':'COMPLETE'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('runtime','python','root','parent','bundle'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--expected-device-name',required=True)
    parser.add_argument('--variants',nargs='+',required=True,choices=('raw','continuous','ordinal','binary'))
    parser.add_argument('--after',nargs='*',default=[],choices=('raw','continuous','ordinal','binary'))
    args=parser.parse_args()
    with (args.root/('isolated_'+'_'.join(args.variants)+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
