"""Finite systemd ExecCondition: closed pinned predecessor and free resources."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from tools.curated_rl_campaign import digest, _verify_artifacts
from tools.verify_rl_static_engineering import save_status


def predecessor_ready(root, pin):
    root=Path(root)
    if not re.fullmatch('[0-9a-f]{64}',pin):
        raise ValueError('PREDECESSOR_PIN_REQUIRED')
    if not (root/'CLOSE.json').is_file():
        return 'WAIT_PREDECESSOR'
    if digest(root/'CONTRACT.json')!=pin:
        raise ValueError('PREDECESSOR_CONTRACT_CHANGED')
    body=json.loads((root/'CONTRACT.json').read_text())
    close=json.loads((root/'CLOSE.json').read_text())
    state=json.loads((root/'STATUS.json').read_text())
    expected=[(arm,seed,role) for seed in body['seeds'] for arm in body['arms']
              for role in ('train','validation')]
    box=close['warehouse']
    if (close['state']!='COMPLETED_AND_WAREHOUSE_READBACK_VERIFIED'
            or close['contract_sha256']!=pin or state['contract_sha256']!=pin
            or close['completed_tasks']!=len(expected)
            or [(t['arm'],t['seed'],t['split']) for t in state['tasks']]!=expected
            or any(t['state']!='ENQUEUED' for t in state['tasks'])
            or box['expected']!=len(expected) or box['delivered']!=len(expected)
            or box['pending'] or box['mismatch']
            or close['physical_copy']!='metrics-backup.sqlite'
            or digest(root/'metrics-backup.sqlite')!=close['physical_sha256']):
        raise ValueError('PREDECESSOR_NOT_AUTHENTICATED_CLOSED')
    _verify_artifacts(state)
    return 'READY'


def resources_ready(root, uuid):
    memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(memory['MemAvailable'].split()[0])*1024 < 6*1024**3:
        return 'WAIT_RAM'
    if shutil.disk_usage(root).free < 10*1024**3:
        return 'WAIT_DISK'
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    if any(line.split(',')[0].strip()==uuid for line in result.stdout.splitlines()):
        return 'WAIT_GPU_OWNER'
    return 'READY'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predecessor-root',type=Path,required=True)
    p.add_argument('--predecessor-sha256',required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    a=p.parse_args(argv)
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',a.gpu_uuid):
        return 78
    try:
        if (a.root/'CLOSE.json').is_file():
            return 1
        if (a.root/'STATUS.json').is_file():
            state=json.loads((a.root/'STATUS.json').read_text())
            if state.get('state')=='ATTEMPTS_EXHAUSTED':
                save_status(a.root/'DISPATCH_STATUS.json',dict(state='ATTEMPTS_EXHAUSTED'))
                return 1
        status=predecessor_ready(a.predecessor_root,a.predecessor_sha256)
        if status=='READY':
            status=resources_ready(a.root,a.gpu_uuid)
        save_status(a.root/'DISPATCH_STATUS.json',dict(state=status,
            predecessor_sha256=a.predecessor_sha256))
        return 0 if status=='READY' else 1
    except (OSError,ValueError,KeyError,subprocess.SubprocessError):
        save_status(a.root/'DISPATCH_STATUS.json',dict(state='PREDECESSOR_OR_RESOURCE_REFUSAL'))
        return 78


if __name__=='__main__':
    raise SystemExit(main())
