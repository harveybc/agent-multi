"""Run one authorized regularization successor after a successful service exit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def wait_for_success(unit, status, run=subprocess.run, sleep=time.sleep):
    while True:
        proc = run(['systemctl','--user','show',unit,'-p','LoadState','-p','ActiveState',
                    '-p','Result','-p','ExecMainStatus'],check=True,capture_output=True,text=True)
        info = dict(line.split('=',1) for line in proc.stdout.splitlines() if '=' in line)
        if info.get('LoadState') != 'loaded':
            raise ValueError('PREDECESSOR_SERVICE_MISSING')
        if info.get('ActiveState') in ('active','activating','deactivating','reloading'):
            status('WAITING_FOR_PREDECESSOR')
            sleep(30)
            continue
        if (info.get('ActiveState') != 'inactive' or info.get('Result') != 'success'
                or info.get('ExecMainStatus') != '0'):
            raise ValueError('PREDECESSOR_SERVICE_FAILED')
        return


def execute(args):
    from tools.run_structural_sac_refit import authenticate, read, write
    from app.structural_refit import load_bundle
    root = args.root.resolve()
    status_path = root.with_suffix('.chain.json')
    def status(stage):
        write(status_path,dict(stage=stage,profile='core_first_batchnorm',
                              parent_identity=args.parent_identity))
        print(stage,flush=True)
    with root.with_suffix('.chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        wait_for_success(args.after_service,status)
        parent = authenticate(args.source_root,args.source_runtime)
        result = read(args.source_root/'runs/seed_7/RESULT.json')
        if parent['identity'] != args.parent_identity or result['identity'] != parent['identity']:
            raise ValueError('PREDECESSOR_RESULT_IDENTITY')
        load_bundle(args.source_root/'runs/seed_7'/result['best_bundle']['relative_path'],'cpu')
        status('PREDECESSOR_AUTHENTICATED')
        if not root.exists():
            staging = root.with_name(root.name+'.preparing')
            # Only our private, unpromoted preparation is disposable after interruption.
            if staging.exists(): shutil.rmtree(staging)
            subprocess.run([sys.executable,'-m','tools.run_structural_sac_refit',
                'prepare-regularized','--source-root',str(args.source_root),
                '--source-runtime',str(args.source_runtime),'--root',str(staging),
                '--profile','core_first_batchnorm'],cwd=ROOT,check=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            staging.rename(root)
        spec = authenticate(root)
        if (spec['effective']['critic_regularization'] != {'kind':'core_first_batchnorm'}
                or spec['structural_round']['parent_identity'] != parent['identity']):
            raise ValueError('SUCCESSOR_CONFIG_IDENTITY')
        status('INPUTS_AUTHENTICATED')
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not gpu.startswith('GPU-') or ',' in gpu:
            raise ValueError('ONE_EXPLICIT_GPU_REQUIRED')
        subprocess.run([sys.executable,'tools/gpu_readiness_probe.py','--gate','--uuid',gpu,
            '--python',sys.executable,'--output-fs',str(root),
            '--expected-artifact-bytes',str(4*1024**3)],cwd=ROOT,check=True)
        status('GPU_ACCEPTANCE')
        subprocess.run([sys.executable,'-m','pytest','-q',
            'tests/unit/test_critic_regularization.py',
            'tests/unit/test_causal_patch_attention.py'],cwd=ROOT,check=True)
        status('TRAINING')
        subprocess.run([sys.executable,'-m','tools.run_structural_sac_refit',
                        'worker','--root',str(root),'--seed','7'],cwd=ROOT,check=True)
        completed = read(root/'runs/seed_7/RESULT.json')
        if completed['identity'] != spec['identity']:
            raise ValueError('SUCCESSOR_RESULT_IDENTITY')
        status('COMPLETE')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-service',required=True)
    parser.add_argument('--parent-identity',required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--source-runtime',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
