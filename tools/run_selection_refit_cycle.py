"""One authenticated deletion/reinsertion pass followed by a fresh structural fit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.run_generalization_queue import read, write, sha, remote, fetch_receipt


def validated_config(path):
    cfg=read(path);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    pin=out/'CONFIG_SHA.json';value={'sha256':sha(path)}
    if pin.exists() and read(pin)!=value:
        raise ValueError('CYCLE_CONFIG_CHANGED')
    write(pin,value)
    return cfg,out


def selection_receipt(root):
    root=Path(root);r=read(root/'RESULT.json');c=read(root/'CLOSE.json')
    if (sha(root/'RESULT.json')!=c['result_sha256'] or r['identity']!=c['identity']
            or c['termination']!='TWO_PASSES_COMPLETE'
            or r['result']['termination']!='TWO_PASSES_COMPLETE'
            or r['test_read'] or not r['original_support_preserved']
            or not r['unmasked_after_pass_exact']):
        raise ValueError('CYCLE_SELECTION_RECEIPT')
    return c


def run(config):
    cfg,out=validated_config(config)
    with (out/'cycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'COMPLETE.json').exists(): return
        def execute(module,*args):
            subprocess.run([sys.executable,'-m',module,*map(str,args)],check=True)
        def stage(name,**extra):
            write(out/'STATUS.json',dict(stage=name,updated_unix=time.time(),**extra))
        selection=Path(cfg['selection']);refit=Path(cfg['refit'])
        try:
            stage('SELECTION')
            if not (selection/'SPEC.private.json').exists():
                execute('tools.run_refit_mask_pass','prepare','--parent',cfg['parent'],
                        '--root',selection,'--seed',cfg['seed'],'--device','cuda','--cross-device-probe')
            if not (selection/'CLOSE.json').exists():
                execute('tools.run_refit_mask_pass','run','--root',selection)
            receipt=selection_receipt(selection)
            stage('PREPARE_REFIT',selection_receipt=receipt)
            if not (refit/'PREFLIGHT.json').exists():
                # Staging prevents interruption during preparation from leaving
                # a partially prepared root that looks runnable on restart.
                import shutil
                staging=refit.with_name(refit.name+'.preparing')
                if staging.exists(): shutil.rmtree(staging)
                execute('tools.run_structural_sac_refit','prepare-selected',
                        '--source-root',cfg['parent'],'--source-runtime',cfg['parent_runtime'],
                        '--selection',selection,'--selection-runtime',cfg['runtime'],
                        '--root',staging,'--seeds',cfg['seed'])
                staging.rename(refit)
            from tools.run_structural_sac_refit import authenticate
            spec=authenticate(refit)
            if spec['selection_identity']!=receipt['identity']:
                raise ValueError('CYCLE_REFIT_SELECTION_IDENTITY')
            stage('TRAINING',identity=spec['identity'],features=spec['structural_round']['features_after'])
            execute('tools.run_generalization_queue','task','--root',refit,'--seed',cfg['seed'])
            from tools.run_generalization_queue import collect
            training=collect(refit,cfg['seed'])
            write(out/'COMPLETE.json',dict(selection=receipt,training=training))
            stage('COMPLETE',identity=spec['identity'])
        except Exception as exc:
            stage('FAILED',error=repr(exc));raise


def collect_cycle(config):
    cfg,out=validated_config(config)
    with (out/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'COLLECTED.json').exists(): return
        while True:
            try:
                code=("import json,pathlib; p=pathlib.Path(%r); "
                      "print((p/'STATUS.json').read_text() if (p/'STATUS.json').exists() else '{}')") % cfg['remote_output']
                state=json.loads(remote(cfg['host'],[cfg['python'],'-c',code]))
                write(out/'REMOTE_STATUS.json',state)
                if state.get('stage')=='COMPLETE':
                    job=dict(key='selected-refit',host=cfg['host'],root=cfg['refit'],
                             seed=cfg['seed'],identity=state['identity'],runtime=cfg['runtime'],python=cfg['python'])
                    receipt=fetch_receipt(job,out)
                    dest=out/'selection';dest.mkdir(exist_ok=True)
                    subprocess.run(['rsync','-a',cfg['host']+':'+cfg['selection']+'/',str(dest)+'/'],check=True,timeout=600)
                    selected=selection_receipt(dest)
                    write(out/'COLLECTED.json',dict(selection=selected,training=receipt))
                    return
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired,OSError) as exc:
                write(out/'TRANSFER_ERROR.json',dict(error=repr(exc),at=time.time()))
            time.sleep(cfg.get('poll_seconds',60))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','collect']);p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();(run if args.command=='run' else collect_cycle)(args.config)


if __name__=='__main__': main()
