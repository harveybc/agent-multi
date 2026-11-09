"""Resumable, sequential retained/reset AdamW critic-anchor contrast."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_discrete_curve_probe import read, write, now


def validate_pair(retained, reset):
    keys = ('source_identity','initial_sha256','replay_identity','return_identity',
            'snapshot_sha256','schedule_sha256','updates','batch_size','train_panel_rows',
            'replay_with_targets_digest','full_train_replay','sources',
            'optimizer_diagnostic_sha256','points','arms','device_name')
    if any(retained[k] != reset[k] for k in keys):
        raise ValueError('OPTIMIZER_PAIR_NOT_MATCHED')
    if retained['reset_optimizer'] or not reset['reset_optimizer']:
        raise ValueError('OPTIMIZER_PAIR_MODES')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-runtime','replay','calibration','anchor-control','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,action='append',required=True)
    parser.add_argument('--expected-device-name')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    config=dict(arguments={k:([str(p) for p in v] if isinstance(v,list) else str(v))
                           for k,v in vars(args).items()},
        sources={n:file_sha(root/n) for n in ('tools/run_optimizer_memory_contrast.py',
            'tools/audit_fixed_actor_critics.py','app/actor_learning_diagnostics.py')},
        order=['retained','reset'],updates=4096,batch_size=128)
    config['identity']=digest(config)
    with (args.output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'QUEUE_CONFIG.private.json').exists():
            if read(args.output/'QUEUE_CONFIG.private.json')!=config:
                raise ValueError('OPTIMIZER_QUEUE_CHANGED')
        else:write(args.output/'QUEUE_CONFIG.private.json',config)
        try:
            for arm in config['order']:
                destination=args.output/arm
                if not (destination/'DONE.json').exists():
                    command=[sys.executable,str(root/'tools/audit_fixed_actor_critics.py')]
                    for name in ('source','source_runtime','replay','calibration','anchor_control'):
                        command += ['--'+name.replace('_','-'),str(getattr(args,name))]
                    for path in args.snapshot:command+=['--snapshot',str(path)]
                    command += ['--output',str(destination),'--start-epoch','40',
                                '--full-train-replay','--return-anchor','--updates','4096']
                    if arm=='reset':command+=['--reset-optimizer']
                    if args.expected_device_name:
                        command+=['--expected-device-name',args.expected_device_name]
                    write(args.output/'PROGRESS.json',dict(stage='RUNNING',arm=arm,at=now()))
                    with (args.output/(arm+'.log')).open('a') as log:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=root)
                verify_collection(destination)
            configs=[read(args.output/a/'CONFIG.json') for a in config['order']]
            validate_pair(*configs)
            records={a:read(args.output/a/'return_anchor.json') for a in config['order']}
            for a,r in records.items():
                if not all(p['actor_exact'] and p['frozen_parameters_exact'] for p in r['points']):
                    raise ValueError('OPTIMIZER_FROZEN_FAILURE')
            write(args.output/'COMPARISON.json',dict(identity=config['identity'],matched=True,
                arms={a:dict(optimizer=r['optimizer_reset'],points=[dict(updates=p['updates'],
                    calibration=p['fixed_policy_train_panel'],alternate=p['alternate_policy_trajectories'])
                    for p in r['points']]) for a,r in records.items()},
                model_promoted=False,actor_unchanged=True))
            write(args.output/'QUEUE_CLOSE.json',dict(identity=config['identity'],finished_at=now(),
                comparison_sha256=file_sha(args.output/'COMPARISON.json'),
                units={a:file_sha(args.output/a/'DONE.json') for a in config['order']}))
            write(args.output/'PROGRESS.json',dict(stage='COMPLETE',at=now()))
        except Exception as exc:
            write(args.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()))
            raise


if __name__=='__main__':main()
