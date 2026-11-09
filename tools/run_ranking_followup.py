"""Authenticated attribution then two bounded ranking replications and reports."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import time

from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from tools.audit_fixed_actor_critics import verify_collection
from tools.report_action_ranking_contrast import export, trading_rows
from tools.run_discrete_curve_probe import read, write, now

ROOT = Path(__file__).resolve().parents[1]
REPLICAS = ('r19', 'r43')


def paired_rows(results):
    rows = []
    for replica in REPLICAS:
        raw = trading_rows(results[replica])
        controls = {(r['actor_updates'], r['period']): r for r in raw if r['arm'] == 'control'}
        for row in raw:
            control = controls[(row['actor_updates'], row['period'])]
            rows.append(dict(replica=replica, **row,
                delta_net_usd=row['net_profit_usd'] - control['net_profit_usd'],
                delta_return_pp=row['return_pct'] - control['return_pct'],
                delta_sharpe=row['sharpe'] - control['sharpe'],
                delta_drawdown_pp=row['max_drawdown_pct'] - control['max_drawdown_pct']))
    return rows


def summarize(root):
    results = {}
    for replica in REPLICAS:
        receipt = verify_collection(root / replica / 'fit')
        value = read(root / replica / 'fit/RESULT.json')
        if value['identity'] != receipt['identity'] or value['replica'] != replica:
            raise ValueError('FOLLOWUP_RESULT_IDENTITY')
        results[replica] = value
    rows = paired_rows(results)
    with (root / 'PAIRED_TRADING.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summaries = []
    for arm in results['r19']['arms']:
        for period in ('TRAIN', 'VALIDATION'):
            selected = [r for r in rows if r['arm'] == arm and r['period'] == period and r['actor_updates'] == 32]
            if len(selected) != len(REPLICAS):
                raise ValueError('FOLLOWUP_ENDPOINT_COVERAGE')
            delta = [r['delta_return_pp'] for r in selected]
            summaries.append(dict(arm=arm, period=period, actor_updates=32,
                mean_paired_return_pp=sum(delta) / len(delta), min_paired_return_pp=min(delta),
                max_paired_return_pp=max(delta), all_eligible=all(r['eligible'] for r in selected)))
    write(root / 'SUMMARY.json', dict(fixed_endpoint=32, summaries=summaries,
        independent_market_replications=False, model_promoted=False))


def run(args):
    start = time.monotonic()
    if any(args.output.resolve() == getattr(args, n).resolve() for n in
           ('source', 'source_runtime', 'probe', 'attribution', 'calibration', 'ranking')):
        raise ValueError('FOLLOWUP_OUTPUT_IS_SOURCE')
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(replicas=list(REPLICAS), source=str(args.source), ranking=str(args.ranking),
        sources={n: file_sha(ROOT / n) for n in ('tools/run_ranking_followup.py',
            'tools/attribute_ranking_improvement.py', 'tools/run_action_ranking_contrast.py',
            'app/action_ranking_contrast.py', 'tools/report_action_ranking_contrast.py')})
    config['identity'] = digest(config)
    if (args.output / 'CONFIG.json').exists() and read(args.output / 'CONFIG.json') != config:
        raise ValueError('FOLLOWUP_CONFIG_CHANGED')
    write(args.output / 'CONFIG.json', config)
    if (args.output / 'DONE.json').exists():
        verify_collection(args.output); return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def progress(stage):
        write(args.output / 'PROGRESS.json', dict(stage=stage, at=now(), seconds=time.monotonic()-start))

    progress('ATTRIBUTION')
    out = args.output / 'attribution'
    command = ['env', f'PYTHONPATH={args.worker_runtime}', 'CUDA_VISIBLE_DEVICES=',
        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', args.worker_python, '-u', '-m',
        'tools.attribute_ranking_improvement']
    for name in ('source', 'source_runtime', 'probe', 'ranking'):
        command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    command.extend(['--output', str(out)])
    with (args.output / 'ATTRIBUTION.log').open('a') as log:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', args.worker_host,
            shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(['rsync', '-a', f'{args.worker_host}:{out}/', str(out) + '/'], check=True)
    verify_collection(out)
    for replica in REPLICAS:
        progress(replica)
        command = [args.worker_python, '-u', '-m', 'tools.run_action_ranking_contrast', 'queue']
        for name in ('source', 'source_runtime', 'probe', 'attribution', 'calibration',
                     'worker_host', 'worker_runtime', 'worker_python', 'gpu_id'):
            command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
        command.extend(['--output', str(args.output / replica), '--replica', replica])
        with (args.output / f'{replica}.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        export(args.output / replica, args.output / replica / 'reports')
    progress('COMBINED_REPORT')
    summarize(args.output)
    write(args.output / 'DONE.json', dict(identity=config['identity'], finished_at=now(),
        seconds=time.monotonic()-start, model_promoted=False,
        files={str(p.relative_to(args.output)): file_sha(p) for p in args.output.rglob('*')
            if p.is_file() and p.name not in ('PROGRESS.json', 'worker.lock', 'queue.lock')
            and p != args.output / 'DONE.json' and p.suffix != '.log'}))
    progress('COMPLETE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'source-runtime', 'probe', 'attribution', 'calibration', 'ranking', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('worker-host', 'worker-runtime', 'worker-python', 'gpu-id'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except Exception as exc:
            write(args.output / 'FAILED.json', dict(at=now(), error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
