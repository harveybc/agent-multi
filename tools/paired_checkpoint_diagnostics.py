"""Read-only, full-support diagnostic of existing paired SAC checkpoints."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tempfile
import os
from zoneinfo import ZoneInfo

PROTOCOL = 'paired_checkpoint_diagnostics.v1'


def report_key(result, step):
    identity = ':'.join((result['contract_sha256'], str(step), result['analyzer_sha256']))
    return 'paired-diag:' + hashlib.sha256(identity.encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(body, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def trace_metrics(path, initial_cash, timezone):
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError('BAD_INITIAL_EQUITY')
    zone, days, count, previous = ZoneInfo(timezone), set(), 0, None
    support = hashlib.sha256()
    first, last, equity = None, None, None
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            stamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            equity = float(row['equity'])
            if (stamp.tzinfo is None or not math.isfinite(equity) or equity <= 0
                    or (previous is not None and stamp <= previous)):
                raise ValueError('BAD_TRACE')
            first = first or stamp.isoformat()
            last = stamp.isoformat()
            support.update((last + '\n').encode())
            days.add(stamp.astimezone(zone).date())
            previous, count = stamp, count + 1
    if not count:
        raise ValueError('EMPTY_TRACE')
    return dict(growth=math.log(equity / initial_cash) / len(days), rows=count,
                days=len(days), return_pct=100 * (equity / initial_cash - 1),
                support_sha256=support.hexdigest(), first=first, last=last)


def compare_curve(points, min_delta=1e-8, patience=10):
    if not points:
        raise ValueError('NO_COMPLETE_PAIRS')
    best, previous, used, rows = None, None, 0, []
    for p in points:
        if (not all(math.isfinite(p[k]) for k in ('train_growth', 'validation_growth'))
                or p['step'] <= 0 or (previous and p['step'] <= previous['step'])):
            raise ValueError('BAD_CURVE')
        score = -.5 * (p['train_growth'] + p['validation_growth'])
        improved = best is None or best['score'] - score > min_delta
        di = None if best is None else p['validation_growth'] - best['validation_growth']
        dp = None if previous is None else p['validation_growth'] - previous['validation_growth']
        row = dict(p, score=score, improved=improved,
                   gap=p['train_growth'] - p['validation_growth'],
                   previous_selected_step=best['step'] if best else None,
                   validation_delta_incumbent=di, validation_delta_previous=dp,
                   reset_worse_incumbent=bool(improved and di is not None and di < 0),
                   reset_worse_previous=bool(improved and dp is not None and dp < 0))
        used = 0 if improved else used + 1
        if improved:
            best = row
        row.update(best_step=best['step'], patience_used=used, stop=used >= patience)
        rows.append(row)
        previous = row
    return dict(rows=rows, selected_step=best['step'],
                resets_after_first=sum(r['improved'] for r in rows[1:]),
                resets_worse_than_incumbent=sum(r['reset_worse_incumbent'] for r in rows),
                resets_worse_than_previous=sum(r['reset_worse_previous'] for r in rows),
                positive_validation_checkpoints=sum(r['validation_growth'] > 0 for r in rows))


def compare_fixed_curve(points):
    expected=(25000,50000,75000,100000)
    if not points or tuple(p['step'] for p in points)!=expected[:len(points)]:
        raise ValueError('FIXED_BUDGET_CURVE_STEP_SEQUENCE')
    rows=[]
    for p in points:
        t,v=p['train_growth'],p['validation_growth']
        if not all(math.isfinite(x) for x in (t,v)):
            raise ValueError('FIXED_BUDGET_NONFINITE_CURVE')
        previous=rows[-1] if rows else None
        rows.append(dict(p,score=-.5*(t+v),gap=t-v,stop=False,
            selected_step=100000 if p['step']==100000 else None,
            checkpoint_selection='FINAL_100000',
            validation_delta_previous=None if previous is None else v-previous['validation_growth'],
            possible_overtraining_step=bool(previous and t>previous['train_growth'] and v<previous['validation_growth'])))
    return dict(rows=rows,selected_step=rows[-1]['selected_step'],resets_after_first=None,
        resets_worse_than_incumbent=None,resets_worse_than_previous=None,
        positive_validation_checkpoints=sum(r['validation_growth']>0 for r in rows),
        checkpoint_selection='FINAL_100000')


def analyze(root):
    root = Path(root).resolve()
    body = read(root / 'CONTRACT.json')
    settings = body['common_config']['paired_early_stopping']
    paths = sorted(root.glob('*/*/paired-monitor/step-*/PAIR.json'))
    if len({p.parent.parent for p in paths}) > 1:
        raise ValueError('MULTIPLE_MONITOR_ATTEMPTS_REQUIRE_EXPLICIT_DISPOSITION')
    points, inputs, support, decisions, templates = [], {}, {}, [], []
    for path in paths:
        record = read(path)
        point = dict(step=record['step'], pair_sha256=digest(path),
                     policy_sha256=record['policy_tensor_sha256'])
        if {t['split'] for t in record['reports']} != {'train', 'validation'} or len(record['reports']) != 2:
            raise ValueError('ROLE_MISMATCH')
        for task in record['reports']:
            role = task['split']
            for key in ('config', 'summary', 'effective_config', 'return_trace_file', 'checkpoint'):
                if digest(task[key]) != task['artifacts'][key]:
                    raise ValueError('ARTIFACT_HASH_MISMATCH:' + key)
            cfg, summary = read(task['config']), read(task['summary'])
            identity = summary['curated_input_identity']
            if (identity['manifest_sha256'] != body['splits'][role]['curated_input_manifest_sha256']
                    or identity['split'] != role):
                raise ValueError('INPUT_IDENTITY_MISMATCH')
            if cfg['mode'] != 'inference' or summary.get('evaluation_truncated_by_cap') is not False:
                raise ValueError('NOT_FULL_INFERENCE')
            static = summary['static_policy']
            if static['before'] != record['policy_tensor_sha256'] or static['after'] != static['before']:
                raise ValueError('POLICY_CHANGED')
            measured = trace_metrics(task['return_trace_file'], cfg['initial_cash'], cfg['entry_day_timezone'])
            expected = identity['row_count'] - ((cfg['window_size'] - 1) * cfg['observation_stride_bars'] + 1)
            stored = record['growth'][role]
            if (measured['rows'] != expected or measured['rows'] != stored['rows']
                    or measured['days'] != stored['days']
                    or not math.isclose(measured['growth'], stored['mean_daily_log_return'], rel_tol=0, abs_tol=1e-14)
                    or not math.isclose(measured['return_pct'] / 100, summary['total_return'], rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError('GROWTH_OR_SUPPORT_MISMATCH')
            if role in support and (support[role] != measured['support_sha256'] or inputs[role] != identity):
                raise ValueError('SUPPORT_CHANGED_BETWEEN_CHECKPOINTS')
            support[role], inputs[role] = measured['support_sha256'], identity
            point[role + '_growth'] = measured['growth']
            point[role + '_return_pct'] = measured['return_pct']
            point[role + '_support'] = measured
        templates.append(path.parent / 'validation' / 'warehouse_report.json')
        decisions.append(record['decision'])
        points.append(point)
    fixed=settings.get('early_stopping') is False
    result = compare_fixed_curve(points) if fixed else compare_curve(points, settings['min_delta'], settings['patience'])
    for row, decision in zip(result['rows'], decisions):
        for key in (('step','selected_step','checkpoint_selection','stop') if fixed else ('step', 'improved', 'best_step', 'patience_used', 'stop')):
            if row[key] != decision[key]:
                raise ValueError('DECISION_REPLAY_MISMATCH:' + key)
        if not math.isclose(row['score'], decision['score'], rel_tol=0, abs_tol=1e-14):
            raise ValueError('SCORE_REPLAY_MISMATCH')
    final_path = paths[0].parent.parent / 'FINAL.json'
    final = read(final_path) if final_path.exists() else None
    if final and (final['selected_step' if fixed else 'best_step'] != result['selected_step'] or final['evaluations'] != len(points)):
        raise ValueError('FINAL_REPLAY_MISMATCH')
    result.update(protocol=PROTOCOL, contract_sha256=digest(root / 'CONTRACT.json'),
                  analyzer_sha256=digest(__file__), input_identities=inputs,
                  full_support_sha256=support, settings=settings, final=final,
                  campaign_closed=(root / 'CLOSE.json').exists(),
                  interpretation='DEVELOPMENT_DIAGNOSTIC_NO_NEW_CHECKPOINT_SELECTION',
                  daily_unit='log_fraction_per_observed_New_York_calendar_date')
    return result, body, templates


def publish(result, body, templates, destination):
    from tools import rl_warehouse
    publication = body['publication']
    rl_warehouse.prepare(publication, destination)
    for row, template in zip(result['rows'], templates):
        report = read(template)
        report['experiment_key'] = report_key(result, row['step'])
        report['phase'] = 'PAIRED_CHECKPOINT_DIAGNOSTICS'
        report['tags'].update(diagnostic_protocol=PROTOCOL, analyzer_sha256=result['analyzer_sha256'],
                              pair_sha256=row['pair_sha256'], classification=result['interpretation'],
                              split='paired', diagnostic_step=str(row['step']))
        report['metrics'] = []
        for key in ('train_growth', 'validation_growth', 'gap', 'score', 'validation_delta_incumbent',
                    'validation_delta_previous', 'train_return_pct', 'validation_return_pct',
                    'reset_worse_incumbent', 'reset_worse_previous', 'improved', 'patience_used'):
            if key not in row:continue
            value = row[key]
            unit = 'percent' if key.endswith('_pct') else 'log_fraction_per_observed_day'
            if isinstance(value, bool):
                value, unit = int(value), 'flag'
            if key == 'patience_used':
                unit = 'evaluations'
            report['metrics'].append(dict(metric='diagnostic.' + key, value=value,
                split='train' if key.startswith('train_') else 'validation', horizon=None, unit=unit))
        request = destination / ('request-' + str(row['step']) + '.json')
        write(request, dict(report=report, source_unit=report['experiment_key']))
        rl_warehouse.invoke(publication, destination, 'enqueue', request)
    request = destination / 'pump.json'
    write(request, {k: publication[k] for k in ('warehouse_url', 'token_env')})
    status = rl_warehouse.invoke(publication, destination, 'pump', request)
    write(destination / 'WAREHOUSE_STATUS.json', status)
    if status['warehouse']['pending'] or status['warehouse']['mismatch']:
        raise ValueError('PUBLICATION_NOT_COMPLETE')
    # SQLite backup, rather than copying a live WAL database.
    import sqlite3
    with sqlite3.connect(destination / 'warehouse-outbox.sqlite') as source:
        with sqlite3.connect(destination / 'warehouse-backup.sqlite') as backup:
            source.backup(backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('OUTPUT_MUST_BE_OUTSIDE_SOURCE_CAMPAIGN')
    result, body, templates = analyze(args.root)
    identity = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    destination = args.output / identity
    write(destination / 'RESULT.json', result)
    if args.publish:
        publish(result, body, templates, destination)
    print(json.dumps(dict(output=str(destination), pairs=len(result['rows']),
        selected_step=result['selected_step'], campaign_closed=result['campaign_closed'],
        resets_after_first=result['resets_after_first'],
        resets_worse_than_incumbent=result['resets_worse_than_incumbent'],
        resets_worse_than_previous=result['resets_worse_than_previous'],
        positive_validation_checkpoints=result['positive_validation_checkpoints'])))


if __name__ == '__main__':
    main()
