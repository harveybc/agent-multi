"""Authenticated, read-only chronological SAC generalization panels."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from zoneinfo import ZoneInfo

from tools import paired_checkpoint_diagnostics as checkpoints

PROTOCOL = 'sac_generalization_panels.v1'


def daily_growth(path, initial_cash, timezone):
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError('BAD_INITIAL_CASH')
    closes, previous = {}, None
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            stamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            value = float(row['equity'])
            if (stamp.tzinfo is None or not math.isfinite(value) or value <= 0
                    or (previous is not None and stamp <= previous)):
                raise ValueError('BAD_EQUITY_TRACE')
            closes[stamp.astimezone(ZoneInfo(timezone)).date().isoformat()] = value
            previous = stamp
    if not closes:
        raise ValueError('EMPTY_TRACE')
    result, value = {}, initial_cash
    for day, closing in closes.items():
        result[day] = math.log(closing/value)
        value = closing
    return result


def panels(daily):
    groups = {}
    for day, growth in daily.items():
        quarter = day[:4]+'Q'+str((int(day[5:7])-1)//3+1)
        groups.setdefault(quarter, []).append(growth)
    return {q: dict(days=len(values), log_growth=statistics.mean(values),
                    return_pct=100*math.expm1(math.fsum(values)),
                    zero_return_days=sum(v == 0 for v in values))
            for q, values in groups.items()}


def block_difference(treatment, reference, block, repetitions=1000, seed=7):
    if list(treatment) != list(reference):
        raise ValueError('PAIRED_DAILY_SUPPORT_MISMATCH')
    values = [treatment[k]-reference[k] for k in treatment]
    if (not values or not 1 <= block <= len(values) or repetitions < 2
            or not all(math.isfinite(v) for v in values)):
        raise ValueError('INVALID_BOOTSTRAP')
    rng = random.Random(seed)
    # Moving contiguous blocks, with no circular jump across the year boundary.
    estimates = []
    for _ in range(repetitions):
        draw = []
        while len(draw) < len(values):
            start = rng.randrange(len(values)-block+1)
            draw.extend(values[start:start+block])
        estimates.append(math.fsum(draw[:len(values)])/len(values))
    estimates.sort()
    return dict(delta=statistics.mean(values), lower=estimates[int(.025*(repetitions-1))],
                upper=estimates[int(.975*(repetitions-1))], block_observed_days=block,
                repetitions=repetitions, seed=seed, days=len(values),
                scope='DESCRIPTIVE_DEVELOPMENT_AFTER_ADAPTIVE_SELECTION')


def analyze(sources):
    reference = sources['reference']
    if reference not in sources['campaigns']:
        raise ValueError('REFERENCE_REQUIRED')
    results, traces, publications = {}, {}, {}
    for name, root in sources['campaigns'].items():
        curve, body, templates = checkpoints.analyze(root)
        if not curve['campaign_closed']:
            raise ValueError('CLOSED_CAMPAIGN_REQUIRED')
        chosen = next(r for r in curve['rows'] if r['step'] == curve['selected_step'])
        paths = list(Path(root).glob(f'*/*/paired-monitor/step-{chosen["step"]:06d}/PAIR.json'))
        if len(paths) != 1:
            raise ValueError('EXACT_SELECTED_PAIR_REQUIRED')
        pair = checkpoints.read(paths[0]); roles, daily = {}, {}
        for task in pair['reports']:
            role = task['split']; cfg = checkpoints.read(task['config'])
            daily[role] = daily_growth(task['return_trace_file'], cfg['initial_cash'], cfg['entry_day_timezone'])
            if not math.isclose(statistics.mean(daily[role].values()), chosen[role+'_growth'], abs_tol=1e-14):
                raise ValueError('DAILY_GROWTH_REPLAY_MISMATCH')
            roles[role] = dict(quarters=panels(daily[role]), days=len(daily[role]),
                daily_log_growth=statistics.mean(daily[role].values()),
                selected_metrics={k: checkpoints.read(task['summary']).get(k) for k in (
                    'net_profit', 'total_return', 'max_drawdown_pct', 'sharpe_ratio',
                    'trades_closed', 'win_rate_pct', 'profit_factor')})
        traces[name] = daily
        results[name] = dict(contract_sha256=curve['contract_sha256'],
            source_pair_sha256=chosen['pair_sha256'], input_identities=curve['input_identities'],
            selected_step=curve['selected_step'], stop_reason=curve['final']['stop_reason'],
            complete_checkpoint_pairs=len(curve['rows']),
            positive_validation_checkpoints=curve['positive_validation_checkpoints'],
            objective_resets_worsening_validation=curve['resets_worse_than_incumbent'],
            daily_growth_gap=chosen['gap'], roles=roles,
            unavailable=dict(trade_pnl_concentration='NO_AUTHENTICATED_CLOSED_TRADE_PNL_VECTOR_IN_PAIR',
                             critic_td_gradient_replay='NOT_RETAINED_IN_HISTORICAL_TELEMETRY'))
        publications[name] = (body, templates[-1])
    ref = results[reference]
    for name, result in results.items():
        if result['input_identities'] != ref['input_identities']:
            raise ValueError('CROSS_ARM_INPUT_IDENTITY_MISMATCH')
        result['validation_vs_reference'] = {str(b): block_difference(
            traces[name]['validation'], traces[reference]['validation'], b) for b in (5,10,20)}
        result['validation_vs_no_trade'] = {str(b): block_difference(
            traces[name]['validation'], dict.fromkeys(traces[name]['validation'], 0.), b) for b in (5,10,20)}
    return dict(protocol=PROTOCOL, analyzer_sha256=checkpoints.digest(__file__),
        reference=reference, classification='DEVELOPMENT_DIAGNOSTIC_NOT_CAUSAL_ATTRIBUTION',
        experiments=results), publications


def publish(result, publications, output):
    from tools import rl_warehouse
    from tools.verify_rl_static_engineering import save_status
    import sqlite3
    for name, row in result['experiments'].items():
        body, template = publications[name]
        report = checkpoints.read(template)
        identity = hashlib.sha256(json.dumps(dict(protocol=PROTOCOL, name=name,
            analyzer=result['analyzer_sha256'], source=row['source_pair_sha256']), sort_keys=True).encode()).hexdigest()
        report.update(experiment_key='sac-panels:'+identity, phase='SAC_GENERALIZATION_PANELS', metrics=[])
        report['tags'].update(diagnostic_protocol=PROTOCOL, source_pair_sha256=row['source_pair_sha256'],
                             analyzer_sha256=result['analyzer_sha256'], profile=name,
                             interval_scope='DESCRIPTIVE_DEVELOPMENT')
        for role, detail in row['roles'].items():
            report['metrics'].append(dict(metric='daily_log_growth', value=detail['daily_log_growth'],
                split=role, horizon=None, unit='log_fraction_per_observed_day'))
            for quarter, panel in detail['quarters'].items():
                for key in ('log_growth', 'return_pct', 'days', 'zero_return_days'):
                    unit = 'percent' if key == 'return_pct' else ('days' if 'days' in key else 'log_fraction_per_observed_day')
                    report['metrics'].append(dict(metric=quarter+'.'+key, value=panel[key], split=role, horizon=None, unit=unit))
        for control in ('validation_vs_reference', 'validation_vs_no_trade'):
            for block, interval in row[control].items():
                for key in ('delta', 'lower', 'upper'):
                    report['metrics'].append(dict(metric=control+'.block'+block+'.'+key, value=interval[key],
                        split='validation', horizon=None, unit='log_fraction_per_observed_day'))
        publication = body['publication']; rl_warehouse.prepare(publication, output)
        request = output/(name+'-request.json')
        save_status(request, dict(report=report, source_unit=report['experiment_key']))
        rl_warehouse.invoke(publication, output, 'enqueue', request)
    request = output/'pump.json'
    save_status(request, {k: publication[k] for k in ('warehouse_url','token_env')})
    status = rl_warehouse.invoke(publication, output, 'pump', request)
    save_status(output/'WAREHOUSE_STATUS.json', status)
    if status['warehouse']['pending'] or status['warehouse']['mismatch']:
        raise ValueError('PANELS_PUBLICATION_PENDING')
    with sqlite3.connect(output/'warehouse-outbox.sqlite') as source:
        with sqlite3.connect(output/'warehouse-backup.sqlite') as target:
            source.backup(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--publish', action='store_true')
    a = p.parse_args(); sources = checkpoints.read(a.sources)
    if any(a.output.resolve().is_relative_to(Path(r).resolve()) or
           Path(r).resolve().is_relative_to(a.output.resolve()) for r in sources['campaigns'].values()):
        raise ValueError('DIAGNOSTIC_OUTPUT_MUST_NOT_OVERLAP_SOURCE')
    result, publications = analyze(sources)
    identity = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    output = a.output/identity
    checkpoints.write(output/'RESULT.json', result)
    if a.publish: publish(result, publications, output)
    print(json.dumps(dict(output=str(output), experiments=len(result['experiments']),
                         pairs=sum(v['complete_checkpoint_pairs'] for v in result['experiments'].values()))))


if __name__ == '__main__':
    main()
