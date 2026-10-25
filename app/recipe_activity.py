"""Activity-constrained selection, independent of observation-year outcomes."""
import math


ENTROPY_GRID = {'ACT_ALPHA001': .01, 'ACT_ALPHA010': .1, 'ACT_ALPHA100': 1.}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def activity(report):
    trades, weeks = report.get('trades'), report.get('weeks_with_decisions')
    denominator_ok = finite(weeks) and weeks > 0 and weeks == int(weeks)
    count_ok = finite(trades) and trades >= 0 and trades == int(trades)
    rate = trades / weeks if denominator_ok and count_ok else None
    stated = report.get('orders_per_week')
    rate_ok = rate is not None and finite(stated) and math.isclose(rate, stated, rel_tol=1e-9, abs_tol=1e-12)
    cap = report.get('max_entries_per_day')
    eligible = bool(rate_ok and rate > 2 and report.get('valid') is True
        and report.get('ruined') is False and finite(cap) and 0 < cap <= 1
        and finite(report.get('mean_daily_log_return')) and finite(report.get('max_drawdown_pct')))
    return dict(activity_eligible=eligible, activity_orders_per_week=rate,
                activity_rule='closed trades / all scored weeks > 2; original daily cap and validity retained')


def choose(rows):
    grouped = {}
    for r in rows:
        if r.get('step') == 100000 and r.get('period') in ('TRAIN','INNER_A'):
            grouped.setdefault(r['arm'], []).append(r)
    eligible = []
    for arm, values in sorted(grouped.items()):
        if len(values) != 2 or {r['period'] for r in values} != {'TRAIN','INNER_A'}:
            continue
        if not all(activity(r)['activity_eligible'] for r in values):
            continue
        train = next(r for r in values if r['period']=='TRAIN')
        inner = next(r for r in values if r['period']=='INNER_A')
        a,b = train['mean_daily_log_return'],inner['mean_daily_log_return']
        eligible.append(dict(arm=arm,balanced_daily_log_return=(a+b)/2,
                             absolute_gap=abs(a-b),inner_drawdown_pct=inner['max_drawdown_pct']))
    eligible.sort(key=lambda r:(-r['balanced_daily_log_return'],r['absolute_gap'],r['inner_drawdown_pct'],r['arm']))
    return dict(state='ACTIVITY_QUALIFIED' if eligible else 'ACTIVITY_NOT_MET',
                selected=eligible[0]['arm'] if eligible else None, eligible=eligible,
                selection_periods=['TRAIN','INNER_A'], fixed_step=100000,
                scope='DEVELOPMENT_CALIBRATION_SINGLE_SEED_NOT_PROMOTION')
