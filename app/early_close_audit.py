"""Audit executed early closes against retained synthetic decision references.

This checks opportunity capture, not oracle optimality or learned performance.
"""
from collections import Counter
import math


def audit_episode(row):
    if row['mode'] not in ('ideal','position_long','position_short','position_optimal_long','position_optimal_short'):
        raise ValueError('IDEAL_ONLY')
    counts = Counter()
    failures = []
    trades = row['trades']
    for trade in trades:
        side = 'long' if trade['direction'] == 1 else 'short'
        counts['trades'] += 1
        counts[f'{side}.trades'] += 1
        counts['exit.' + trade['close_reason']] += 1
        if trade['close_reason'] == 'AGENT_CLOSE':
            counts['early_closes'] += 1
            counts[f'{side}.early_closes'] += 1
    accounted = set()
    for decision in row['decisions']:
        b = decision['bar']
        expected = decision['expected']
        cmd = decision['command']
        event = decision['event']
        if 'close_net' not in expected:
            if cmd in (3, 4):
                failures.append([b, 'CLOSE_WITHOUT_COMPARABLE_REFERENCE'])
            if expected.get('reason') == 'BROKER_OPEN_PROTECTION_PRIORITY':
                counts['broker_protection_before_execution'] += 1
            continue
        active = [(i, t) for i, t in enumerate(trades)
                  if t['entry_bar_abs'] <= b < t['exit_bar_abs']]
        if len(active) != 1:
            failures.append([b, 'POSITION_RECEIPT_MISMATCH'])
            continue
        index, trade = active[0]
        side = 'long' if trade['direction'] == 1 else 'short'
        counts[f'{side}.positioned_decisions'] += 1
        close, hold = expected['close_net'], expected['hold_net']
        if not all(math.isfinite(x) for x in (close, hold)):
            failures.append([b, 'NONFINITE_REFERENCE'])
            continue
        required = close >= hold if expected.get('tie_policy')=='EARLIEST_CLOSE' else close > hold
        if not required:
            counts[f'{side}.hold_preferred'] += 1
            if cmd != 0:
                failures.append([b, 'CLOSE_WHEN_HOLD_IS_BETTER_OR_EQUAL'])
            continue
        counts['opportunities'] += 1
        counts[f'{side}.opportunities'] += 1
        category = 'preserve_gain' if close > 0 else 'reduce_loss'
        counts[f'{side}.{category}'] += 1
        if expected['reference']['reason'] == 'SL':
            counts[f'{side}.avoid_future_sl'] += 1
        wanted = 3 if trade['direction'] == 1 else 4
        valid = (cmd == wanted and event.get('close') == 'AGENT_CLOSE'
                 and trade['close_reason'] == 'AGENT_CLOSE'
                 and trade['exit_bar_abs'] == b + 1)
        if not valid:
            failures.append([b, 'REQUIRED_EARLY_CLOSE_NOT_EXECUTED'])
            continue
        if not math.isclose(trade['net_usd'], close, rel_tol=1e-10, abs_tol=1e-7):
            failures.append([b, 'CLOSE_NET_REFERENCE_MISMATCH'])
            continue
        accounted.add(index)
        counts['captured'] += 1
        counts[f'{side}.captured'] += 1
    for i, trade in enumerate(trades):
        if trade['close_reason'] == 'AGENT_CLOSE' and i not in accounted:
            failures.append([trade['exit_bar_abs'], 'UNJUSTIFIED_EARLY_CLOSE_RECEIPT'])
    counts['cells'] = 1
    counts['cells_without_close_opportunity'] = int(not counts['opportunities'])
    counts['cells_without_trades'] = int(not counts['trades'])
    if row['mode'].startswith('position_'):
        counts['position_probe_cells'] = 1
        counts['position_probe_without_entry'] = int(not row['position_probe_entry'])
    return dict(counts=dict(counts), failures=failures,
                state='FAIL' if failures else 'CAPTURE_VERIFIED' if counts['opportunities']
                else 'NO_EARLY_CLOSE_WITNESS',
                financial_admission=False, learning_qualified=False)
