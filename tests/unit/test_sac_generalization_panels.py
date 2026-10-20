import csv
import math
import json
from datetime import datetime, timedelta, timezone

import pytest


def test_daily_growth_first_day_zero_day_and_quarter_boundary(tmp_path):
    from tools.sac_generalization_panels import daily_growth, panels
    path = tmp_path/'returns.csv'
    with path.open('w') as stream:
        writer = csv.writer(stream); writer.writerow(['timestamp', 'equity'])
        writer.writerows([['2024-03-28T20:00:00+00:00', 110],
                          ['2024-03-29T20:00:00+00:00', 110],
                          ['2024-04-01T20:00:00+00:00', 121]])
    daily = daily_growth(path, 100, 'America/New_York')
    assert list(daily.values()) == pytest.approx([math.log(1.1), 0, math.log(1.1)])
    p = panels(daily)
    assert p['2024Q1']['days'] == 2
    assert p['2024Q1']['log_growth'] == pytest.approx(math.log(1.1)/2)
    assert p['2024Q2']['return_pct'] == pytest.approx(10)


def test_bootstrap_determinism_pairing_and_zero_control():
    from tools.sac_generalization_panels import block_difference
    a = {f'd{i}': .01 for i in range(40)}
    zero = {k: 0 for k in a}
    first = block_difference(a, zero, 5, 100, 7)
    assert first == block_difference(a, zero, 5, 100, 7)
    assert first['lower'] == pytest.approx(.01)
    assert first['upper'] == pytest.approx(.01)
    with pytest.raises(ValueError, match='SUPPORT'):
        block_difference(a, {'wrong': 0}, 5, 100, 7)


def test_nonfinite_and_nonchronological_trace_rejected(tmp_path):
    from tools.sac_generalization_panels import daily_growth
    for rows in [('2024-01-02T00:00:00Z,nan',),
                 ('2024-01-02T00:00:00Z,100', '2024-01-01T00:00:00Z,100')]:
        path = tmp_path/'bad.csv'; path.write_text('timestamp,equity\n'+'\n'.join(rows)+'\n')
        with pytest.raises(ValueError): daily_growth(path, 100, 'UTC')


def test_analyzer_composes_authenticated_curve_and_daily_panels(tmp_path, monkeypatch):
    from tools import sac_generalization_panels as module
    root = tmp_path/'campaign'; pair_root = root/'task/attempt-1/paired-monitor/step-005000'
    pair_root.mkdir(parents=True)
    trace = tmp_path/'trace.csv'
    with trace.open('w') as stream:
        writer = csv.writer(stream); writer.writerow(['timestamp','equity'])
        for i in range(40):
            writer.writerow([(datetime(2024,1,1, tzinfo=timezone.utc)+timedelta(days=i)).isoformat(),100])
    cfg = tmp_path/'cfg.json'; cfg.write_text(json.dumps(dict(initial_cash=100, entry_day_timezone='UTC')))
    summary = tmp_path/'summary.json'; summary.write_text('{}')
    (pair_root/'PAIR.json').write_text(json.dumps(dict(reports=[dict(split=role,
        config=str(cfg), summary=str(summary), return_trace_file=str(trace)) for role in ('train','validation')])))
    curve = dict(campaign_closed=True, selected_step=5000, contract_sha256='a'*64,
        input_identities={'identity':'same'}, positive_validation_checkpoints=0,
        resets_worse_than_incumbent=0, final=dict(stop_reason='PATIENCE'),
        rows=[dict(step=5000, gap=0, pair_sha256='b'*64, train_growth=0, validation_growth=0)])
    monkeypatch.setattr(module.checkpoints,'analyze', lambda _: (curve,{},[tmp_path/'template.json']))
    result, _ = module.analyze(dict(reference='base',campaigns={'base':str(root)}))
    row = result['experiments']['base']
    assert row['roles']['train']['daily_log_growth'] == 0
    assert row['roles']['validation']['days'] == 40
    assert row['validation_vs_no_trade']['5']['upper'] == 0
