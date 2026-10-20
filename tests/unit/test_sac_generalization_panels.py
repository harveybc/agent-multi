import csv
import math

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
