import copy

import pytest

from tools.native_wrapper_audit import ledger_checks, local_artifact


def ledger():
    trade = dict(direction=1, units=100, percentual_order_size=0.2,
                 stop_loss=0.9, entry_price=1.0, take_profit=1.1,
                 entry_time='2023-08-01', exit_time='2023-08-02',
                 gross_usd=10.0, net_usd=6.0,
                 spread_usd=1.0, slippage_usd=1.0,
                 commission_usd=1.0, financing_usd=1.0)
    report = dict(trades=1, net_profit_usd=6.0,
                  cost_spread_usd=1.0, cost_slippage_usd=1.0,
                  cost_commission_usd=1.0, cost_financing_usd=1.0)
    return report, [trade]


def test_ledger_accepts_reconciled_orders():
    report, trades = ledger()
    assert ledger_checks(report, trades)['state'] == 'PASS'
    trades[0].update(direction=-1, stop_loss=1.1, take_profit=0.9)
    assert ledger_checks(report, trades)['state'] == 'PASS'


@pytest.mark.parametrize('field,value', [
    ('units', float('inf')), ('direction', 2), ('percentual_order_size', 0),
    ('stop_loss', 1.2), ('take_profit', float('inf')),
    ('net_usd', 7), ('gross_usd', 11), ('spread_usd', 2),
    ('exit_time', '2023-07-01'),
])
def test_ledger_rejects_mutated_orders(field, value):
    report, trades = ledger()
    trades[0][field] = value
    with pytest.raises((ValueError, RuntimeError)):
        ledger_checks(report, trades)


def test_ledger_rejects_mutated_report():
    report, trades = ledger()
    for field in ('trades', 'net_profit_usd', 'cost_financing_usd'):
        changed = copy.deepcopy(report)
        changed[field] += 1
        with pytest.raises((ValueError, RuntimeError)):
            ledger_checks(changed, trades)


def test_direction_cannot_fall_through_to_short():
    report, trades = ledger()
    trades[0].update(direction=2, stop_loss=1.1, take_profit=0.9)
    with pytest.raises((ValueError, RuntimeError), match='ORDER_DIRECTION'):
        ledger_checks(report, trades)


def test_mirror_mapping_rejects_escape(tmp_path):
    source = tmp_path / 'source'
    mirror = tmp_path / 'mirror'
    mirror.mkdir()
    assert local_artifact(source / 'a.json', source, mirror) == mirror / 'a.json'
    for path in (source / '../secret', tmp_path / 'other'):
        with pytest.raises((ValueError, RuntimeError)):
            local_artifact(path, source, mirror)
    (mirror / 'link').symlink_to(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        local_artifact(source / 'link/secret', source, mirror)
