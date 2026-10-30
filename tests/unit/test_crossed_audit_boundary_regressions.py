"""Discriminating native witnesses for Satoshi CX-05 and CX-07, both sides."""
from dataclasses import replace
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from app.absolute_exit_corpus import create_case, specs, _funding
from app.close_hold_controller import value_paths
from app.close_hold_verification import build_report, position_view
from app.crossed_account_panel import CrossedWorld
from app.dynamic_account_panel import HOUR
from app.zigzag_opportunities import engine

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'docs/audits/evidence/crossed_account_final_20261029'


@pytest.fixture
def config():
    root = os.environ.get('STRATEGY_ENGINE_ROOT')
    if not root:
        pytest.skip('set STRATEGY_ENGINE_ROOT to the pinned native engine')
    engine(Path(root))
    return json.loads((ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())


@pytest.mark.parametrize('direction', [1, -1])
def test_deadline_spread_changes_exactly_at_flat_and_changes_native_cash(config, direction):
    e, risk = create_case(dict(next(specs()), direction=direction), config)
    p = e.pos
    start = e.next_decision+1
    for name in ('o', 'h', 'l', 'c'):
        values = getattr(e, name).copy()
        values[start:] = p.entry_price
        setattr(e, name, values)
    # spread_at uses elapsed hours since market open, not UTC wall-clock hour.
    flat = int(e.ff[p.seg])
    opening = int(e.mo[p.seg])
    hour = int((flat-opening)//HOUR % 24)
    assert hour == 12
    e.cfg = replace(e.cfg, profile=replace(e.cfg.profile,
        spread_schedule=((0, hour, .00005), (hour, 24, .0004))))
    r = build_report(e, 'ideal', 0)
    assert r.spreads[0, -1] == .00005
    assert r.terminal_spreads[0] == .0004
    times = r.first_fill+np.arange(r.paths.shape[1])*HOUR
    fo = np.array([_funding(e, p, int(t))/p.units for t in times])
    fe = np.array([_funding(e, p, int(t)+HOUR)/p.units for t in times])
    view = position_view(e, risk)
    correct = value_paths(view, r, fo, fe)[0, -1]
    legacy = value_paths(view, replace(r, terminal_spreads=r.spreads[:, -1]), fo, fe)[0, -1]
    assert legacy-correct == pytest.approx(p.units*(.0004-.00005)/2, abs=1e-8)
    while e.pos is not None:
        e.step(e.next_decision, 0)
    trade = e.trades[-1]
    assert trade['close_reason'] == 'FORCED_FLAT'
    assert trade['net_usd'] == pytest.approx(correct, abs=1e-8)
    assert abs(trade['net_usd']-legacy) > 1


@pytest.mark.parametrize('mirror', [1, -1])
def test_archived_account_close_request_is_preempted_by_native_gap(config, mirror):
    front = json.loads((EVIDENCE/'FRONTIER.json').read_text())['data']
    key = front['representatives'][0]
    genes = next(row['genes'] for row in front['rows'] if row['key'] == key)
    archive = json.loads(gzip.decompress((EVIDENCE/f'REPLAY_correlated_{key}.json.gz').read_bytes()))
    spec = dict(bank='confirmation', quality='correlated', seed=60100,
                pattern='trend', volatility='increasing', spread_pips=.5, mirror=mirror)
    expected = next(w for w in archive['worlds'] if w['spec'] == spec)
    actual = CrossedWorld(spec, config).run(genes, verify=True, details=True)
    # Dot-product rounding varies across CPU/BLAS; accounts and actions must not.
    assert 0 <= actual['cache_max_error'] <= 1e-10
    assert actual['cache_max_error'] == pytest.approx(expected['cache_max_error'],rel=0,abs=1e-12)
    assert {k:v for k,v in actual.items() if k!='cache_max_error'} == {
        k:v for k,v in expected.items() if k!='cache_max_error'}
    request = next(t for t in actual['trace'] if t['bar'] == 26)
    assert request['action'] == 3 and request['close'] == 'SL'
    trade = next(t for t in actual['ledger'] if t['exit_bar'] == 27)
    assert trade['direction'] == mirror and trade['reason'] == 'SL'
