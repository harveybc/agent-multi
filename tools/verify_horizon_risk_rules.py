#!/usr/bin/env python3
"""Sequential synthetic verification of horizon risk rules; no optimization."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.horizon_risk_rules import coefficients, propose
from app.protection_reference import protection_exit
from app.zigzag_opportunities import engine, sha
from tools.zigzag_opportunity_deap import write


def make_config(profiles, profile_sha):
    rows = []
    for row in profiles['rows']:
        for side, stats in row['directions'].items():
            p = stats['positive_calendar']
            rows.append(dict(horizon=row['horizon'], direction=side, zigzag=row['params'],
                             n=p['n'], reference={q: dict(tp_pips=p['tp_pips'][q],
                                                       mae_pips=p['mae_pips'][q])
                                                  for q in ('median', 'p75')}))
    return dict(schema='shared_horizon_risk_controls.v1', status='ENGINEERING_CONTROLS_NOT_OPTIMIZED',
                source_profile_sha256=profile_sha, source_run_identity=profiles['source_identity'],
                instrument='EURUSD', account_currency='USD', rel_volume=.1, leverage=50,
                quality=1., baseline=coefficients(), rows=rows,
                proposed_search_bounds=dict(a_sl=[.5, 2], a_tp=[.5, 2], b_sl=[0, 1], b_tp=[0, 1],
                                            buffer_pips=[0, 5], rho=[.001, .01]),
                baseline_reference='median', baseline_rho_status='simulation control only, not a selected risk budget',
                future_deap=dict(population=20, generations=300, status='after full mechanism qualification'),
                axes=dict(reference=['median', 'p75'], sigma_pips=[5, 15, 30],
                          spread_pips=[.5, 2, 5], rho=[.001, .005, .01],
                          a_sl=[.75, 1.25], a_tp=[.75, 1.25], buffer_pips=[.5, 2]),
                probe_fixed=dict(b_sl=.25, b_tp=.1, quality=1.),
                synthetic_execution=dict(balance=10000., tick=.00001, min_distance=.00005,
                    slippage_market=.00002, slippage_stop=.00005,
                    commission_per_lot_side=3.5, lot_units=100000., funding_reserve_per_unit=0.,
                    status='ASSUMED_SYNTHETIC_NOT_BROKER_VERIFIED'))


def protection_witnesses(X, order, liquidation):
    d, sl, tp = order['direction'], order['sl'], order['tp']
    span = abs(tp-sl)
    mid = (tp+sl)/2
    # Exit-side quotes. Required outcomes are fixed, not inferred from the engine.
    fixtures = dict(TP=(tp, tp+.1*span, tp-.1*span, 'TP'),
                    SL=(sl, sl+.1*span, sl-.1*span, 'SL'),
                    BOTH=(mid, max(tp, sl)+span, min(tp, sl)-span, 'SL'),
                    GAP=(sl-d*.5*span, sl-d*.5*span, sl-d*.5*span, 'SL'),
                    HOLD=(mid, mid+.1*span, mid-.1*span, None))
    for name, (o, h, l, wanted) in fixtures.items():
        native = X._protect(d, tp, sl, o, h, l, through=0.)
        ref = protection_exit(d, tp, sl, [o], [h], [l], [o], through=0.)
        if (native[0] if native else None) != wanted:
            raise AssertionError('NATIVE_EXPECTATION:'+name)
        if ref[2] != (wanted if wanted else 'DEADLINE'):
            raise AssertionError('REFERENCE_EXPECTATION:'+name)
        if native and not np.isclose(native[1], ref[1], rtol=0, atol=1e-12):
            raise AssertionError('NATIVE_REFERENCE_PRICE:'+name)
    # Widening spread beyond the level before execution is not an admissible entry.
    if not min(sl, tp) < liquidation < max(sl, tp):
        raise AssertionError('ENTRY_LIQUIDATION_BRACKET')
    return len(fixtures)


def grid_cells(config):
    axes = config['axes']; keys = sorted(axes)
    for values in itertools.product(*(axes[k] for k in keys)):
        yield dict(zip(keys, values))


def run(config, X):
    ex = config['synthetic_execution']
    sizing = X.Sizing(rel_volume=config['rel_volume'], leverage=config['leverage'], basis='margin')
    records = []; protections = 0
    for row in config['rows']:
        d = 1 if row['direction'] == 'buy' else -1
        for cell in grid_cells(config):
            ref = row['reference'][cell['reference']]
            g = coefficients(**{k: cell[k] for k in ('rho', 'a_sl', 'a_tp', 'buffer_pips')},
                             b_sl=config['probe_fixed']['b_sl'], b_tp=config['probe_fixed']['b_tp'])
            entry = 1.1; liquidation = entry-d*cell['spread_pips']*.0001
            cost = ex['slippage_market']+ex['slippage_stop']+2*ex['commission_per_lot_side']/ex['lot_units']
            order = propose(direction=d, entry_quote=entry, liquidation_quote=liquidation,
                balance=ex['balance'], tp_reference=ref['tp_pips']*.0001,
                mae_reference=ref['mae_pips']*.0001, sigma=cell['sigma_pips']*.0001,
                genes=g, sizing=sizing, tick=ex['tick'], min_distance=ex['min_distance'],
                quality=config['quality'], loss_cost_per_unit=cost,
                funding_reserve_per_unit=ex['funding_reserve_per_unit'])
            if order['status'] == 'ORDER':
                # Independent integer and monetary identities, not the sizing helper.
                maximum = int(np.floor(ex['balance']*.1*50/entry))
                expected_units = int(np.floor(order['q']*maximum))
                assert order['units'] == expected_units
                assert np.isclose(order['margin'], expected_units*entry/50)
                expected_loss = expected_units*(abs(entry-order['sl'])+cost+ex['funding_reserve_per_unit'])
                assert np.isclose(order['estimated_loss'], expected_loss)
                assert expected_loss <= g['rho']*ex['balance']+1e-9
                protections += protection_witnesses(X, order, liquidation)
            records.append(dict(horizon=row['horizon'], direction=row['direction'], cell=cell, order=order))
    summary = []
    for row in config['rows']:
        subset = [r['order'] for r in records if r['horizon'] == row['horizon'] and r['direction'] == row['direction']]
        orders = [r for r in subset if r['status'] == 'ORDER']
        summary.append(dict(horizon=row['horizon'], direction=row['direction'], proposals=len(subset),
            orders=len(orders), nop=len(subset)-len(orders),
            q_min=min((r['q'] for r in orders), default=None),
            q_max=max((r['q'] for r in orders), default=None),
            max_margin=max((r['margin'] for r in orders), default=None),
            max_risk_budget_fraction=max((r['estimated_loss']/r['risk_budget'] for r in orders), default=None)))
    return records, dict(status='PASS', proposals=len(records), protection_witnesses=protections,
        counts=dict(Counter(r['order']['status'] for r in records)), rows=summary,
        scope='SYNTHETIC_SIZING_AND_PROTECTION_PRIMITIVES_NOT_ACCOUNT_OR_EARLY_CLOSE_QUALIFICATION')


def refusal_controls(X):
    reports = []
    for d in (1, -1):
        sizing = X.Sizing(rel_volume=.1, leverage=50, basis='margin')
        args = dict(direction=d, entry_quote=1.1, liquidation_quote=1.1-d*.0001,
                    balance=10000., tp_reference=.003, mae_reference=.001, sigma=.001,
                    genes=coefficients(rho=.001), sizing=sizing, tick=.00001,
                    min_distance=.00005, quality=1., loss_cost_per_unit=.00014,
                    funding_reserve_per_unit=0.)
        fixtures = [
            ('zero_quality', dict(quality=0.), 'NOP'),
            ('zero_budget', dict(genes=coefficients(rho=0.)), 'NOP'),
            ('eroded_tp', dict(genes=coefficients(b_tp=4)), 'NOP'),
            ('spread_through_sl', dict(liquidation_quote=1.1-d*.01), 'NOP'),
            ('below_minimum', dict(sizing=X.Sizing(rel_volume=.1, leverage=50, min_units=100000)), 'NOP'),
            ('zero_mae_unbuffered', dict(mae_reference=0., genes=coefficients(buffer_pips=0.)), 'NOP'),
            ('zero_mae_buffered', dict(mae_reference=0., liquidation_quote=1.1-d*.00002), 'ORDER'),
        ]
        for name, changes, expected in fixtures:
            out = propose(**(args | changes))
            assert out['status'] == expected, name
            reports.append(dict(direction=d, case=name, expected=expected, actual=out['status']))
        baseline = propose(**args)
        funded = propose(**(args | dict(funding_reserve_per_unit=.0005)))
        assert funded['q'] < baseline['q']
        assert funded['estimated_loss'] <= funded['risk_budget']
        reports.append(dict(direction=d, case='funding_reduces_q', q_before=baseline['q'], q_after=funded['q']))
        gap_price = baseline['sl']-d*.01
        hit = X._protect(d, baseline['tp'], baseline['sl'], gap_price, gap_price, gap_price)
        gap_loss = baseline['units']*(d*(args['entry_quote']-hit[1])+args['loss_cost_per_unit'])
        assert hit[0] == 'SL' and gap_loss > baseline['risk_budget']
        reports.append(dict(direction=d, case='gap_exceeds_nominal_budget_as_expected',
                            nominal_loss=baseline['estimated_loss'], gap_loss=gap_loss,
                            budget=baseline['risk_budget']))
    return reports


def validate_config(config, references):
    for key in ('schema', 'source_profile_sha256', 'source_run_identity', 'rows',
                'instrument', 'account_currency', 'rel_volume', 'leverage', 'quality'):
        if config[key] != references[key]:
            raise ValueError('CONFIG_FIXED_CONTRACT:'+key)
    if set(config['axes']) != set(references['axes']):
        raise ValueError('CONFIG_AXES')
    if (set(config['probe_fixed']) != {'b_sl', 'b_tp', 'quality'}
            or config['probe_fixed']['quality'] != config['quality']):
        raise ValueError('CONFIG_PROBE_QUALITY_OR_KEYS')
    coefficients(b_sl=config['probe_fixed']['b_sl'], b_tp=config['probe_fixed']['b_tp'])
    for key, values in config['axes'].items():
        if not values or len({json.dumps(v) for v in values}) != len(values):
            raise ValueError('CONFIG_EMPTY_OR_DUPLICATE_AXIS')
        if key == 'reference':
            if not set(values) <= {'median', 'p75'}:
                raise ValueError('CONFIG_REFERENCE_QUANTILE')
        elif not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('CONFIG_NUMERICAL_AXIS')
    ex = config['synthetic_execution']
    if (not all(np.isfinite(v) for k, v in ex.items() if k != 'status')
            or min(ex[k] for k in ('balance', 'tick', 'lot_units')) <= 0
            or min(ex[k] for k in ('min_distance', 'slippage_market', 'slippage_stop',
                                  'commission_per_lot_side', 'funding_reserve_per_unit')) < 0):
        raise ValueError('CONFIG_SYNTHETIC_EXECUTION')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profiles', type=Path, required=True)
    p.add_argument('--profiles-sha', required=True)
    p.add_argument('--engine-root', type=Path, required=True)
    p.add_argument('--engine-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, help='Optional JSON sensitivity config; references and business constants must match')
    a = p.parse_args()
    source = a.engine_root/'app/short_horizon/execution.py'
    if a.output.exists():
        raise ValueError('NEW_OUTPUT_REQUIRED')
    if sha(a.profiles) != a.profiles_sha or sha(source) != a.engine_sha:
        raise ValueError('SOURCE_PIN')
    profiles = json.loads(a.profiles.read_text())
    if profiles['schema'] != 'zigzag_train_risk_profiles.v1' or profiles['train'] != ['2020-01-01', '2024-01-01']:
        raise ValueError('PROFILE_CONTRACT')
    references = make_config(profiles, a.profiles_sha)
    config = json.loads(a.config.read_text()) if a.config else references
    validate_config(config, references)
    _, _, X = engine(a.engine_root)
    records, report = run(config, X)
    report['additional_controls'] = refusal_controls(X)
    if sha(a.profiles) != a.profiles_sha or sha(source) != a.engine_sha:
        raise ValueError('SOURCE_CHANGED')
    a.output.mkdir(parents=True)
    write(a.output/'CONFIG.json', config)
    write(a.output/'CASES.json', records)
    report.update(config_sha256=sha(a.output/'CONFIG.json'), cases_sha256=sha(a.output/'CASES.json'),
                  engine_execution_sha256=a.engine_sha,
                  code_sha256={f: sha(ROOT/f) for f in ('app/horizon_risk_rules.py', 'tools/verify_horizon_risk_rules.py')})
    write(a.output/'REPORT.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
