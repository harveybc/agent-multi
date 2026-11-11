"""Reconstruct available pre-TRAIN feature history with frozen original producers.

This exports a candidate, not an admitted store. Calendar and supervised-activity
history require their own contracts; no synthetic substitutes are emitted.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, path, package=False):
    kwargs = dict(submodule_search_locations=[str(Path(path).parent)]) if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_columns(frame, columns):
    result = {}
    for name in columns:
        base, separator, op = name.partition('@')
        if base not in frame:
            continue
        x = frame[base].to_numpy(float)
        if separator:
            if op == 'diff':
                x = np.r_[np.nan, np.diff(x)]
            elif op == 'asinh':
                x = np.arcsinh(x)
            elif op == 'log':
                with np.errstate(divide='ignore', invalid='ignore'):
                    x = np.log(x)
                x[~np.isfinite(x)] = np.nan
            else:
                raise ValueError('WARMUP_UNKNOWN_FROZEN_TRANSFORM:'+op)
        result[name] = x
    return pd.DataFrame(result, index=frame.index)


def compare(actual, reference):
    out = {}
    for name in actual:
        a, b = actual[name].to_numpy(float), reference[name].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        out[name] = dict(mask_equal=bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
                         max_error=float(np.max(np.abs(a[both]-b[both]))) if both.any() else None,
                         equal=bool(np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)))
    return out


def _run(spec_path, output):
    spec = json.loads(spec_path.read_text())
    for path, expected in spec['pins'].items():
        if sha(path) != expected:
            raise ValueError('WARMUP_INPUT_DIGEST')
    binding = dict(spec_sha256=sha(spec_path), runtime_sha256=sha(__file__))
    binding_path = output/'BINDING.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError('WARMUP_RESUME_IDENTITY')
    else:
        if any(p.name != 'worker.lock' for p in output.iterdir()):
            raise ValueError('WARMUP_UNRECOGNIZED_OUTPUT')
        binding_path.write_text(json.dumps(binding, indent=2)+'\n')
    if (output/'DONE.json').exists():
        done = json.loads((output/'DONE.json').read_text())
        if done['binding'] != binding or any(sha(output/p) != h for p, h in done['files'].items()):
            raise ValueError('WARMUP_TERMINAL_DIGEST')
        return
    def emit(stage, **values):
        record = dict(stage=stage, **values)
        (output/'PROGRESS.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    # Load only the pinned package, not another environment's entire site-packages.
    sys.path.append(spec['compatibility_path'])
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan
    load_module('pandas_ta', spec['technical_package']+'/__init__.py', package=True)
    generator = load_module('warmup_technical', spec['technical_generator'])
    cross = load_module('warmup_cross', spec['cross_module'])
    sys.path.insert(0, spec['predictor_root'])
    from tools import layer1_pool_h1 as l1, layer2_pool_h1 as l2, timing_price_pool as pp
    recipe = json.loads(Path(spec['recipe']).read_text())
    scope = json.loads(Path(spec['scope']).read_text())
    support = json.loads(Path(spec['support_audit']).read_text())
    needed = [c for c in recipe['list']['columns']
              if support['invalid_windows_by_column'][c]['2020-01'] > 0]
    end = pd.Timestamp(spec['overlap_end'])
    if end.tzinfo is None or end > pd.Timestamp('2024-01-01', tz='UTC'):
        raise ValueError('WARMUP_TRAIN_ONLY')
    original = pd.read_parquet(spec['store'], filters=[('timestamp', '<', end)]).set_index('timestamp')
    native = pd.read_parquet(spec['eurusd'], filters=[
        ('timestamp', '>=', pd.Timestamp(spec['history_start'])), ('timestamp', '<', end)]).set_index('timestamp')
    bars = pd.read_parquet(spec['bars'], filters=[('timestamp', '<', end)]).set_index('timestamp')
    bars = bars[['BidOpen', 'BidHigh', 'BidLow', 'BidClose']].rename(columns={
        'BidOpen': 'OPEN', 'BidHigh': 'HIGH', 'BidLow': 'LOW', 'BidClose': 'CLOSE'})
    if not bars.index.equals(original.index):
        raise ValueError('WARMUP_ADMITTED_GRID')
    native = native[['open', 'high', 'low', 'close']].rename(columns=str.upper)
    matched = native.reindex(bars.index)
    if not np.array_equal(matched.to_numpy(), bars.to_numpy()):
        raise ValueError('WARMUP_PRICE_OVERLAP')
    history = native.loc[native.index < pd.Timestamp('2020-01-01', tz='UTC')]
    extended = pd.concat([history, bars])
    if (not extended.index.is_unique or not extended.index.is_monotonic_increasing
            or not np.isfinite(extended.to_numpy()).all()):
        raise ValueError('WARMUP_REAL_PRICE_GRID')
    factor = json.loads(Path(spec['factor_state']).read_text())
    for k in ('mu', 'sd', 'W'):
        factor[k] = np.asarray(factor[k], float)
    prices = {pair: pd.read_parquet(Path(spec['cross_prices'])/(pair+'.parquet'), filters=[
        ('timestamp', '>=', pd.Timestamp(spec['history_start'])), ('timestamp', '<', end)]
        ).set_index('timestamp').close for pair in factor['columns']}

    def build(frame):
        technical = generator.add_derived_features(frame.assign(DATE_TIME=frame.index+pd.Timedelta(hours=1)))
        technical = technical[['Stochastic_D', 'CCI', 'BB_position']]
        pframe = frame.rename(columns={'OPEN': 'BidOpen', 'HIGH': 'BidHigh',
                                       'LOW': 'BidLow', 'CLOSE': 'BidClose'})
        pframe.index.name = 'timestamp'
        parents = l2.parents(pframe.reset_index())
        per = {pair: cross.hourly_features(close.reindex(frame.index), 'fx.'+pair)
               for pair, close in prices.items()}
        returns = pd.DataFrame({p: per[p]['fx.'+p+'.ret_1'] for p in factor['columns']}, index=frame.index)
        f1 = cross.pca_apply(returns, factor).f1
        strength = pd.DataFrame({
            'usd_strength_ret': cross.composite(per, cross.USD_SIGN, 'usd_strength')['fx.usd_strength.ret_1'],
            'eur_strength_ret': cross.composite(per, cross.EUR_SIGN, 'eur_strength')['fx.eur_strength.ret_1']},
            index=frame.index)
        ssa = l1.causal_ssa(frame.CLOSE.to_numpy())
        pieces = [technical, pp.price_pool(frame, clock='utc'), l1.candle_pips(frame),
                  pd.DataFrame({'ssa.c2_pips': ssa[:, 1]/l1.PIP, 'ssa.c3_pips': ssa[:, 2]/l1.PIP,
                                'fac.fx_f1.ret_4': f1.rolling(4).sum(),
                                'fac.fx_f1.ret_24': f1.rolling(24).sum()}, index=frame.index),
                  l2.fam_wavelet(parents)[0], l2.fam_hilbert(parents)[0],
                  l2.apply_gmm(parents, scope['params']['gmm'])[0], l2.fam_strength(strength)[0]]
        return frozen_columns(pd.concat(pieces, axis=1), needed)

    emit('ORIGINAL_PRODUCER_REPLAY', required=len(needed))
    baseline = build(bars)
    parity = compare(baseline, original)
    (output/'BASELINE_PARITY.json').write_text(json.dumps(parity, indent=2)+'\n')
    if not all(x['equal'] and x['mask_equal'] for x in parity.values()):
        emit('BASELINE_MISMATCH', failures={k: v for k, v in parity.items() if not v['equal'] or not v['mask_equal']})
        raise ValueError('WARMUP_BASELINE_MISMATCH')
    emit('PREHISTORY_RECONSTRUCTION', pre_train_rows=len(history), supported_columns=len(baseline.columns))
    candidate = build(extended)
    cut = len(extended)-37
    prefix = build(extended.iloc[:cut])
    checks = compare(prefix, candidate.iloc[:cut])
    if not all(x['equal'] and x['mask_equal'] for x in checks.values()):
        raise ValueError('WARMUP_PREFIX_MISMATCH')
    candidate.index.name = 'timestamp'
    target = output/'WARMUP_CANDIDATE.parquet'
    candidate.reset_index().to_parquet(target, index=False)
    pre = candidate.loc[candidate.index < pd.Timestamp('2020-01-01', tz='UTC')]
    report = dict(schema='stl_price_warmup_candidate.v1', spec_sha256=sha(spec_path),
                  source_pins=spec['pins'], required_columns=needed, reconstructed_columns=list(candidate),
                  unresolved_columns=[c for c in needed if c not in candidate],
                  finite_pre_train={c: int(np.isfinite(pre[c]).sum()) for c in pre},
                  baseline_parity=parity, prefix_parity=checks, pre_train_rows=len(history),
                  admitted_calendar_extended=False, new_store_admitted=False,
                  source_store_unchanged=True, training_released=False,
                  no_imputation=True, supervised_activity_extended=False,
                  note='Real observed-row candidate; 2019 calendar admission and activity scope are separate requirements.')
    for path, expected in spec['pins'].items():
        if sha(path) != expected:
            raise ValueError('WARMUP_SOURCE_CHANGED')
    (output/'RESULT.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    done = dict(binding=binding, files={name: sha(output/name) for name in
                      ('RESULT.json', 'WARMUP_CANDIDATE.parquet', 'BASELINE_PARITY.json')})
    (output/'DONE.json').write_text(json.dumps(done, indent=2)+'\n')
    emit('CANDIDATE_COMPLETE', reconstructed=len(candidate.columns), unresolved=report['unresolved_columns'])


def run(spec_path, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(spec_path, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.spec, args.output)
