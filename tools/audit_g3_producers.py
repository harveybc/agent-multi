"""Qualify current G3 dependencies and real TRAIN producers, without training."""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

from app.g3_producer_audit import (
    CROSSES, TECHNICAL, dependency_inventory, identity, parity, require_grid, support_report,
)
from tools.recover_stl_price_warmup import frozen_columns, load_module, sha


def write(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def public_receipt(output):
    """Small derived receipt: no private paths, market rows or fitted weights."""
    done = json.loads((output/'DONE.json').read_text())
    if any(sha(output/p) != h for p,h in done['files'].items()):
        raise ValueError('G3_TERMINAL_DIGEST')
    report = json.loads((output/'RESULT.json').read_text())
    baseline = json.loads((output/'BASELINE_PARITY.json').read_text())
    prefixes = json.loads((output/'PREFIX_PARITY.json').read_text())
    inventory = json.loads((output/'DEPENDENCIES.json').read_text())
    return dict(schema='g3_producer_public_receipt.v1',identity=report['identity'],
                status=report['status'],selected_columns=report['selected_columns'],
                replayed_columns=report['replayed_columns'],
                baseline_passed=report['baseline_passed'],baseline_failures=report['baseline_failures'],
                baseline_bit_exact=sum(v['exact'] for v in baseline.values()),
                max_absolute_error=max(v['max_error'] for v in baseline.values() if v['max_error'] is not None),
                prefix_checks={cut:dict(positive_control_changed=v['positive_control_changed'],
                    **{k:dict(passed=sum(x['passed'] for x in v[k].values()),
                              bit_exact=sum(x['exact'] for x in v[k].values()))
                       for k in ('prefix','suffix_perturbation')}) for cut,v in prefixes.items()},
                activity_predictors=inventory['activity_predictors'],
                primitive_dependencies=inventory['primitive_dependencies'],
                actual_learned_scope=report['actual_learned_scope'],
                train_coverage={k:v for k,v in report['train_coverage'].items() if k != 'by_column'},
                elapsed_seconds=report['elapsed_seconds'],receipt_files=done['files'],
                external_columns_not_replayed=report['external_columns_not_replayed'],
                synthetic_rows_generated=0,optimizer_steps=0,training_released=False,
                joint_generator_qualified=False,real_history_extended=False,
                scope='Numerical replay on original TRAIN under frozen states and legacy publication rules; not source/live certification')


def authenticate(spec):
    for path, expected in spec['pins'].items():
        if sha(path) != expected:
            raise ValueError('G3_INPUT_DIGEST:' + Path(path).name)


def producers(spec):
    warm = spec['producer_settings']
    sys.path.append(warm['compatibility_path'])
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan
    load_module('pandas_ta', warm['technical_package'] + '/__init__.py', package=True)
    technical = load_module('g3_technical', warm['technical_generator'])
    core = load_module('g3_core', spec['core_producer'])
    sys.path.insert(0, warm['predictor_root'])
    # Both repositories call their package "tools". Resolve the pinned producer
    # modules explicitly within this isolated audit process, not a cached sibling.
    import tools as tools_package
    tools_package.__path__ = [str(Path(warm['predictor_root'])/'tools'), *tools_package.__path__]
    modules = {}
    for name in ('layer1_pool_h1', 'layer2_pool_h1', 'timing_price_pool', 'timing_event_clock'):
        mod = importlib.import_module('tools.' + name)
        expected = Path(warm['predictor_root']) / 'tools' / (name + '.py')
        if Path(mod.__file__).resolve() != expected.resolve():
            raise ValueError('G3_PRODUCER_IMPORT:' + name)
        modules[name] = mod
    return technical, core, modules


def load_external(spec):
    cfg = spec['external']['config']
    end = pd.Timestamp(spec['train_end'])
    fx = {pair:pd.read_parquet(spec['external']['paths']['fx'][pair],
        filters=[('timestamp','<',end)]).set_index('timestamp').close for pair in CROSSES}
    daily = {}
    for name in ('us10y','dax','spx'):
        raw = pd.read_parquet(spec['external']['paths']['daily'][name])
        item = cfg['daily'][name]
        dates = pd.DatetimeIndex(pd.to_datetime(raw[item['date_column']])).tz_localize(None)
        values = pd.to_numeric(raw[item['value_column']],errors='coerce').to_numpy(float)
        series = pd.Series(values,index=dates).sort_index()
        series = series[series.index < end.tz_localize(None)]
        daily[name] = series[~series.index.duplicated(keep='last')]
    cot = pd.read_csv(spec['external']['paths']['cot'])
    cot = cot[pd.to_datetime(cot.Date) < end.tz_localize(None)].copy()
    factor = json.loads(Path(spec['producer_settings']['factor_state']).read_text())
    for name in ('mu','sd','W'):
        factor[name] = np.asarray(factor[name],float)
    activity = joblib.load(spec['external']['activity_state'])
    cross = load_module('g3_cross',spec['producer_settings']['cross_module'])
    return dict(fx=fx,daily=daily,cot=cot,factor=factor,activity=activity,cross=cross,cfg=cfg)


def external_features(frame, raw, pp, ec, l2, external):
    cross = external['cross']
    params = external['cfg']['params']
    per = {pair:cross.hourly_features(external['fx'][pair].reindex(frame.index),'fx.'+pair)
           for pair in CROSSES}
    returns = pd.DataFrame({pair:per[pair]['fx.'+pair+'.ret_1']
                            for pair in external['factor']['columns']},index=frame.index)
    f1 = cross.pca_apply(returns,external['factor']).f1
    factor = pd.DataFrame({'fac.fx_f1.ret_4':f1.rolling(4).sum(),
                           'fac.fx_f1.ret_24':f1.rolling(24).sum()},index=frame.index)
    strength = pd.DataFrame({
        'usd_strength_ret':cross.composite(per,cross.USD_SIGN,'usd_strength')['fx.usd_strength.ret_1'],
        'eur_strength_ret':cross.composite(per,cross.EUR_SIGN,'eur_strength')['fx.eur_strength.ret_1']},
        index=frame.index)
    daily = []
    for name,series in external['daily'].items():
        item = external['cfg']['daily'][name]
        features = cross.daily_features(series,name,item['kind'])
        available = cross.daily_available(features.index,params['daily_lag_days'],
                                          params['release_offset_hours'][name])
        daily.append(cross.asof_join(frame.index,features,available,
                                     pd.Timedelta(days=params['daily_max_stale_days'])))
    cot = cross.cot_features(external['cot'])
    cot = cross.asof_join(frame.index,cot,cross.cot_available(cot.index,
        params['release_offset_hours']['cot']),pd.Timedelta(days=params['cot_max_stale_days']))
    pool = pd.concat([pp,ec,raw[['dow_cos','dow_sin']],*daily],axis=1)
    bundle = external['activity']
    if not np.array_equal(frame.index.asi8,bundle['times_ns'][:len(frame)]):
        raise ValueError('G3_ACTIVITY_PREFIX_GRID')
    values = pd.DataFrame({name:pool[name.rsplit('@',1)[0]].shift(int(name.rsplit('@',1)[1]))
                           for name in bundle['features']},index=frame.index).to_numpy(float)
    p = bundle['estimators'][0].predict_proba(values)[:,1]
    for model,rows in zip(bundle['estimators'][1:],bundle['prediction_rows'],strict=True):
        rows = rows[rows < len(frame)]
        if len(rows):
            p[rows] = model.predict_proba(values[rows])[:,1]
    return [factor,l2.fam_strength(strength)[0],cot,*daily,
            pd.DataFrame({'act.p_hit':p},index=frame.index)]


def perturb_external_after(external, cutoff):
    future = dict(external,fx={k:v.copy() for k,v in external['fx'].items()},
                  daily={k:v.copy() for k,v in external['daily'].items()},cot=external['cot'].copy())
    for series in future['fx'].values():
        series.loc[series.index >= cutoff] *= 1.2
    for series in future['daily'].values():
        series.loc[series.index >= cutoff.tz_localize(None)] *= 1.2
    mask = pd.to_datetime(future['cot'].Date) >= cutoff.tz_localize(None)
    for c in ('EUR_Net_Long','EUR_Long','EUR_Short'):
        future['cot'][c] = future['cot'][c].astype(float)
        future['cot'].loc[mask,c] *= 1.2
    return future


def rebuild(frame, columns, state, closures, loaded, external=None):
    technical, core, modules = loaded
    l1, l2 = modules['layer1_pool_h1'], modules['layer2_pool_h1']
    price, event = modules['timing_price_pool'], modules['timing_event_clock']
    raw = core.candidate_frame(frame.assign(DATE_TIME=frame.index + pd.Timedelta(hours=1)),
                               technical.add_derived_features, (*TECHNICAL,'dow_cos','dow_sin')).drop(columns='DATE_TIME')
    parents_frame = frame.rename(columns={'OPEN':'BidOpen','HIGH':'BidHigh',
                                         'LOW':'BidLow','CLOSE':'BidClose'})
    parents_frame.index.name = 'timestamp'
    parents = l2.parents(parents_frame.reset_index())
    ssa = l1.causal_ssa(frame.CLOSE.to_numpy())
    pp = price.price_pool(frame, clock='utc', closures=closures)
    ec = event.event_clock(frame.index)
    pieces = [raw, pp, l1.candle_pips(frame), ec,
              pd.DataFrame({'ssa.c2_pips':ssa[:,1]/l1.PIP,'ssa.c3_pips':ssa[:,2]/l1.PIP},
                           index=frame.index), l2.fam_wavelet(parents)[0],
              l2.fam_hilbert(parents)[0], l2.apply_kalman(parents,state['params']['kalman'])[0],
              l2.apply_gmm(parents,state['params']['gmm'])[0]]
    if external is not None:
        pieces.extend(external_features(frame,raw,pp,ec,l2,external))
    result = frozen_columns(pd.concat(pieces, axis=1), columns)
    if list(result) != columns:
        raise ValueError('G3_REBUILD_COLUMN_COVERAGE')
    return result


def prepare(warmup_spec, recipe, activity_receipt, core_producer, closures, manifest,
            cross_config=None, financial_root=None, fx_root=None):
    warm = json.loads(Path(warmup_spec).read_text())
    pins = dict(warm['pins'])
    # Pin the complete producer Python tree, including imported local helpers.
    for path in (Path(warm['predictor_root'])/'tools').rglob('*.py'):
        pins[str(path)] = sha(path)
    for path in (warmup_spec,recipe,activity_receipt,core_producer,closures,manifest):
        pins[str(Path(path).resolve())] = sha(path)
    state_file = Path(activity_receipt).parent/'CONSUMER_ACTIVITY.joblib'
    receipt = json.loads(Path(activity_receipt).read_text())
    if sha(state_file) != receipt['weights_sha256']:
        raise ValueError('G3_ACTIVITY_STATE_DIGEST')
    pins[str(state_file)] = receipt['weights_sha256']
    external = None
    if any(x is not None for x in (cross_config,financial_root,fx_root)):
        if not all(x is not None for x in (cross_config,financial_root,fx_root)):
            raise ValueError('G3_EXTERNAL_ARGUMENTS')
        cfg = json.loads(Path(cross_config).read_text())
        pins[str(Path(cross_config).resolve())] = sha(cross_config)
        paths = dict(fx={},daily={})
        for group,names,root in (('fx',CROSSES,fx_root),('daily',('us10y','dax','spx'),financial_root)):
            for name in names:
                path = Path(root)/cfg[group][name]['file']
                pins[str(path)] = cfg[group][name]['sha256']
                paths[group][name] = str(path)
        path = Path(financial_root)/cfg['cot']['file']
        pins[str(path)] = cfg['cot']['sha256']
        paths['cot'] = str(path)
        external = dict(config=cfg,paths=paths,activity_state=str(state_file))
    return dict(schema='g3_producer_audit_spec.v1', producer_settings=warm,external=external,
                recipe=str(Path(recipe).resolve()), activity_receipt=str(Path(activity_receipt).resolve()),
                core_producer=str(Path(core_producer).resolve()), closures=str(Path(closures).resolve()),
                manifest=str(Path(manifest).resolve()), pins=pins,
                train_start='2020-01-01T00:00:00Z', train_end='2024-01-01T00:00:00Z',
                prefix_lengths=[1024,8192], parity_tolerance=1e-9)


def run(spec, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return locked_run(spec,output)


def locked_run(spec, output):
    started = time.monotonic()
    authenticate(spec)
    binding = dict(spec_sha256=identity(spec), code={str(p):sha(p) for p in (
        Path(__file__).resolve(), Path(__file__).resolve().parents[1]/'app/g3_producer_audit.py',
        Path(__file__).resolve().parent/'recover_stl_price_warmup.py')})
    binding_path = output/'BINDING.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError('G3_RESUME_IDENTITY')
        if (output/'DONE.json').exists():
            done = json.loads((output/'DONE.json').read_text())
            if done['binding'] != binding or any(sha(output/p) != h for p,h in done['files'].items()):
                raise ValueError('G3_TERMINAL_DIGEST')
            return json.loads((output/'RESULT.json').read_text())
    else:
        if any(p.name != 'worker.lock' for p in output.iterdir()):
            raise ValueError('G3_UNRECOGNIZED_OUTPUT')
        write(binding_path,binding)
        write(output/'SPEC.json',spec)

    def emit(stage, **kw):
        value = dict(stage=stage,elapsed_seconds=time.monotonic()-started,**kw)
        write(output/'PROGRESS.json',value)
        print(json.dumps(value),flush=True)

    warm = spec['producer_settings']
    recipe = json.loads(Path(spec['recipe']).read_text())
    manifest = json.loads(Path(spec['manifest']).read_text())
    state = json.loads(Path(warm['scope']).read_text())
    activity = json.loads(Path(spec['activity_receipt']).read_text())
    if (recipe['store']['sha256'] != sha(warm['store']) or
            manifest['file_sha256'] != sha(warm['store']) or
            recipe['store']['manifest_sha256'] != sha(spec['manifest'])):
        raise ValueError('G3_RECIPE_STORE_BINDING')
    if (manifest['consumer_scope']['report_sha256'] != sha(warm['scope']) or
            activity['source_sha256']['store'] != sha(warm['store']) or
            activity['source_sha256']['scope'] != sha(warm['scope'])):
        raise ValueError('G3_FIT_SCOPE_BINDING')
    start,end = pd.Timestamp(spec['train_start']),pd.Timestamp(spec['train_end'])
    if (start.tzinfo is None or end.tzinfo is None or not start < end or
            start != pd.Timestamp('2020-01-01',tz='UTC') or
            end != pd.Timestamp('2024-01-01',tz='UTC')):
        raise ValueError('G3_TRAIN_CONTRACT')
    columns = recipe['list']['columns']
    inventory = dependency_inventory(columns,activity['features'])
    write(output/'DEPENDENCIES.json',inventory)
    filters = [('timestamp','>=',start),('timestamp','<',end)]
    store = pd.read_parquet(warm['store'],columns=['timestamp']+columns,filters=filters).set_index('timestamp')
    bars = pd.read_parquet(warm['bars'],filters=filters).set_index('timestamp')
    require_grid(store)
    require_grid(bars)
    if not bars.index.equals(store.index):
        raise ValueError('G3_PRICE_STORE_GRID')
    coverage = support_report(store,columns,recipe['tensor']['window']['slot_offsets_rows'])
    write(output/'COVERAGE.json',coverage)
    price_columns = columns if spec.get('external') else [row['column'] for row in
                    inventory['columns'] if row['price_calendar_replay']]
    frame = bars[['BidOpen','BidHigh','BidLow','BidClose']].rename(columns={
        'BidOpen':'OPEN','BidHigh':'HIGH','BidLow':'LOW','BidClose':'CLOSE'})
    if (not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any() or
            (frame.HIGH < frame[['OPEN','CLOSE','LOW']].max(axis=1)).any() or
            (frame.LOW > frame[['OPEN','CLOSE','HIGH']].min(axis=1)).any()):
        raise ValueError('G3_INVALID_OHLC')
    closures = json.loads(Path(spec['closures']).read_text())
    loaded = producers(spec)
    external_data = load_external(spec) if spec.get('external') else None
    emit('REAL_TRAIN_PRODUCER_REPLAY',columns=len(price_columns),rows=len(frame))
    rebuilt = rebuild(frame,price_columns,state,closures,loaded,external_data)
    comparison = parity(rebuilt,store[price_columns],tolerance=spec['parity_tolerance'])
    write(output/'BASELINE_PARITY.json',comparison)
    prefix_checks = {}
    for cut in spec['prefix_lengths']:
        if not 512 <= cut < len(frame):
            raise ValueError('G3_PREFIX_LENGTH')
        emit('PREFIX_AND_SUFFIX_PERTURBATION',cut=cut)
        prefix = rebuild(frame.iloc[:cut].copy(),price_columns,state,closures,loaded,external_data)
        checks = parity(prefix,rebuilt.iloc[:cut],tolerance=spec['parity_tolerance'])
        # A positive common scaling preserves OHLC algebra; it changes every
        # suffix price without modifying the prefix or frozen producer state.
        changed = frame.iloc[:cut+73].copy()
        changed.iloc[cut:] *= 1.2
        future = external_data
        if external_data is not None:
            future = perturb_external_after(external_data,frame.index[cut])
        perturbed = rebuild(changed,price_columns,state,closures,loaded,future)
        suffix = parity(prefix,perturbed.iloc[:cut],tolerance=spec['parity_tolerance'])
        changed_after = not np.allclose(perturbed.iloc[cut:].to_numpy(),
                                        rebuilt.iloc[cut:cut+73].to_numpy(),equal_nan=True)
        prefix_checks[str(cut)] = dict(prefix=checks,suffix_perturbation=suffix,
                                      positive_control_changed=bool(changed_after))
    write(output/'PREFIX_PARITY.json',prefix_checks)
    external = [r['column'] for r in inventory['columns'] if r['column'] not in price_columns]
    passed = all(r['passed'] for r in comparison.values()) and all(
        v['positive_control_changed'] and all(r['passed'] for k in ('prefix','suffix_perturbation')
                                            for r in v[k].values()) for v in prefix_checks.values())
    scope = 'ALL_PRODUCERS' if external_data is not None else 'PRICE_CALENDAR'
    report = dict(schema='g3_producer_audit.v1',identity=identity(binding),
                  status=scope+'_QUALIFIED' if passed else scope+'_REPLAY_MISMATCH',
                  selected_columns=len(columns),replayed_columns=len(price_columns),
                  baseline_passed=sum(r['passed'] for r in comparison.values()),
                  baseline_failures=[c for c,r in comparison.items() if not r['passed']],
                  external_columns_not_replayed=external,train_coverage=coverage,
                  actual_learned_scope=state['scope'],activity_inputs=activity['features'],
                  synthetic_rows_generated=0,optimizer_steps=0,training_released=False,
                  joint_generator_qualified=False,real_history_extended=False,stl_resumed=False,
                  source_store_changed=False,elapsed_seconds=time.monotonic()-started)
    authenticate(spec)
    write(output/'RESULT.json',report)
    write(output/'DONE.json',dict(binding=binding,files={p:sha(output/p) for p in (
        'SPEC.json','DEPENDENCIES.json','COVERAGE.json','BASELINE_PARITY.json','PREFIX_PARITY.json','RESULT.json')}))
    emit(report['status'],baseline_passed=report['baseline_passed'],external_remaining=len(external))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup-spec',required=True)
    parser.add_argument('--recipe',required=True)
    parser.add_argument('--activity-receipt',required=True)
    parser.add_argument('--core-producer',required=True)
    parser.add_argument('--closures',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--cross-config')
    parser.add_argument('--financial-root')
    parser.add_argument('--fx-root')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--summary-output',type=Path)
    args = vars(parser.parse_args())
    output = args.pop('output')
    summary_output = args.pop('summary_output')
    result = run(prepare(**args),output)
    if summary_output is not None:
        if summary_output.exists():
            raise ValueError('G3_SUMMARY_ALREADY_EXISTS')
        summary_output.parent.mkdir(parents=True,exist_ok=True)
        write(summary_output,public_receipt(output))
    print(json.dumps({k:result[k] for k in ('status','selected_columns','replayed_columns',
                                          'baseline_failures','training_released')}))
    return 0 if result['status'].endswith('_QUALIFIED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
