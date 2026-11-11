"""Qualify historical candidates without mutating any admitted source."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

OHLC = ['open', 'high', 'low', 'close']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate_complete(frame, start, minutes):
    if minutes <= 0 or 60 % minutes:
        raise ValueError('INVALID_INTERVAL')
    if frame.index.has_duplicates:
        raise ValueError('DUPLICATE_OBSERVATION')
    end = start + pd.Timedelta(hours=1)
    rows = frame.loc[(frame.index >= start) & (frame.index < end), OHLC]
    expected = pd.date_range(start, periods=60 // minutes, freq=f'{minutes}min')
    if not rows.index.equals(expected) or not np.isfinite(rows.to_numpy()).all():
        raise ValueError('INCOMPLETE_REAL_OBSERVATIONS')
    if ((rows.high < rows[['open', 'close', 'low']].max(axis=1)) |
            (rows.low > rows[['open', 'close', 'high']].min(axis=1))).any():
        raise ValueError('INVALID_OHLC')
    return dict(open=float(rows.open.iloc[0]), high=float(rows.high.max()),
                low=float(rows.low.min()), close=float(rows.close.iloc[-1]))


def lake_frame(path):
    f = pd.read_parquet(path, filters=[('datetime', '<', pd.Timestamp('2025-01-01', tz='UTC'))])
    wall = pd.DatetimeIndex(f.pop('datetime')).tz_convert('UTC').tz_localize(None)
    f.index = (wall + pd.Timedelta(hours=5)).tz_localize(
        'Europe/London', ambiguous='NaT', nonexistent='NaT').tz_convert('UTC')
    f = f[f.index.notna()]
    if not f.index.is_unique or not f.index.is_monotonic_increasing:
        raise ValueError('INVALID_SOURCE_GRID')
    return f


def repair_interval(native, hourly, fine, window):
    """Replace only a declared interval after exact surrounding-source checks."""
    if (not len(window) or not window.equals(pd.date_range(window[0], window[-1], freq='h'))
            or any(not f.index.is_unique or not f.index.is_monotonic_increasing
                   for f in [native, hourly, *fine.values()])):
        raise ValueError('INVALID_REPAIR_GRID')
    if not native.index.union(window).equals(hourly.index):
        raise ValueError('REPAIR_GRID_MISMATCH')
    anchors = [window[0]-pd.Timedelta(hours=1), window[-1]+pd.Timedelta(hours=1)]
    if any(t not in native.index or t not in hourly.index for t in anchors):
        raise ValueError('BOUNDARY_ANCHORS_MISSING')
    outside = native.index.difference(window)
    if not np.array_equal(native.loc[outside, OHLC], hourly.loc[outside, OHLC]):
        raise ValueError('OUTSIDE_INTERVAL_DISAGREEMENT')
    if set(fine) != {5, 15}:
        raise ValueError('BOTH_FINE_RESOLUTIONS_REQUIRED')
    for stamp in window:
        actual = hourly.loc[stamp, OHLC].to_dict()
        for minutes, frame in fine.items():
            if aggregate_complete(frame, stamp, minutes) != actual:
                raise ValueError('FINE_SOURCE_OHLC_DISAGREEMENT')
    fixed = native[OHLC].reindex(hourly.index).copy()
    fixed.loc[window, OHLC] = hourly.loc[window, OHLC]
    if not np.array_equal(fixed, hourly[OHLC]):
        raise ValueError('REPAIR_PARITY')
    return fixed, dict(inserted_rows=len(window.difference(native.index)),
                       replaced_rows=len(window.intersection(native.index)),
                       outside_exact=True, outside_rows=len(outside),
                       fine_resolutions_minutes=[5, 15], no_imputation=True)


def qualify(lake, recovered, download_root, output, candidate_dir=None):
    lake, recovered, download_root = map(Path, (lake, recovered, download_root))
    stamp = pd.Timestamp('2023-11-29T13:00Z')
    november = {}
    destination = Path(candidate_dir) if candidate_dir else None
    if destination:
        if any(destination.resolve().is_relative_to(p.resolve())
               for p in [lake, recovered]):
            raise ValueError('OUTPUT_OVERLAPS_SOURCE')
        destination.mkdir(parents=True, exist_ok=False)
    for pair in ['gbpusd', 'usdchf', 'usdjpy', 'eurgbp', 'eurjpy', 'audusd']:
        paths = {tf: lake / pair / f'{tf}.parquet' for tf in ['1h', '15m', '5m']}
        native = pd.read_parquet(recovered / f'{pair}.parquet',
                                 filters=[('timestamp', '<', pd.Timestamp('2025-01-01', tz='UTC'))])
        native = native.set_index('timestamp')
        if not native.index.is_unique:
            raise ValueError('DUPLICATE_RECOVERED_GRID')
        frames = {tf: lake_frame(p) for tf, p in paths.items()}
        actual = {k: float(v) for k, v in frames['1h'].loc[stamp, OHLC].items()}
        derived = {tf: aggregate_complete(frames[tf], stamp, step)
                   for tf, step in [('5m', 5), ('15m', 15)]}
        exact = all(actual == row for row in derived.values())
        joins = []
        for offset in [-1, 1]:
            t = stamp + pd.Timedelta(hours=offset)
            joins.append(dict(timestamp=str(t), native_close=float(native.loc[t, 'close']),
                              lake_close=float(frames['1h'].loc[t, 'close'])))
        if stamp in native.index:
            raise ValueError('WOULD_OVERWRITE_EXISTING_BAR')
        month = native.loc['2023-11-01':'2023-11-30', OHLC]
        alternate = frames['1h'].loc['2023-11-01':'2023-11-30', OHLC]
        window = pd.date_range(stamp-pd.Timedelta(hours=1), periods=3, freq='h')
        repaired, interval = repair_interval(month, alternate,
                                              {5: frames['5m'], 15: frames['15m']}, window)
        interval['timestamps'] = list(map(str, window))
        interval['ohlc'] = repaired.loc[window].to_dict('records')
        if destination:
            # Candidate observations only. Original admission and metadata stay untouched.
            candidate = native[OHLC].reindex(native.index.union(window)).copy()
            candidate.loc[window, OHLC] = repaired.loc[window, OHLC]
            candidate.index.name = 'timestamp'
            target = destination / f'{pair}.parquet'
            candidate.reset_index().to_parquet(target, index=False)
            interval['candidate_sha256'] = sha(target)
            interval['candidate_rows'] = len(candidate)
        november[pair] = dict(source_sha256={k: sha(v) for k, v in paths.items()},
                              recovered_sha256=sha(recovered / f'{pair}.parquet'),
                              timestamp=str(stamp), actual=actual, aggregated=derived,
                              exact_ohlc_agreement=exact, adjacent_observations=joins,
                              interval_repair=interval,
                              no_imputation=True, source_splice_approved=False)
    p = download_root / 'fxcm/USDCHF_2022_20_m1.csv.gz'
    fx = pd.read_csv(p)
    fx.index = pd.to_datetime(fx.DateTime, format='%m/%d/%Y %H:%M:%S.%f', utc=True)
    missing = pd.Timestamp('2022-05-19T21:00Z')
    count = int(((fx.index >= missing) & (fx.index < missing + pd.Timedelta(hours=1))).sum())
    zpath = download_root / 'forexite_190522.zip'
    with zipfile.ZipFile(zpath) as archive:
        names = [n for n in archive.namelist() if n.endswith('.txt')]
        if len(names) != 1:
            raise ValueError('AMBIGUOUS_ARCHIVE')
        alt = pd.read_csv(archive.open(names[0]))
    alt = alt[alt['<TICKER>'] == 'USDCHF'].copy()
    native_time = pd.to_datetime(alt['<DTYYYYMMDD>'].astype(str) + alt['<TIME>'].astype(str).str.zfill(6),
                                 format='%Y%m%d%H%M%S')
    alt.index = pd.DatetimeIndex(native_time).tz_localize('Europe/Berlin').tz_convert('UTC')
    rows = alt.loc[(alt.index >= missing) & (alt.index < missing + pd.Timedelta(hours=1))]
    result = dict(schema='stl_recovery_source_qualification.v1', november=november,
                  usdchf=dict(timestamp=str(missing), fxcm_sha256=sha(p), fxcm_minutes=count,
                              forexite_sha256=sha(zpath), forexite_observations=len(rows),
                              forexite_unique_closes=int(rows['<CLOSE>'].nunique()),
                              quote_side='NOT_VERIFIED_FOR_DIRECT_ARCHIVE',
                              minute_label='START_OR_END_UNRESOLVED', admitted=False),
                  source_changes=0, no_imputation=True, training_released=False)
    out = Path(output)
    with out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(november_exact=sum(x['exact_ohlc_agreement'] for x in november.values()),
                         fxcm_missing_hour_minutes=count, forexite_observations=len(rows),
                         source_changes=0, training_released=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('lake', 'recovered', 'download-root', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--candidate-dir')
    qualify(**vars(p.parse_args()))
