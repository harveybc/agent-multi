"""Observed-price qualification and bounded recomputation of causal STL endpoints."""
import lzma
import struct

import numpy as np
import pandas as pd

from app.causal_stl_inputs import CausalSTLColumn


def hourly_bid_archive(payload, month, price_divisor):
    """Decode a Dukascopy monthly H1 archive; zero-volume rows are not observations.

    BI5 candles are big-endian (seconds, open, close, low, high, volume).
    The month and instrument's price divisor must come from the source contract.
    """
    start = pd.Timestamp(month)
    if (start.tzinfo is None or str(start.tzinfo) != 'UTC'
            or start != start.normalize().replace(day=1)
            or not np.isfinite(price_divisor) or price_divisor <= 0):
        raise ValueError('ARCHIVE_CONTRACT')
    raw = lzma.decompress(payload)
    if not raw or len(raw) % 24:
        raise ValueError('ARCHIVE_RECORD_SIZE')
    frame = pd.DataFrame(struct.iter_unpack('>IIIIIf', raw),
                         columns=['seconds', 'open', 'close', 'low', 'high', 'volume'])
    end = start + pd.offsets.MonthBegin(1)
    seconds = frame.pop('seconds')
    if ((seconds % 3600 != 0).any() or seconds.duplicated().any()
            or not seconds.is_monotonic_increasing
            or (seconds >= (end-start).total_seconds()).any()):
        raise ValueError('ARCHIVE_CLOCK')
    frame.index = pd.DatetimeIndex(start + pd.to_timedelta(seconds, unit='s'), name='timestamp')
    prices = ['open', 'high', 'low', 'close']
    frame[prices] = frame[prices] / price_divisor
    if (not np.isfinite(frame.to_numpy()).all() or (frame[prices] <= 0).any().any()
            or (frame.volume < 0).any()
            or (frame.high < frame[prices].max(axis=1)).any()
            or (frame.low > frame[prices].min(axis=1)).any()):
        raise ValueError('ARCHIVE_VALUES')
    frame['observed'] = frame.volume > 0
    return frame


def repair_endpoints(original, repaired, seasonal, parameters):
    """Keep the frozen estimator; recompute exactly its changed dependency support."""
    old, new, prior = (np.asarray(a, dtype=float) for a in (original, repaired, seasonal))
    if old.ndim != 1 or old.shape != new.shape or old.shape != prior.shape:
        raise ValueError('REPAIR_SHAPE')
    if np.isinf(old).any() or np.isinf(new).any() or np.isinf(prior).any():
        raise ValueError('REPAIR_INFINITY')
    CausalSTLColumn(parameters)  # Validate even when there are no changes.
    changed = ~((old == new) | (np.isnan(old) & np.isnan(new)))
    history = parameters['history']
    # Difference-array union of [changed_row, changed_row + history).
    bounds = np.zeros(len(old)+1, dtype=np.int64)
    starts = np.flatnonzero(changed)
    np.add.at(bounds, starts, 1)
    np.add.at(bounds, np.minimum(starts+history, len(old)), -1)
    affected = np.cumsum(bounds[:-1]) > 0
    result = prior.copy()
    for i in np.flatnonzero(affected):
        stream = CausalSTLColumn(parameters, new[max(0, i-history+1):i])
        result[i] = stream.update(new[i])
    return result, affected, changed
