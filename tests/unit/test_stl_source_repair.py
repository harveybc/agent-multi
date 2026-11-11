import lzma
import struct

import numpy as np
import pytest

from app.causal_stl_inputs import endpoint_series
from app.stl_source_repair import hourly_bid_archive, repair_endpoints
from tools.propagate_stl_source_repair import window_mask


PARAMS = dict(period=6, seasonal=7, robust=False, history=24)


def test_repair_matches_full_replay_and_preserves_unaffected_bytes():
    x = np.sin(np.arange(90)/3.)
    original = x.copy(); original[30] = np.nan
    before = endpoint_series(original, PARAMS)
    got, affected, changed = repair_endpoints(original, x, before, PARAMS)
    np.testing.assert_array_equal(got, endpoint_series(x, PARAMS))
    assert np.flatnonzero(changed).tolist() == [30]
    assert np.flatnonzero(affected).tolist() == list(range(30, 54))
    assert got[~affected].tobytes() == before[~affected].tobytes()


def test_multiple_changes_tail_and_initial_warmup():
    old = np.sin(np.arange(90)/3.)
    new = old.copy(); new[[0, 23, 70, 89]] += .2
    prior = endpoint_series(old, PARAMS)
    got, _, _ = repair_endpoints(old, new, prior, PARAMS)
    np.testing.assert_array_equal(got, endpoint_series(new, PARAMS))
    prefix, _, _ = repair_endpoints(old[:50], new[:50], prior[:50], PARAMS)
    np.testing.assert_array_equal(prefix, got[:50])


def test_no_change_does_not_refit():
    x = np.arange(30.)
    prior = endpoint_series(x, PARAMS)
    got, affected, changed = repair_endpoints(x, x.copy(), prior, PARAMS)
    assert not changed.any() and not affected.any()
    assert prior.tobytes() == got.tobytes()


def archive(rows):
    return lzma.compress(b''.join(struct.pack('>IIIIIf', *r) for r in rows))


def test_zero_volume_provider_bar_is_not_an_observation():
    frame = hourly_bid_archive(archive([(0, 100, 101, 99, 102, 1),
                                        (3600, 101, 101, 101, 101, 0)]),
                                '2022-05-01T00:00Z', 100)
    assert frame.observed.tolist() == [True, False]
    assert frame.index[1].isoformat() == '2022-05-01T01:00:00+00:00'
    assert frame.close.tolist() == [1.01, 1.01]


@pytest.mark.parametrize('rows', [
    [(1, 100, 101, 99, 102, 1)],
    [(0, 100, 101, 99, 102, 1)]*2,
    [(31*86400, 100, 101, 99, 102, 1)],
    [(0, 100, 101, 99, 98, 1)],
    [(0, 100, 101, 99, 102, -1)],
])
def test_reject_bad_archive(rows):
    with pytest.raises(ValueError):
        hourly_bid_archive(archive(rows), '2022-05-01T00:00Z', 100)


def test_reject_truncated_record():
    with pytest.raises(ValueError, match='RECORD_SIZE'):
        hourly_bid_archive(lzma.compress(b'broken'), '2022-05-01T00:00Z', 100)


def test_window_gate_does_not_fill_a_gap_or_shorten_window():
    values = np.ones((12, 2)); values[5, 0] = np.nan
    mask = window_mask(values, 3, 2)
    assert np.flatnonzero(mask).tolist() == [4, 6, 8, 10, 11]
