import json
import numpy as np
import pytest

from app.head_spillover import decompose_final_layer, spectrum, nearest_rows


def test_final_layer_identity_with_interaction():
    r = np.random.default_rng(309)
    h0, h1 = r.normal(size=(2, 21, 8))
    w0, w1 = r.normal(size=(2, 3, 8))
    b0, b1 = r.normal(size=(2, 3))
    parts = decompose_final_layer(h0, h1, w0, w1, b0, b1)
    np.testing.assert_allclose(sum(parts.values()),
        (h1@w1.T+b1)-(h0@w0.T+b0), atol=1e-14, rtol=1e-14)
    assert np.abs(parts['interaction']).max() > 0


def test_bias_only_and_unchanged():
    h = np.ones((5, 2)); w = np.ones((3, 2)); b = np.zeros(3)
    parts = decompose_final_layer(h, h, w, w, b, b+1)
    for k in ('weight', 'hidden', 'interaction'):
        assert not parts[k].any()
    assert (parts['bias'] == 1).all()
    assert not sum(decompose_final_layer(h, h, w, w, b, b).values()).any()


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_reject_nonfinite(bad):
    with pytest.raises(ValueError):
        decompose_final_layer([[bad]], [[1]], [[1]], [[1]], [0], [0])
    with pytest.raises(ValueError):
        spectrum([[0], [bad]])


def test_shapes_and_constant_spectrum():
    with pytest.raises(ValueError):
        decompose_final_layer([[1]], [[1, 2]], [[1]], [[1]], [0], [0])
    result = spectrum(np.ones((10, 3)))
    assert result['effective_rank'] == 0
    assert result['dimensions_99pct'] == 0
    assert result['constant_dimensions'] == 3
    rank1 = spectrum(np.arange(10)[:, None]*np.ones((1, 3)))
    assert rank1['effective_rank'] == pytest.approx(1)
    json.dumps(rank1, allow_nan=False)


def test_neighbours_exclude_anchor_and_stable_ties():
    ids, distances = nearest_rows([[0, 0], [1, 0], [-1, 0], [0, 0]], 0, 3)
    assert ids.tolist() == [3, 1, 2]
    assert distances.tolist() == [0, 1, 1]
