import pytest
import torch

from app.capacity_measurements import spectrum, activation_spectrum, low_rank, signal_change


def test_spectrum_known_rank_and_invariance():
    x = torch.eye(4, dtype=torch.float64)
    a = spectrum(x)
    assert a['stable_rank'] == pytest.approx(4)
    assert a['participation_rank'] == pytest.approx(4)
    assert a['rank99'] == 4
    b = spectrum(7*x[:, [2, 0, 3, 1]])
    assert b['stable_rank'] == pytest.approx(a['stable_rank'])
    assert spectrum(torch.ones(4, 3))['rank99'] == 1
    z = activation_spectrum(torch.ones(8, 4))
    assert z['rank99'] == 0 and z['constant_channels'] == 4


def test_activation_rank_and_temporal_shape():
    x = torch.arange(30, dtype=torch.float64).reshape(10, 3)
    assert activation_spectrum(x)['rank99'] == 1
    assert activation_spectrum(x)['channels'] == 3
    with pytest.raises(ValueError):
        activation_spectrum(torch.ones(2, 3, 4))
    with pytest.raises(ValueError):
        spectrum(torch.tensor([[float('nan')]]))


def test_low_rank_identity_reconstruction_and_source_unchanged():
    x = torch.diag(torch.tensor([4., 3., 2., 1.]))
    before = x.clone()
    assert torch.equal(low_rank(x, 1.), x)
    z = low_rank(x, .5)
    assert torch.equal(x, before)
    assert torch.allclose(z, torch.diag(torch.tensor([4., 3., 0., 0.])))
    with pytest.raises(ValueError):
        low_rank(x, 0)


def test_functional_positive_and_negative_controls():
    x = torch.tensor([[0., 1., 2.], [2., 1., 0.]])
    unchanged = signal_change(x, x)
    assert unchanged['flips'] == 0 and unchanged['mean_kl'] == 0
    changed = signal_change(x, x.flip(-1))
    assert changed['flips'] == 2 and changed['mean_kl'] > 0


def test_predeclared_point_epochs():
    from tools.diagnose_component_capacity import point_epochs
    assert point_epochs(None, 7, [1, 5, 38]) == [0, 1, 38]
    assert point_epochs({'7': [76, 77, 78]}, 7, [76, 77, 78]) == [76, 77, 78]
    with pytest.raises(ValueError):
        point_epochs({'7': [76, 77, 79]}, 7, [76, 77, 78])
    with pytest.raises(ValueError):
        point_epochs({'7': [77, 77, 78]}, 7, [77, 78])
