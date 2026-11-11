import copy

import numpy as np
import pytest
import torch

from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.rl_subset import subset_recipe
from tests.unit.test_double_tcn_probe import probe_recipe
from tests.unit.test_frozen_recipe_inputs import seal
from app.frozen_actor_compaction import (
    ConstantResponse, compact_actor, effective_actor, PackedActor,
    save_compact, load_compact, compare_logits,
)


def two_group_recipe():
    r = probe_recipe()
    r['representation']['resolved'] = dict(names=['keep', 'remove'],
        indices=[[0], [1]], units=[2, 2], kinds=['conv', 'conv'])
    return seal(r)


@pytest.mark.parametrize('groups', [1, 2])
def test_partial_or_whole_branch_preserves_function_and_source(groups):
    torch.set_num_threads(1)
    r = probe_recipe() if groups == 1 else two_group_recipe()
    model = ThreeSignalSAC(r)
    before = {k: v.clone() for k, v in model.loss.state_dict().items()}
    actor = effective_actor(model)
    x = torch.randn(7, 13, 1)
    full = torch.cat([x, torch.zeros_like(x)], dim=2)
    packed = PackedActor(actor, [0], 2)
    compact, info = compact_actor(actor, r, ['x'])
    with torch.inference_mode():
        expected = model.surfaces(full)
        assert torch.equal(packed.surface(x), expected)
        torch.testing.assert_close(compact.surface(x), expected, atol=1e-5, rtol=1e-5)
        changed = x.clone(); changed[:, 7:] += 100
        a = compact.encoder.sequence({'features': x})
        b = compact.encoder.sequence({'features': changed})
        assert torch.equal(a[:, :, :7], b[:, :, :7])
    assert info['parameters_after'] < info['parameters_before']
    assert len(info['constant_groups']) == groups - 1
    assert compact.encoder.shape == (13, 1)
    assert compact.encoder.transport_shape == (13, 1)
    assert actor.encoder.transport_shape == (13, 2)
    assert all(not p.requires_grad for p in compact.parameters())
    assert all(torch.equal(v, model.loss.state_dict()[k]) for k, v in before.items())


def test_empty_branch_keeps_temporal_response_not_mean_or_zero():
    r = two_group_recipe()
    actor = effective_actor(ThreeSignalSAC(r))
    compact, _ = compact_actor(actor, r, ['x'])
    branch = compact.encoder.branches[1]
    assert isinstance(branch, ConstantResponse)
    assert torch.count_nonzero(branch.response) > 0
    assert not torch.equal(branch.response[:, :, 0], branch.response[:, :, -1])
    assert not list(branch.parameters())
    with torch.inference_mode():
        expected = actor.encoder.branches[1](torch.zeros(5, 1, 13))
        torch.testing.assert_close(branch(torch.zeros(5, 0, 13)), expected)
        zeroed = copy.deepcopy(compact)
        zeroed.encoder.branches[1].response.zero_()
        x = torch.randn(5, 13, 1)
        assert not torch.equal(compact.surface(x), zeroed.surface(x))


def test_effective_copy_follows_functional_parameters_and_is_independent():
    model = ThreeSignalSAC(probe_recipe())
    with torch.no_grad():
        for p in model.loss.actor_network_params.parameters():
            p.add_(.17)
    actor = effective_actor(model)
    x = torch.randn(3, 13, 2)
    with torch.inference_mode():
        expected = model.surfaces(x).clone()
        assert torch.equal(actor.surface(x), expected)
        next(actor.parameters()).add_(1)
        assert torch.equal(model.surfaces(x), expected)


def test_long_context_cannot_be_silently_compacted():
    r = probe_recipe()
    actor = effective_actor(ThreeSignalSAC(r))
    actor.encoder.context = torch.nn.Identity()
    with pytest.raises(ValueError, match='COMPACT_LONG_CONTEXT_UNSUPPORTED'):
        compact_actor(actor, r, ['x'])
    assert actor.encoder.transport_shape == (13, 2)


@pytest.mark.parametrize('columns', [[], ['y', 'x'], ['x', 'x'], ['unknown']])
def test_invalid_subset_rejected(columns):
    r = probe_recipe()
    with pytest.raises(ValueError, match='SUBSET_ORDER'):
        compact_actor(effective_actor(ThreeSignalSAC(r)), r, columns)


def test_compact_reload_and_tamper(tmp_path):
    r = two_group_recipe(); model = ThreeSignalSAC(r)
    compact, info = compact_actor(effective_actor(model), r, ['x'])
    x = np.random.default_rng(9).normal(size=(5, 13, 1)).astype('float32')
    folder = tmp_path / 'bundle'
    save_compact(folder, compact, r, subset_recipe(r, ['x']), model.effective,
                 {'source_checkpoint_sha256': 'fixture'}, x, info)
    loaded, manifest = load_compact(folder)
    assert manifest['schema'] == 'frozen_actor.compact.v1'
    assert manifest['training_supported'] is False
    with torch.inference_mode():
        assert torch.equal(compact.surface(torch.tensor(x)), loaded.surface(torch.tensor(x)))
    with (folder / 'actor.pt').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='BUNDLE_HASH'):
        load_compact(folder)


def test_input_shape_and_batch_independent_constant():
    r = two_group_recipe()
    compact, _ = compact_actor(effective_actor(ThreeSignalSAC(r)), r, ['x'])
    with pytest.raises(ValueError, match='OBSERVATION_SHAPE'):
        compact.surface(torch.zeros(1, 12, 1))
    with torch.inference_mode():
        a = compact.surface(torch.zeros(1, 13, 1))
        b = compact.surface(torch.zeros(6, 13, 1))
        torch.testing.assert_close(a.expand_as(b), b)


def test_parity_uses_native_tie_to_nop_and_detects_small_action_change():
    a = np.array([[0., 1., 1.], [0., 2., -1.]], dtype='float32')
    b = a.copy(); b[0, 2] += 1e-6
    result = compare_logits(a, b)
    assert result['within_tolerance']
    assert result['action_disagreements'] == 1
    assert not result['qualified']
    assert compare_logits(a, a)['bitwise_exact']
    b[0, 1] = np.nan
    with pytest.raises(ValueError, match='FINITE'):
        compare_logits(a, b)
