import copy

import pytest
import torch

from app.actor_checkpoint_qualification import (
    actor_difference, development_gate, exact_state, exact_trace, one_update_point,
)


def test_actor_scope_and_critic_changes_are_distinct():
    ref = {'actor_network.encoder.weight': torch.ones(2),
           'actor_network.head.0.weight': torch.ones(2), 'critic.weight': torch.ones(2)}
    new = copy.deepcopy(ref)
    new['actor_network.head.0.weight'] += .1
    new['critic.weight'] += .5
    assert actor_difference(ref, new)['unchanged_actor_tensors'] == 1
    new['actor_network.encoder.weight'] += .01
    with pytest.raises(ValueError, match='ACTOR_SCOPE'):
        actor_difference(ref, new)


def test_no_update_is_rejected():
    ref = {'actor_network.head.0.weight': torch.ones(2)}
    with pytest.raises(ValueError, match='ACTOR_SCOPE'):
        actor_difference(ref, ref)


@pytest.mark.parametrize('points', [[], [{'updates': 32}], [{'updates': 1}, {'updates': 1}]])
def test_exactly_one_saved_point(points):
    with pytest.raises(ValueError, match='ONE_UPDATE_POINT'):
        one_update_point(points)


def test_saved_point():
    assert one_update_point([{'updates': 0}, {'updates': 1, 'tag': 'correct'}])['tag'] == 'correct'


def test_activity_and_validation_are_not_compensated_by_train():
    ref = {p: dict(return_pct=10., orders_per_week=3.) for p in ('TRAIN', 'VALIDATION')}
    new = copy.deepcopy(ref); new['TRAIN']['return_pct'] = 100.
    assert development_gate(new, ref)['qualified']
    new['VALIDATION']['return_pct'] = 9.99
    assert not development_gate(new, ref)['qualified']
    new['VALIDATION']['return_pct'] = 10.; new['TRAIN']['orders_per_week'] = 2.
    assert not development_gate(new, ref)['qualified']
    new['TRAIN']['return_pct'] = float('nan')
    with pytest.raises(ValueError, match='NONFINITE'):
        development_gate(new, ref)


def test_trace_and_tensor_tampering():
    exact_trace([{'signal': 1}], [{'signal': 1}])
    with pytest.raises(ValueError, match='TRACE_PARITY'):
        exact_trace([{'signal': 1}], [{'signal': 2}])
    exact_state({'a': torch.ones(2)}, {'a': torch.ones(2)})
    with pytest.raises(ValueError, match='RELOAD_STATE'):
        exact_state({'a': torch.ones(2)}, {'a': torch.zeros(2)})
