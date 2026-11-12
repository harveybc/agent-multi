import copy

import pytest
import torch

from app.actor_checkpoint_qualification import (
    actor_difference, development_gate, exact_state, exact_trace, one_update_point,
    qualification_profile, saved_point,
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


def test_boundary_profile_pins_endpoint_not_best_saved_metric():
    source = dict(schema='opening_boundary_refinement.v1', primary_endpoint=8,
                  points=[0, 1, 8], seeds=[19, 43], test_read=False)
    assert qualification_profile(source, 'boundary_absolute8') == ('greedy_calibrated_native', 8)
    points = [{'updates': 1, 'score': 100}, {'updates': 8, 'score': 1}]
    assert saved_point(points, 8)['score'] == 1
    for key, value in [('schema', 'other'), ('primary_endpoint', 1),
                       ('points', [0, 8]), ('seeds', [43]), ('test_read', True)]:
        with pytest.raises(ValueError, match='QUALIFY_PROFILE'):
            qualification_profile(dict(source, **{key: value}), 'boundary_absolute8')
    for bad in [[], [{'updates': 1}], [{'updates': 8}, {'updates': 8}]]:
        with pytest.raises(ValueError, match='QUALIFY_SAVED_POINT'):
            saved_point(bad, 8)


def test_bounded_critic_profile_rejects_wrong_experiment_or_missing_arm():
    source=dict(schema='native_target_geometry.v1',primary_endpoint=8,
                points=[0,1,8],seeds=[19,43],test_read=False,
                modes=['critic_adamw','return_adamw','critic_project','return_project'])
    assert qualification_profile(source,'bounded_critic8')==('critic_adamw',8)
    for key,value in [('schema','bounded_functional_actor.v1'),('primary_endpoint',1),
                      ('points',[0,8]),('seeds',[43]),('test_read',True),('modes',['return_adamw'])]:
        with pytest.raises(ValueError,match='QUALIFY_PROFILE'):
            qualification_profile(dict(source,**{key:value}),'bounded_critic8')


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


def test_coverage_profile_pins_narrow_and_fixed_endpoint():
    source=dict(schema='train_coverage_refinement.v1',primary_endpoint=8,
                points=[0,1,8],seeds=[19,43],test_read=False,
                modes=['critic_narrow','critic_broad'])
    assert qualification_profile(source,'coverage_narrow8')==('critic_narrow',8)
    for key,value in [('schema','native_target_geometry.v1'),('primary_endpoint',1),
                      ('points',[0,8]),('seeds',[43]),('test_read',True),('modes',['critic_broad'])]:
        with pytest.raises(ValueError,match='QUALIFY_PROFILE'):
            qualification_profile(dict(source,**{key:value}),'coverage_narrow8')


def test_interior_profile_pins_corrected_run_and_fixed_endpoint():
    source=dict(schema='interior_fit_error_correction.v1',primary_endpoint=8,
                points=[0,1,8],seeds=[19,43],test_read=False,
                modes=['interior_error_project_guard'])
    assert qualification_profile(source,'interior_error8')==('interior_error_project_guard',8)
    for key,value in [('schema','targeted_fit_error_correction.v1'),('primary_endpoint',1),
                      ('points',[0,8]),('seeds',[43]),('test_read',True),('modes',['error_project_guard'])]:
        with pytest.raises(ValueError,match='QUALIFY_PROFILE'):
            qualification_profile(dict(source,**{key:value}),'interior_error8')


def test_fresh72_profile_cannot_consume_historical69_correction():
    source=dict(schema='fresh_calibrated_interior_correction.v1',primary_endpoint=8,
                points=[0,1,8],seeds=[19,43],test_read=False,
                modes=['interior_error_project_guard'])
    assert qualification_profile(source,'fresh_interior72_8')==('interior_error_project_guard',8)
    for key,value in [('schema','interior_fit_error_correction.v1'),('primary_endpoint',1),
                      ('points',[0,8]),('seeds',[43]),('test_read',True),('modes',['error_project_guard'])]:
        with pytest.raises(ValueError,match='QUALIFY_PROFILE'):
            qualification_profile(dict(source,**{key:value}),'fresh_interior72_8')
    with pytest.raises(ValueError,match='QUALIFY_PROFILE'):
        qualification_profile(source,'interior_error8')


def test_trace_and_tensor_tampering():
    exact_trace([{'signal': 1}], [{'signal': 1}])
    with pytest.raises(ValueError, match='TRACE_PARITY'):
        exact_trace([{'signal': 1}], [{'signal': 2}])
    exact_state({'a': torch.ones(2)}, {'a': torch.ones(2)})
    with pytest.raises(ValueError, match='RELOAD_STATE'):
        exact_state({'a': torch.ones(2)}, {'a': torch.zeros(2)})
