"""Narrow gates for saved one-update actor candidates; no learning or promotion."""
import numpy as np
import torch


def one_update_point(points):
    matches = [p for p in points if p['updates'] == 1]
    if len(matches) != 1:
        raise ValueError('QUALIFY_ONE_UPDATE_POINT')
    return matches[0]


def qualification_profile(source, profile):
    if profile == 'one_update':
        return 'control', 1
    if profile in ('interior_error8', 'fresh_interior72_8'):
        schema = ('fresh_calibrated_interior_correction.v1' if profile == 'fresh_interior72_8'
                  else 'interior_fit_error_correction.v1')
        if (source.get('schema') != schema or
                source.get('primary_endpoint') != 8 or source.get('points') != [0,1,8] or
                source.get('seeds') != [19,43] or source.get('test_read') is not False or
                source.get('modes') != ['interior_error_project_guard']):
            raise ValueError('QUALIFY_PROFILE')
        return 'interior_error_project_guard', 8
    if profile == 'coverage_narrow8':
        if (source.get('schema') != 'train_coverage_refinement.v1' or
                source.get('primary_endpoint') != 8 or source.get('points') != [0,1,8] or
                source.get('seeds') != [19,43] or source.get('test_read') is not False or
                source.get('modes') != ['critic_narrow','critic_broad']):
            raise ValueError('QUALIFY_PROFILE')
        return 'critic_narrow', 8
    if profile == 'bounded_critic8':
        if (source.get('schema') != 'native_target_geometry.v1' or
                source.get('primary_endpoint') != 8 or source.get('points') != [0,1,8] or
                source.get('seeds') != [19,43] or source.get('test_read') is not False or
                source.get('modes') != ['critic_adamw','return_adamw','critic_project','return_project']):
            raise ValueError('QUALIFY_PROFILE')
        return 'critic_adamw', 8
    if (profile != 'boundary_absolute8' or
            source.get('schema') != 'opening_boundary_refinement.v1' or
            source.get('primary_endpoint') != 8 or
            source.get('points') != [0, 1, 8] or
            source.get('seeds') != [19, 43] or
            source.get('test_read') is not False):
        raise ValueError('QUALIFY_PROFILE')
    return 'greedy_calibrated_native', 8


def saved_point(points, updates):
    if updates == 1:
        return one_update_point(points)
    matches = [p for p in points if p['updates'] == updates]
    if len(matches) != 1:
        raise ValueError('QUALIFY_SAVED_POINT')
    return matches[0]


def actor_difference(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError('QUALIFY_STATE_KEYS')
    names = [n for n in reference if n.startswith('actor_')]
    if not names:
        raise ValueError('QUALIFY_ACTOR_ABSENT')
    changed = [n for n in names if not torch.equal(reference[n], candidate[n])]
    if not changed or any('.head.' not in n for n in changed):
        raise ValueError('QUALIFY_ACTOR_SCOPE')
    return dict(actor_tensors=len(names), changed_head_tensors=changed,
                unchanged_actor_tensors=len(names)-len(changed))


def development_gate(metrics, reference):
    if set(metrics) != {'TRAIN', 'VALIDATION'} or set(reference) != set(metrics):
        raise ValueError('QUALIFY_PERIODS')
    values = [m[k] for m in metrics.values() for k in ('return_pct', 'orders_per_week')]
    values += [m['return_pct'] for m in reference.values()]
    if not np.isfinite(values).all():
        raise ValueError('QUALIFY_NONFINITE')
    checks = dict(activity=all(m['orders_per_week'] > 2 for m in metrics.values()),
                  train_improved=metrics['TRAIN']['return_pct'] > reference['TRAIN']['return_pct'],
                  validation_not_lower=metrics['VALIDATION']['return_pct'] >= reference['VALIDATION']['return_pct'])
    return dict(checks=checks, qualified=all(checks.values()), model_promoted=False)


def exact_trace(actual, expected):
    if actual != expected:
        raise ValueError('QUALIFY_TRACE_PARITY')


def exact_state(actual, expected):
    if actual.keys() != expected.keys() or any(not torch.equal(actual[k], expected[k]) for k in actual):
        raise ValueError('QUALIFY_RELOAD_STATE')
