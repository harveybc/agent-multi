"""Target-blind, FIT-only negative controls for the native critic diagnostic."""
import copy
from collections import defaultdict

import numpy as np


def state_kind(record):
    if 'kind' in record:
        return record['kind']
    if record['position']:
        return 'open_long' if record['position'] == 1 else 'open_short'
    return 'flat_both_allowed' if record['allowed'] else 'flat_restricted'


def shuffled_fit_records(records, seed):
    groups = defaultdict(list)
    for i, record in enumerate(records):
        if record['role'] not in ('fit', 'monitor'):
            raise ValueError('REPRESENTATION_ROLE')
        if record['role'] == 'fit':
            if record['year'] not in (2020, 2021, 2022):
                raise ValueError('REPRESENTATION_FIT_YEAR')
            groups[(record['year'], state_kind(record))].append(i)
    result = copy.deepcopy(records)
    rng = np.random.default_rng(seed)
    mapping = list(range(len(records)))
    for key in sorted(groups):
        indices = groups[key]
        for dest, src in zip(indices, rng.permutation(indices).tolist()):
            mapping[dest] = src
            for action in range(3):
                for field in ('reward_return', 'soft_return'):
                    result[dest]['outcomes'][action][field] = records[src]['outcomes'][action][field]
    return result, dict(source_indices=mapping,
        moved_rows=sum(i != src for i, src in enumerate(mapping)),
        singleton_strata=[list(k) for k, ix in sorted(groups.items()) if len(ix) == 1])


def recommendation(rows):
    by_key = {(r['seed'], r['scope'], r['labels']): r['metrics']['held']['minimum'] for r in rows}
    if len(rows) != 8 or len(by_key) != 8:
        raise ValueError('REPRESENTATION_INCOMPLETE')
    passed = []
    for seed in (19, 43):
        h = by_key[seed, 'heads', 'true']
        r = by_key[seed, 'critic_representation', 'true']
        n = by_key[seed, 'critic_representation', 'shuffled']
        passed.append(r['pairwise_accuracy'] >= h['pairwise_accuracy']
            and r['mean_regret'] <= h['mean_regret']
            and (r['pairwise_accuracy'] > h['pairwise_accuracy'] or r['mean_regret'] < h['mean_regret'])
            and r['mean_regret'] < n['mean_regret'])
    return ('DESIGN_BOUNDED_OWN_POLICY_ACTOR_STEP' if all(passed)
            else 'DIAGNOSE_PARTIAL_OBSERVATION_AND_NATIVE_COVERAGE')
