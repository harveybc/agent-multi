"""Checkpoint selectors for the paired feature-sensitivity diagnostic."""
import copy
import numpy as np

from app.stopping_contrast import ShadowRule


def make_selectors(design):
    # A observes the entire fixed budget; B freezes at its own stopping time.
    return [ShadowRule('train_best', design['steps'] + 1, design['minimum_steps'],
                       design['min_delta'], False),
            ShadowRule('guarded_p5', 5, design['minimum_steps'], design['min_delta'], True)]


def observe(selectors, report, step):
    for selector in selectors:
        r = copy.deepcopy(report)
        if selector.name == 'train_best':
            r['score'] = float(np.mean([x['return_per_week'] for x in r['train']]))
        selector.observe(r, step)
    return {s.name: s.state() for s in selectors}


def prefix_select(history, design):
    selected = make_selectors(design)
    expected = list(range(design['evaluation_every'], design['steps'] + 1, design['evaluation_every']))
    prefix = [h for h in history if h['step'] <= design['steps']]
    if [h['step'] for h in prefix] != expected:
        raise ValueError('ABLATION_REFERENCE_PREFIX_INCOMPLETE')
    for h in prefix:
        observe(selected, h['report'], h['step'])
    return selected, prefix
