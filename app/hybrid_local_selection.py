"""Finite349 choice: no held-label selection, no adaptive budget extension."""
import numpy as np
from app.function_local_projection import pairwise_qualified
from app.training_forensics import tree_digest


def choose_local(control,local,assessment,control_metrics,drift,control_drift):
    qualified=local is not None and pairwise_qualified(assessment,control_metrics,drift,control_drift)
    selected=local if qualified else control
    return selected,dict(qualified=bool(qualified),control_state=tree_digest(control),
        local_state=tree_digest(local) if local is not None else None,selected_state=tree_digest(selected))


def continuation_decision(units,substitutions):
    if len(units)!=4:raise ValueError('HYBRID_CASE_COUNT')
    wins=0
    for u in units:
        control=u['control']['fit'];final=u['curve'][-1]['metrics']['fit']
        if set(control)!=set(final) or len(control)!=5:raise ValueError('HYBRID_VIEW_COUNT')
        for k in control:
            values=[control[k]['regret'],final[k]['regret']]
            if not np.isfinite(values).all():raise ValueError('HYBRID_NONFINITE')
            if final[k]['regret']>control[k]['regret']:
                return dict(selected='original345',reason='paired_FIT_regret_worse',substitutions=substitutions)
        wins+=int(all(final[k]['regret']<control[k]['regret'] for k in control))
    # Exact ties can still qualify through strictly lower measured collateral drift.
    return dict(selected='hybrid349' if substitutions else 'original345',
        reason='no_paired_regret_loss_with_locality_gain' if substitutions else 'no_local_substitution',
        paired_strict_wins=wins,substitutions=substitutions)
