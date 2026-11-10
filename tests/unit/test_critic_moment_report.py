import copy

import pytest

from tools.report_critic_moments import error_summary, validate_pair


def pair():
    base={k:k for k in ('source_sha256','manifest_sha256','parent_identity','support_sha256',
                       'recipe_identity','torch','device')}
    base['design']=dict(arms=['deterministic','bayesian'],batch=128,lr=3e-5,seeds=[7,19,43])
    candidate=copy.deepcopy(base)
    candidate['design'].update(arms=['bayesian'],variance_weight=.5,critic_objective='mean_var_05')
    return base,candidate


def test_matching_pair_and_residual_identity():
    a,b=pair();validate_pair(a,b)
    r=error_summary(dict(bias=3.,mae=4.,rmse=5.,n=10))
    assert r['residual_std']==4.


@pytest.mark.parametrize('field',['source_sha256','support_sha256','recipe_identity'])
def test_changed_identity_rejected(field):
    a,b=pair();b[field]='changed'
    with pytest.raises(ValueError):validate_pair(a,b)


@pytest.mark.parametrize('field',['batch','lr','seeds'])
def test_changed_training_rejected(field):
    a,b=pair();b['design'][field]=None
    with pytest.raises(ValueError):validate_pair(a,b)
