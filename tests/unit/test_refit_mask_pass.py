import copy
import pytest
import numpy as np

from tools.run_refit_mask_pass import account_equal, score_metrics, authenticated_parent
from tools.run_discrete_curve_probe import write


def test_cross_device_probe_tolerance_never_allows_changed_actions():
    from tools.run_refit_mask_pass import verify_probe
    expected=np.array([[1.,2.,0.]],dtype=np.float32)
    assert verify_probe(expected,expected,False)['exact']
    actual=expected.copy(); actual[0,0]+=1e-7
    with pytest.raises(ValueError,match='PROBE_NOT_EXACT'):
        verify_probe(actual,expected,False)
    assert not verify_probe(actual,expected,True)['exact']
    with pytest.raises(ValueError,match='PROBE_TOLERANCE'):
        verify_probe(expected+0.01,expected,True)
    with pytest.raises(ValueError,match='PROBE_ACTION'):
        verify_probe(np.array([[1.,1.+1e-7,0.]]),np.array([[1.+1e-7,1.,0.]]),True)
    with pytest.raises(ValueError,match='PROBE_SHAPE_OR_FINITE'):
        verify_probe(expected*np.nan,expected,True)


def metrics(train=2., validation=-.1, activity=3.):
    return {p:dict(return_per_week=v,orders_per_week=activity,trades=100)
            for p,v in [('TRAIN',train),('VALIDATION',validation)]}


def test_validation_ranks_not_compensating_train():
    a=score_metrics(metrics(100.,-.2)); b=score_metrics(metrics(1.,-.1))
    assert b['score']>a['score']
    assert b['frozen_weights'] and not b['fresh_training']


def test_activity_is_strict_and_both_partitions_required():
    m=metrics(activity=2.)
    assert not score_metrics(m)['eligible']
    m['VALIDATION']['orders_per_week']=3.
    assert not score_metrics(m)['eligible']
    m['TRAIN']['orders_per_week']=3.
    assert score_metrics(m)['eligible']


@pytest.mark.parametrize('v',[float('nan'),float('inf')])
def test_nonfinite_score_rejected(v):
    with pytest.raises(ValueError,match='NONFINITE_SCORE'):
        score_metrics(metrics(validation=v))


def test_parity_checks_native_account_not_descriptive_entropy():
    m=metrics()['TRAIN']; other=copy.deepcopy(m)
    other['mean_signal_entropy']=1.
    other['early_close_diagnostic']={'harmful':2}
    account_equal(m,other)
    other['trades']+=1
    with pytest.raises(ValueError,match='ACCOUNT_PARITY'):
        account_equal(m,other)


def test_parent_completion_and_bundle_identity(tmp_path,monkeypatch):
    import tools.run_refit_mask_pass as runner
    from app.frozen_recipe_inputs import file_sha
    monkeypatch.setattr(runner,'authenticate',lambda p:dict(identity='parent'))
    out=tmp_path/'runs'/'seed_7'; bundle=out/'bundles'/'epoch_0028'
    bundle.mkdir(parents=True)
    write(bundle/'MANIFEST.json',{'test':1})
    (bundle/'model.pt').write_bytes(b'test-weight-fixture')
    best=dict(relative_path='bundles/epoch_0028',manifest_sha256=file_sha(bundle/'MANIFEST.json'),
              model_sha256=file_sha(bundle/'model.pt'))
    write(out/'RESULT.json',dict(identity='parent',best_bundle=best,activity_admissible=True,seed=7))
    write(out/'TRAINING.json',dict(identity='parent',best=best,restored_best=True))
    assert authenticated_parent(tmp_path,7)[3]==bundle
    (bundle/'model.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='BUNDLE_CHANGED'):
        authenticated_parent(tmp_path,7)
    write(out/'TRAINING.json',dict(identity='parent',best=best,restored_best=False))
    with pytest.raises(ValueError,match='NOT_COMPLETED'):
        authenticated_parent(tmp_path,7)
