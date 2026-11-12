import numpy as np
import pytest

from app.transformed_feature_scope import context_fit_overlap


def test_causal_filter_can_break_cross_fit_isolation():
    fits=[np.arange(6),np.array([4,5]),np.array([0,1])]
    predictions=[np.arange(3),np.arange(3,6)]
    available=np.ones(6,bool)
    direct,_,_=context_fit_overlap(fits,predictions,available,[0])
    assert not direct.any()
    filtered,lag,owners=context_fit_overlap(fits,predictions,available,[0,1,2,3])
    assert np.flatnonzero(filtered).tolist()==[4,5]
    assert lag[4]==2 and lag[5]==3 and owners[4]==2
    # Enlarged producer purge eliminates this particular target-row overlap.
    fixed=[fits[0],np.array([],dtype=int),np.array([],dtype=int)]
    assert not context_fit_overlap(fixed,predictions,available,[0,1,2,3])[0].any()


def test_unobserved_context_is_not_fabricated_and_duplicates_fail():
    fits=[np.arange(4),np.array([2,3]),np.array([0,1])]
    predictions=[np.array([0,1]),np.array([2,3])]
    available=np.array([False,False,True,True])
    assert not context_fit_overlap(fits,predictions,available,[0,1,2])[0].any()
    with pytest.raises(ValueError,match='DUPLICATE'):
        context_fit_overlap(fits,[np.array([0,1]),np.array([1,2])],available,[0])
    with pytest.raises(ValueError,match='OFFSETS'):
        context_fit_overlap(fits,predictions,available,[-1])


def test_composed_context_is_stl_history_plus_window_span():
    from tools.rebuild_stl_activity_scope import downstream_context
    assert downstream_context(480,13,2)==504
    assert downstream_context(64,1,1)==64
    with pytest.raises(ValueError,match='ACTIVITY_CONTEXT'):
        downstream_context(480,13,0)


def test_training_gate_rejects_failed_or_stale_scope_report():
    from app.transformed_feature_scope import require_scope_report
    r=dict(lane_identity='lane',raw_sha256='raw',prepared_receipt_sha256='prepared',
        passed=True,reports={'seasonal':{'paired_evaluation_rows_with_overlap':0}})
    require_scope_report(r,'lane','raw','prepared')
    with pytest.raises(ValueError,match='LINEAGE'):
        require_scope_report(r,'changed','raw','prepared')
    with pytest.raises(ValueError,match='OVERLAP'):
        require_scope_report(dict(r,passed=False),'lane','raw','prepared')
    r['reports']['seasonal']['paired_evaluation_rows_with_overlap']=1
    with pytest.raises(ValueError,match='OVERLAP'):
        require_scope_report(r,'lane','raw','prepared')
