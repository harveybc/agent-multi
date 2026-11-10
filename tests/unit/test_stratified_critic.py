import numpy as np
import pytest

from app.stratified_critic import select_strata, validate_support, grouped_metrics


def test_sampling_is_unique_reproducible_and_reports_shortages():
    rows=[dict(kind=k,id=f'{k}-{i}') for k,n in
          [('flat_both_allowed',40),('open_long',35),('open_short',7),('flat_restricted',20)]
          for i in range(n)]
    selected,counts=select_strata(rows,19)
    assert selected==select_strata(rows,19)[0]
    assert len(selected)==32+32+7+8==len(set(selected))
    assert counts['open_short']==dict(available=7,selected=7,requested=32)
    changed=[dict(r,target=999) for r in rows]
    assert select_strata(changed,19)[0]==selected
    with pytest.raises(ValueError):select_strata([dict(kind='unknown')],19)


def test_temporal_support_rejects_overlap_and_bad_windows():
    fit=dict(role='fit',timestamp=10,terminal=20,window_start=1)
    mon=dict(role='monitor',timestamp=32,terminal=40,window_start=30)
    validate_support([fit,mon],30)
    for key,value in [('terminal',30),('timestamp',31)]:
        with pytest.raises(ValueError):validate_support([dict(fit,**{key:value}),mon],30)
    with pytest.raises(ValueError):validate_support([fit,dict(mon,window_start=29)],30)
    with pytest.raises(ValueError):validate_support([fit,dict(mon,terminal=31)],30)
    with pytest.raises(ValueError):validate_support([dict(fit,role='validation'),mon],30)


def test_state_metrics_do_not_hide_empty_or_tied_groups():
    rows=[dict(role='fit',kind='open_long'),dict(role='monitor',kind='flat_restricted')]
    target=np.array([[1,1,0],[2,2,2]],float)
    metrics=grouped_metrics(np.stack([target,target]),target,rows)
    assert metrics['fit']['open_long']['minimum']['pairwise_accuracy']==1
    assert metrics['fit']['open_short'] is None
    assert metrics['monitor']['flat_restricted']['minimum']['pairwise_accuracy'] is None
    assert metrics['monitor']['flat_restricted']['minimum']['maximum_regret']==0
