import numpy as np
import pytest
import torch
import torch.nn.functional as F

from app.action_ranking_contrast import advantage_loss, calibration_indices, target_loss


def test_loss_matches_all_pairs_and_mean_with_real_gradients():
    q=torch.tensor([[[1.,3.,6.]],[[2.,2.,5.]]],requires_grad=True)
    target=torch.tensor([[0.,2.,7.]])
    actual=advantage_loss(q,target)
    pairs=[(0,1),(0,2),(1,2)]
    predicted=torch.stack([q[...,a]-q[...,b] for a,b in pairs],-1)
    wanted=torch.stack([target[...,a]-target[...,b] for a,b in pairs],-1)
    expected=F.smooth_l1_loss(predicted,wanted.unsqueeze(0).expand_as(predicted))
    expected+=F.smooth_l1_loss(q.mean(-1),target.mean(-1).unsqueeze(0).expand(2,-1))
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0


def test_common_level_is_identified_and_permutation_invariant():
    target=torch.tensor([[1.,4.,8.],[0.,3.,2.]])
    exact=target.unsqueeze(0).repeat(2,1,1)
    assert advantage_loss(exact,target)==0
    shifted=exact+torch.tensor([5.,-3.]).reshape(2,1,1)
    assert advantage_loss(shifted,target)>0
    perm=[2,0,1]
    torch.testing.assert_close(advantage_loss(shifted,target),
                              advantage_loss(shifted[...,perm],target[:,perm]))
    # A common shift of predictions AND labels changes neither calibration loss.
    torch.testing.assert_close(advantage_loss(shifted+9,target+9),advantage_loss(shifted,target))


@pytest.mark.parametrize('case',['nan','shape','empty'])
def test_invalid_loss_refuses(case):
    q=torch.ones(2,3,3);target=torch.zeros(3,3)
    if case=='nan':target[0,0]=float('nan')
    if case=='shape':q=q[:1]
    if case=='empty':q=q[:,:0];target=target[:0]
    with pytest.raises(ValueError):advantage_loss(q,target)


def cohort_fixture():
    rows=[]
    for year in range(2020,2024):
        for role,month in [('fit','01'),('monitor','02')]:
            ts=int(np.datetime64(f'{year}-{month}-01','ns').astype('int64'))
            rows.append(dict(row=len(rows),timestamp=ts,year=year,role=role))
    return rows


def test_fit_indices_exclude_monitor_and_validate_real_timestamps():
    cohort=cohort_fixture();ts=np.array([r['timestamp'] for r in cohort])
    ix=calibration_indices(cohort,ts,np.ones(len(ts),bool),np.array(['TRAIN']*len(ts)))
    assert ix.tolist()==[0,2,4,6]
    cohort[0]['timestamp']+=1
    with pytest.raises(ValueError):calibration_indices(cohort,ts,np.ones(len(ts),bool),np.array(['TRAIN']*len(ts)))


@pytest.mark.parametrize('bad',['period','year','role','duplicate','invalid'])
def test_invalid_cohort_refuses(bad):
    cohort=cohort_fixture();ts=np.array([r['timestamp'] for r in cohort])
    valid=np.ones(len(ts),bool);period=np.array(['TRAIN']*len(ts),dtype=object)
    if bad=='period':period[0]='VALIDATION'
    if bad=='year':cohort[0]['year']=2024
    if bad=='role':cohort[0]['role']='validation'
    if bad=='duplicate':cohort[1]=dict(cohort[0])
    if bad=='invalid':valid[0]=False
    with pytest.raises(ValueError):calibration_indices(cohort,ts,valid,period)
