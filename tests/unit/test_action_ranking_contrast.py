import numpy as np
import pytest
import torch

from app.action_ranking_contrast import select_anchors, target_loss, ranking_metrics, differentiable_q


def trace(year=2020,n=60):
    start=int(np.datetime64(f'{year}-01-01','ns').astype(np.int64))
    return [dict(timestamp=start+i*3600000000000,position=(-1,0,1)[i%3]) for i in range(n)]


def test_anchor_selection_train_only_reproducible_disjoint():
    t=trace()+trace(2021)
    a=select_anchors(t,per_stratum=8,fit_per_stratum=6,seed=41)
    assert a==select_anchors(t,8,6,41)
    assert len(a)==48 and len({x['trace_index'] for x in a})==48
    assert sum(x['role']=='fit' for x in a)==36
    with pytest.raises(ValueError,match='TRAIN'):select_anchors(trace(2024))
    with pytest.raises(ValueError):select_anchors(t,8,8)


def test_paired_and_chosen_losses_have_correct_action_gradients():
    q=torch.zeros(2,4,3,requires_grad=True);target=torch.ones(4,3)
    target_loss(q,target,None).backward()
    assert (q.grad!=0).all()
    q.grad.zero_();chosen=torch.tensor([0,1,2,0])
    target_loss(q,target,chosen).backward()
    assert int((q.grad!=0).sum())==8
    with pytest.raises(ValueError):target_loss(q,torch.ones(3,4),None)
    with pytest.raises(ValueError):target_loss(q,torch.full((4,3),float('nan')),None)


def test_ranking_and_ties_not_misreported_as_profit():
    target=np.array([[0.,2.,1.],[1.,1.,1.]])
    q=np.array([[0.,1.,2.],[0.,1.,2.]])
    r=ranking_metrics(q,target)
    assert r['pairwise_comparisons']==3
    assert r['pairwise_accuracy']==pytest.approx(2/3)
    assert r['mean_regret']==.5
    assert r['target_ties']==3


def test_critic_correction_preserves_actor_encoder_alpha():
    from tests.unit.test_double_tcn_probe import probe_recipe
    from agent_plugins.three_signal_sac import ThreeSignalSAC
    from tools.diagnose_warmstart_drift import set_arm
    torch.set_num_threads(1);model=ThreeSignalSAC(probe_recipe())
    before={n:p.detach().clone() for n,p in model.loss.named_parameters()}
    set_arm(model,'critic_only')
    x=torch.randn(4,13,2);y=torch.ones(4,3)
    model.optimizer.zero_grad(set_to_none=True)
    q=differentiable_q(model,x)
    torch.testing.assert_close(q,model.q_surfaces(x))
    target_loss(q,y,None).backward();model.optimizer.step()
    changed=[n for n,p in model.loss.named_parameters() if not torch.equal(p,before[n])]
    assert changed and all(n.startswith('qvalue_') and '.head.' in n for n in changed)


def test_panel_loader_rejects_incomplete_or_altered_receipts(tmp_path):
    from tools.run_action_ranking_contrast import load_panel
    from tools.run_discrete_curve_probe import write
    from app.frozen_recipe_inputs import file_sha
    from app.rl_subset import digest
    cohort=[dict(id=i,trace_index=i,timestamp=i,year=2020,position=0,role='fit',row=i,chosen_action=0) for i in range(2)]
    for i in range(2):
        p=tmp_path/f'shard_{i}';result=dict(greedy=[dict(reward_return=float(a)) for a in range(3)],
            stochastic={str(s):[dict(soft_return=float(a)+.1,reward_return=float(a)) for a in range(3)] for s in (7,19,43,73)})
        write(p/'COHORT.json',cohort)
        write(p/f'rows/{i:04d}.json',dict(identity='test',anchor=cohort[i],results=result,sha256=digest(result)))
        write(p/'DONE.json',dict(identity='test',files={str(f.relative_to(p)):file_sha(f) for f in p.rglob('*.json')}))
    c,g,s,plain=load_panel(tmp_path,'test')
    assert c==cohort and g.shape==(2,3) and s.shape==(2,4,3)
    assert np.allclose(s-plain,.1)
    write(tmp_path/'shard_1/rows/0001.json',{})
    with pytest.raises(ValueError,match='COLLECTION'):load_panel(tmp_path,'test')
