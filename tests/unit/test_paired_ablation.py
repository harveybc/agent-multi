import copy
import numpy as np
import pytest
import torch

from app.ablation_comparison import make_selectors, observe, prefix_select
from app.rl_subset import subset_recipe, project_inputs, support_hash
from app.frozen_recipe_inputs import FrozenRecipeInputs
from agent_plugins.three_signal_sac import ThreeSignalSAC
from tests.unit.test_rl_subset import split_recipe
from tests.unit.test_frozen_recipe_inputs import frame, seal


def design():
    return dict(steps=100, minimum_steps=10, min_delta=1e-5, evaluation_every=10)


def report(train, val, eligible=True):
    return dict(score=(train+val)/2, eligible=eligible,
                train=[dict(return_per_week=train)], internal_validation=[dict(return_per_week=val)])


def test_train_best_and_guarded_freeze_on_shared_history():
    rules=make_selectors(design())
    observe(rules, report(1,1),10)
    for step in range(20,80,10): observe(rules, report(step,-step),step)
    assert rules[0].selected_step==70 and not rules[0].stopped
    assert rules[1].selected_step==10 and rules[1].stopped
    frozen=copy.deepcopy(rules[1].state())
    observe(rules,report(1000,1000),80)
    assert rules[1].state()==frozen and rules[0].selected_step==80


def test_control_prefix_cannot_see_beyond_budget():
    h=[dict(step=i,report=report(i,i)) for i in range(10,120,10)]
    rules,prefix=prefix_select(h,design())
    assert len(prefix)==10 and all(r.selected_step==100 for r in rules)
    with pytest.raises(ValueError,match='INCOMPLETE'): prefix_select(h[1:],design())
    with pytest.raises(ValueError,match='INCOMPLETE'): prefix_select(h+h[:1],design())


def test_no_activity_is_not_zero_score_or_selected_checkpoint():
    r=make_selectors(design())
    for step in range(10,110,10): observe(r,report(100,100,False),step)
    assert all(x.selected_step is None for x in r)


def test_actual_sac_structurally_pruned_with_same_support_and_scale():
    r=split_recipe(); r['tensor']['window']['observations']=13; seal(r)
    f=frame(); f.loc[8,'x']=np.nan
    full=FrozenRecipeInputs.from_frame(r,f)
    reduced=subset_recipe(r,['y']); projected=project_inputs(full,reduced)
    assert support_hash(full)==support_hash(projected)
    assert reduced['scaler']['mean']==[r['scaler']['mean'][1]]
    a=ThreeSignalSAC(r); b=ThreeSignalSAC(reduced)
    assert len(a.actor_net.encoder.branches)==2 and len(b.actor_net.encoder.branches)==1
    assert sum(p.numel() for p in b.actor_net.parameters()) < sum(p.numel() for p in a.actor_net.parameters())
    assert b.actor_net.surface(torch.zeros(2,13,1)).shape==(2,3)
    assert b.q_surfaces(torch.zeros(2,13,1)).shape==(2,2,3)


def test_smoke_receipt_cannot_be_adopted_as_production(tmp_path):
    from tools.run_discrete_curve_probe import write
    from tools.run_paired_ablation import verified
    path=tmp_path/'RESULT.json'
    write(path,dict(identity='x',unit='a__7',smoke=True))
    with pytest.raises(ValueError,match='SMOKE_NOT_PRODUCTION'):
        verified(path,dict(identity='x'),'a__7')


def test_paired_report_uses_same_dates_and_exposes_disagreement(tmp_path):
    from tools.run_discrete_curve_probe import write
    from tools.run_paired_ablation import summarize
    results={}; paths={}
    for arm in ('full','drop'):
        for seed in (7,19):
            unit=f'{arm}__{seed}'; folder=tmp_path/unit; folder.mkdir()
            account=folder/'accounts'/'10'; account.mkdir(parents=True)
            paths[unit]=folder/'RESULT.json'
            write(folder/'HISTORY.json',[dict(step=10,report=report(1 if arm=='full' else 2,-1 if arm=='full' else -.5))])
            logits=np.array([[1,0,0],[0,1,0]]) if arm=='full' else np.array([[0,1,0],[0,1,0]])
            np.savez_compressed(account/'EMISSIONS.npz',timestamp=np.array([1,2]),logits=logits,
                                period=np.array(['TRAIN','INNER_A'],dtype='U16'))
            results[unit]=dict(seed=seed,feature_count=2,group_count=1,
                selectors={n:dict(selected_step=10) for n in ('train_best','guarded_p5')},accounts={'10':{}})
    spec=dict(identity='x',arms=[dict(name='full'),dict(name='drop')],config=dict(seeds=[7,19]))
    r=summarize(spec,results,paths)
    assert all(x['signs_agree']==dict(train=True,validation=True) for x in r['stability'])
    drops=[x for x in r['rows'] if x['unit'].startswith('drop')]
    assert all(x['paired_monitor_delta']==dict(train=1.,validation=.5) for x in drops)
    assert all(x['signal_disagreement']==dict(TRAIN=1.,INNER_A=0.) for x in drops)
    np.savez_compressed(tmp_path/'drop__7/accounts/10/EMISSIONS.npz',timestamp=np.array([2,3]))
    with pytest.raises(ValueError,match='EMISSION_SUPPORT'): summarize(spec,results,paths)
