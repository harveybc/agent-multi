import torch
import pytest

from app.actor_gradient_coupling import decompose, first_adamw_delta
from app.selected_refinement_diagnosis import adaptive_parts, action_quality, optimal_mass_derivative, validate_diagnostic_source, decision_strata, decision_behavior


def test_adaptive_self_cross_decay_reconstruct_step():
    torch.manual_seed(91)
    z = torch.randn(4, 3, dtype=torch.double)
    q = torch.randn_like(z)
    j = torch.randn(4, 3, 7, dtype=torch.double)
    parts, weights, jm = decompose(z, q, .1, j, torch.tensor([0, 0, 1, 2]))
    theta = torch.randn(7, dtype=torch.double)
    got = adaptive_parts(parts['total'], weights, jm, theta, .001, .02, 1e-8)
    adaptive, decay = first_adamw_delta(theta, parts['total']['batch_gradient'], .001, .02, 1e-8)
    torch.testing.assert_close(got['self'] + got['cross'], jm @ adaptive)
    torch.testing.assert_close(got['total'], jm @ (adaptive + decay))
    assert torch.equal(got['self'][3], torch.zeros(2, dtype=torch.double))


def test_shared_head_can_reverse_helpful_local_buy():
    z = torch.zeros(2, 3, dtype=torch.double)
    q = torch.tensor([[0., 1., 0.], [0., -10., 0.]], dtype=torch.double)
    j = torch.eye(3, dtype=torch.double).expand(2, -1, -1)
    parts, weights, jm = decompose(z, q, 0., j, torch.tensor([0, 1]))
    got = adaptive_parts(parts['total'], weights, jm, torch.zeros(3, dtype=torch.double), .001, 0., 1e-8)
    assert parts['total']['local_logit'][0, 0] > 0
    assert got['self'][0, 0] > 0 and got['total'][0, 0] < 0


def test_optimal_action_not_just_positive_gap_and_ties():
    native = torch.tensor([[0., 1., 2.], [0., 0., 0.]], dtype=torch.double)
    q = torch.tensor([[0., 3., 1.], [1., 1., 1.]], dtype=torch.double)
    got = action_quality(native, q)
    assert got['optimal_count'] == 1
    assert got['mean_regret'] == .5
    assert got['beneficial_gap_sign_errors'] == 0
    assert got['unique_native_optimum'] == 1


def test_native_ties_use_joint_optimal_mass_not_first_argmax():
    native = torch.tensor([[1.,1.,0.],[0.,0.,0.],[0.,2.,1.]],dtype=torch.double)
    change = torch.tensor([[-.1,.2,-.1],[-.1,.2,-.1],[.1,-.2,.1]],dtype=torch.double)
    torch.testing.assert_close(optimal_mass_derivative(native,change),
                               torch.tensor([.1,0.,-.2],dtype=torch.double))
    with pytest.raises(ValueError):optimal_mass_derivative(native,change*float('nan'))


def test_calibration_and_refinement_sources_cannot_be_interchanged():
    validate_diagnostic_source('selected_input_critic_calibration.v1',True)
    validate_diagnostic_source('selected_input_bounded_refinement.v1',False)
    with pytest.raises(ValueError):validate_diagnostic_source('selected_input_critic_calibration.v1',False)
    with pytest.raises(ValueError):validate_diagnostic_source('selected_input_bounded_refinement.v1',True)


def record(position, reward, cash=None, role='fit', allowed=False, executable=True):
    close = 2 if position == 1 else 1
    cash = reward if cash is None else cash
    return dict(position=position, role=role, allowed=allowed, outcomes=[
        dict(reward_return=r, net_usd=c, first_effect=dict(count_changes={
            'close_AGENT_CLOSE':int(position != 0 and a == close and executable)}))
        for a,(r,c) in enumerate(zip(reward,cash))])


def test_strata_partition_and_blocked_opening_are_distinct():
    rows=[record(0,[0,1,2],allowed=True),record(0,[0,0,0]),
          record(1,[1,1,0]),record(-1,[1,0,1],role='monitor')]
    groups=decision_strata(rows)
    assert sum(int(m.sum()) for m in groups.values())==len(rows)
    assert groups['fit:flat_allowed'].tolist()==[True,False,False,False]
    assert groups['fit:flat_blocked'].tolist()==[False,True,False,False]
    with pytest.raises(ValueError):decision_strata([dict(rows[0],position=3)])


def test_close_and_hold_aliases_in_both_directions():
    rows=[record(1,[0,0,2]),record(-1,[0,2,0]),record(1,[2,2,0]),record(-1,[2,0,2])]
    result=decision_behavior(rows,torch.tensor([[0,0,1],[0,1,0],[0,1,0],[0,0,1]]))
    assert result['reward_quality']['optimal_count']==4
    assert result['closing']['helpful_closes']==2
    assert result['closing']['missed_beneficial_closes']==0
    wrong=decision_behavior(rows,torch.tensor([[1,0,0],[0,0,1],[0,0,1],[0,1,0]]))
    assert wrong['closing']['missed_beneficial_closes']==2
    assert wrong['closing']['harmful_closes']==2


def test_preempted_close_and_tied_outcomes_are_not_harmful_closes():
    rows=[record(1,[1,1,1]),record(-1,[2,0,2],executable=False)]
    result=decision_behavior(rows,torch.tensor([[0,0,1],[0,1,0]]))
    assert result['closing']['tie_closes']==1
    assert result['closing']['nonexecuted_close_signals']==1
    assert result['closing']['harmful_closes']==0


def test_reward_cash_disagreement_and_empty_groups_are_explicit():
    rows=[record(0,[0,2,1],[0,1,2],allowed=True)]
    result=decision_behavior(rows,torch.tensor([[0,1,0]]))
    assert result['reward_optimal_cash_suboptimal']==1
    assert result['reward_cash_optimum_set_disagreements']==1
    assert decision_behavior([],torch.empty(0,3))['rows']==0
    with pytest.raises(ValueError):decision_behavior(rows,torch.ones(1,2))


def test_cash_quality_and_cash_tie_set_share_the_same_usd_tolerance():
    rows=[record(0,[0,1,2],[0,1,1+1e-7],allowed=True)]
    result=decision_behavior(rows,torch.tensor([[0,1,0]]))
    assert result['cash_quality']['optimal_count']==1
    assert result['cash_optimal_reward_suboptimal']==1


def test_diagnostic_witness_legacy_strict_guard_does_not_cover_hold_tie():
    from app.fit_decision_preservation import protected_actions, margins, preservation
    z=torch.tensor([[0.,3.,0.],[2.,1.,0.]])
    native=torch.tensor([[0.,2.,1.],[1.,1.,0.]])
    rows,actions=protected_actions(z,native,[{'role':'fit'},{'role':'fit'}])
    assert rows.tolist()==[0]
    changed=z.clone();changed[1]=torch.tensor([0.,0.,3.])
    assert preservation(changed,margins(z,rows,actions),rows,actions)['passed']
    assert action_quality(native,z)['optimal_count']==2
    assert action_quality(native,changed)['optimal_count']==1


def test_diagnostic_witness_legacy_error_target_omits_bad_close_with_two_good_holds():
    from app.targeted_fit_errors import error_targets
    z=torch.tensor([[0.,0.,3.]])
    native=torch.tensor([[1.,1.,0.]])
    target,weight=error_targets(z,native,[{'role':'fit'}])
    assert action_quality(native,z)['optimal_count']==0
    assert target.tolist()==[-1] and weight.tolist()==[0.]
