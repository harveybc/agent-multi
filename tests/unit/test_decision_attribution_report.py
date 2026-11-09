from tools.summarize_decision_attribution import event_row


def test_report_does_not_confuse_greedy_loss_and_soft_critic_error():
    cell=dict(period='TRAIN',arm='fresh',updates=8)
    effect=dict(command=0,entries=0,closes=0)
    event=dict(old_signal=0,new_signal=2,utc='2020-01-01T00:00:00Z',
        prefix_old=dict(first_effect=effect),prefix_new=dict(first_effect=effect),
        chronological_contribution_usd=-10,q_gap=.2,reference_logits=[1,0,.9],
        candidate_logits=[.8,0,.9],isolated=dict(delta_usd=-11,
            stochastic=[dict(soft_delta=v) for v in [.1,.2,.3]]))
    r=event_row(cell,event)
    assert r['chronological_delta_usd']==-10 and r['isolated_delta_usd']==-11
    assert not r['positive_q_negative_all_three']
    event['isolated']['stochastic']=[dict(soft_delta=-v) for v in [.1,.2,.3]]
    assert event_row(cell,event)['positive_q_negative_all_three']
