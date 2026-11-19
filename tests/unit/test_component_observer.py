import pytest
import torch

from app.component_observer import matrix_ranks, selected_epochs, policy_stats


def test_epoch_selection_does_not_use_returns():
    rows=[{'epoch':i,'metrics':{'anything':-i}} for i in range(1,44)]
    assert selected_epochs(rows)==[1,20,40,43]
    assert selected_epochs([{'epoch':102},{'epoch':103}])==[102,103]
    assert selected_epochs([])==[]


def test_matrix_ranks_known_cases():
    assert matrix_ranks(torch.eye(4))==pytest.approx([4.])
    assert matrix_ranks(torch.ones(4,4))==pytest.approx([1.])
    assert matrix_ranks(torch.zeros(4,4))==[0.]
    assert matrix_ranks(7*torch.eye(4))==pytest.approx([4.])
    assert matrix_ranks(torch.stack([torch.eye(4),torch.ones(4,4)]))==pytest.approx([4.,1.])
    assert matrix_ranks(torch.ones(4))==[]


def test_policy_probe_separates_flips_margins_and_critic_disagreement():
    logits=torch.tensor([[4.,0.,0.],[0.,0.,4.]])
    q=torch.stack([logits,logits])
    result=policy_stats(logits,q,torch.tensor([0,1]))
    assert result['action_flip_fraction']==.5
    assert result['critic_actor_disagreement']==0.
    assert result['probability_margin_median']>.9
    assert result['near_tie_fraction']==0.
