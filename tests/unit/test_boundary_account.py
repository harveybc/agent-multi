import pytest

from tools.replay_boundary_account import fixed_candidates


def test_account_list_uses_only_prequalified_fit_witnesses():
    probe=dict(source_restored_exactly=True,account_evaluated=False,results=[
        dict(panel_index=63,target=2,qualified=True),dict(panel_index=2,target=1,qualified=False)])
    assert fixed_candidates(probe)==[(63,2)]
    with pytest.raises(ValueError,match='FIT_ONLY'):
        fixed_candidates(dict(probe,account_evaluated=True))
    with pytest.raises(ValueError,match='WITNESS_LIST'):
        fixed_candidates(dict(probe,results=[probe['results'][0]]*2))
    with pytest.raises(ValueError,match='WITNESS_LIST'):
        fixed_candidates(dict(probe,results=[]))
