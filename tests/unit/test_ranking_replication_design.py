from types import SimpleNamespace

import numpy as np
import pytest

from app.action_ranking_contrast import replication_design, shard_owner, calibration_target


def test_default_and_replica_seeds_are_explicit_and_distinct():
    default=replication_design('original');a=replication_design('r19');b=replication_design('r43')
    assert default['update_seed']==7 and default['anchor_seed']==41
    assert a['update_seed']==19 and b['update_seed']==43
    assert a['seeds']!=b['seeds']!=default['seeds']
    assert 'paired_stochastic_reward' in a['arms']
    with pytest.raises(ValueError):replication_design('unknown')


def test_weighted_ownership_covers_rows_once_and_defaults_are_unchanged():
    for n in (0,1,6,7,379,384):
        groups=[{i for i in range(n) if shard_owner(i,[5,2])==s} for s in (0,1)]
        assert not groups[0]&groups[1] and groups[0]|groups[1]==set(range(n))
    assert [shard_owner(i,[1,1]) for i in range(10)]==[i%2 for i in range(10)]
    assert sum(shard_owner(i,[5,2])==0 for i in range(700))==500
    with pytest.raises(ValueError):shard_owner(1,[0,2])


def test_plain_target_cannot_accidentally_use_entropy_or_greedy():
    g=np.full((4,3),10.);s=np.full((4,4,3),3.);p=np.full((4,4,3),2.)
    assert np.array_equal(calibration_target('paired_greedy',g,s,p),g)
    assert np.array_equal(calibration_target('paired_stochastic',g,s,p),s.mean(1))
    assert np.array_equal(calibration_target('paired_stochastic_reward',g,s,p),p.mean(1))
    with pytest.raises(ValueError):calibration_target('unknown',g,s,p)


def test_replica_forwarded_to_worker_cli():
    from tools.run_action_ranking_contrast import shared_args
    args=SimpleNamespace(**{n:'fixture' for n in ('source','source_runtime','probe','attribution','calibration','output')},replica='r43')
    command=shared_args(args)
    assert command[command.index('--replica')+1]=='r43'
