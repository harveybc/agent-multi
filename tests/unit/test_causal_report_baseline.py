import numpy as np
import pytest

from app.causal_report_baseline import sampled_prefix_report


def sample(prefix,**kwargs):
    return sampled_prefix_report(prefix,now=10,step=1,deadline=34,
        spreads=np.full(24,.0001),terminal_spread=.0005,seed=19,**kwargs)


def prefix():
    close=1.1+.001*np.sin(np.arange(100))
    return np.column_stack((close,close+.0003,close-.0003,close))


def test_only_passed_prefix_matters_and_sampling_replays():
    data=prefix(); first=sample(data[:30])
    data[30:]+=50
    second=sample(data[:30])
    assert np.array_equal(first.paths,second.paths)
    assert first.paths.shape==(32,24,4) and first.provenance=='causal'
    assert not np.array_equal(first.paths[0],first.paths[1])
    assert np.all(first.paths[:,0,0]==data[29,3])


def test_reflected_prefix_reflects_report_without_clipping():
    data=prefix()
    a=sample(data); b=sample(2.2-data[:,[0,2,1,3]])
    assert np.allclose(b.paths,2.2-a.paths[:,:,[0,2,1,3]],rtol=0,atol=1e-12)


def test_no_three_sample_oracle_disguised_as_baseline():
    with pytest.raises(ValueError,match='BASELINE_CONTRACT'):
        sample(prefix(),samples=3)
