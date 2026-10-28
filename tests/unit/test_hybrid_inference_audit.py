import numpy as np
import pytest
from tools.hybrid_inference_audit import timestamp_uniform, categorical_command, frozen_probabilities


def test_common_random_numbers_are_timestamp_bound_and_reproducible():
    values=[timestamp_uniform(101,t) for t in range(1000)]
    assert values==[timestamp_uniform(101,t) for t in range(1000)]
    assert all(0<=u<1 for u in values)
    assert values!=[timestamp_uniform(202,t) for t in range(1000)]


def test_masked_sampling_does_not_force_directions_or_use_illegal_actions():
    p=frozen_probabilities([.1,.2,.7,0,0],[1,1,1,0,0])
    assert [categorical_command(p,u) for u in (.05,.2,.9)]==[0,1,2]
    assert np.array_equal(frozen_probabilities(p,[1,0,0,0,0]),[1,0,0,0,0])
    assert all(categorical_command([0,1,0,0,0],timestamp_uniform(101,t))==1 for t in range(100))
    with pytest.raises(ValueError):frozen_probabilities([0,1,0,0,0],[0,0,1,0,0])
