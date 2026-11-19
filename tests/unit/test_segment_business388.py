import numpy as np
import torch
from tools.segment_business388 import path_metrics,stress
from tools.verify_business388 import scalar


def test_fork_and_gap_contract():
    stress('cpu')


def test_independent_scalar_parity():
    r=np.random.default_rng(388)
    c=1+np.cumsum(r.normal(0,.001,(32,12)),axis=1)
    o=c+r.normal(0,.0005,c.shape)
    v=np.stack([o,np.maximum(o,c)+.0001,np.minimum(o,c)-.0001,c],axis=-1)
    for d in (-1,1):
        actual=path_metrics(torch.from_numpy(v),d,8.4,17.1).numpy()
        expected=np.array([scalar(row,d,8.4,17.1) for row in v])
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-10,equal_nan=True)
