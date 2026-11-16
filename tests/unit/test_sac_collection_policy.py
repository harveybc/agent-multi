import numpy as np
import pytest

from tools.diagnose_sac_collection_policy import sampled_signals


def test_categorical_cdf_and_invalid_rows():
    logits=np.log(np.tile([.2,.3,.5],(5,1)))
    logits[-1]=np.nan
    uniforms=np.array([.1,.25,.6,.999,.5])
    before=logits.copy()
    out=sampled_signals(logits,uniforms)
    assert out[0][:4].tolist()==[0,1,2,2]
    assert not out[3][-1]
    assert np.array_equal(logits,before,equal_nan=True)
    assert np.allclose(out[1][:4],-sum(p*np.log(p) for p in (.2,.3,.5)))


def test_uniform_quantiles_recover_exact_frequencies():
    logits=np.log(np.tile([.2,.3,.5],(1000,1)))
    actions=sampled_signals(logits,(np.arange(1000)+.5)/1000)[0]
    assert np.bincount(actions).tolist()==[200,300,500]
    with pytest.raises(ValueError):sampled_signals(logits,np.ones(1000))
