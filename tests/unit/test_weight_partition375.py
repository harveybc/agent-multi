import pytest
from app.sac_training_reliability import module_key


@pytest.mark.parametrize('prefix,role',[('actor_network_params','actor'),('qvalue_network_params','critic')])
def test_patch_adapter_is_not_fusion_or_other(prefix,role):
    for field in ('gate','patch_projection.weight','encoder.self_attn.in_proj_weight','output.bias'):
        assert module_key(prefix+'.encoder.post_fusion.'+field)==role+'.attention'
    assert module_key(prefix+'.encoder.fusion.conv.weight')==role+'.fusion'
