import pytest

from tools.run_bounded_functional_actor import calibration_only_config


def test_calibration_only_disables_actor_updates_without_changing_budget():
    original = dict(modes=['calibrated_adamw'], points=[0, 1, 8],
                    primary_endpoint=8, seeds=[19, 43], critic_updates=4096)
    result = calibration_only_config(original, True, True)
    assert result['modes'] == []
    assert result['points'] == [0]
    assert result['actor_updates'] == result['primary_endpoint'] == 0
    assert result['critic_updates'] == 4096
    assert result['seeds'] == [19, 43]
    assert original['modes'] == ['calibrated_adamw']


def test_default_preserves_existing_contract_and_invalid_combination_fails():
    original = dict(modes=['last_layer'])
    assert calibration_only_config(original, False, False) is original
    with pytest.raises(ValueError, match='REQUIRES_REFINE_EXPORT'):
        calibration_only_config(original, False, True)
