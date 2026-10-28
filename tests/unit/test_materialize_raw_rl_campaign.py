import json
from pathlib import Path

import pytest

from tools.materialize_raw_rl_campaign import materialize


DESIGN = Path(__file__).resolve().parents[2]/'examples/config/rl_raw_eurusd_static.v1.json'


def test_frozen_raw_design_has_business_rules_small_models_and_finite_budget():
    body = json.loads(DESIGN.read_text())
    cfg = body['common_config']
    assert body['scope'] == 'EURUSD_CORE_PRICES_ONLY_NOT_FINAL_FEATURE_SELECTION'
    assert body['arms'][:2] == ['SAC_FLAT', 'SAC_PER_FEATURE_CONV1D']
    assert body['arms'][2:] == ['NO_TRADE', 'PASSIVE_LONG', 'PASSIVE_SHORT', 'RANDOM']
    assert body['refit_cadence'] == 'NONE'
    assert cfg['window_size'] == 13 and cfg['observation_stride_bars'] == 2
    assert cfg['total_timesteps'] == cfg['training_transition_cap'] == 100000
    assert cfg['net_arch'] == [32, 32]
    assert cfg['session_exposure_policy']['forced_flatten_hours'] == 12
    assert cfg['session_exposure_policy']['reopen_min_hours'] == 12
    assert cfg['max_entries_per_day'] == cfg['execution_envelope']['max_entries_per_day'] == 1
    assert cfg['commission'] > 0 and cfg['slippage_perc'] > 0
    assert cfg['financing_enabled'] is True and cfg['financing_annual_rate'] > 0
    assert cfg['timestamp_semantics'] == 'completed_bar_available_at'
    assert cfg['include_price_window'] is False


def test_invalid_design_cannot_create_campaign(tmp_path):
    design = tmp_path/'design.json'
    design.write_text('{}')
    output = tmp_path/'output'
    with pytest.raises(ValueError, match='STATIC_RAW_BASELINE'):
        materialize(design, tmp_path, tmp_path, tmp_path, tmp_path, output, tmp_path, 'GPU-test')
    assert not output.exists()
