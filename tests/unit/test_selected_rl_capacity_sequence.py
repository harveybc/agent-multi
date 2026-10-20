import copy
import json
from pathlib import Path

import pytest

from tests.unit.test_selected_rl_comparison import fixture, write


def paired_request(tmp_path):
    from agent_plugins.paired_early_stopping import SETTINGS
    path, request, _ = fixture(tmp_path)
    request.update(schema='selected_rl_comparison_request.v4', leverage=50,
        order_budget=dict(rel_volume=.1, min_order_volume=1, max_order_volume=1e12, volume_step=1),
        early_stopping=SETTINGS)
    request['execution']['gpu_evidence'] = dict(assigned_gpu_uuid='GPU-' + 'a'*36)
    write(path, request)
    return path, request


def test_capacity_only_changes_architecture_and_binds_restart(tmp_path):
    from tools import selected_rl_comparison as selected
    path, request = paired_request(tmp_path)
    old, _ = selected._build(path, tmp_path/'original')
    request.update(schema='selected_rl_comparison_request.v6', net_arch=[8])
    write(path, request)
    body, _ = selected._build(path, tmp_path/'capacity')
    expected = copy.deepcopy(old['common_config']); expected['net_arch'] = [8]
    assert body['common_config'] == expected
    assert body['splits'] == old['splits']
    assert body['selected_input_binding'] == old['selected_input_binding']
    root = tmp_path/'capacity'
    selected.materialize(path, root)
    selected.validate(root)
    request['net_arch'] = [16]; write(path, request)
    with pytest.raises(ValueError): selected.validate(root)


@pytest.mark.parametrize('arch', [[], [0], [True], [8, 8], [32, 32], [1024], '8'])
def test_undeclared_capacity_refused(tmp_path, arch):
    from tools import selected_rl_comparison as selected
    path, request = paired_request(tmp_path)
    request.update(schema='selected_rl_comparison_request.v6', net_arch=arch)
    write(path, request)
    with pytest.raises(ValueError): selected._build(path, tmp_path/'out')


def test_no_regularization_or_extra_overrides(tmp_path):
    from tools import selected_rl_comparison as selected
    path, request = paired_request(tmp_path)
    request.update(schema='selected_rl_comparison_request.v6', net_arch=[8], learning_rate=.1)
    write(path, request)
    with pytest.raises(ValueError): selected._build(path, tmp_path/'out')
    request.pop('learning_rate'); request['regularization'] = {}
    write(path, request)
    with pytest.raises(ValueError): selected._build(path, tmp_path/'out')


def test_sequence_and_units_are_pinned_ordered_and_idempotent(tmp_path):
    from tools import selected_rl_capacity_sequence as capacity
    path, request = paired_request(tmp_path)
    root = tmp_path/'sequence'; previous = tmp_path/'closed'; pin = 'f'*64
    spec = capacity.prepare(path, root, previous, pin)
    assert [n['net_arch'] for n in spec['nodes']] == [[8], [16], [16, 16]]
    assert capacity.prepare(path, root, previous, pin) == spec
    assert len({json.loads(Path(n['request_file']).read_text())['publication']['experiment_set_key'] for n in spec['nodes']}) == 3
    for node in spec['nodes']:
        assert node['predecessor_root'] == str(previous)
        assert node['predecessor_sha256'] == pin
        body = json.loads((Path(node['root'])/'CONTRACT.json').read_text())
        assert body['common_config']['net_arch'] == node['net_arch']
        assert 'sac_regularization' not in body['common_config']
        previous = Path(node['root']); pin = node['contract_sha256']
    units = capacity.render(spec, root/'units', tmp_path/'private.env', root/'SEQUENCE.json')
    assert len(units) == 8
    first = (root/'units/rl-selected-capacity-n8.service').read_text()
    publisher = (root/'units/rl-selected-capacity-publication.service').read_text()
    assert 'tools.verify_rl_predecessor' in first
    assert 'MemoryMax=6G' in first and 'StartLimitIntervalSec=0' in first
    assert 'tools.selected_rl_capacity_sequence' in first and '--smoke-root' in first
    assert 'tools.selected_rl_capacity_sequence' in publisher and 'publish' in publisher
    assert 'selected_rl_regularization_sequence' not in publisher
