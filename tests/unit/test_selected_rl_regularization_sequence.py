import copy
import json
from pathlib import Path

import pytest

from tests.unit.test_selected_rl_comparison import fixture,write


def test_prepare_four_configs_chain_units_and_idempotence(tmp_path):
    from agent_plugins.paired_early_stopping import SETTINGS
    from agent_plugins.sac_regularization import PROFILES
    from tools import selected_rl_regularization_sequence as sequence
    path,request,_=fixture(tmp_path)
    request.update(schema='selected_rl_comparison_request.v4',leverage=50,
        order_budget=dict(rel_volume=.1,min_order_volume=1,max_order_volume=1e12,volume_step=1),
        early_stopping=SETTINGS)
    request['execution']['gpu_evidence']=dict(assigned_gpu_uuid='GPU-'+'a'*36)
    write(path,request)
    root=tmp_path/'sequence';previous=tmp_path/'previous';pin='f'*64
    spec=sequence.prepare(path,root,previous,pin)
    assert [n['profile'] for n in spec['nodes']]==list(PROFILES)
    old=(root/'SEQUENCE.json').read_bytes()
    assert sequence.prepare(path,root,previous,pin)==spec
    assert old==(root/'SEQUENCE.json').read_bytes()
    for node in spec['nodes']:
        cfg=json.loads((Path(node['root'])/'CONTRACT.json').read_text())
        assert cfg['common_config']['sac_regularization']==PROFILES[node['profile']]
        assert cfg['common_config']['paired_early_stopping']==SETTINGS
        assert cfg['common_config']['total_timesteps']==100000
        assert node['predecessor_root']==str(previous)
        assert node['predecessor_sha256']==pin
        previous=Path(node['root']);pin=node['contract_sha256']
    units=sequence.render(spec,root/'units',tmp_path/'private.env')
    assert len(units)==10
    first=(root/'units/rl-selected-reg-dropout005.service').read_text()
    assert 'tools.verify_rl_predecessor' in first
    assert first.count('ExecCondition=')==2
    assert 'agent_plugins.sac_regularization' in first
    assert 'CUDA_VISIBLE_DEVICES=GPU-' in first
    assert json.dumps('--predecessor-sha256')+' '+json.dumps('f'*64) in first
    assert 'MemoryMax=6G' in first
    assert 'rl-selected-reg-dropout005.service' in (root/'units/rl-selected-reg-dropout005.timer').read_text()
    publisher=(root/'units/rl-selected-reg-publication.service').read_text()
    assert 'selected_rl_regularization_sequence' in publisher and 'publish' in publisher
    changed=copy.deepcopy(request);changed['leverage']=100;write(path,changed)
    with pytest.raises(ValueError):sequence.prepare(path,root,tmp_path/'previous','f'*64)


def test_closed_node_pin_is_rechecked_before_publication_or_status(tmp_path,monkeypatch):
    from tools import selected_rl_regularization_sequence as sequence
    root=tmp_path/'node';root.mkdir();(root/'CONTRACT.json').write_text('{}')
    spec=dict(schema='selected_rl_regularization_sequence.v1',nodes=[dict(
        profile='dropout005',root=str(root),contract_sha256='a'*64)])
    with pytest.raises(ValueError,match='SEQUENCE_CONTRACT_DRIFT'):
        sequence.status(spec)
