import pytest


def test_window_normalization_does_not_hide_scientific_changes():
    from tools.audit_window_evidence import scientific_config,differences
    base=dict(window_size=13,commission=.00005,financing_enabled=False,include_price_window=False)
    newer=dict(base,window_size=11,temporal_core_layers=1,save_model='/new/model.zip')
    assert scientific_config(base)==scientific_config(newer)
    newer['commission']=0
    assert differences(scientific_config(base),scientific_config(newer))==['commission']
    newer=dict(base,temporal_core_layers=2)
    assert differences(scientific_config(base),scientific_config(newer))==['temporal_core_layers']


def test_operational_normalization_keeps_seed_costs_and_gpu_assignment():
    from tools.audit_window_evidence import scientific_config
    a=dict(train_seed=7,queue_execution=dict(device='cuda',gpu_evidence=dict(
        assigned_gpu_uuid='same',path='/old',sha256='old',max_age_seconds=86400)))
    b=dict(train_seed=7,queue_execution=dict(device='cuda',gpu_evidence=dict(
        assigned_gpu_uuid='same',path='/new',sha256='new',max_age_seconds=86400)))
    assert scientific_config(a)==scientific_config(b)
    b['train_seed']=17
    assert scientific_config(a)!=scientific_config(b)


def test_relocation_refuses_paths_outside_source(tmp_path):
    from tools.audit_window_evidence import relocated
    assert relocated(tmp_path,'/source','/source/a.json')==tmp_path/'a.json'
    with pytest.raises(ValueError):relocated(tmp_path,'/source','/other/a.json')


def test_policy_identity_checks_frozen_evaluation_and_train_digest():
    from tools.audit_window_evidence import policy_identity
    assert policy_identity(dict(split='validation'),dict(static_policy=dict(before='a',after='a')))=='a'
    with pytest.raises(ValueError):
        policy_identity(dict(split='validation'),dict(static_policy=dict(before='a',after='b')))
    with pytest.raises(ValueError):
        policy_identity(dict(split='train',policy_tensor_sha256='b'),dict(checkpoint_policy_tensor_sha256='a'))


def test_business_trace_ignores_only_identifiers(tmp_path):
    from tools.audit_window_evidence import business_trace_hash
    a=tmp_path/'a.csv';b=tmp_path/'b.csv'
    a.write_text('timestamp,run_id,episode_id,equity\n2024-01-01,a,a,100\n')
    b.write_text('timestamp,run_id,episode_id,equity\n2024-01-01,b,b,100\n')
    assert business_trace_hash(a)==business_trace_hash(b)
    b.write_text('timestamp,run_id,episode_id,equity\n2024-01-01,b,b,101\n')
    assert business_trace_hash(a)!=business_trace_hash(b)


def test_initialization_probe_does_not_train_and_shapes_change_critics():
    from tools.audit_window_evidence import initialization_probe
    rows=initialization_probe([11,13])['runs']
    assert all(r['gradient_updates']==0 for r in rows)
    assert rows[0]['actor_extractor_sha256']==rows[1]['actor_extractor_sha256']
    assert rows[0]['critic_extractor_sha256']!=rows[1]['critic_extractor_sha256']
    assert (rows[1]['actor_parameters'],rows[1]['critic_parameters'])==(1062,1638)
