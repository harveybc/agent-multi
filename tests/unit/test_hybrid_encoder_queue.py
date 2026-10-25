import pytest
import torch
from app.frozen_recipe_inputs import file_sha
from tools.recipe_sac_queue import write
from tools.hybrid_encoder_queue import study_design, authenticate, build_model
from tools.qualify_hybrid_learning import ObservableSignal
from tests.unit.test_frozen_recipe_inputs import recipe


def test_frozen_matrix_and_initial_function_parity():
    d=study_design()
    assert d['periods']==['TRAIN','INNER_A']
    assert len(d['units'])==6 and len(set(d['units']))==6
    assert d['training']['total_timesteps']==102400
    env=ObservableSignal()
    a=build_model(env,recipe(),7,'cpu','SHARED',n_steps=8)
    b=build_model(env,recipe(),7,'cpu','INDEPENDENT',n_steps=8)
    for key,value in a.policy.state_dict().items():
        assert torch.equal(value,b.policy.state_dict()[key]),key
    obs,_=env.reset(seed=7);t,_=a.policy.obs_to_tensor(obs)
    x,v=a.policy._parts(t);y,w=b.policy._parts(t)
    assert torch.equal(x.cat.probs,y.cat.probs) and torch.equal(v,w)


def test_receipt_cannot_resume_wrong_unit_or_tampered_artifact(tmp_path):
    write(tmp_path/'QUALIFICATION.json',{'passed':True})
    result=tmp_path/'RESULT.json'
    write(result,dict(state='COMPLETE',spec_sha256='abc',unit='QUAL_7_0',
        artifacts={'QUALIFICATION.json':file_sha(tmp_path/'QUALIFICATION.json')}))
    authenticate(result,'abc','QUAL_7_0')
    with pytest.raises(ValueError):authenticate(result,'abc','INDEPENDENT_s7')
    write(tmp_path/'QUALIFICATION.json',{'passed':False})
    with pytest.raises(ValueError):authenticate(result,'abc','QUAL_7_0')


def test_empty_financial_receipt_is_not_completion(tmp_path):
    result=tmp_path/'RESULT.json'
    write(result,dict(state='COMPLETE',spec_sha256='abc',unit='SHARED_s7',artifacts={}))
    with pytest.raises(ValueError,match='FINANCIAL_REPORTS_REQUIRED'):
        authenticate(result,'abc','SHARED_s7')


def test_failed_qualification_retries_bounded_and_never_trains_financial(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import tools.hybrid_encoder_queue as q
    from tools.recipe_sac_queue import read
    spec=dict(study=q.study_design(),max_attempts=2,task_timeout_seconds=1)
    write(tmp_path/'SPEC.private.json',spec)
    monkeypatch.setattr(q,'verify',lambda _:spec)
    calls=[]
    def fail(command,**kwargs):
        calls.append(command[command.index('--unit')+1]);return SimpleNamespace(returncode=2)
    monkeypatch.setattr(q.subprocess,'run',fail)
    q.tick(tmp_path)
    assert calls==['QUAL_7_0','QUAL_7_0']
    assert read(tmp_path/'CLOSE.json')['state']=='FAILED_QUALIFICATION'
    q.tick(tmp_path)
    assert len(calls)==2


@pytest.mark.parametrize('arm',['SHARED','INDEPENDENT'])
def test_instrumentation_does_not_change_training(arm,tmp_path):
    from tools.recipe_hybrid_queue import callback
    from tools.recipe_sac_queue import tensor_hash
    torch.set_num_threads(1)
    plain=build_model(ObservableSignal(),recipe(),7,'cpu',arm,n_steps=16)
    plain.learn(64,callback=callback(None,'plain',0))
    observed=build_model(ObservableSignal(),recipe(),7,'cpu',arm,n_steps=16)
    observed.diagnostic_output=str(tmp_path/'trace')
    observed.learn(64,callback=callback(None,'observed',0))
    assert tensor_hash(plain)==tensor_hash(observed)
    assert len(list((tmp_path/'trace').glob('UPDATE_*.json')))==4
