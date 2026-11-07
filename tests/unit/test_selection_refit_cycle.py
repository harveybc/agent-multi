import json
import pytest
from tools.run_selection_refit_cycle import selection_receipt, validated_config
from tools.run_generalization_queue import write, sha


def test_selection_receipt_requires_both_passes_and_hash(tmp_path):
    result={'identity':'x','result':{'termination':'TWO_PASSES_COMPLETE','selected':['a']},
            'test_read':False,'original_support_preserved':True,'unmasked_after_pass_exact':True}
    write(tmp_path/'RESULT.json',result)
    write(tmp_path/'CLOSE.json',{'identity':'x','termination':'TWO_PASSES_COMPLETE',
                               'result_sha256':sha(tmp_path/'RESULT.json')})
    assert selection_receipt(tmp_path)['identity']=='x'
    result['result']['termination']='PARTIAL';write(tmp_path/'RESULT.json',result)
    with pytest.raises(ValueError,match='SELECTION_RECEIPT'):
        selection_receipt(tmp_path)


def test_cycle_config_is_immutable_on_resume(tmp_path):
    path=tmp_path/'config.json';write(path,{'output':str(tmp_path/'state')})
    validated_config(path)
    write(path,{'output':str(tmp_path/'state'),'changed':True})
    with pytest.raises(ValueError,match='CYCLE_CONFIG_CHANGED'):
        validated_config(path)


def test_chain_prepares_before_training_and_does_not_retrain_complete(tmp_path,monkeypatch):
    import tools.run_selection_refit_cycle as cycle
    import tools.run_structural_sac_refit as refit
    import tools.run_generalization_queue as queue
    selection=tmp_path/'selection';selection.mkdir()
    write(selection/'SPEC.private.json',{})
    result={'identity':'selected','result':{'termination':'TWO_PASSES_COMPLETE','selected':['a']},
            'test_read':False,'original_support_preserved':True,'unmasked_after_pass_exact':True}
    write(selection/'RESULT.json',result)
    write(selection/'CLOSE.json',{'identity':'selected','termination':'TWO_PASSES_COMPLETE',
                                'result_sha256':sha(selection/'RESULT.json')})
    cfg=dict(output=str(tmp_path/'state'),selection=str(selection),refit=str(tmp_path/'fit'),
             parent='parent',parent_runtime='parent_runtime',runtime='runtime',seed=7)
    path=tmp_path/'config.json';write(path,cfg);calls=[]
    def execute(argv,check):
        calls.append(argv)
        if 'prepare-selected' in argv:
            stage=cycle.Path(argv[argv.index('--root')+1]);stage.mkdir()
            write(stage/'PREFLIGHT.json',{})
        elif 'task' in argv:
            assert (tmp_path/'fit'/'PREFLIGHT.json').exists()
        else: pytest.fail('Unexpected subprocess')
    monkeypatch.setattr(cycle.subprocess,'run',execute)
    monkeypatch.setattr(refit,'authenticate',lambda p:{'identity':'fit','selection_identity':'selected',
                                                    'structural_round':{'features_after':1}})
    monkeypatch.setattr(queue,'collect',lambda root,seed:{'identity':'fit'})
    cycle.run(path)
    assert len(calls)==2
    assert json.loads((tmp_path/'state'/'STATUS.json').read_text())['stage']=='COMPLETE'
    cycle.run(path)
    assert len(calls)==2
