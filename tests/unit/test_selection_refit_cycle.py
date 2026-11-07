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
