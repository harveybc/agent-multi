import pytest

from app.frozen_recipe_inputs import file_sha
from tools.recipe_sac_collect import bind_report
from tools.recipe_sac_queue import write


def test_report_binds_actual_attempt_checkpoint_and_native_values(tmp_path):
    path = tmp_path / 'GROUPED/attempt-2/step-25000/model.zip'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'model2')
    write(path.parent/'TRAIN.json', {'net_profit_usd':-1.,'policy_tensor_sha256':'policy2'})
    request = {'source_unit':'unit','report':{'experiment_key':'unit', 'tags':{
        'arm':'GROUPED','checkpoint':'25000','checkpoint_archive_sha256':file_sha(path),
        'policy_tensor_sha256':'policy2'},'metrics':[{'metric':'net_profit_usd','value':-1.,'split':'train'}]}}
    bound = bind_report(tmp_path,request)
    assert bound['source_unit']=='unit-attempt-2'
    assert bound['report']['tags']['attempt']=='attempt-2'
    assert request['source_unit']=='unit'
    request['report']['metrics'][0]['value']=1.
    with pytest.raises(ValueError,match='REPORT_NATIVE_VALUES'):
        bind_report(tmp_path,request)
    path.write_bytes(b'other model')
    with pytest.raises(ValueError,match='REPORT_CHECKPOINT_BINDING'):
        bind_report(tmp_path,request)
